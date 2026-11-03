"""FOR-19 fixed baseline contracts, determinism, visibility and CLI wiring."""

import hashlib
import json
from dataclasses import FrozenInstanceError, replace

import pytest

from formicmind.actions import ActionMaskV2, ActionV2, get_action_mask, validate_action
from formicmind.baselines import (
    BaselineConfig,
    baseline_names,
    baseline_selectors,
    baseline_spec,
    make_baseline,
    _decode_u_integer,
)
from formicmind.cli import EXIT_OK, main as sim_main
from formicmind.decoder import decode_action_logits
from formicmind.evaluation import controller_pair, evaluate_episodes
from formicmind.events import OrderEvent, Phase
from formicmind.observation import OBS_V2_OFFSETS, OwnGlobalIndex, ObservationV2, build_observation, f32
from formicmind.policy import decide
from formicmind.rng import new_stream
from formicmind.reset import reset
from formicmind.rules import Q_TOTAL, default_config
from formicmind.snapshot import begin_policy_phase
from formicmind.tick import ScriptedController, step_tick

from helpers import fresh_state, mirror_state

V2_BASELINES = tuple(f"{name}:v2" for name in baseline_names())
TEST_BASELINES = (*baseline_names(), *V2_BASELINES)


def _observations(state):
    return build_observation(begin_policy_phase(state), 0)


def _edit(obs: ObservationV2, block: str, edits: dict[int, float]) -> ObservationV2:
    start, length = OBS_V2_OFFSETS[block]
    values = list(obs.values)
    for index, value in edits.items():
        assert 0 <= index < length
        values[start + index] = f32(float(value))
    return ObservationV2(tuple(values))


def _threat_observation(obs: ObservationV2, *, current: bool, age: float = 0.0) -> ObservationV2:
    # Five enemy soldiers in local N1, encoded with the ObservationV2 u(x, 64) rule.
    obs = _edit(obs, "enemy_count_last", {3: 5 / 69})
    obs = _edit(obs, "enemy_known", {1: 1.0})
    obs = _edit(obs, "enemy_current", {1: float(current)})
    return _edit(obs, "enemy_age_norm", {1: age})


def _action(policy_name, observation, seed=11):
    spec = baseline_spec(policy_name)
    policy = make_baseline(policy_name)
    return decide(policy, observation, new_stream(seed, f"policy:{spec.policy_id}:0")).action


def test_baselines_v1_logits_match_pre_v2_golden_vectors():
    observation = _observations(fresh_state(60))
    mask = get_action_mask(observation)
    expected = {
        "balanced": "527601be360c45091ef124ba0ae5918b9ad578187a5e499e7f814b85bacd0428",
        "raider": "0d6761f96714f7fb5cbe69a633db4155c94b3965e5ea8b844af9a433fedf1e71",
        "random": "6549d9f13e5c2b44c26dfd1517808be48e81d3d96053024645c53e8b0d19e472",
        "rush": "2f6a487664d376ebe5896faafe660b63ef202df9b52ad6582c4cae24bdc1566d",
        "turtle": "64d4d550faf2204bb28ea11904d3e0ca5c229f376c6c88b3744b546e78012a41",
    }
    actual = {}
    for name in baseline_names():
        logits = make_baseline(name)(observation, mask, new_stream(81, f"policy:{name}-v1:0"))
        actual[name] = hashlib.sha256(json.dumps(logits.values, separators=(",", ":")).encode()).hexdigest()
    assert actual == expected


def test_baselines_v1_paired_match_trajectories_match_pre_v2_golden_hashes():
    rows, _ = evaluate_episodes(
        episodes=2, seed_start=20261103, scenario=None, a="balanced", b="rush",
        paired_seat=True, max_ticks=90,
    )
    assert [(row["state_hash"], row["checkpoint_hash"]) for row in rows] == [
        ("393940d2556f54812450c8eab94fd265473f50e2c40b465059b08e3aa757d4de",
         "d3f5b855abc696a4a352d39fdec3fb5080ecdcb397e109947db7eb0dee46361e"),
        ("3cd21d7a37f76d3942d4dc1fcb22ed7630b00fb9064f00b4d172608898eaa2a8",
         "557731db3cad98581fd3fa1f5e149e35aaf060465d988096304ac3ef2fc1b19a"),
        ("393940d2556f54812450c8eab94fd265473f50e2c40b465059b08e3aa757d4de",
         "34222226c066b79cb20f8ebb7e9d7c2b1067807121c239c85d168cb17917fc30"),
        ("3cd21d7a37f76d3942d4dc1fcb22ed7630b00fb9064f00b4d172608898eaa2a8",
         "039ad6b7178a58224bbee47c7ac7b11c7cb90cef3a8b28b28d187d930ad73ac2"),
    ]


@pytest.mark.parametrize("scale,maximum", [(10.0, 512), (32.0, 2048), (64.0, 4096),
                                           (250.0, 32768), (300.0, 32768), (1200.0, 32768)])
def test_float32_normalized_integer_metrics_recover_exactly_in_supported_range(scale, maximum):
    for original in range(maximum + 1):
        normalized = f32(original / (original + scale))
        estimate = scale * normalized / max(1e-9, 1.0 - normalized)
        assert abs(estimate - original) < 0.5
        assert _decode_u_integer(normalized, scale) == original
    assert _decode_u_integer(f32((maximum + 100) / (maximum + 100 + scale)), scale) == maximum + 1


def test_v2_integer_recovery_does_not_round_continuous_observation_fields():
    observation = _edit(_observations(fresh_state()), "own_global", {
        8: f32(0.1234567), 10: f32(0.3141592), 11: f32(0.2718281), 16: f32(-0.093817),
    })
    own = observation.own_global
    assert own.worker_brood_eta == f32(0.1234567)
    assert own.lay_cooldown == f32(0.3141592)
    assert own.upkeep_remainder == f32(0.2718281)
    assert own.starvation_delta == f32(-0.093817)


def test_v2_configs_are_immutable_and_hash_the_config_the_policy_uses():
    spec = baseline_spec("raider:v2")
    observation = _edit(_observations(fresh_state()), "own_global", {2: 6 / 70})
    mask = get_action_mask(observation)
    default_policy = spec.build()
    default_action = decide(default_policy, observation, new_stream(11, "policy:raider-v2:0")).action
    assert default_action.army_choice == 6

    changed_profile = replace(spec.config.profile, minimum_soldiers_to_raid=7)
    changed_config = BaselineConfig(shared=spec.config.shared, profile=changed_profile)
    changed_spec_hash = spec.digest(changed_config)
    changed_policy = spec.build(changed_config)
    assert changed_policy.baseline_config is changed_config
    assert changed_policy.config == changed_profile
    assert changed_spec_hash != spec.digest()
    changed_action = decide(changed_policy, observation, new_stream(11, "policy:raider-v2:0")).action
    assert changed_action.army_choice != default_action.army_choice

    with pytest.raises(FrozenInstanceError):
        changed_profile.minimum_soldiers_to_raid = 8


@pytest.mark.parametrize("soldiers,should_raid", [(4, False), (5, True)])
def test_raider_v2_minimum_soldier_threshold_boundary(soldiers, should_raid):
    observation = _edit(_observations(fresh_state()), "own_global", {2: soldiers / (soldiers + 64)})
    action = _action("raider:v2", observation)
    assert (action.army_choice == 6) is should_raid


@pytest.mark.parametrize("side", (0, 1))
@pytest.mark.parametrize("selector,threshold", (("rush:v2", 4), ("raider:v2", 5), ("balanced:v2", 6)))
def test_v2_raid_soldier_thresholds_are_exact_for_both_sides(side, selector, threshold):
    snapshot = begin_policy_phase(fresh_state(60))
    base_observation = build_observation(snapshot, side)
    for soldiers, should_raid in ((threshold - 1, False), (threshold, True)):
        observation = _edit(base_observation, "own_global", {
            OwnGlobalIndex.WORKERS: 8 / 72,
            OwnGlobalIndex.SOLDIERS: soldiers / (soldiers + 64),
        })
        assert _decode_u_integer(observation.own_global.soldiers, 64.0) == soldiers
        assert (_action(selector, observation).army_choice == 6) is should_raid


@pytest.mark.parametrize("side", (0, 1))
@pytest.mark.parametrize("selector,threshold,field,before,at", (
    ("raider:v2", 25, "lay", 0, 2),
        ("rush:v2", 35, "lay", 0, 2),
    ("raider:v2", 45, "lay", 2, 2),
    ("balanced:v2", 45, "lay", 1, 2),
    ("turtle:v2", 55, "lay", 0, 1),
    ("balanced:v2", 65, "forage", 3, 2),
    ("turtle:v2", 80, "forage", 2, 1),
))
def test_v2_food_thresholds_are_exact_for_both_sides(side, selector, threshold, field, before, at):
    snapshot = begin_policy_phase(fresh_state(60))
    base_observation = build_observation(snapshot, side)
    values = []
    for food in (threshold - 1, threshold):
        observation = _edit(base_observation, "own_global", {
            OwnGlobalIndex.FOOD: food / (food + 250),
            OwnGlobalIndex.WORKERS: 8 / 72,
            OwnGlobalIndex.SOLDIERS: 3 / 67,
        })
        assert _decode_u_integer(observation.own_global.food, 250.0) == food
        action = _action(selector, observation)
        values.append(action.lay_order if field == "lay" else sum(q > 0 for q in action.forage_share_q))
    assert values == [before, at]


@pytest.mark.parametrize("side", (0, 1))
@pytest.mark.parametrize("selector,expected_lay", (
    ("turtle:v2", (1, 0, 0)),
    ("rush:v2", (2, 2, 0)),
    ("balanced:v2", (2, 2, 0)),
    ("raider:v2", (2, 2, 2)),
))
def test_v2_starvation_debt_thresholds_use_normalized_signal_at_0_1_2(side, selector, expected_lay):
    snapshot = begin_policy_phase(fresh_state(60))
    base_observation = build_observation(snapshot, side)
    observed_lay = []
    legacy_name = selector.removesuffix(":v2")
    for debt, expected in zip((0, 1, 2), expected_lay, strict=True):
        observation = _edit(base_observation, "own_global", {
            OwnGlobalIndex.FOOD: 80 / 330,
            OwnGlobalIndex.WORKERS: 8 / 72,
            OwnGlobalIndex.SOLDIERS: 3 / 67,
            OwnGlobalIndex.STARVATION_DEBT: debt / (debt + 10),
            OwnGlobalIndex.STARVATION_DELTA: 0.0,
        })
        assert observation.own_global.starvation_debt == f32(debt / (debt + 10))
        action = _action(selector, observation)
        assert action.lay_order == expected
        assert action == _action(legacy_name, observation)
        observed_lay.append(action.lay_order)
    assert tuple(observed_lay) == expected_lay


def test_v2_starvation_threshold_sha_matches_the_normalized_config_used_by_policy():
    spec = baseline_spec("balanced:v2")
    observation = _edit(_observations(fresh_state(60)), "own_global", {
        OwnGlobalIndex.FOOD: 80 / 330,
        OwnGlobalIndex.WORKERS: 8 / 72,
        OwnGlobalIndex.SOLDIERS: 3 / 67,
        OwnGlobalIndex.STARVATION_DEBT: 1 / 11,
        OwnGlobalIndex.STARVATION_DELTA: 0.0,
    })
    default_action = _action("balanced:v2", observation)
    assert default_action.lay_order == 2

    changed_profile = replace(spec.config.profile, lay_none_starvation_debt=0.08)
    changed_config = BaselineConfig(shared=spec.config.shared, profile=changed_profile)
    changed_policy = spec.build(changed_config)
    assert changed_policy.baseline_config is changed_config
    assert spec.parameter_object(changed_config)["profile"]["lay_none_starvation_debt"] == 0.08
    assert spec.digest(changed_config) != spec.digest()
    changed_action = decide(
        changed_policy, observation, new_stream(11, "policy:balanced-v2:0"),
    ).action
    assert changed_action.lay_order == 0


def test_baseline_selectors_keep_legacy_aliases_on_v1_and_expose_both_versions():
    assert all(baseline_spec(name).version == "v1" for name in baseline_names())
    assert set(baseline_selectors()) == set(baseline_names()) | {
        f"{name}:v{version}" for name in baseline_names() for version in (1, 2)
    }
    assert baseline_spec("raider:v1").digest() == baseline_spec("raider").digest()
    assert baseline_spec("raider:v2").full_id == "raider:v2"


def test_policy_descriptors_hash_the_immutable_config_used_by_each_version():
    pair = controller_pair(None, a="raider:v2", b="turtle:v1")
    for descriptor, selector in zip(pair.descriptors, ("raider:v2", "turtle:v1"), strict=True):
        spec = baseline_spec(selector)
        assert descriptor["id"] == spec.full_id
        assert descriptor["parameters"] == spec.parameter_object()
        assert descriptor["parameter_sha256"] == spec.digest()
        assert descriptor["policy_id"] == spec.policy_id


@pytest.mark.parametrize("name", TEST_BASELINES)
def test_baselines_return_finite_legal_actions_for_normal_and_restricted_masks(name):
    observation = _observations(fresh_state())
    normal = get_action_mask(observation)
    policy = make_baseline(name)
    stream_id = f"policy:{baseline_spec(name).policy_id}:0"
    logits = policy(observation, normal, new_stream(1, stream_id))
    action = validate_action(observation, normal, decode_action_logits(logits, normal))
    assert sum(share > 0 for share in action.action.forage_share_q) <= 3

    # Even when a requested army route and the center-avoid route are forbidden,
    # the policy remains on the common masked decoder/validator path.
    forced_route = next(index for index in range(1, 7) if normal.army_choice[index])
    army_choices = tuple(index == forced_route for index in range(7))
    avoid_options = tuple(
        (True, False) if index == forced_route else (False, False) for index in range(7)
    )
    restricted = ActionMaskV2(
        lay=normal.lay,
        forage_target=normal.forage_target,
        army_choice=army_choices,
        army_avoid_center=avoid_options,
    )
    action = validate_action(
        observation,
        restricted,
        decode_action_logits(policy(observation, restricted, new_stream(1, stream_id)), restricted),
    )
    assert action.action.army_choice == forced_route
    assert action.action.avoid_center is False
    assert sum(share > 0 for share in action.action.forage_share_q) <= 3


def test_policy_rng_is_seeded_role_local_and_random_baseline_varies_by_seed():
    observation = _observations(fresh_state())
    mask = get_action_mask(observation)
    policy = make_baseline("random")
    first = policy(observation, mask, new_stream(31, "policy:random-v1:0"))
    again = policy(observation, mask, new_stream(31, "policy:random-v1:0"))
    assert first == again
    actions = {
        decode_action_logits(policy(observation, mask, new_stream(seed, "policy:random-v1:0")), mask)
        for seed in range(40, 48)
    }
    assert len(actions) > 1

    pair = controller_pair(None, a="random", b="random")
    assert pair.descriptors[0]["stream_id"] == "policy:random-v1:0"
    assert pair.descriptors[1]["stream_id"] == "policy:random-v1:1"
    swapped = pair.swapped()
    assert swapped.descriptors == tuple(reversed(pair.descriptors))

    random_pair = controller_pair(None, a="random", b="balanced")
    streams = [descriptor["stream_id"] for descriptor in random_pair.descriptors]
    state, run = reset(default_config(), 32, streams)
    observation = _observations(state)
    world_before = run.stream("world").clone()
    role_before = run.stream(streams[0]).clone()
    random_pair.controllers[0].act(0, observation, run)
    assert run.stream("world") == world_before
    assert run.stream(streams[0]) != role_before


@pytest.mark.parametrize("name", TEST_BASELINES)
def test_baseline_decisions_are_mirror_equivariant(name):
    state = fresh_state(60)
    obs_a = _observations(state)
    obs_mirrored = _observations(mirror_state(state))
    assert obs_a == obs_mirrored
    policy = make_baseline(name)
    seed = 81
    stream_id = f"policy:{baseline_spec(name).policy_id}:0"
    action_a = decide(policy, obs_a, new_stream(seed, stream_id)).action
    action_b = decide(policy, obs_mirrored, new_stream(seed, stream_id)).action
    assert action_a == action_b


def test_profiles_react_to_food_deficit_and_current_threat_but_not_stale_threat():
    calm = _observations(fresh_state(60))
    current_threat = _threat_observation(calm, current=True)
    stale_threat = _threat_observation(calm, current=False, age=0.7)
    food_deficit = _edit(calm, "own_global", {0: 10 / 260, 12: 0.3, 16: 0.3})

    for name in ("turtle", "rush", "balanced", "raider", *V2_BASELINES[1:]):
        calm_action = _action(name, calm)
        danger_action = _action(name, current_threat)
        assert danger_action.army_choice == 1, name
        hungry_action = _action(name, food_deficit)
        assert (hungry_action.forage_share_q != calm_action.forage_share_q
                or hungry_action.lay_order != calm_action.lay_order)

    balanced_stale = _action("balanced", stale_threat)
    assert balanced_stale.army_choice != 1
    raider_stale = _action("raider", stale_threat)
    assert raider_stale.army_choice != 1

    # The three strategic profiles have intentionally distinct default choices.
    assert len({_action(name, calm) for name in ("turtle", "rush", "balanced", "raider")}) >= 3


@pytest.mark.parametrize("name", ("balanced", "raider", "balanced:v2", "raider:v2"))
def test_ready_raid_ignores_center_workers_without_queen_intel(name):
    observation = _observations(fresh_state(60))
    observation = _edit(observation, "own_global", {2: 6 / 70})
    observation = _edit(observation, "enemy_count_last", {6: 3 / 67})
    observation = _edit(observation, "enemy_known", {3: 1.0})
    observation = _edit(observation, "enemy_current", {3: 1.0})

    action = _action(name, observation)
    assert observation.block("enemy_queen_known")[0] == 0.0
    assert action.army_choice == 6


@pytest.mark.parametrize("name", ("balanced", "raider"))
def test_raiding_profiles_reach_legal_raid_from_reset_without_queen_intel(name):
    pair = controller_pair(None, a=name, b="turtle")
    state, run = reset(default_config(), 20261103, [d["stream_id"] for d in pair.descriptors])
    # A passive scripted opponent leaves the public resource routes open. The policy
    # under test still develops and issues its orders through the real tick engine.
    passive = ScriptedController({
        0: ActionV2(0, (0, Q_TOTAL, 0, 0, 0, 0), 0, False),
    })
    controllers = (pair.controllers[0], passive)

    for _ in range(state.config.t_max):
        observation = build_observation(
            begin_policy_phase(state.clone(), tuple(run.windows)), 0
        ) if state.tick % state.config.policy_interval_k == 0 else None
        result = step_tick(state, run, controllers)
        raid = next((event for event in result.events
                     if isinstance(event, OrderEvent) and event.side == 0
                     and event.army_choice == 6), None)
        if raid is not None:
            assert observation is not None
            assert observation.block("enemy_queen_known")[0] == 0.0
            assert get_action_mask(observation).army_choice[6]
            assert raid.phase == Phase.POLICY
            assert raid.army_target_node == 6
            return
        if run.outcome is not None:
            break

    pytest.fail(f"{name} did not issue a raid order from the reset state")


def test_raider_v2_raids_with_exactly_five_soldiers_through_step_tick():
    base = default_config()
    config = replace(base, balance=replace(base.balance, start_soldiers=5))
    pair = controller_pair(None, a="raider:v2", b="turtle:v1")
    state, run = reset(config, 20261103, [d["stream_id"] for d in pair.descriptors])
    assert sum(unit.caste == 1 for unit in state.colonies[0].units) == 5

    class RecordingController:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.observations = []

        def act(self, tick, observation, run):
            self.observations.append(observation)
            return self.wrapped.act(tick, observation, run)

    recording = RecordingController(pair.controllers[0])
    passive = ScriptedController({
        0: ActionV2(0, (0, Q_TOTAL, 0, 0, 0, 0), 0, False),
    })
    observation = build_observation(begin_policy_phase(state.clone(), tuple(run.windows)), 0)
    assert _decode_u_integer(observation.own_global.soldiers, 64.0) == 5
    mask = get_action_mask(observation)
    assert mask.army_choice[6]
    logits = make_baseline("raider:v2")(observation, mask, new_stream(1, "policy:raider-v2:0"))
    legal_action = validate_action(observation, mask, decode_action_logits(logits, mask))
    assert legal_action.action.army_choice == 6

    result = step_tick(state, run, (recording, passive))
    raid = next((event for event in result.events if isinstance(event, OrderEvent)
                 and event.side == 0 and event.army_choice == 6), None)
    assert recording.observations == [observation]
    assert raid is not None
    assert raid.phase == Phase.POLICY
    assert raid.army_target_node == 6


def test_paired_policy_matchups_are_repeatable_with_role_keyed_streams():
    first, aggregate = evaluate_episodes(
        episodes=2, seed_start=20261103, scenario=None, a="balanced", b="rush",
        paired_seat=True, max_ticks=90,
    )
    again, again_aggregate = evaluate_episodes(
        episodes=2, seed_start=20261103, scenario=None, a="balanced", b="rush",
        paired_seat=True, max_ticks=90,
    )
    assert first == again
    assert aggregate == again_aggregate
    assert [row["seat_assignment"]["A"] for row in first] == [
        "balanced:v1#replicate-0", "rush:v1#replicate-0",
        "balanced:v1#replicate-0", "rush:v1#replicate-0",
    ]
    assert all(row["truncated"] for row in first)
    assert all(row["state_hash"] and row["checkpoint_hash"] for row in first)
    assert first[0]["controller_config"]["A"]["parameters"] == baseline_spec("balanced").parameter_object()


def test_eval_manifest_round_trip_and_single_match_replay(tmp_path, capsys):
    manifest = tmp_path / "baselines.manifest.json"
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    args = [
        "eval", "--episodes", "1", "--seed-start", "20261103", "--a", "random", "--b", "balanced",
        "--paired", "--max-ticks", "45", "--manifest-out", str(manifest), "--json-out", str(first),
    ]
    assert sim_main(args) == EXIT_OK
    capsys.readouterr()
    assert sim_main(["eval", "--manifest", str(manifest), "--json-out", str(second)]) == EXIT_OK
    capsys.readouterr()
    first_doc = json.loads(first.read_text(encoding="utf-8"))
    second_doc = json.loads(second.read_text(encoding="utf-8"))
    assert first_doc["manifest"] == second_doc["manifest"]
    assert first_doc["results"] == second_doc["results"]
    assert first_doc["aggregate"] == second_doc["aggregate"]

    replay_a, replay_b = tmp_path / "baseline-a.jsonl", tmp_path / "baseline-b.jsonl"
    for path in (replay_a, replay_b):
        assert sim_main([
            "run", "--seed", "20261103", "--a", "random", "--b", "turtle", "--max-ticks", "90",
            "--replay-out", str(path),
        ]) == EXIT_OK
        capsys.readouterr()
    assert replay_a.read_bytes() == replay_b.read_bytes()
    assert sim_main(["replay", "--verify", str(replay_a)]) == EXIT_OK
    assert "verified" in capsys.readouterr().out


def test_eval_manifest_round_trips_explicit_baseline_versions(tmp_path, capsys):
    manifest = tmp_path / "versioned.manifest.json"
    first = tmp_path / "versioned-first.json"
    second = tmp_path / "versioned-second.json"
    args = [
        "eval", "--episodes", "1", "--seed-start", "912", "--a", "raider:v2", "--b", "turtle:v1",
        "--max-ticks", "30", "--manifest-out", str(manifest), "--json-out", str(first),
    ]
    assert sim_main(args) == EXIT_OK
    capsys.readouterr()
    assert sim_main(["eval", "--manifest", str(manifest), "--json-out", str(second)]) == EXIT_OK
    capsys.readouterr()
    first_doc = json.loads(first.read_text(encoding="utf-8"))
    second_doc = json.loads(second.read_text(encoding="utf-8"))
    assert first_doc["manifest"]["controllers"]["A"]["id"] == "raider:v2"
    assert first_doc["manifest"]["controllers"]["B"]["id"] == "turtle:v1"
    assert first_doc["results"] == second_doc["results"]


def test_strategic_policies_reject_scripted_scenarios_and_mixed_scripted_pairs(capsys):
    assert sim_main([
        "run", "--seed", "1", "--a", "balanced", "--b", "rush", "--scenario", "raid_win",
    ]) != EXIT_OK
    assert "scenario" in capsys.readouterr().err
    assert sim_main(["run", "--seed", "1", "--a", "scripted", "--b", "rush"]) != EXIT_OK
    assert "mixed" in capsys.readouterr().err


def test_benchmark_accepts_baseline_policies_without_changing_tick_budget(tmp_path):
    output = tmp_path / "baseline-benchmark.json"
    assert sim_main([
        "benchmark", "--runs", "1", "--ticks-per-run", "12", "--warmup-ticks", "0",
        "--seed-start", "77", "--a", "random", "--b", "turtle", "--json-out", str(output),
    ]) == EXIT_OK
    from json import loads

    document = loads(output.read_text(encoding="utf-8"))
    assert document["measurement"]["samples"][0]["ticks"] == 12
    assert document["manifest"]["controllers"]["A"]["id"] == "random:v1"
    assert document["measurement"]["throughput"]["gate0_status"] == "NOT MEASURED"
