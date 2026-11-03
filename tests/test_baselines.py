"""FOR-19 fixed baseline contracts, determinism, visibility and CLI wiring."""

import json

import pytest

from formicmind.actions import ActionMaskV2, get_action_mask, validate_action
from formicmind.baselines import baseline_names, baseline_spec, make_baseline
from formicmind.cli import EXIT_OK, main as sim_main
from formicmind.decoder import decode_action_logits
from formicmind.evaluation import controller_pair, evaluate_episodes
from formicmind.observation import OBS_V2_OFFSETS, ObservationV2, build_observation, f32
from formicmind.policy import decide
from formicmind.rng import new_stream
from formicmind.reset import reset
from formicmind.rules import default_config
from formicmind.snapshot import begin_policy_phase

from helpers import fresh_state, mirror_state


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
    policy = make_baseline(policy_name)
    return decide(policy, observation, new_stream(seed, f"policy:{policy_name}-v1:0")).action


@pytest.mark.parametrize("name", baseline_names())
def test_baselines_return_finite_legal_actions_for_normal_and_restricted_masks(name):
    observation = _observations(fresh_state())
    normal = get_action_mask(observation)
    policy = make_baseline(name)
    logits = policy(observation, normal, new_stream(1, f"policy:{name}-v1:0"))
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
        decode_action_logits(policy(observation, restricted, new_stream(1, f"policy:{name}-v1:0")), restricted),
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


@pytest.mark.parametrize("name", baseline_names())
def test_baseline_decisions_are_mirror_equivariant(name):
    state = fresh_state(60)
    obs_a = _observations(state)
    obs_mirrored = _observations(mirror_state(state))
    assert obs_a == obs_mirrored
    policy = make_baseline(name)
    seed = 81
    action_a = decide(policy, obs_a, new_stream(seed, f"policy:{name}-v1:0")).action
    action_b = decide(policy, obs_mirrored, new_stream(seed, f"policy:{name}-v1:0")).action
    assert action_a == action_b


def test_profiles_react_to_food_deficit_and_current_threat_but_not_stale_threat():
    calm = _observations(fresh_state(60))
    current_threat = _threat_observation(calm, current=True)
    stale_threat = _threat_observation(calm, current=False, age=0.7)
    food_deficit = _edit(calm, "own_global", {0: 10 / 260, 12: 0.3, 16: 0.3})

    for name in ("turtle", "rush", "balanced", "raider"):
        calm_action = _action(name, calm)
        danger_action = _action(name, current_threat)
        assert danger_action.army_choice == 1, name
        hungry_action = _action(name, food_deficit)
        assert hungry_action.forage_share_q != calm_action.forage_share_q or hungry_action.lay_order != calm_action.lay_order

    balanced_stale = _action("balanced", stale_threat)
    assert balanced_stale.army_choice != 1
    raider_stale = _action("raider", stale_threat)
    assert raider_stale.army_choice != 1

    # The three strategic profiles have intentionally distinct default choices.
    assert len({_action(name, calm) for name in ("turtle", "rush", "balanced", "raider")}) >= 3


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
