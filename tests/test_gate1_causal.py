from __future__ import annotations

import pytest

from formicmind.actions import ActionV2, get_action_mask, validate_action
from formicmind.baselines import baseline_spec, make_baseline
from formicmind.canonical import canonical_dumps
from formicmind.events import OrderEvent, events_hash
from formicmind.gate1_causal import (
    AXES,
    _ArrayDigest,
    _EventDigest,
    _MirrorNormalizer,
    _TickBudget,
    _arm,
    _candidate,
    _capture_checkpoint,
    _full_episode_physical_delta,
    _make_manifest,
    _physical_state_object,
    _run_baseline,
    _sampled_metric_physical_delta,
    _unique_clusters,
)
from formicmind.observation import build_observation
from formicmind.policy import decide
from formicmind.reset import reset
from formicmind.rng import policy_stream_id
from formicmind.rules import Q_TOTAL, default_config
from formicmind.snapshot import begin_policy_phase
from formicmind.tick import PolicyController, ScriptedController, step_tick


def _action(*, lay=0, forage=None, army=0, avoid=False):
    return ActionV2(
        lay_order=lay,
        forage_share_q=tuple(forage or (Q_TOTAL, 0, 0, 0, 0, 0)),
        army_choice=army,
        avoid_center=avoid,
    )


def test_candidates_are_valid_and_change_exactly_one_declared_axis():
    config = default_config()
    stream_id = policy_stream_id(baseline_spec("balanced").policy_id, 0)
    state, run = reset(config, 20261201, (stream_id,))
    snapshot = begin_policy_phase(state, tuple(run.windows))
    observation = build_observation(snapshot, 0)
    mask = get_action_mask(observation)
    view = snapshot.view(0)

    controls = {
        "lay_order": _action(),
        "forage_share_q": _action(forage=(0, Q_TOTAL, 0, 0, 0, 0)),
        "army_choice": _action(army=1),
    }
    for axis in AXES:
        candidate, reason = _candidate(
            axis, controls[axis], mask, queen_alive=view.queen_hp > 0,
            worker_count=sum(unit.caste == 0 for unit in view.units),
            soldier_count=sum(unit.caste == 1 for unit in view.units),
        )
        assert reason is None
        assert candidate is not None
        assert [
            field for field in ("lay_order", "forage_share_q", "army_choice", "avoid_center")
            if getattr(candidate, field) != getattr(controls[axis], field)
        ] == [axis]
        assert validate_action(observation, mask, candidate).action == candidate


@pytest.mark.parametrize(
    ("control", "treatment", "changed_axis"),
    [
        (_action(), _action(lay=1), "lay_order"),
        (_action(), _action(forage=(0, Q_TOTAL, 0, 0, 0, 0)), "forage_share_q"),
        (_action(army=0), _action(army=6), "army_choice"),
    ],
)
def test_each_intervention_has_a_real_engine_effect(control, treatment, changed_axis):
    config = default_config()
    state_control, run_control = reset(config, 20261202)
    state_treatment, run_treatment = reset(config, 20261202)
    result_control = step_tick(
        state_control, run_control,
        (ScriptedController({0: control}), ScriptedController()),
    )
    result_treatment = step_tick(
        state_treatment, run_treatment,
        (ScriptedController({0: treatment}), ScriptedController()),
    )
    assert [
        field for field in ("lay_order", "forage_share_q", "army_choice", "avoid_center")
        if getattr(control, field) != getattr(treatment, field)
    ] == [changed_axis]
    orders_control = [event for event in result_control.events if isinstance(event, OrderEvent) and event.side == 0]
    orders_treatment = [event for event in result_treatment.events if isinstance(event, OrderEvent) and event.side == 0]
    assert len(orders_control) == len(orders_treatment) == 1
    assert orders_control[0] != orders_treatment[0]

    if changed_axis == "lay_order":
        assert len(state_control.colonies[0].brood) == 0
        assert len(state_treatment.colonies[0].brood) == 1
    elif changed_axis == "forage_share_q":
        assert not any(unit.caste == 0 and unit.node < 0 for unit in state_control.colonies[0].units)
        assert any(unit.caste == 0 and unit.node < 0 for unit in state_treatment.colonies[0].units)
    else:
        assert not any(unit.caste == 1 and unit.node < 0 for unit in state_control.colonies[0].units)
        assert any(unit.caste == 1 and unit.node < 0 for unit in state_treatment.colonies[0].units)


def test_independent_control_arms_match_every_tick_and_terminal_hash():
    config = default_config()
    stream_ids = (
        policy_stream_id(baseline_spec("balanced").policy_id, 0),
        policy_stream_id(baseline_spec("rush").policy_id, 0),
    )
    state, run = reset(config, 20261203, stream_ids)
    checkpoint = _capture_checkpoint(
        state, run, tick=0, target_side=0, target_policy="balanced",
        opponent_policy="rush", matchup="balanced-rush", seed=20261203, swapped=False,
        mirror_normalizer=_MirrorNormalizer(0),
    )
    left = _arm(checkpoint, "lay_order", treatment=False, budget=_CallCounter())
    right = _arm(checkpoint, "lay_order", treatment=False, budget=_CallCounter())
    assert left["per_tick_trace_sha256"] == right["per_tick_trace_sha256"]
    assert left["events_sha256"] == right["events_sha256"]
    assert left["final_state_sha256"] == right["final_state_sha256"]
    assert left["final_checkpoint_sha256"] == right["final_checkpoint_sha256"]
    assert left["terminal"] == right["terminal"]


class _CallCounter:
    def __init__(self):
        self.calls = 0

    def step(self, state, run, controllers):
        from formicmind.tick import step_tick
        self.calls += 1
        return step_tick(state, run, controllers)


def test_mirror_baseline_policy_gets_equivalent_observation_and_action():
    config = default_config()
    stream_a = policy_stream_id(baseline_spec("balanced").policy_id, 0)
    stream_b = policy_stream_id(baseline_spec("balanced").policy_id, 1)
    state, run = reset(config, 20261204, (stream_a, stream_b))
    snapshot = begin_policy_phase(state, tuple(run.windows))
    observation_a = build_observation(snapshot, 0)
    observation_b = build_observation(snapshot, 1)
    assert observation_a == observation_b
    action_a = decide(make_baseline("balanced"), observation_a, run.stream(stream_a))
    action_b = decide(make_baseline("balanced"), observation_b, run.stream(stream_b))
    assert action_a == action_b


def test_event_digest_matches_the_engine_event_hash():
    config = default_config()
    state, run = reset(config, 20261205)
    result = step_tick(state, run, (ScriptedController(), ScriptedController()))
    digest = _EventDigest()
    digest.add_all(event.to_canonical() for event in result.events)
    assert digest.hexdigest() == events_hash(list(result.events))
    assert digest.count == len(result.events)


def test_manifest_records_fixed_protocol_and_policy_descriptors():
    manifest = _make_manifest(default_config(), "a" * 40)
    assert manifest["study"]["open_seeds"] == list(range(20261201, 20261209))
    assert [policy["id"] for policy in manifest["policies"]] == [
        "balanced:v1", "raider:v1", "random:v1", "rush:v1", "turtle:v1",
    ]
    assert all(len(policy["parameters_sha256"]) == 64 for policy in manifest["policies"])
    assert canonical_dumps(manifest)


def test_seat_swap_does_not_create_a_second_mirror_trajectory_cluster():
    config = default_config()
    base, _ = _run_baseline(config, "balanced-rush", "balanced", "rush", 20261201, False, _TickBudget())
    swapped, _ = _run_baseline(config, "balanced-rush", "balanced", "rush", 20261201, True, _TickBudget())
    assert base["trajectory_sha256"] != swapped["trajectory_sha256"]
    assert base["mirror_trajectory_sha256"] == swapped["mirror_trajectory_sha256"]
    rows = [
        {
            "axis": "lay_order", "eligible": True, "matchup": "balanced-rush",
            "baseline_trajectory_sha256": base["trajectory_sha256"],
            "mirror_baseline_trajectory_sha256": base["mirror_trajectory_sha256"],
            "pre_state_sha256": "world-a", "mirror_pre_state_sha256": "relative-state",
            "primary": {"delta_score_millipoints": 0},
        },
        {
            "axis": "lay_order", "eligible": True, "matchup": "balanced-rush",
            "baseline_trajectory_sha256": swapped["trajectory_sha256"],
            "mirror_baseline_trajectory_sha256": swapped["mirror_trajectory_sha256"],
            "pre_state_sha256": "world-b", "mirror_pre_state_sha256": "relative-state",
            "primary": {"delta_score_millipoints": 0},
        },
    ]
    clusters = _unique_clusters(rows, "lay_order")
    assert sum(len(matchup) for matchup in clusters.values()) == 1


def test_random_mirror_grouping_includes_rng_and_keeps_distinct_seeds_separate():
    config = default_config()
    base, _ = _run_baseline(config, "random-balanced", "random", "balanced", 20261201, False, _TickBudget())
    swapped, _ = _run_baseline(config, "random-balanced", "random", "balanced", 20261201, True, _TickBudget())
    other_seed, _ = _run_baseline(config, "random-balanced", "random", "balanced", 20261202, False, _TickBudget())
    assert base["trajectory_sha256"] != swapped["trajectory_sha256"]
    assert base["mirror_trajectory_sha256"] == swapped["mirror_trajectory_sha256"]
    assert base["mirror_trajectory_sha256"] != other_seed["mirror_trajectory_sha256"]


def test_physical_trace_detects_transient_difference_after_endpoint_convergence():
    config = default_config()
    initial, _ = reset(config, 20261203)
    transient = initial.clone()
    transient.node_food[1] -= 1
    control = _ArrayDigest()
    treatment = _ArrayDigest()
    for state in (initial, initial):
        control.add(_physical_state_object(state, 0))
    for state in (transient, initial):
        treatment.add(_physical_state_object(state, 0))
    control_arm = {
        "physical_trace_sha256": control.hexdigest(),
        "horizons": {"30": {"metrics": {"food": 7}}, "150": {"metrics": {"food": 7}}},
        "terminal_metrics": {"food": 7},
    }
    treatment_arm = {
        "physical_trace_sha256": treatment.hexdigest(),
        "horizons": {"30": {"metrics": {"food": 7}}, "150": {"metrics": {"food": 7}}},
        "terminal_metrics": {"food": 7},
    }
    assert not _sampled_metric_physical_delta(control_arm, treatment_arm)
    assert _full_episode_physical_delta(control_arm, treatment_arm)


def test_physical_projection_excludes_order_only_state():
    state, _ = reset(default_config(), 20261204)
    before = _physical_state_object(state, 0)
    state.colonies[0].lay_order = 1
    state.colonies[0].forage_q = [0, Q_TOTAL, 0, 0, 0, 0]
    state.colonies[0].army_choice = 6
    after = _physical_state_object(state, 0)
    assert before == after
