from __future__ import annotations

import struct
from fractions import Fraction

from formicmind.actions import get_action_mask
from formicmind.baselines import baseline_spec
from formicmind.decoder import PolicyLogitsV2
from formicmind.gate1_feedback import (
    EXPECTED_MATCHES,
    MAP_IDS,
    OPPONENTS,
    SEEDS,
    TARGETS,
    _EXPECTED_CONFIG_HASHES,
    _FrozenObservationController,
    _RecordingPolicy,
    _new_episode,
    _block_from_rows,
    _build_summary,
    _frozen_observation,
    _policy_streams,
    _schedule,
    _target_label,
)
from formicmind.observation import OBS_V2_LENGTH, ObservationV2
from formicmind.rng import policy_stream_id
from formicmind.runstate import RunState
from formicmind.gate1_context import _build_configs
from formicmind.serialization import config_hash


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _observation(*, time: tuple[float, float] = (0.0, 1.0), food: float = 0.0,
                 queen_hp: float = 1.0) -> ObservationV2:
    values = [0.0] * OBS_V2_LENGTH
    values[0:2] = [_f32(value) for value in time]
    values[12] = _f32(food)  # own_global food ratio
    values[17] = _f32(queen_hp)  # own_global queen HP ratio
    return ObservationV2(tuple(values))


class _ProbePolicy:
    def __init__(self) -> None:
        self.inputs: list[ObservationV2] = []
        self.masks = []
        self.logits: list[PolicyLogitsV2] = []

    def __call__(self, observation, mask, policy_rng):
        del policy_rng
        self.inputs.append(observation)
        self.masks.append(mask)
        values = [-2.0] * 17
        values[2] = 2.0 + observation.values[12]
        values[3] = 1.0
        values[9] = 2.0
        logits = PolicyLogitsV2(values)
        self.logits.append(logits)
        return logits


def test_fixed_schedule_has_full_preregistered_factorial_count() -> None:
    assert len(MAP_IDS) == 6
    assert len(TARGETS) == 4
    assert len(OPPONENTS) == 3
    assert len(SEEDS) == 6
    assert len(_schedule()) == EXPECTED_MATCHES == 1728
    assert EXPECTED_MATCHES == 6 * 4 * 3 * 6 * 2 * 2


def test_reused_for22_map_configs_keep_their_canonical_hashes() -> None:
    configs = _build_configs()
    assert {map_id: config_hash(configs[map_id]) for map_id in MAP_IDS} == _EXPECTED_CONFIG_HASHES


def test_frozen_observation_updates_only_time_and_ignores_live_world_values() -> None:
    initial = _observation(food=0.0, queen_hp=1.0)
    live_a = _observation(time=(0.5, 0.5), food=0.8, queen_hp=0.0)
    live_b = _observation(time=(0.5, 0.5), food=0.2, queen_hp=0.25)

    frozen_a = _frozen_observation(initial, live_a)
    frozen_b = _frozen_observation(initial, live_b)

    assert frozen_a.values[:2] == live_a.values[:2]
    assert frozen_a.values[2:] == initial.values[2:]
    assert frozen_a == frozen_b


def test_frozen_controller_keeps_initial_mask_and_uses_live_mask_only_for_legality() -> None:
    stream_id = policy_stream_id(baseline_spec("balanced").policy_id, 0)
    run = RunState.new(12345, (stream_id,))
    probe = _ProbePolicy()
    recorded = _RecordingPolicy(probe)
    controller = _FrozenObservationController(recorded, stream_id)
    initial = _observation(food=0.0, queen_hp=1.0)
    live = _observation(time=(0.5, 0.5), food=0.75, queen_hp=0.0)

    initial_action = controller.act(0, initial, run).action
    live_action = controller.act(30, live, run).action

    assert initial_action.lay_order == 2
    assert live_action.lay_order == 0
    assert probe.inputs[1].values[:2] == live.values[:2]
    assert probe.inputs[1].values[2:] == initial.values[2:]
    assert probe.inputs[0].values[2:] == probe.inputs[1].values[2:]
    assert probe.masks[0] == probe.masks[1] == get_action_mask(initial)
    assert probe.logits[0] == probe.logits[1]
    assert get_action_mask(live).lay == (True, False, False)
    assert recorded.pending is not None
    assert recorded.pending["mask_changed_from_initial"] is True
    assert recorded.pending["mask_only_legality_change"] is True


def test_self_play_assigns_distinct_role_streams_and_keeps_them_policy_keyed() -> None:
    target_stream, opponent_stream = _policy_streams("balanced", "balanced")
    assert target_stream == "policy:balanced-v1:0"
    assert opponent_stream == "policy:balanced-v1:1"
    assert target_stream != opponent_stream

    config = _build_configs()["R0-T0"]
    target_a = _new_episode("R0-T0", config, "balanced", "balanced", SEEDS[0], True, "closed")
    target_b = _new_episode("R0-T0", config, "balanced", "balanced", SEEDS[0], False, "frozen")
    assert target_a.controllers[0].stream_id == target_stream
    assert target_a.controllers[1].stream_id == opponent_stream
    assert target_b.controllers[0].stream_id == opponent_stream
    assert target_b.controllers[1].stream_id == target_stream


def test_matched_block_collapses_seats_before_exact_delta() -> None:
    rows = []
    for seat, mode, score in (
        ("target_A", "closed", 2),
        ("target_A", "frozen", 1),
        ("target_B", "closed", 1),
        ("target_B", "frozen", 0),
    ):
        rows.append({
            "status": "complete",
            "t0_control_pass": True,
            "seat_assignment": seat,
            "target_mode": mode,
            "target_score_x2": score,
            "paired_trajectory_block_sha256": "a" * 64,
        })
    result = _block_from_rows(rows)
    assert result["closed_mean_score"] == Fraction(3, 4)
    assert result["frozen_mean_score"] == Fraction(1, 4)
    assert result["delta"] == Fraction(1, 2)


def test_summary_deduplicates_repeated_seed_trajectories_before_labeling() -> None:
    rows = []
    for map_id in MAP_IDS:
        for target in TARGETS:
            for opponent in OPPONENTS:
                digest = f"{target}:{map_id}:{opponent}"
                for seed in SEEDS:
                    for seat in ("target_A", "target_B"):
                        for mode in ("closed", "frozen"):
                            rows.append({
                                "target": target,
                                "map_id": map_id,
                                "opponent": opponent,
                                "seed": seed,
                                "seat_assignment": seat,
                                "target_mode": mode,
                                "status": "complete",
                                "t0_control_pass": True,
                                "target_score_x2": 2 if mode == "closed" else 1,
                                "paired_trajectory_block_sha256": digest,
                                "terminal": {"reason": "queen_death"},
                                "decision_counts": {"target": 1},
                                "target_feedback_diagnostics": {
                                    "live_mask_changed_from_initial_decisions": 0,
                                    "mask_only_legality_changes": 0,
                                },
                                "matched_mode_diagnostics": {
                                    "shared_decision_ticks": 1,
                                    "different_logits_decisions": 1,
                                    "different_actions_decisions": 1,
                                    "different_live_observation_decisions": 1,
                                    "first_logits_difference_tick": 30,
                                    "first_action_difference_tick": 30,
                                    "first_live_observation_difference_tick": 30,
                                    "first_physical_state_difference_tick": 30,
                                },
                            })
    summary = _build_summary(rows, calls=0)
    target = summary["target_results"][0]
    assert summary["scheduled_matches"] == EXPECTED_MATCHES
    assert target["unique_paired_blocks_total"] == len(MAP_IDS) * len(OPPONENTS)
    assert target["target_level_delta_equal_weight_cells"] == {"numerator": 1, "denominator": 2}
    assert target["coverage"]["adequate_for_candidate_label"] is False
    assert target["label"] == "INCONCLUSIVE"


def test_feedback_candidate_threshold_is_exact_and_zero_maps_do_not_count() -> None:
    maps = {
        "m1": Fraction(1, 10),
        "m2": Fraction(1, 10),
        "m3": Fraction(1, 10),
        "m4": Fraction(1, 10),
        "m5": Fraction(0),
        "m6": Fraction(0),
    }
    assert _target_label(
        Fraction(1, 10), maps, adequate_coverage=True, complete_schedule=True,
    ) == "FEEDBACK_BENEFIT_CANDIDATE"
    assert _target_label(
        Fraction(1, 10), maps, adequate_coverage=False, complete_schedule=True,
    ) == "INCONCLUSIVE"
    maps["m4"] = Fraction(0)
    assert _target_label(
        Fraction(1, 10), maps, adequate_coverage=True, complete_schedule=True,
    ) == "NO_CLEAR_FEEDBACK_SIGNAL_IN_SCHEDULE"
