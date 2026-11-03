"""Preregistered Gate 1C matched live-versus-frozen feedback ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import struct
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

from .actions import ActionMaskV2, get_action_mask, validate_action
from .baselines import baseline_spec, make_baseline
from .canonical import canonical_dumps, canonical_loads
from .decoder import PolicyLogitsV2, decode_action_logits
from .events import HarvestEvent, OrderEvent, StarvationEvent, events_hash
from .gate1_context import _build_configs, _colony_metrics
from .observation import OBS_V2_OFFSETS, ObservationV2, flatten_v2
from .reset import reset
from .rng import policy_stream_id
from .serialization import checkpoint_hash, config_hash, state_hash
from .tick import PolicyController, step_tick
from .versions import (
    ACTION_VERSION,
    BALANCE_VERSION,
    DECODER_VERSION,
    OBSERVATION_VERSION,
    RNG_VERSION,
    RULES_VERSION,
    SERIALIZATION_VERSION,
    STREAM_DERIVATION_VERSION,
)

PROTOCOL_ID = "formicmind.gate1-feedback/v1"
PROTOCOL_PATH = Path("docs/gate1_feedback_protocol.md")
OUT_RELATIVE = Path("results/gate1_feedback")
SCHEMA_MANIFEST = "formicmind.gate1-feedback-manifest/v1"
SCHEMA_MATCH = "formicmind.gate1-feedback-match/v1"
SCHEMA_SUMMARY = "formicmind.gate1-feedback-summary/v1"
BASE_MAIN_SHA = "af4d6ff5364e29add002aab504d2609b3d797fb8"
MAP_IDS = tuple(f"{family}-{terrain}" for family in ("R0", "R1")
                for terrain in ("T0", "T1", "T2"))
TARGETS = ("balanced", "raider", "turtle", "rush")
OPPONENTS = ("random", "balanced", "raider")
SEEDS = tuple(range(20270111, 20270117))
SEATS = (True, False)  # True: target=A; False: target=B
MODES = ("closed", "frozen")
EXPECTED_MATCHES = 1_728
STEP_BUDGET = 5_184_000
HORIZONS = (100, 300, 1000)
SCORE_SCALE = 2
MDE = Fraction(1, 10)
_ROOT = Path(__file__).resolve().parents[2]
_TIME_OFFSET, _TIME_LENGTH = OBS_V2_OFFSETS["time"]
_OBS_OFFSET_AFTER_TIME = _TIME_OFFSET + _TIME_LENGTH
_EXPECTED_CONFIG_HASHES = {
    "R0-T0": "3f6aaa36a0d3d15888fa42f142df1151376794e62b553ab8e5f7659c9794145d",
    "R0-T1": "78e9c6979a2fb3973ae9a275f79ad5b62cb4b6ed3893a8f8469e4cc599e5fb34",
    "R0-T2": "b9e43b2aec34f303f371a9f9e5a2bcd8b5c5d894e608a3ca9e71386339f632f1",
    "R1-T0": "407bb3907099533adfe10b7c300a480b45b5d604dc019693b1ac99f47502303f",
    "R1-T1": "3bf8dabf6011ed88275c3fcfec3875c4da46ef849e7da7b0ac17bb2cd0e9e605",
    "R1-T2": "1b6f1f999246711919fb4dee9238e5fed31c5baac0dca779ea76c47afe167ceb",
}
_EXPECTED_POLICY_HASHES = {
    "random": "254037fbbe5f089fc1f909405ebcc3567573682553d3715d9bab8cf292b07f2c",
    "turtle": "6c9fa1c676a553294c26ffb4f3372953f93ecec73b5d8669f99a451a05e0ee4e",
    "rush": "cfea106a71257d8ebe79cfa6ac090b1927fd3010ca905516e2e41512460ea496",
    "balanced": "03c6760da2b2fc42108133c8b81a91b70e3c086327fbabedf9d3f30eb7dcdd0f",
    "raider": "6d4a23b0071c6ee308b8149850debe488e83d79c7fda50f19ee5053310c39b19",
}


class Gate1FeedbackError(ValueError):
    """Invalid schedule, controller behavior, provenance, or strict replay."""


class _Digest:
    """Streaming SHA-256 over a deterministic canonical JSON array."""

    def __init__(self) -> None:
        self._hash = hashlib.sha256(b"[")
        self._count = 0

    def add(self, value: Any) -> None:
        if self._count:
            self._hash.update(b",")
        self._hash.update(canonical_dumps(value))
        self._count += 1

    def hexdigest(self) -> str:
        value = self._hash.copy()
        value.update(b"]")
        return value.hexdigest()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _mask_bytes(mask: ActionMaskV2) -> bytes:
    return bytes(1 if value else 0 for value in mask.flatten())


def _float32_bytes(values: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _canonical_value(value: Any) -> Any:
    """Convert descriptive metadata to the canonical JSON type set."""
    if type(value) is float:
        return format(value, ".17g")
    if type(value) is tuple:
        return [_canonical_value(item) for item in value]
    if type(value) is list:
        return [_canonical_value(item) for item in value]
    if type(value) is dict:
        return {key: _canonical_value(item) for key, item in value.items()}
    return value


def _frozen_observation(initial: ObservationV2, live: ObservationV2) -> ObservationV2:
    """Copy the initial 223 floats, replacing only the two allowed time floats."""
    values = list(initial.values)
    values[_TIME_OFFSET:_OBS_OFFSET_AFTER_TIME] = live.values[_TIME_OFFSET:_OBS_OFFSET_AFTER_TIME]
    frozen = ObservationV2(tuple(values))
    if frozen.values[_OBS_OFFSET_AFTER_TIME:] != initial.values[_OBS_OFFSET_AFTER_TIME:]:
        raise Gate1FeedbackError("frozen observation changed a non-time field")
    return frozen


class _RecordingPolicy:
    """Transparent policy adapter that records policy-visible data and output hashes."""

    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self.pending: dict[str, Any] | None = None
        self.trace = _Digest()

    def __call__(self, observation: ObservationV2, mask: ActionMaskV2, policy_rng: Any) -> PolicyLogitsV2:
        logits = PolicyLogitsV2.coerce(self.policy(observation, mask, policy_rng))
        self.pending = {
            "policy_input_sha256": _sha(flatten_v2(observation)),
            "policy_mask_sha256": _sha(_mask_bytes(mask)),
            "logits_sha256": _sha(_float32_bytes(logits.values)),
            "live_observation_sha256": _sha(flatten_v2(observation)),
            "live_mask_sha256": _sha(_mask_bytes(mask)),
            "mask_changed_from_initial": False,
            "mask_only_legality_change": False,
        }
        return logits


@dataclass(slots=True)
class _FrozenObservationController:
    """Frozen policy input with only time refreshed; legality still uses live data."""

    policy: _RecordingPolicy
    stream_id: str
    initial_observation: ObservationV2 | None = None
    initial_mask: ActionMaskV2 | None = None

    def act(self, tick: int, observation: ObservationV2, run: Any) -> Any:
        live_mask = get_action_mask(observation)
        if self.initial_observation is None:
            if tick != 0:
                raise Gate1FeedbackError("frozen controller was not initialized at t=0")
            self.initial_observation = observation
            self.initial_mask = get_action_mask(observation)
        elif self.initial_mask is None:
            raise Gate1FeedbackError("frozen controller has an incomplete initial snapshot")
        frozen = _frozen_observation(self.initial_observation, observation)
        frozen_mask = self.initial_mask
        logits = PolicyLogitsV2.coerce(
            self.policy(frozen, frozen_mask, run.stream(self.stream_id))
        )
        frozen_mask_action = decode_action_logits(logits, frozen_mask)
        live_mask_action = decode_action_logits(logits, live_mask)
        validated = validate_action(observation, live_mask, live_mask_action)
        if self.policy.pending is None:
            raise Gate1FeedbackError("frozen policy call was not recorded")
        self.policy.pending.update({
            "live_observation_sha256": _sha(flatten_v2(observation)),
            "live_mask_sha256": _sha(_mask_bytes(live_mask)),
            "mask_changed_from_initial": _mask_bytes(live_mask) != _mask_bytes(frozen_mask),
            "mask_only_legality_change": frozen_mask_action != live_mask_action,
        })
        return validated


def _policy_descriptor(name: str) -> dict[str, Any]:
    spec = baseline_spec(name)
    return {
        "name": name,
        "id": spec.full_id,
        "policy_id": spec.policy_id,
        "parameters": _canonical_value(spec.parameter_object()),
        "parameters_sha256": spec.digest(),
        "stream_ids": [policy_stream_id(spec.policy_id, replicate) for replicate in (0, 1)],
    }


def _policy_streams(target: str, opponent: str) -> tuple[str, str]:
    target_spec, opponent_spec = baseline_spec(target), baseline_spec(opponent)
    target_rep = 0
    opponent_rep = 1 if target_spec.policy_id == opponent_spec.policy_id else 0
    return (
        policy_stream_id(target_spec.policy_id, target_rep),
        policy_stream_id(opponent_spec.policy_id, opponent_rep),
    )


@dataclass(slots=True)
class _Episode:
    map_id: str
    config: Any
    target: str
    opponent: str
    seed: int
    target_on_a: bool
    mode: str
    controllers: tuple[Any, Any]
    state: Any
    run: Any
    target_recorder: _RecordingPolicy
    opponent_recorder: _RecordingPolicy
    target_side: int
    opponent_side: int
    target_stream_id: str
    opponent_stream_id: str
    ticks_executed: int = 0
    event_counts: Counter[str] = field(default_factory=Counter)
    target_events: Counter[str] = field(default_factory=Counter)
    target_behavior: dict[str, Any] = field(default_factory=dict)
    target_actions: Counter[str] = field(default_factory=Counter)
    decisions: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: {"target": [], "opponent": []}
    )
    horizons: dict[str, dict[str, Any]] = field(default_factory=dict)
    events_trace: _Digest = field(default_factory=_Digest)
    t0_checkpoint_sha256: str = "not_measured"
    last_tick: int = -1


def _new_episode(
    map_id: str,
    config: Any,
    target: str,
    opponent: str,
    seed: int,
    target_on_a: bool,
    mode: str,
) -> _Episode:
    target_stream, opponent_stream = _policy_streams(target, opponent)
    target_recorder = _RecordingPolicy(make_baseline(target))
    opponent_recorder = _RecordingPolicy(make_baseline(opponent))
    target_controller: Any
    if mode == "closed":
        target_controller = PolicyController(target_recorder, target_stream)
    else:
        target_controller = _FrozenObservationController(target_recorder, target_stream)
    opponent_controller = PolicyController(opponent_recorder, opponent_stream)
    target_side = 0 if target_on_a else 1
    opponent_side = 1 - target_side
    controllers_by_role = {"target": target_controller, "opponent": opponent_controller}
    controllers = (
        (controllers_by_role["target"], controllers_by_role["opponent"])
        if target_on_a else
        (controllers_by_role["opponent"], controllers_by_role["target"])
    )
    stream_ids = tuple(controller.stream_id for controller in controllers)
    state, run = reset(config, seed, stream_ids)
    return _Episode(
        map_id=map_id, config=config, target=target, opponent=opponent, seed=seed,
        target_on_a=target_on_a, mode=mode, controllers=controllers,
        state=state, run=run, target_recorder=target_recorder,
        opponent_recorder=opponent_recorder, target_side=target_side,
        opponent_side=opponent_side, target_stream_id=target_stream,
        opponent_stream_id=opponent_stream,
        target_behavior={
            "harvested": 0,
            "starvation_deaths": 0,
            "combat_events": 0,
            "order_count": 0,
            "lay_orders": Counter(),
            "forage_vectors": Counter(),
            "army_choices": Counter(),
        },
    )


def _action_object(event: OrderEvent) -> dict[str, Any]:
    return {
        "lay_order": event.lay_order,
        "forage_share_q": list(event.forage_share_q),
        "army_choice": event.army_choice,
        "avoid_center": event.avoid_center,
    }


def _record_policy_decisions(episode: _Episode, tick: int, events: Sequence[Any]) -> None:
    orders = {event.side: event for event in events if isinstance(event, OrderEvent)}
    role_info = (
        ("target", episode.target_side, episode.target_recorder),
        ("opponent", episode.opponent_side, episode.opponent_recorder),
    )
    for role, side, recorder in role_info:
        pending = recorder.pending
        if pending is None:
            continue
        order = orders.get(side)
        if order is None:
            raise Gate1FeedbackError(f"policy decision at tick {tick} produced no OrderEvent for {role}")
        action = _action_object(order)
        action_sha = _sha(canonical_dumps(action))
        trace_item = {
            "tick": tick,
            "policy_input_sha256": pending["policy_input_sha256"],
            "policy_mask_sha256": pending["policy_mask_sha256"],
            "decoder_mask_sha256": pending["live_mask_sha256"],
            "logits_sha256": pending["logits_sha256"],
            "action_sha256": action_sha,
        }
        recorder.trace.add(trace_item)
        episode.decisions[role].append({
            **trace_item,
            "live_observation_sha256": pending["live_observation_sha256"],
            "live_mask_sha256": pending["live_mask_sha256"],
            "mask_changed_from_initial": pending["mask_changed_from_initial"],
            "mask_only_legality_change": pending["mask_only_legality_change"],
        })
        if role == "target":
            episode.target_actions[action_sha] += 1
        recorder.pending = None
    if any(recorder.pending is not None for recorder in (episode.target_recorder, episode.opponent_recorder)):
        raise Gate1FeedbackError("policy trace record was not consumed by the matching engine order")


def _record_events(episode: _Episode, tick: int, events: Sequence[Any]) -> None:
    for event in events:
        episode.event_counts[event.kind] += 1
        if getattr(event, "side", None) == episode.target_side:
            episode.target_events[event.kind] += 1
        if isinstance(event, HarvestEvent) and event.side == episode.target_side:
            episode.target_behavior["harvested"] += event.harvested
        elif isinstance(event, StarvationEvent) and event.side == episode.target_side:
            episode.target_behavior["starvation_deaths"] += 1
        elif (
            event.kind == "combat"
            and episode.target_side in (getattr(event, "attacker_side", -1), getattr(event, "defender_side", -1))
        ):
            episode.target_behavior["combat_events"] += 1
        if isinstance(event, OrderEvent) and event.side == episode.target_side:
            behavior = episode.target_behavior
            behavior["order_count"] += 1
            behavior["lay_orders"][str(event.lay_order)] += 1
            forage_key = ",".join(str(value) for value in event.forage_share_q)
            behavior["forage_vectors"][forage_key] += 1
            behavior["army_choices"][str(event.army_choice)] += 1
    episode.events_trace.add({"tick": tick, "events_sha256": events_hash(list(events))})


def _behavior_summary(episode: _Episode) -> dict[str, Any]:
    behavior = episode.target_behavior
    return {
        "harvested_food": behavior["harvested"],
        "starvation_deaths": behavior["starvation_deaths"],
        "combat_events": behavior["combat_events"],
        "order_count": behavior["order_count"],
        "lay_order_counts": dict(sorted(behavior["lay_orders"].items())),
        "forage_vector_counts": dict(sorted(behavior["forage_vectors"].items())),
        "army_choice_counts": dict(sorted(behavior["army_choices"].items())),
        "distinct_lay_orders": len(behavior["lay_orders"]),
        "distinct_forage_vectors": len(behavior["forage_vectors"]),
        "distinct_army_choices": len(behavior["army_choices"]),
        "distinct_validated_actions": len(episode.target_actions),
    }


def _capture_horizon(episode: _Episode) -> dict[str, Any]:
    return {
        "state_tick": episode.state.tick,
        "terminal_already_reached": episode.run.outcome is not None,
        "sides": {
            "target": _colony_metrics(episode.state, episode.target_side),
            "opponent": _colony_metrics(episode.state, episode.opponent_side),
        },
        "target_behavior_cumulative": _behavior_summary(episode),
    }


def _step_episode(episode: _Episode, calls: list[int]) -> None:
    if episode.run.outcome is not None or episode.state.tick >= episode.config.t_max:
        return
    if calls[0] >= STEP_BUDGET:
        raise Gate1FeedbackError(f"step_tick budget exceeded ({STEP_BUDGET})")
    tick = episode.state.tick
    result = step_tick(episode.state, episode.run, episode.controllers)
    calls[0] += 1
    episode.ticks_executed += 1
    episode.last_tick = tick
    _record_events(episode, tick, result.events)
    if tick % episode.config.policy_interval_k == 0:
        _record_policy_decisions(episode, tick, result.events)
    if tick == 0:
        episode.t0_checkpoint_sha256 = checkpoint_hash(episode.state, episode.run)
    if episode.state.tick in HORIZONS and str(episode.state.tick) not in episode.horizons:
        episode.horizons[str(episode.state.tick)] = _capture_horizon(episode)


def _target_action_score(episode: _Episode) -> int | str:
    outcome = episode.run.outcome
    if outcome is None:
        return "not_measured"
    value = int(round(outcome.score[episode.target_side] * SCORE_SCALE))
    if value not in (0, 1, 2):
        raise Gate1FeedbackError(f"unexpected engine target score {outcome.score[episode.target_side]!r}")
    return value


def _episode_row(episode: _Episode) -> dict[str, Any]:
    outcome = episode.run.outcome
    complete = outcome is not None
    terminal = "not_measured" if outcome is None else {
        "winner": outcome.winner.value,
        "reason": outcome.reason.value,
        "terminal_tick": outcome.terminal_tick,
        "tick_completed": outcome.tick_completed,
        "state_tick": episode.state.tick,
    }
    for horizon in HORIZONS:
        key = str(horizon)
        if key not in episode.horizons:
            episode.horizons[key] = {
                "status": "not_measured",
                "requested_completed_ticks": horizon,
                "reason": (
                    "episode_ended_before_horizon" if complete
                    else "engine_cutoff_without_terminal"
                ),
                "terminal_reason": "not_measured" if outcome is None else outcome.reason.value,
            }
        else:
            episode.horizons[key] = {
                "status": "reached",
                "requested_completed_ticks": horizon,
                **episode.horizons[key],
            }
    target_trace_sha = episode.target_recorder.trace.hexdigest()
    opponent_trace_sha = episode.opponent_recorder.trace.hexdigest()
    episode_trace_sha = _sha(canonical_dumps({
        "target": target_trace_sha,
        "opponent": opponent_trace_sha,
    }))
    target_decisions = episode.decisions["target"]
    return {
        "schema": SCHEMA_MATCH,
        "status": "complete" if complete else "censored_incomplete",
        "map_id": episode.map_id,
        "resource_family": episode.map_id.split("-")[0],
        "terrain_profile": episode.map_id.split("-")[1],
        "config_sha256": config_hash(episode.config),
        "target": episode.target,
        "opponent": episode.opponent,
        "target_policy_parameters_sha256": baseline_spec(episode.target).digest(),
        "opponent_policy_parameters_sha256": baseline_spec(episode.opponent).digest(),
        "seed": episode.seed,
        "seat_assignment": "target_A" if episode.target_on_a else "target_B",
        "target_mode": episode.mode,
        "target_side": "A" if episode.target_side == 0 else "B",
        "stream_ids": {
            "target": episode.target_stream_id,
            "opponent": episode.opponent_stream_id,
            "A": episode.controllers[0].stream_id,
            "B": episode.controllers[1].stream_id,
        },
        "ticks_executed": episode.ticks_executed,
        "target_score_x2": _target_action_score(episode),
        "terminal": terminal,
        "time_zero_control": {
            "policy_input_sha256": target_decisions[0]["policy_input_sha256"],
            "policy_mask_sha256": target_decisions[0]["policy_mask_sha256"],
            "logits_sha256": target_decisions[0]["logits_sha256"],
            "action_sha256": target_decisions[0]["action_sha256"],
            "checkpoint_sha256": episode.t0_checkpoint_sha256,
        },
        "decision_counts": {
            "target": len(target_decisions),
            "opponent": len(episode.decisions["opponent"]),
        },
        "target_policy_trace_sha256": target_trace_sha,
        "opponent_policy_trace_sha256": opponent_trace_sha,
        "episode_policy_trace_sha256": episode_trace_sha,
        "target_live_observation_trace_sha256": _hash_trace_field(
            target_decisions, "live_observation_sha256"
        ),
        "target_feedback_diagnostics": {
            "live_mask_changed_from_initial_decisions": sum(
                bool(item["mask_changed_from_initial"]) for item in target_decisions
            ),
            "mask_only_legality_changes": sum(
                bool(item["mask_only_legality_change"]) for item in target_decisions
            ),
        },
        "horizons": {key: episode.horizons[key] for key in map(str, HORIZONS)},
        "event_counts": dict(sorted(episode.event_counts.items())),
        "target_event_counts": dict(sorted(episode.target_events.items())),
        "target_behavior": _behavior_summary(episode),
        "events_trace_sha256": episode.events_trace.hexdigest(),
        "final_state_sha256": state_hash(episode.state),
        "final_checkpoint_sha256": checkpoint_hash(episode.state, episode.run),
        "t0_control_pass": False,
        "paired_trajectory_block_sha256": "not_measured",
    }


def _hash_trace_field(decisions: Sequence[dict[str, Any]], name: str) -> str:
    digest = _Digest()
    for item in decisions:
        digest.add({"tick": item["tick"], "sha256": item[name]})
    return digest.hexdigest()


def _compare_mode_decisions(closed: _Episode, frozen: _Episode, first_state_divergence: int | None) -> dict[str, Any]:
    closed_by_tick = {item["tick"]: item for item in closed.decisions["target"]}
    frozen_by_tick = {item["tick"]: item for item in frozen.decisions["target"]}
    shared = sorted(closed_by_tick.keys() & frozen_by_tick.keys())
    logit_differences = [
        tick for tick in shared
        if closed_by_tick[tick]["logits_sha256"] != frozen_by_tick[tick]["logits_sha256"]
    ]
    action_differences = [
        tick for tick in shared
        if closed_by_tick[tick]["action_sha256"] != frozen_by_tick[tick]["action_sha256"]
    ]
    observation_differences = [
        tick for tick in shared
        if closed_by_tick[tick]["live_observation_sha256"]
        != frozen_by_tick[tick]["live_observation_sha256"]
    ]
    return {
        "shared_decision_ticks": len(shared),
        "unmatched_closed_decisions": len(closed_by_tick.keys() - frozen_by_tick.keys()),
        "unmatched_frozen_decisions": len(frozen_by_tick.keys() - closed_by_tick.keys()),
        "different_logits_decisions": len(logit_differences),
        "different_actions_decisions": len(action_differences),
        "different_live_observation_decisions": len(observation_differences),
        "first_logits_difference_tick": logit_differences[0] if logit_differences else "not_observed",
        "first_action_difference_tick": action_differences[0] if action_differences else "not_observed",
        "first_live_observation_difference_tick": (
            observation_differences[0] if observation_differences else "not_observed"
        ),
        "first_physical_state_difference_tick": (
            first_state_divergence if first_state_divergence is not None else "not_observed"
        ),
    }


def _control_fields(row: dict[str, Any]) -> tuple[str, ...]:
    item = row["time_zero_control"]
    return tuple(item[key] for key in (
        "policy_input_sha256", "policy_mask_sha256", "logits_sha256", "action_sha256",
        "checkpoint_sha256",
    ))


def _episode_order(row: dict[str, Any]) -> tuple[str, bool, str]:
    return (
        row["target_mode"],
        row["seat_assignment"] == "target_A",
        row["target"],
    )


def _execute_block(
    map_id: str,
    config: Any,
    target: str,
    opponent: str,
    seed: int,
    calls: list[int],
) -> list[dict[str, Any]]:
    arms = {
        (seat, mode): _new_episode(map_id, config, target, opponent, seed, seat, mode)
        for seat in SEATS for mode in MODES
    }
    ordered_arms = [
        arms[(seat, mode)]
        for seat in SEATS for mode in MODES
    ]
    first_state_divergence: dict[bool, int | None] = {seat: None for seat in SEATS}
    while True:
        active = [
            episode for episode in ordered_arms
            if episode.run.outcome is None and episode.state.tick < episode.config.t_max
        ]
        if not active:
            break
        stepped: dict[int, int] = {}
        for episode in active:
            current_tick = episode.state.tick
            _step_episode(episode, calls)
            stepped[id(episode)] = current_tick
        for seat in SEATS:
            if first_state_divergence[seat] is not None:
                continue
            closed, frozen = arms[(seat, "closed")], arms[(seat, "frozen")]
            if id(closed) not in stepped or id(frozen) not in stepped:
                continue
            closed_tick, frozen_tick = stepped[id(closed)], stepped[id(frozen)]
            if closed_tick != frozen_tick:
                first_state_divergence[seat] = min(closed_tick, frozen_tick)
                continue
            if state_hash(closed.state) != state_hash(frozen.state):
                first_state_divergence[seat] = closed_tick

    block_rows = [
        _episode_row(arms[(seat, mode)])
        for seat in SEATS for mode in MODES
    ]
    rows_by_arm = {
        (row["seat_assignment"] == "target_A", row["target_mode"]): row
        for row in block_rows
    }
    for seat in SEATS:
        closed, frozen = rows_by_arm[(seat, "closed")], rows_by_arm[(seat, "frozen")]
        fields = _control_fields(closed)
        if fields != _control_fields(frozen):
            raise Gate1FeedbackError(
                f"time-zero control check failed for {map_id}/{target}/{opponent}/{seed}/"
                f"{'target_A' if seat else 'target_B'}"
            )
        diagnostics = _compare_mode_decisions(
            arms[(seat, "closed")], arms[(seat, "frozen")], first_state_divergence[seat],
        )
        closed["t0_control_pass"] = frozen["t0_control_pass"] = True
        closed["matched_mode_diagnostics"] = frozen["matched_mode_diagnostics"] = diagnostics

    by_arm = {
        (row["seat_assignment"] == "target_A", row["target_mode"]): row
        for row in block_rows
    }
    trajectory_order = [
        (True, "closed"), (True, "frozen"), (False, "closed"), (False, "frozen"),
    ]
    paired_digest = _sha(canonical_dumps([
        by_arm[key]["episode_policy_trace_sha256"] for key in trajectory_order
    ]))
    for row in block_rows:
        row["paired_trajectory_block_sha256"] = paired_digest
    return block_rows


def _schedule() -> list[tuple[str, str, str, int, bool, str]]:
    return [
        (map_id, target, opponent, seed, target_on_a, mode)
        for map_id in MAP_IDS
        for target in TARGETS
        for opponent in OPPONENTS
        for seed in SEEDS
        for target_on_a in SEATS
        for mode in MODES
    ]


def _fraction(value: Fraction | int) -> dict[str, int]:
    exact = Fraction(value)
    return {"numerator": exact.numerator, "denominator": exact.denominator}


def _mean(values: Sequence[Fraction]) -> Fraction | None:
    return None if not values else sum(values, Fraction(0)) / len(values)


def _block_from_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 4:
        raise Gate1FeedbackError("matched block must have exactly four scheduled episodes")
    by_arm = {
        (row["seat_assignment"] == "target_A", row["target_mode"]): row
        for row in rows
    }
    if len(by_arm) != 4:
        raise Gate1FeedbackError("matched block does not contain exactly four arms")
    if any(row.get("t0_control_pass") is not True for row in rows):
        raise Gate1FeedbackError("matched block is missing a passing time-zero control check")
    if any(row.get("paired_trajectory_block_sha256") != rows[0].get("paired_trajectory_block_sha256")
           for row in rows):
        raise Gate1FeedbackError("matched block arms disagree on the composite trajectory digest")
    if any(row["status"] != "complete" or type(row["target_score_x2"]) is not int for row in rows):
        raise Gate1FeedbackError("censored or incomplete arm in matched block")
    closed_sum = sum(by_arm[(seat, "closed")]["target_score_x2"] for seat in SEATS)
    frozen_sum = sum(by_arm[(seat, "frozen")]["target_score_x2"] for seat in SEATS)
    delta = Fraction(closed_sum - frozen_sum, 4)
    return {
        "paired_trajectory_block_sha256": rows[0]["paired_trajectory_block_sha256"],
        "closed_mean_score": Fraction(closed_sum, 4),
        "frozen_mean_score": Fraction(frozen_sum, 4),
        "delta": delta,
        "closed_score_x2_by_seat": {
            "target_A": by_arm[(True, "closed")]["target_score_x2"],
            "target_B": by_arm[(False, "closed")]["target_score_x2"],
        },
        "frozen_score_x2_by_seat": {
            "target_A": by_arm[(True, "frozen")]["target_score_x2"],
            "target_B": by_arm[(False, "frozen")]["target_score_x2"],
        },
    }


def _target_label(
    estimate: Fraction | None,
    map_means: dict[str, Fraction | None],
    *,
    adequate_coverage: bool,
    complete_schedule: bool,
) -> str:
    if not complete_schedule or not adequate_coverage or estimate is None:
        return "INCONCLUSIVE"
    positive_maps = sum(value is not None and value > 0 for value in map_means.values())
    negative_maps = sum(value is not None and value < 0 for value in map_means.values())
    if estimate >= MDE and positive_maps >= 4:
        return "FEEDBACK_BENEFIT_CANDIDATE"
    if estimate <= -MDE and negative_maps >= 4:
        return "FEEDBACK_HARM_CANDIDATE"
    return "NO_CLEAR_FEEDBACK_SIGNAL_IN_SCHEDULE"


def _diagnostic_summary(
    mode_pairs: Sequence[dict[str, Any]], frozen_arms: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    def histogram(key: str) -> dict[str, int]:
        counts = Counter(str(item[key]) for item in mode_pairs)
        return dict(sorted(counts.items()))

    return {
        "unique_matched_seat_pairs": len(mode_pairs),
        "shared_decision_ticks": sum(item["shared_decision_ticks"] for item in mode_pairs),
        "different_logits_decisions": sum(item["different_logits_decisions"] for item in mode_pairs),
        "different_actions_decisions": sum(item["different_actions_decisions"] for item in mode_pairs),
        "different_live_observation_decisions": sum(
            item["different_live_observation_decisions"] for item in mode_pairs
        ),
        "first_logits_difference_tick_counts": histogram("first_logits_difference_tick"),
        "first_action_difference_tick_counts": histogram("first_action_difference_tick"),
        "first_live_observation_difference_tick_counts": histogram(
            "first_live_observation_difference_tick"
        ),
        "first_physical_state_difference_tick_counts": histogram(
            "first_physical_state_difference_tick"
        ),
        "frozen_target_decision_count": sum(item["decision_count"] for item in frozen_arms),
        "live_mask_changed_from_initial_decisions": sum(
            item["live_mask_changed_from_initial_decisions"] for item in frozen_arms
        ),
        "mask_only_legality_changes": sum(
            item["mask_only_legality_changes"] for item in frozen_arms
        ),
    }


def _build_summary(rows: list[dict[str, Any]], calls: int) -> dict[str, Any]:
    blocks_by_cell: dict[tuple[str, str, str], list[list[dict[str, Any]]]] = defaultdict(list)
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["target"], row["map_id"], row["opponent"], row["seed"])].append(row)
    for (target, map_id, opponent, _seed), block_rows in sorted(grouped.items()):
        blocks_by_cell[(target, map_id, opponent)].append(block_rows)

    cells: list[dict[str, Any]] = []
    cell_exact: dict[tuple[str, str, str], dict[str, Any]] = {}
    all_unique_diagnostics: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_unique_frozen_arms: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for target in TARGETS:
        for map_id in MAP_IDS:
            for opponent in OPPONENTS:
                scheduled = blocks_by_cell[(target, map_id, opponent)]
                unique: dict[str, dict[str, Any]] = {}
                duplicate_count = 0
                for block_rows in scheduled:
                    block = _block_from_rows(block_rows)
                    digest = block["paired_trajectory_block_sha256"]
                    if digest in unique:
                        prior = unique[digest]
                        if (prior["delta"], prior["closed_mean_score"], prior["frozen_mean_score"]) != (
                            block["delta"], block["closed_mean_score"], block["frozen_mean_score"]
                        ):
                            raise Gate1FeedbackError("identical trajectory digest has different outcomes")
                        duplicate_count += 1
                    else:
                        unique[digest] = block
                        block_rows_by_seat = {
                            (row["seat_assignment"], row["target_mode"]): row for row in block_rows
                        }
                        for seat_name in ("target_A", "target_B"):
                            all_unique_diagnostics[target].append(
                                block_rows_by_seat[(seat_name, "closed")]["matched_mode_diagnostics"]
                            )
                            frozen_row = block_rows_by_seat[(seat_name, "frozen")]
                            all_unique_frozen_arms[target].append({
                                "decision_count": frozen_row["decision_counts"]["target"],
                                **frozen_row["target_feedback_diagnostics"],
                            })
                unique_blocks = list(unique.values())
                delta_mean = _mean([item["delta"] for item in unique_blocks])
                closed_mean = _mean([item["closed_mean_score"] for item in unique_blocks])
                frozen_mean = _mean([item["frozen_mean_score"] for item in unique_blocks])
                cell = {
                    "target": target,
                    "map_id": map_id,
                    "opponent": opponent,
                    "scheduled_seed_blocks": len(scheduled),
                    "complete_seed_blocks": sum(
                        all(row["status"] == "complete" for row in block_rows)
                        for block_rows in scheduled
                    ),
                    "unique_paired_trajectory_blocks": len(unique_blocks),
                    "duplicate_seed_blocks": duplicate_count,
                    "mean_closed_target_score": "not_measured" if closed_mean is None else _fraction(closed_mean),
                    "mean_frozen_target_score": "not_measured" if frozen_mean is None else _fraction(frozen_mean),
                    "mean_closed_minus_frozen_delta": "not_measured" if delta_mean is None else _fraction(delta_mean),
                    "unique_paired_block_deltas": [
                        {
                            "trajectory_sha256": item["paired_trajectory_block_sha256"],
                            "delta": _fraction(item["delta"]),
                            "closed_mean_score": _fraction(item["closed_mean_score"]),
                            "frozen_mean_score": _fraction(item["frozen_mean_score"]),
                        }
                        for item in unique_blocks
                    ],
                }
                cells.append(cell)
                cell_exact[(target, map_id, opponent)] = {
                    "delta": delta_mean,
                    "closed": closed_mean,
                    "frozen": frozen_mean,
                    "unique": len(unique_blocks),
                    "complete": cell["complete_seed_blocks"] == len(SEEDS),
                }

    targets = []
    for target in TARGETS:
        target_cells = [cell_exact[(target, map_id, opponent)]
                        for map_id in MAP_IDS for opponent in OPPONENTS]
        complete = all(item["complete"] for item in target_cells)
        cell_means = [item["delta"] for item in target_cells]
        estimate = _mean([value for value in cell_means if value is not None])
        map_means = {
            map_id: _mean([
                cell_exact[(target, map_id, opponent)]["delta"]
                for opponent in OPPONENTS
                if cell_exact[(target, map_id, opponent)]["delta"] is not None
            ])
            for map_id in MAP_IDS
        }
        opponent_means = {
            opponent: _mean([
                cell_exact[(target, map_id, opponent)]["delta"]
                for map_id in MAP_IDS
                if cell_exact[(target, map_id, opponent)]["delta"] is not None
            ])
            for opponent in OPPONENTS
        }
        supported_cells = [
            (map_id, opponent)
            for map_id in MAP_IDS for opponent in OPPONENTS
            if cell_exact[(target, map_id, opponent)]["unique"] >= 2
        ]
        supported_maps = sorted({map_id for map_id, _ in supported_cells})
        supported_opponents = sorted({opponent for _, opponent in supported_cells})
        adequate = len(supported_maps) >= 3 and len(supported_opponents) >= 2
        label = _target_label(
            estimate, map_means, adequate_coverage=adequate, complete_schedule=complete,
        )
        targets.append({
            "target": target,
            "scheduled_matches": sum(row["target"] == target for row in rows),
            "scheduled_matched_blocks": len(SEEDS) * len(MAP_IDS) * len(OPPONENTS),
            "unique_paired_blocks_total": sum(item["unique"] for item in target_cells),
            "coverage": {
                "cells_with_at_least_two_unique_blocks": len(supported_cells),
                "maps_with_at_least_one_supported_cell": supported_maps,
                "opponent_families_with_at_least_one_supported_cell": supported_opponents,
                "adequate_for_candidate_label": adequate,
            },
            "target_level_delta_equal_weight_cells": "not_measured" if estimate is None else _fraction(estimate),
            "map_mean_deltas": {
                key: "not_measured" if value is None else _fraction(value)
                for key, value in map_means.items()
            },
            "opponent_mean_deltas": {
                key: "not_measured" if value is None else _fraction(value)
                for key, value in opponent_means.items()
            },
            "label": label,
            "unique_trajectory_feedback_diagnostics": _diagnostic_summary(
                all_unique_diagnostics[target], all_unique_frozen_arms[target],
            ),
            "coverage_limit": (
                "at least two distinct paired trajectories inside a target×map×opponent cell; "
                "qualifying cells must span 3 maps and 2 opponent families"
            ),
        })

    complete_rows = [row for row in rows if row["status"] == "complete"]
    timeouts = sum(
        row["terminal"]["reason"] == "timeout" for row in complete_rows
    )
    return {
        "schema": SCHEMA_SUMMARY,
        "status": "complete" if len(rows) == EXPECTED_MATCHES and len(complete_rows) == EXPECTED_MATCHES else "incomplete",
        "scheduled_matches": len(rows),
        "complete_matches": len(complete_rows),
        "censored_or_incomplete_matches": len(rows) - len(complete_rows),
        "step_tick_calls": calls,
        "step_tick_call_budget": STEP_BUDGET,
        "total_engine_timeout_draws": timeouts,
        "target_results": targets,
        "target_map_opponent_cells": cells,
        "interpretation": "descriptive_on_fixed_maps_and_open_seeds_no_confidence_intervals_or_p_values",
        "independence": (
            "collapse two seats inside each (map,target,opponent,seed) block; deduplicate by "
            "the four policy-relative matched episode traces; root seed is not part of the digest"
        ),
    }


def _versions() -> dict[str, str]:
    return {
        "rules_version": RULES_VERSION,
        "balance_version": BALANCE_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "action_version": ACTION_VERSION,
        "decoder_version": DECODER_VERSION,
        "rng_version": RNG_VERSION,
        "stream_derivation_version": STREAM_DERIVATION_VERSION,
        "serialization_version": SERIALIZATION_VERSION,
    }


def _study_descriptor() -> dict[str, Any]:
    return {
        "maps": list(MAP_IDS),
        "targets": list(TARGETS),
        "opponents": list(OPPONENTS),
        "open_evaluation_seeds": list(SEEDS),
        "seat_assignments": ["target_A", "target_B"],
        "target_modes": list(MODES),
        "expected_matches": EXPECTED_MATCHES,
        "step_tick_call_budget": STEP_BUDGET,
        "max_ticks_per_episode": 3000,
        "policy_interval_k": 30,
        "primary_estimand": "target_relative_mean_score_closed_minus_frozen_within_seed_block_after_seat_collapse",
        "score_encoding": "score_x2 in {0,1,2}; exact rational arithmetic",
        "mde": {"numerator": MDE.numerator, "denominator": MDE.denominator},
        "candidate_rule": "target_delta_at_least_mde_and_correct_sign_on_at_least_four_of_six_map_means",
        "coverage_rule": "at_least_two_unique_joint_trajectories_within_cells_spanning_three_maps_and_two_opponent_families",
        "random_seed_in_trajectory_digest": False,
        "policy_stream_assignment": (
            "target uses replicate 0; opponent uses replicate 1 only in self-play, otherwise 0; "
            "streams follow policy identity across seat swaps"
        ),
        "confidence_intervals": "none; fixed maps and trajectory duplicates are not independent",
        "frozen_input": "initial ObservationV2 with only time[0:2] updated; initial ActionMaskV2",
        "frozen_legality": "live ActionMaskV2 only at shared decoder/validator; mask-only legality adaptation",
        "control_check": "same mode t0 inputs/mask/logits/action and post-t0 full checkpoint hash per seat",
        "horizons_completed_ticks": list(HORIZONS),
    }


def _source_fingerprint() -> str:
    package = _ROOT / "src" / "formicmind"
    entries = [
        {
            "path": path.relative_to(_ROOT).as_posix(),
            "sha256": _sha(path.read_bytes()),
        }
        for path in sorted(package.rglob("*.py"))
    ]
    return _sha(canonical_dumps(entries))


def _git_value(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=_ROOT, check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Gate1FeedbackError(f"git {' '.join(args)} failed: {exc}") from exc
    return result.stdout.strip()


def _protocol_sha() -> str:
    try:
        return _sha((_ROOT / PROTOCOL_PATH).read_bytes())
    except OSError as exc:
        raise Gate1FeedbackError(f"cannot read frozen protocol: {exc}") from exc


def _manifest(source_commit: str) -> dict[str, Any]:
    configs = _build_configs()
    config_hashes = {map_id: config_hash(configs[map_id]) for map_id in MAP_IDS}
    if config_hashes != _EXPECTED_CONFIG_HASHES:
        raise Gate1FeedbackError("FOR-22 canonical config hashes changed")
    policies = {name: _policy_descriptor(name) for name in sorted(set((*TARGETS, *OPPONENTS)))}
    for name, expected in _EXPECTED_POLICY_HASHES.items():
        if policies[name]["parameters_sha256"] != expected:
            raise Gate1FeedbackError(f"baseline parameters changed for {name}")
    return {
        "schema": SCHEMA_MANIFEST,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": _protocol_sha(),
        "source_commit_sha": source_commit,
        "source_fingerprint_sha256": _source_fingerprint(),
        "base_main_sha": BASE_MAIN_SHA,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "versions": _versions(),
        "maps": [{"id": map_id, "config_sha256": config_hashes[map_id]}
                 for map_id in MAP_IDS],
        "policies": policies,
        "study": _study_descriptor(),
        "command": "PYTHONPATH=src python -m formicmind.gate1_feedback run --out results/gate1_feedback",
        "status": "complete",
    }


def _json_bytes(value: Any) -> bytes:
    return canonical_dumps(value)


def _write_json(path: Path, value: Any) -> bytes:
    data = _json_bytes(value) + b"\n"
    path.write_bytes(data)
    return data


def _write_rows(rows: Sequence[dict[str, Any]]) -> bytes:
    return b"".join(_json_bytes(row) + b"\n" for row in rows)


def _parse_rows(data: bytes) -> list[dict[str, Any]]:
    rows = []
    for index, line in enumerate(data.splitlines(), start=1):
        try:
            item = canonical_loads(line)
        except Exception as exc:
            raise Gate1FeedbackError(f"invalid JSONL row {index}: {exc}") from exc
        if not isinstance(item, dict):
            raise Gate1FeedbackError(f"JSONL row {index} is not an object")
        rows.append(item)
    return rows


def _assert_manifest(manifest: dict[str, Any]) -> None:
    expected_keys = _manifest(manifest.get("source_commit_sha", ""))
    for key, expected in expected_keys.items():
        if manifest.get(key) != expected:
            raise Gate1FeedbackError(f"manifest {key} differs from frozen protocol/current source")
    if manifest.get("status") != "complete":
        raise Gate1FeedbackError("cannot strictly verify an incomplete study")
    source_commit = manifest.get("source_commit_sha")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise Gate1FeedbackError("manifest source commit SHA is invalid")
    try:
        _git_value("merge-base", "--is-ancestor", BASE_MAIN_SHA, source_commit)
        _git_value("merge-base", "--is-ancestor", source_commit, "HEAD")
    except Gate1FeedbackError as exc:
        raise Gate1FeedbackError("base/source commit is not in verification history") from exc


def run(out: str | Path = OUT_RELATIVE) -> dict[str, Any]:
    output = Path(out)
    if not output.is_absolute():
        output = _ROOT / output
    if output.exists():
        raise Gate1FeedbackError(f"output directory already exists: {output}")
    source_commit = _git_value("rev-parse", "HEAD")
    _git_value("merge-base", "--is-ancestor", BASE_MAIN_SHA, source_commit)
    output.mkdir(parents=True)
    calls = [0]
    rows: list[dict[str, Any]] = []
    configs = _build_configs()
    completed_blocks = 0
    try:
        for map_id in MAP_IDS:
            config = configs[map_id]
            for target in TARGETS:
                for opponent in OPPONENTS:
                    for seed in SEEDS:
                        block_rows = _execute_block(map_id, config, target, opponent, seed, calls)
                        if any(row["status"] != "complete" for row in block_rows):
                            raise Gate1FeedbackError(
                                f"incomplete/censored episode in {map_id}/{target}/{opponent}/{seed}"
                            )
                        rows.extend(block_rows)
                        completed_blocks += 1
                        if completed_blocks % 24 == 0:
                            print(
                                f"gate1_feedback: {completed_blocks}/432 matched blocks, "
                                f"{len(rows)}/{EXPECTED_MATCHES} matches, {calls[0]} step_tick calls",
                                flush=True,
                            )
        if len(rows) != EXPECTED_MATCHES:
            raise Gate1FeedbackError(f"scheduled {len(rows)} rows, expected {EXPECTED_MATCHES}")
        summary = _build_summary(rows, calls[0])
        if summary["status"] != "complete":
            raise Gate1FeedbackError("summary is incomplete")
        manifest = _manifest(source_commit)
        rows_bytes = _write_rows(rows)
        summary_bytes = _json_bytes(summary) + b"\n"
        manifest.update({
            "step_tick_calls": calls[0],
            "scheduled_matches": len(rows),
            "artifact_sha256": {
                "matches.jsonl": _sha(rows_bytes),
                "summary.json": _sha(summary_bytes),
            },
        })
        manifest_bytes = _json_bytes(manifest) + b"\n"
        (output / "matches.jsonl").write_bytes(rows_bytes)
        (output / "summary.json").write_bytes(summary_bytes)
        (output / "manifest.json").write_bytes(manifest_bytes)
        return {
            "status": "complete",
            "scheduled_matches": len(rows),
            "matched_seed_blocks": completed_blocks,
            "step_tick_calls": calls[0],
            "source_fingerprint_sha256": manifest["source_fingerprint_sha256"],
            "matches_jsonl_sha256": manifest["artifact_sha256"]["matches.jsonl"],
            "summary_sha256": manifest["artifact_sha256"]["summary.json"],
        }
    except Exception as exc:
        failure = {
            "protocol_id": PROTOCOL_ID,
            "source_commit_sha": source_commit,
            "completed_matched_blocks": completed_blocks,
            "partial_rows_in_memory": len(rows),
            "step_tick_calls": calls[0],
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        (output / "failure.json").write_bytes(_json_bytes(failure) + b"\n")
        raise


def _expected_schedule_rows(rows: Sequence[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [
        (
            row.get("map_id"), row.get("target"), row.get("opponent"), row.get("seed"),
            row.get("seat_assignment") == "target_A", row.get("target_mode"),
        )
        for row in rows
    ]


def verify(input_path: str | Path = OUT_RELATIVE) -> dict[str, Any]:
    source = Path(input_path)
    if not source.is_absolute():
        source = _ROOT / source
    required = {"manifest.json", "matches.jsonl", "summary.json"}
    if not source.is_dir() or {path.name for path in source.iterdir()} != required:
        raise Gate1FeedbackError("input must contain exactly manifest.json, matches.jsonl, and summary.json")
    try:
        manifest = canonical_loads((source / "manifest.json").read_bytes().rstrip(b"\n"))
    except Exception as exc:
        raise Gate1FeedbackError(f"invalid manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise Gate1FeedbackError("manifest must be an object")
    _assert_manifest(manifest)
    matches_bytes = (source / "matches.jsonl").read_bytes()
    summary_bytes = (source / "summary.json").read_bytes()
    if manifest.get("artifact_sha256") != {
        "matches.jsonl": _sha(matches_bytes),
        "summary.json": _sha(summary_bytes),
    }:
        raise Gate1FeedbackError("saved artifact hash mismatch")
    stored_rows = _parse_rows(matches_bytes)
    if len(stored_rows) != EXPECTED_MATCHES or manifest.get("scheduled_matches") != EXPECTED_MATCHES:
        raise Gate1FeedbackError("saved schedule is missing or has extra rows")
    if _expected_schedule_rows(stored_rows) != _schedule():
        raise Gate1FeedbackError("saved rows are missing, extra, reordered, or outside the fixed schedule")
    configs = _build_configs()
    calls = [0]
    replayed: list[dict[str, Any]] = []
    completed_blocks = 0
    for map_id in MAP_IDS:
        for target in TARGETS:
            for opponent in OPPONENTS:
                for seed in SEEDS:
                    block_rows = _execute_block(map_id, configs[map_id], target, opponent, seed, calls)
                    if any(row["status"] != "complete" for row in block_rows):
                        raise Gate1FeedbackError("strict replay encountered a censored/incomplete episode")
                    replayed.extend(block_rows)
                    completed_blocks += 1
                    if completed_blocks % 24 == 0:
                        print(
                            f"gate1_feedback verify: {completed_blocks}/432 matched blocks, "
                            f"{len(replayed)}/{EXPECTED_MATCHES} matches, {calls[0]} step_tick calls",
                            flush=True,
                        )
    replayed_bytes = _write_rows(replayed)
    if replayed_bytes != matches_bytes:
        for index, (expected, actual) in enumerate(zip(stored_rows, replayed)):
            if expected != actual:
                raise Gate1FeedbackError(f"strict replay differs at scheduled row {index}: {expected.get('map_id')}")
        raise Gate1FeedbackError("strict replay row bytes differ")
    expected_summary = _build_summary(replayed, calls[0])
    expected_summary_bytes = _json_bytes(expected_summary) + b"\n"
    if expected_summary_bytes != summary_bytes:
        raise Gate1FeedbackError("recomputed matched analysis differs from saved summary")
    if manifest.get("step_tick_calls") != calls[0]:
        raise Gate1FeedbackError("replayed step_tick call count differs from manifest")
    return {
        "status": "verified",
        "verified_matches": len(replayed),
        "matched_seed_blocks": completed_blocks,
        "step_tick_calls": calls[0],
        "source_fingerprint_sha256": _source_fingerprint(),
        "matches_jsonl_sha256": _sha(matches_bytes),
        "summary_sha256": _sha(summary_bytes),
        "target_labels": {item["target"]: item["label"] for item in expected_summary["target_results"]},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m formicmind.gate1_feedback", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="run the fixed 1728-match feedback ablation")
    runner.add_argument("--out", required=True, help="new, not-yet-existing output directory")
    verifier = commands.add_parser("verify", help="strictly replay the full 1728-match study")
    verifier.add_argument("--input", required=True, help="saved study output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args.out) if args.command == "run" else verify(args.input)
    except (Gate1FeedbackError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"gate1_feedback: error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0 if result.get("status") in {"complete", "verified"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
