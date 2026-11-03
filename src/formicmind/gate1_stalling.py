"""Preregistered Gate 1D timeout/stalling audit over the unchanged L0 engine."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import platform
import subprocess
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

from .actions import ActionMaskV2, ActionV2, ValidatedActionV2, flatten_mask_v2, get_action_mask, validate_action
from .baselines import baseline_spec, make_baseline
from .canonical import canonical_dumps, canonical_loads
from .events import CombatEvent, Event, HarvestEvent, LayEvent, OrderEvent, QueenDamageEvent, RetargetEvent, StarvationEvent, UpkeepEvent, events_hash
from .gate1_causal import _ArrayDigest, _MirrorNormalizer, _action_diff, _mask_sha256, _observation_sha256, _physical_state_object
from .gate1_context import _build_configs, _colony_metrics, _finish_policy_metrics, _map_descriptors, _new_policy_metrics, _policy_descriptor, _record_event, _versions
from .observation import ObservationV2, build_observation, flatten_v2
from .policy import decide
from .reset import reset
from .rng import policy_stream_id
from .rules import ARMY_RAID_ENEMY_HOME, MIRROR, Q_TOTAL
from .runstate import RunState
from .serialization import checkpoint_hash, config_hash, state_hash
from .snapshot import begin_policy_phase
from .tick import Controller, EpisodeFinishedError, PolicyController, TickResult, step_tick

PROTOCOL_ID = "formicmind.gate1-stalling/v1"
PROTOCOL_PATH = Path("docs/gate1_stalling_protocol.md")
SCHEMA_MANIFEST = "formicmind.gate1-stalling-manifest/v1"
SCHEMA_BASELINE = "formicmind.gate1-stalling-baseline/v1"
SCHEMA_PROBE = "formicmind.gate1-stalling-probe/v1"
SCHEMA_SUMMARY = "formicmind.gate1-stalling-summary/v1"
MAPS = ("R0-T0", "R0-T1", "R0-T2", "R1-T0", "R1-T1", "R1-T2")
PAIRS = (("turtle", "balanced"), ("turtle", "raider"), ("rush", "balanced"), ("random", "turtle"))
SEEDS = tuple(range(20270201, 20270207))
SEATS = (False, True)  # first policy on A, then first policy on B
PROFILE_TICKS = (0, 100, 300, 600, 900, 1500, 2100, 2700)
PROBE_TICKS = (300, 900)
INTERVENTIONS = ("raid_once", "home_forage_once")
STEP_BUDGET = 2_016_000
BASELINE_BUDGET = 864_000
PROBE_BUDGET = 1_152_000
EXPECTED_BASELINES = 288
EXPECTED_PROBES = 384
_ROOT = Path(__file__).resolve().parents[2]


class Gate1StallingError(ValueError):
    """Invalid fixed-study input or strict verification mismatch."""


class _Digest:
    """Incremental SHA-256 of canonical JSON array entries."""

    def __init__(self) -> None:
        self.hash = hashlib.sha256(b"[")
        self.count = 0

    def add(self, value: Any) -> None:
        if self.count:
            self.hash.update(b",")
        self.hash.update(canonical_dumps(value))
        self.count += 1

    def hexdigest(self) -> str:
        result = self.hash.copy()
        result.update(b"]")
        return result.hexdigest()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _protocol_sha256() -> str:
    return _sha((_ROOT / PROTOCOL_PATH).read_bytes())


def _source_fingerprint() -> str:
    package = _ROOT / "src" / "formicmind"
    entries = [
        {"path": path.relative_to(_ROOT).as_posix(), "sha256": _sha(path.read_bytes())}
        for path in sorted(package.rglob("*.py"))
    ]
    return _sha(canonical_dumps(entries))


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=_ROOT, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _git_head() -> str:
    value = _git("rev-parse", "HEAD")
    if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise Gate1StallingError("git HEAD is not a lowercase commit SHA")
    return value


def _safe(value: Any) -> Any:
    if value is None:
        return "not_measured"
    if isinstance(value, float):
        return format(value, ".17g")
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [_safe(v) for v in value]
    if isinstance(value, Counter):
        return {str(k): int(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    return value


def _action_object(action: ActionV2) -> dict[str, Any]:
    return {
        "lay_order": action.lay_order,
        "forage_share_q": list(action.forage_share_q),
        "army_choice": action.army_choice,
        "avoid_center": action.avoid_center,
    }


def _candidate_action(intervention: str, control: ActionV2, mask: ActionMaskV2) -> tuple[ActionV2 | None, str, str]:
    if intervention == "raid_once":
        if control.army_choice == ARMY_RAID_ENEMY_HOME:
            return None, "no_effect", "already_army_choice_6"
        if not mask.army_choice[ARMY_RAID_ENEMY_HOME]:
            return None, "ineligible", "army_choice_6_masked"
        candidate = ActionV2(control.lay_order, control.forage_share_q, ARMY_RAID_ENEMY_HOME, control.avoid_center)
        return candidate, "eligible", "eligible"
    if intervention == "home_forage_once":
        shares = (Q_TOTAL, 0, 0, 0, 0, 0)
        if control.forage_share_q == shares:
            return None, "no_effect", "already_home_only"
        return ActionV2(control.lay_order, shares, control.army_choice, control.avoid_center), "eligible", "eligible"
    raise Gate1StallingError(f"unknown one-shot intervention {intervention}")


def _event_action(event: OrderEvent) -> ActionV2:
    return ActionV2(event.lay_order, tuple(event.forage_share_q), event.army_choice, event.avoid_center)


def _local_node(side: int, node: int) -> int:
    return node if side == 0 or node < 0 else MIRROR[node]


def _policy_streams(left: str, right: str) -> tuple[str, str]:
    return (
        policy_stream_id(baseline_spec(left).policy_id, 0),
        policy_stream_id(baseline_spec(right).policy_id, 0),
    )


def _controllers(left: str, right: str, swapped: bool) -> tuple[PolicyController, PolicyController]:
    first, second = (right, left) if swapped else (left, right)
    return tuple(PolicyController(make_baseline(name), policy_stream_id(baseline_spec(name).policy_id, 0))
                 for name in (first, second))  # type: ignore[return-value]


def _stream_state(run: RunState, policy: str) -> Any:
    return run.stream(policy_stream_id(baseline_spec(policy).policy_id, 0)).to_canonical()


def _normalized_checkpoint_sha(normalizer: _MirrorNormalizer, state: Any, run: RunState, policies: tuple[str, str]) -> str:
    value: dict[str, Any] = {
        "state": normalizer.state_object(state),
        "windows": [
            {"harvest": w.harvest, "upkeep_paid": w.upkeep_paid, "egg_spend": w.egg_spend,
             "starvation_debt_delta": w.starvation_debt_delta, "window_ticks": w.window_ticks}
            for w in (run.windows[normalizer.target_side], run.windows[1 - normalizer.target_side])
        ],
    }
    if "random" in policies:
        value["random_policy_stream"] = _stream_state(run, "random")
    if run.outcome is not None:
        outcome = run.outcome.to_canonical()
        if outcome.get("winner") in ("A", "B"):
            side = 0 if outcome["winner"] == "A" else 1
            outcome["winner"] = "target" if side == normalizer.target_side else "opponent"
        outcome["sides"] = sorted(0 if s == normalizer.target_side else 1 for s in outcome.get("sides", []))
        value["outcome"] = outcome
    return _sha(canonical_dumps(value))


def _metric_snapshot(state: Any, policies_by_side: tuple[str, str]) -> dict[str, Any]:
    return {
        policy: _colony_metrics(state, side)
        for side, policy in enumerate(policies_by_side)
    }


def _profile_snapshot(state: Any, policies_by_side: tuple[str, str], metrics: dict[str, Any],
                      history: deque[dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": "reached", "completed_ticks": state.tick,
        "colonies": _metric_snapshot(state, policies_by_side),
        "cumulative_event_metrics": _finish_metrics(metrics),
        "rolling_events_last_600_completed_ticks": _window_totals(history, 600),
    }


def _finish_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result = _finish_policy_metrics(metrics)
    for item in result.values():
        item["ever_ordered_raid"] = item["army_choice_counts"].get("6", 0) > 0
    return _safe(result)


def _matchup_id(left: str, right: str) -> str:
    return f"{left}-{right}"


def _baseline_id(map_id: str, left: str, right: str, seed: int, swapped: bool) -> str:
    seat = "first_B" if swapped else "first_A"
    return f"{map_id}|{left}-{right}|{seed}|{seat}"


def _schedule() -> list[tuple[str, str, str, int, bool]]:
    return [
        (map_id, left, right, seed, swapped)
        for map_id in MAPS for left, right in PAIRS for seed in SEEDS for swapped in SEATS
    ]


def _probe_schedule() -> list[tuple[dict[str, Any], int, str, str]]:
    probes = []
    eligible_maps = {"R0-T0", "R1-T0"}
    eligible_pairs = {("turtle", "balanced"), ("turtle", "raider")}
    for row in _schedule():
        map_id, left, right, seed, swapped = row
        if map_id not in eligible_maps or (left, right) not in eligible_pairs:
            continue
        descriptor = {
            "baseline_id": _baseline_id(map_id, left, right, seed, swapped),
            "map_id": map_id,
            "matchup": _matchup_id(left, right),
            "seed": seed,
            "seat_assignment": "first_B" if swapped else "first_A",
        }
        for tick in PROBE_TICKS:
            for target in (left, right):
                for intervention in INTERVENTIONS:
                    probes.append((descriptor, tick, target, intervention))
    return probes


@dataclass(slots=True)
class _Checkpoint:
    state: Any
    run: RunState
    normalizer: _MirrorNormalizer
    tick: int
    target_policy: str
    opponent_policy: str
    target_side: int
    control_action: ActionV2
    candidate_action: ActionV2 | None
    eligibility_status: str
    eligibility_reason: str
    state_sha256: str
    checkpoint_sha256: str
    normalized_checkpoint_sha256: str
    observation_sha256: str
    mask_sha256: str
    decision_sha256: str
    stream_before: Any
    stream_after: Any
    control_suffix_sha256: str = "not_measured"


def _capture_checkpoint(
    state: Any, run: RunState, *, tick: int, target_policy: str, opponent_policy: str,
    target_side: int, normalizer: _MirrorNormalizer, intervention: str,
) -> _Checkpoint:
    state_copy, run_copy = state.clone(), run.clone()
    pre_state_sha = state_hash(state_copy)
    pre_checkpoint_sha = checkpoint_hash(state_copy, run_copy)
    phase_state, phase_run = state_copy.clone(), run_copy.clone()
    snapshot = begin_policy_phase(phase_state, (phase_run.windows[0], phase_run.windows[1]))
    observation = build_observation(snapshot, target_side)
    mask = get_action_mask(observation)
    stream_id = policy_stream_id(baseline_spec(target_policy).policy_id, 0)
    stream_before = phase_run.stream(stream_id).to_canonical()
    decision = decide(make_baseline(target_policy), observation, phase_run.stream(stream_id))
    stream_after = phase_run.stream(stream_id).to_canonical()
    control = validate_action(observation, mask, decision.action).action
    action, status, reason = _candidate_action(intervention, control, mask)
    if action is not None:
        if _action_diff(control, action) != (["army_choice"] if intervention == "raid_once" else ["forage_share_q"]):
            raise Gate1StallingError("counterfactual changes more than its preregistered action axis")
        try:
            validate_action(observation, mask, action)
        except (TypeError, ValueError) as exc:
            status, reason, action = "ineligible", f"action_invalid:{type(exc).__name__}:{exc}", None
    decision_sha = _sha(canonical_dumps({
        "observation_sha256": _observation_sha256(observation),
        "mask_sha256": _mask_sha256(mask),
        "action": _action_object(control),
        "stream_before": stream_before,
        "stream_after": stream_after,
        "time": [format(value, ".17g") for value in observation.block("time")],
    }))
    return _Checkpoint(
        state=state_copy, run=run_copy, normalizer=copy.deepcopy(normalizer), tick=tick,
        target_policy=target_policy, opponent_policy=opponent_policy, target_side=target_side,
        control_action=control, candidate_action=action, eligibility_status=status,
        eligibility_reason=reason, state_sha256=pre_state_sha, checkpoint_sha256=pre_checkpoint_sha,
        normalized_checkpoint_sha256=_normalized_checkpoint_sha(normalizer, state, run, (target_policy, opponent_policy)),
        observation_sha256=_observation_sha256(observation), mask_sha256=_mask_sha256(mask),
        decision_sha256=decision_sha, stream_before=stream_before, stream_after=stream_after,
    )


def _terminal_object(state: Any, run: RunState, target_side: int) -> dict[str, Any]:
    outcome = run.outcome
    if outcome is None:
        return {"status": "incomplete", "reason": "not_measured", "winner": "not_measured",
                "score_x2": "not_measured", "terminal_tick": "not_measured", "tick_completed": "not_measured"}
    return {
        "status": "complete", "reason": outcome.reason.value,
        "winner": outcome.winner.value,
        "target_score_x2": int(outcome.score[target_side] * 2),
        "opponent_score_x2": int(outcome.score[1 - target_side] * 2),
        "terminal_tick": outcome.terminal_tick,
        "terminal_phase": int(outcome.terminal_phase),
        "tick_completed": outcome.tick_completed,
        "state_tick": state.tick,
    }


def _record_window_tick(history: deque[dict[str, Any]], tick: int, completed: bool, events: Iterable[Event], policies_by_side: tuple[str, str]) -> None:
    item: dict[str, Any] = {
        "tick": tick, "completed": completed, "combat_events": 0,
        "queen_damage": {policy: 0 for policy in policies_by_side},
        "starvation_deaths": {policy: 0 for policy in policies_by_side},
        "lay_food_rejections": {policy: 0 for policy in policies_by_side},
        "raid_orders": {policy: 0 for policy in policies_by_side},
    }
    for event in events:
        if isinstance(event, CombatEvent):
            item["combat_events"] += 1
        if isinstance(event, QueenDamageEvent):
            item["queen_damage"][policies_by_side[event.side]] += event.damage
        elif isinstance(event, StarvationEvent):
            item["starvation_deaths"][policies_by_side[event.side]] += 1
        elif isinstance(event, LayEvent) and not event.laid:
            item["lay_food_rejections"][policies_by_side[event.side]] += 1
        elif isinstance(event, OrderEvent) and event.army_choice == ARMY_RAID_ENEMY_HOME:
            item["raid_orders"][policies_by_side[event.side]] += 1
    history.append(item)


def _window_totals(history: deque[dict[str, Any]], length: int) -> dict[str, Any]:
    selected = [item for item in reversed(history) if item["completed"]][:length]
    selected.reverse()
    policies = sorted(history[-1]["queen_damage"]) if history else []
    return {
        "requested_completed_ticks": length,
        "available_completed_ticks": len(selected),
        "start_tick": selected[0]["tick"] if selected else "not_measured",
        "end_tick": selected[-1]["tick"] if selected else "not_measured",
        "combat_events": sum(item["combat_events"] for item in selected),
        "queen_damage": {p: sum(item["queen_damage"][p] for item in selected) for p in policies},
        "starvation_deaths": {p: sum(item["starvation_deaths"][p] for item in selected) for p in policies},
        "lay_food_rejections": {p: sum(item["lay_food_rejections"][p] for item in selected) for p in policies},
        "raid_orders": {p: sum(item["raid_orders"][p] for item in selected) for p in policies},
    }


@dataclass(slots=True)
class _RunBudget:
    limit: int
    calls: int = 0

    def step(self, state: Any, run: RunState, controllers: Sequence[Controller]) -> TickResult:
        if self.calls >= self.limit:
            raise Gate1StallingError(f"step_tick call budget exceeded ({self.limit})")
        result = step_tick(state, run, controllers)
        self.calls += 1
        return result


def _run_baseline(map_id: str, config: Any, left: str, right: str, seed: int, swapped: bool,
                  budget: _RunBudget) -> tuple[dict[str, Any], dict[tuple[Any, ...], _Checkpoint], dict[str, dict[int, str]]]:
    controllers = _controllers(left, right, swapped)
    streams = tuple(c.stream_id for c in controllers)
    state, run = reset(config, seed, streams)
    policies_by_side = (right, left) if swapped else (left, right)
    side_by_policy = {name: side for side, name in enumerate(policies_by_side)}
    normalizers = {name: _MirrorNormalizer(side_by_policy[name]) for name in (left, right)}
    first_normalizer = normalizers[left]
    checkpoints: dict[tuple[Any, ...], _Checkpoint] = {}
    profiles: dict[str, Any] = {str(tick): None for tick in PROFILE_TICKS}
    metrics = {left: _new_policy_metrics(), right: _new_policy_metrics()}
    event_counts: Counter[str] = Counter()
    event_digest = _Digest()
    raw_trace = _Digest()
    raw_tick_rows: list[dict[str, Any]] = []
    mirror_trace = _Digest()
    random_trace = _Digest()
    physical_hashes = {name: {} for name in (left, right)}
    physical_digests = {name: _Digest() for name in (left, right)}
    orders: dict[str, dict[str, Any]] = {left: {}, right: {}}
    window_history: deque[dict[str, Any]] = deque(maxlen=900)
    profiles["0"] = _profile_snapshot(state, policies_by_side, metrics, window_history)
    executed = 0
    baseline_key = _baseline_id(map_id, left, right, seed, swapped)
    while run.outcome is None and state.tick < config.t_max:
        tick = state.tick
        if tick in PROBE_TICKS and map_id in {"R0-T0", "R1-T0"} and (left, right) in {("turtle", "balanced"), ("turtle", "raider")}:
            for target in (left, right):
                checkpoints[(tick, target, "raid_once")] = _capture_checkpoint(
                    state, run, tick=tick, target_policy=target,
                    opponent_policy=right if target == left else left,
                    target_side=side_by_policy[target], normalizer=normalizers[target],
                    intervention="raid_once",  # action-specific eligibility is recomputed below
                )
                # Capture the checkpoint once; each alternative is independently preflighted.
                checkpoints[(tick, target, "home_forage_once")] = _capture_checkpoint(
                    state, run, tick=tick, target_policy=target,
                    opponent_policy=right if target == left else left,
                    target_side=side_by_policy[target], normalizer=normalizers[target],
                    intervention="home_forage_once",
                )
        start_state_sha = state_hash(state)
        start_checkpoint_sha = checkpoint_hash(state, run)
        normalized_start = first_normalizer.state_object(state)
        for name in (left, right):
            normalizers[name].state_object(state)
        normalized_start_checkpoint = _normalized_checkpoint_sha(first_normalizer, state, run, (left, right))
        random_before = _stream_state(run, "random") if "random" in (left, right) and tick % config.policy_interval_k == 0 else "not_measured"
        result = budget.step(state, run, controllers)
        executed += 1
        tick_events = list(result.events)
        event_counts.update(event.kind for event in tick_events)
        event_digest.add({"tick": tick, "events_sha256": events_hash(tick_events)})
        for event in tick_events:
            _record_event(metrics, policies_by_side, event)
            if isinstance(event, OrderEvent):
                policy = policies_by_side[event.side]
                action_obj = _action_object(_event_action(event))
                orders[policy][str(tick)] = action_obj
                if tick in PROBE_TICKS:
                    for intervention in INTERVENTIONS:
                        candidate_key = (tick, policy, intervention)
                        checkpoint = checkpoints.get(candidate_key)
                        if checkpoint is not None and checkpoint.control_action != _event_action(event):
                            raise Gate1StallingError("checkpoint policy decision differs from baseline OrderEvent")
        normalized_events = [first_normalizer.event_object(event) for event in tick_events]
        normalized_events.sort(key=canonical_dumps)
        random_after = _stream_state(run, "random") if "random" in (left, right) and tick % config.policy_interval_k == 0 else "not_measured"
        mirror_entry: dict[str, Any] = {
            "tick": tick, "start_state": normalized_start,
            "start_checkpoint_sha256": normalized_start_checkpoint,
            "events": normalized_events,
        }
        if random_before != "not_measured":
            mirror_entry["random_stream_before"] = random_before
            mirror_entry["random_stream_after"] = random_after
            random_trace.add({"tick": tick, "before": random_before, "after": random_after})
        mirror_entry["end_state"] = first_normalizer.state_object(state)
        mirror_entry["end_checkpoint_sha256"] = _normalized_checkpoint_sha(first_normalizer, state, run, (left, right))
        mirror_trace.add(mirror_entry)
        trace_row = {
            "tick": tick, "start_state_sha256": start_state_sha,
            "start_checkpoint_sha256": start_checkpoint_sha,
            "end_state_sha256": state_hash(state),
            "end_checkpoint_sha256": checkpoint_hash(state, run),
            "events_sha256": events_hash(tick_events),
            "tick_completed": state.tick == tick + 1,
        }
        raw_trace.add(trace_row)
        raw_tick_rows.append(trace_row)
        physical_events = [
            first_normalizer.event_object(event) for event in tick_events
            if event.kind not in ("order", "retarget")
        ]
        physical_events.sort(key=canonical_dumps)
        phys = _sha(canonical_dumps({
            "state": _physical_state_object(state, side_by_policy[left]),
            "events": physical_events,
        }))
        physical_hashes[left][tick] = phys
        physical_digests[left].add({"tick": tick, "physical_sha256": phys})
        # The physical state projection is target-relative; normalize the event stream for the other role too.
        other_norm = normalizers[right]
        other_norm.state_object(state)
        other_events = [other_norm.event_object(event) for event in tick_events if event.kind not in ("order", "retarget")]
        other_events.sort(key=canonical_dumps)
        physical_hashes[right][tick] = _sha(canonical_dumps({
            "state": _physical_state_object(state, side_by_policy[right]), "events": other_events,
        }))
        physical_digests[right].add({"tick": tick, "physical_sha256": physical_hashes[right][tick]})
        completed = state.tick == tick + 1
        _record_window_tick(window_history, tick, completed, tick_events, policies_by_side)
        if state.tick in PROFILE_TICKS and str(state.tick) != "0":
            profiles[str(state.tick)] = _profile_snapshot(state, policies_by_side, metrics, window_history)
    terminal = _terminal_object(state, run, side_by_policy[left])
    for checkpoint in checkpoints.values():
        suffix = _Digest()
        for trace_row in raw_tick_rows:
            if trace_row["tick"] >= checkpoint.tick:
                suffix.add(trace_row)
        checkpoint.control_suffix_sha256 = suffix.hexdigest()
    if run.outcome is None:
        status = "incomplete"
    else:
        status = "complete"
    for tick in PROFILE_TICKS:
        if profiles[str(tick)] is None:
            profiles[str(tick)] = {"status": "not_measured", "reason": "episode_ended_before_checkpoint",
                                   "terminal_tick": terminal["terminal_tick"], "terminal_reason": terminal["reason"]}
    final_metrics = _metric_snapshot(state, policies_by_side)
    row = {
        "schema": SCHEMA_BASELINE, "baseline_id": baseline_key, "status": status,
        "map_id": map_id, "config_sha256": config_hash(config),
        "matchup": _matchup_id(left, right), "policies": [left, right],
        "policy_parameter_sha256": {name: baseline_spec(name).digest() for name in (left, right)},
        "seed": seed, "seat_assignment": {"A": policies_by_side[0], "B": policies_by_side[1]},
        "stream_ids": {"A": controllers[0].stream_id, "B": controllers[1].stream_id},
        "ticks_executed": executed, "step_tick_calls": executed,
        "terminal": terminal, "profile_ticks": profiles,
        "terminal_metrics": {
            "colonies": final_metrics,
            "cumulative_event_metrics": _finish_metrics(metrics),
            "last_600_completed_ticks": _window_totals(window_history, 600),
            "last_900_completed_ticks": _window_totals(window_history, 900),
        },
        "event_counts": dict(sorted(event_counts.items())),
        "policy_event_metrics": _finish_metrics(metrics),
        "last_600_completed_ticks": _window_totals(window_history, 600),
        "last_900_completed_ticks": _window_totals(window_history, 900),
        "order_actions": {name: [{"tick": int(tick), "action": action} for tick, action in sorted(actions.items(), key=lambda kv: int(kv[0]))]
                          for name, actions in orders.items()},
        "physical_trace_sha256_by_policy": {name: digest.hexdigest() for name, digest in physical_digests.items()},
        "events_trace_sha256": event_digest.hexdigest(), "integrity_trace_sha256": raw_trace.hexdigest(),
        "policy_relative_trajectory_sha256": mirror_trace.hexdigest(),
        "random_policy_stream_trajectory_sha256": random_trace.hexdigest() if "random" in (left, right) else "not_applicable",
        "final_state_sha256": state_hash(state), "final_checkpoint_sha256": checkpoint_hash(state, run),
        "paired_trajectory_block_sha256": "not_measured",
    }
    row["raw_review_flags"] = _episode_flags(row)
    return row, checkpoints, physical_hashes


def _episode_flags(row: dict[str, Any]) -> dict[str, bool]:
    terminal = row["terminal"]
    metrics = row["policy_event_metrics"]
    timeout = terminal.get("reason") == "timeout"
    last600, last900 = row["last_600_completed_ticks"], row["last_900_completed_ticks"]
    return {
        "timeout": timeout,
        "repetitive_orders": any(item["repetitive_orders_flag"] for item in metrics.values()),
        "late_zero_queen_damage": timeout and all(v == 0 for v in last600["queen_damage"].values()),
        "passive_raid_avoidance": timeout and any(v == 0 for v in last900["raid_orders"].values()),
        "economic_stress": timeout and (
            any(v > 0 for v in last600["starvation_deaths"].values())
            or any(v > 0 for v in last600["lay_food_rejections"].values())
        ),
        "food_stockpile": timeout and all(
            row["terminal_metrics"]["colonies"][name]["queen_hp"] > 0
            and row["terminal_metrics"]["colonies"][name]["food"] >= 300
            and row["terminal_metrics"]["colonies"][name]["soldiers"] >= 1
            for name in row["policies"]
        ),
    }


def _review_flag(numerator: int, denominator: int = 12) -> dict[str, Any]:
    return {
        "numerator": numerator, "denominator": denominator,
        "flag": numerator >= 3, "unit": "descriptive_raw_scheduled_episodes",
    }


class _ProbeController:
    def __init__(self, policy_name: str, checkpoint: _Checkpoint, intervention: str,
                 *, treat: bool, decision_log: list[dict[str, Any]]) -> None:
        self.policy_name = policy_name
        self.checkpoint = checkpoint
        self.intervention = intervention
        self.treat = treat
        self.stream_id = policy_stream_id(baseline_spec(policy_name).policy_id, 0)
        self.policy = make_baseline(policy_name)
        self.decision_log = decision_log
        self.altered_decisions = 0

    def act(self, tick: int, observation: ObservationV2, run: RunState) -> ValidatedActionV2:
        stream = run.stream(self.stream_id)
        before = stream.to_canonical()
        answer = decide(self.policy, observation, stream)
        after = stream.to_canonical()
        action = answer.action
        if tick == self.checkpoint.tick:
            if _observation_sha256(observation) != self.checkpoint.observation_sha256:
                raise Gate1StallingError("probe first observation differs from frozen checkpoint")
            mask = get_action_mask(observation)
            if _mask_sha256(mask) != self.checkpoint.mask_sha256:
                raise Gate1StallingError("probe first action mask differs from frozen checkpoint")
            if _action_object(action) != _action_object(self.checkpoint.control_action):
                raise Gate1StallingError("normal action/RNG continuation differs at intervention tick")
            if before != self.checkpoint.stream_before or after != self.checkpoint.stream_after:
                raise Gate1StallingError("policy RNG stream differs before the one-shot intervention")
            actual_decision_sha = _sha(canonical_dumps({
                "observation_sha256": _observation_sha256(observation),
                "mask_sha256": _mask_sha256(mask),
                "action": _action_object(action),
                "stream_before": before, "stream_after": after,
                "time": [format(value, ".17g") for value in observation.block("time")],
            }))
            if actual_decision_sha != self.checkpoint.decision_sha256:
                raise Gate1StallingError("first decision/time hashes differ from the baseline checkpoint")
            if self.treat:
                if self.checkpoint.candidate_action is None:
                    raise Gate1StallingError("ineligible probe was run as an intervention")
                fields = _action_diff(action, self.checkpoint.candidate_action)
                expected = ["army_choice"] if self.intervention == "raid_once" else ["forage_share_q"]
                if fields != expected:
                    raise Gate1StallingError("one-shot action changes a field outside its fixed axis")
                action = self.checkpoint.candidate_action
                self.altered_decisions += 1
            action = validate_action(observation, mask, action).action
        self.decision_log.append({"tick": tick, "policy": self.policy_name, "action": _action_object(action)})
        return validate_action(observation, get_action_mask(observation), action)


def _probe_id(descriptor: dict[str, Any], tick: int, target: str, intervention: str) -> str:
    return f"{descriptor['baseline_id']}|t={tick}|target={target}|{intervention}"


def _unique_probe_context_sha256(descriptor: dict[str, Any], tick: int, target: str,
                                intervention: str, normalized_checkpoint_sha256: str) -> str:
    return _sha(canonical_dumps({
        "map_id": descriptor["map_id"], "matchup": descriptor["matchup"],
        "target_policy": target, "checkpoint_tick": tick, "intervention": intervention,
        "normalized_checkpoint_sha256": normalized_checkpoint_sha256,
    }))


def _probe_ineligible(descriptor: dict[str, Any], tick: int, target: str, intervention: str,
                      checkpoint: _Checkpoint | None, reason: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA_PROBE, "probe_id": _probe_id(descriptor, tick, target, intervention),
        **descriptor, "checkpoint_tick": tick, "target_policy": target,
        "intervention": intervention, "eligibility_status": "ineligible",
        "eligibility_reason": reason, "label": "INELIGIBLE", "step_tick_calls": 0, "ticks_executed": 0,
        "altered_decision_count": 0,
        "checkpoint_state_sha256": "not_measured" if checkpoint is None else checkpoint.state_sha256,
        "checkpoint_sha256": "not_measured" if checkpoint is None else checkpoint.checkpoint_sha256,
        "normalized_checkpoint_sha256": "not_measured" if checkpoint is None else checkpoint.normalized_checkpoint_sha256,
        "control_suffix_sha256": "not_measured" if checkpoint is None else checkpoint.control_suffix_sha256,
        "first_changed_order_tick": "not_measured", "first_physical_divergence_tick": "not_measured",
        "target_score_delta_x2": "not_measured", "terminal": {"status": "not_measured"},
        "unique_context_sha256": "not_measured",
    }


def _verify_control_suffix(checkpoint: _Checkpoint, left: str, right: str, swapped: bool,
                           baseline: dict[str, Any], budget: _RunBudget) -> int:
    state, run = checkpoint.state.clone(), checkpoint.run.clone()
    controllers = _controllers(left, right, swapped)
    digest = _Digest()
    calls = 0
    while run.outcome is None and state.tick < state.config.t_max:
        tick = state.tick
        start_state = state_hash(state)
        start_checkpoint = checkpoint_hash(state, run)
        result = budget.step(state, run, controllers)
        calls += 1
        digest.add({
            "tick": tick, "start_state_sha256": start_state,
            "start_checkpoint_sha256": start_checkpoint,
            "end_state_sha256": state_hash(state), "end_checkpoint_sha256": checkpoint_hash(state, run),
            "events_sha256": events_hash(list(result.events)),
            "tick_completed": state.tick == tick + 1,
        })
    left_side = 1 if swapped else 0
    replay_terminal = _terminal_object(state, run, left_side)
    if digest.hexdigest() != checkpoint.control_suffix_sha256:
        raise Gate1StallingError("control continuation from checkpoint differs from the baseline suffix")
    if replay_terminal != baseline["terminal"]:
        raise Gate1StallingError("control continuation terminal differs from its full baseline")
    if state_hash(state) != baseline["final_state_sha256"] or checkpoint_hash(state, run) != baseline["final_checkpoint_sha256"]:
        raise Gate1StallingError("control continuation final hashes differ from its full baseline")
    return calls


def _baseline_order_map(row: dict[str, Any], left: str, right: str) -> dict[str, dict[int, str]]:
    return {
        policy: {item["tick"]: _sha(canonical_dumps(item["action"])) for item in row["order_actions"][policy]}
        for policy in (left, right)
    }


def _run_probe(descriptor: dict[str, Any], baseline: dict[str, Any], checkpoint: _Checkpoint | None,
               tick: int, target: str, intervention: str, budget: _RunBudget,
               baseline_physical: dict[str, dict[int, str]]) -> dict[str, Any]:
    if checkpoint is None:
        return _probe_ineligible(descriptor, tick, target, intervention, None, "checkpoint_not_reached_before_terminal")
    if checkpoint.eligibility_status == "no_effect":
        row = _probe_ineligible(descriptor, tick, target, intervention, checkpoint, checkpoint.eligibility_reason)
        row["eligibility_status"], row["label"] = "no_effect", "NO_EFFECT"
        return row
    if checkpoint.eligibility_status != "eligible" or checkpoint.candidate_action is None:
        return _probe_ineligible(descriptor, tick, target, intervention, checkpoint, checkpoint.eligibility_reason)
    state, run = checkpoint.state.clone(), checkpoint.run.clone()
    if state_hash(state) != checkpoint.state_sha256 or checkpoint_hash(state, run) != checkpoint.checkpoint_sha256:
        raise Gate1StallingError("probe checkpoint clone hash differs from the baseline checkpoint")
    left, right = baseline["policies"]
    policies_by_side = (baseline["seat_assignment"]["A"], baseline["seat_assignment"]["B"])
    target_side = 0 if policies_by_side[0] == target else 1
    opponent = right if target == left else left
    decision_log: list[dict[str, Any]] = []
    target_controller = _ProbeController(target, checkpoint, intervention, treat=True, decision_log=decision_log)
    opponent_controller = PolicyController(make_baseline(opponent), policy_stream_id(baseline_spec(opponent).policy_id, 0))
    controllers = (target_controller, opponent_controller) if target_side == 0 else (opponent_controller, target_controller)
    metrics = {left: _new_policy_metrics(), right: _new_policy_metrics()}
    event_counts: Counter[str] = Counter()
    event_digest = _Digest()
    integrity_trace = _Digest()
    physical_trace = _Digest()
    physical_hashes: dict[int, str] = {}
    profiles: dict[str, Any] = {}
    for horizon in PROFILE_TICKS:
        if horizon < tick:
            profiles[str(horizon)] = {
                "status": "before_intervention",
                "baseline_profile": baseline["profile_ticks"][str(horizon)],
            }
        elif horizon == tick:
            profiles[str(horizon)] = {
                "status": "intervention_start",
                "baseline_profile": baseline["profile_ticks"][str(horizon)],
            }
        else:
            profiles[str(horizon)] = None
    order_map = _baseline_order_map(baseline, left, right)
    probe_orders: dict[str, dict[int, str]] = {left: {}, right: {}}
    history: deque[dict[str, Any]] = deque(maxlen=900)
    first_target_order = "not_measured"
    first_any_order = "not_measured"
    first_physical = "not_measured"
    cp_normalizer = checkpoint.normalizer
    calls = 0
    while run.outcome is None and state.tick < state.config.t_max:
        current_tick = state.tick
        start_state = state_hash(state)
        start_checkpoint = checkpoint_hash(state, run)
        try:
            result = budget.step(state, run, controllers)
        except EpisodeFinishedError as exc:
            raise Gate1StallingError(f"probe attempted a tick after terminal: {exc}") from exc
        calls += 1
        tick_events = list(result.events)
        event_counts.update(event.kind for event in tick_events)
        event_digest.add({"tick": current_tick, "events_sha256": events_hash(tick_events)})
        for event in tick_events:
            _record_event(metrics, policies_by_side, event)
            if isinstance(event, OrderEvent):
                policy = policies_by_side[event.side]
                action_obj = _action_object(_event_action(event))
                digest = _sha(canonical_dumps(action_obj))
                probe_orders[policy][current_tick] = digest
                baseline_digest = order_map[policy].get(current_tick)
                if baseline_digest != digest:
                    if first_any_order == "not_measured":
                        first_any_order = current_tick
                    if policy == target and first_target_order == "not_measured":
                        first_target_order = current_tick
                if current_tick == tick and policy == target:
                    if self_action := _event_action(event):
                        expected = checkpoint.candidate_action
                        if self_action != expected:
                            raise Gate1StallingError("engine OrderEvent does not contain the one-shot action")
        normalizer = cp_normalizer
        normalizer.state_object(state)
        normalized_events = [normalizer.event_object(event) for event in tick_events if event.kind not in ("order", "retarget")]
        normalized_events.sort(key=canonical_dumps)
        physical = _sha(canonical_dumps({
            "state": _physical_state_object(state, target_side), "events": normalized_events,
        }))
        physical_hashes[current_tick] = physical
        baseline_physical_hash = baseline_physical[target].get(current_tick, "not_measured")
        if first_physical == "not_measured" and baseline_physical_hash != "not_measured" and physical != baseline_physical_hash:
            first_physical = current_tick
        physical_trace.add({"tick": current_tick, "physical_sha256": physical})
        _record_window_tick(history, current_tick, state.tick == current_tick + 1, tick_events, policies_by_side)
        integrity_trace.add({
            "tick": current_tick, "start_state_sha256": start_state,
            "start_checkpoint_sha256": start_checkpoint, "end_state_sha256": state_hash(state),
            "end_checkpoint_sha256": checkpoint_hash(state, run),
            "events_sha256": events_hash(tick_events), "tick_completed": state.tick == current_tick + 1,
        })
        if state.tick in PROFILE_TICKS and str(state.tick) in profiles:
            profiles[str(state.tick)] = _profile_snapshot(state, policies_by_side, metrics, history)
    if target_controller.altered_decisions != 1:
        raise Gate1StallingError("eligible one-shot probe did not alter exactly one policy decision")
    terminal = _terminal_object(state, run, target_side)
    if run.outcome is None:
        label = "INCOMPLETE"
    else:
        base_terminal = baseline["terminal"]
        base_reason, probe_reason = base_terminal["reason"], terminal["reason"]
        left_policy = baseline["policies"][0]
        base_score = base_terminal["target_score_x2"] if target == left_policy else base_terminal["opponent_score_x2"]
        probe_score = terminal["target_score_x2"]
        if base_reason == "timeout" and probe_reason == "timeout":
            label = "TIMEOUT_PERSISTED"
        elif base_reason == "timeout" and probe_reason in {"queen_death", "economic_collapse"}:
            label = "TIMEOUT_ESCAPE"
        elif base_reason != probe_reason or base_score != probe_score:
            label = "OTHER_OUTCOME_CHANGE"
        elif first_physical == "not_measured":
            label = "NO_PHYSICAL_EFFECT"
        else:
            label = "NO_OUTCOME_CHANGE"
    for horizon in PROFILE_TICKS:
        if profiles.get(str(horizon)) is None:
            profiles[str(horizon)] = {"status": "not_measured", "reason": "episode_ended_before_checkpoint",
                                      "terminal_tick": terminal["terminal_tick"], "terminal_reason": terminal["reason"]}
    base_score_key = "target_score_x2" if target == baseline["policies"][0] else "opponent_score_x2"
    score_delta = "not_measured" if run.outcome is None else terminal["target_score_x2"] - baseline["terminal"][base_score_key]
    if label == "TIMEOUT_ESCAPE" and score_delta == 1:
        detailed_label = "BENEFICIAL_TIMEOUT_ESCAPE"
    elif label == "TIMEOUT_ESCAPE" and score_delta == -1:
        detailed_label = "HARMFUL_TIMEOUT_ESCAPE"
    else:
        detailed_label = label
    unique_context = _unique_probe_context_sha256(
        descriptor, tick, target, intervention, checkpoint.normalized_checkpoint_sha256,
    )
    return {
        "schema": SCHEMA_PROBE, "probe_id": _probe_id(descriptor, tick, target, intervention),
        **descriptor, "checkpoint_tick": tick, "target_policy": target, "intervention": intervention,
        "eligibility_status": "eligible", "eligibility_reason": "eligible",
        "checkpoint_state_sha256": checkpoint.state_sha256,
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "normalized_checkpoint_sha256": checkpoint.normalized_checkpoint_sha256,
        "observation_sha256": checkpoint.observation_sha256, "mask_sha256": checkpoint.mask_sha256,
        "decision_sha256": checkpoint.decision_sha256,
        "control_action": _action_object(checkpoint.control_action),
        "treatment_action": _action_object(checkpoint.candidate_action),
        "altered_decision_count": target_controller.altered_decisions,
        "first_changed_order_tick": first_target_order,
        "first_any_policy_order_divergence_tick": first_any_order,
        "first_physical_divergence_tick": first_physical,
        "physical_trace_sha256": physical_trace.hexdigest(),
        "physical_effect": first_physical != "not_measured",
        "ticks_executed": calls, "step_tick_calls": calls,
        "event_counts": dict(sorted(event_counts.items())),
        "policy_event_metrics": _finish_metrics(metrics),
        "last_600_completed_ticks": _window_totals(history, 600),
        "last_900_completed_ticks": _window_totals(history, 900),
        "profile_ticks": profiles, "terminal": terminal,
        "terminal_metrics": {
            "colonies": _metric_snapshot(state, policies_by_side),
            "cumulative_event_metrics_since_intervention": _finish_metrics(metrics),
            "last_600_completed_ticks_since_intervention": _window_totals(history, 600),
        },
        "target_score_delta_x2": score_delta,
        "label": detailed_label, "classification": label,
        "events_trace_sha256": event_digest.hexdigest(),
        "integrity_trace_sha256": integrity_trace.hexdigest(),
        "final_state_sha256": state_hash(state), "final_checkpoint_sha256": checkpoint_hash(state, run),
        "unique_context_sha256": unique_context,
        "decisions_sha256": _sha(canonical_dumps(decision_log)),
    }


def _attach_block_digests(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["map_id"], row["matchup"], row["seed"])].append(row)
    if len(groups) != EXPECTED_BASELINES // 2 or any(len(group) != 2 for group in groups.values()):
        raise Gate1StallingError("each fixed seed block must contain exactly two paired seat rows")
    for key, group in groups.items():
        by_seat = {row["seat_assignment"]["A"] == row["policies"][0]: row for row in group}
        if set(by_seat) != {True, False}:
            raise Gate1StallingError(f"missing mirror seat in {key}")
        first_a, first_b = by_seat[True], by_seat[False]
        digest = _sha(canonical_dumps({
            "map_id": key[0], "matchup": key[1],
            "policy_relative_seat_traces": [
                first_a["policy_relative_trajectory_sha256"],
                first_b["policy_relative_trajectory_sha256"],
            ],
        }))
        for row in group:
            row["paired_trajectory_block_sha256"] = digest


def _score_pair(group: list[dict[str, Any]]) -> dict[str, Any]:
    policies = group[0]["policies"]
    result: dict[str, Any] = {}
    for policy, key in ((policies[0], "target_score_x2"), (policies[1], "opponent_score_x2")):
        values = [row["terminal"][key] for row in group]
        if any(not isinstance(value, int) for value in values):
            result[policy] = "not_measured"
        else:
            fraction = sum((Fraction(value, 2) for value in values), Fraction()) / len(values)
            result[policy] = {"numerator": fraction.numerator, "denominator": fraction.denominator}
    return result


def _build_summary(baselines: list[dict[str, Any]], probes: list[dict[str, Any]],
                   baseline_calls: int, probe_calls: int, baselines_sha: str, probes_sha: str,
                   manifest_identity_sha: str) -> dict[str, Any]:
    _attach_block_digests(baselines)
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_seed: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in baselines:
        by_cell[(row["map_id"], row["matchup"])].append(row)
        by_seed[(row["map_id"], row["matchup"], row["seed"])].append(row)
    matchup_map = []
    for map_id in MAPS:
        for left, right in PAIRS:
            key = (map_id, _matchup_id(left, right))
            rows = by_cell[key]
            if len(rows) != 12:
                raise Gate1StallingError(f"{key} has {len(rows)} rows, expected 12")
            flag_names = (
                "timeout_concentration", "repetitive_orders", "late_zero_queen_damage",
                "passive_raid_avoidance", "economic_stress", "food_stockpile",
            )
            numerators = {
                "timeout_concentration": sum(row["raw_review_flags"]["timeout"] for row in rows),
                "repetitive_orders": sum(row["raw_review_flags"]["repetitive_orders"] for row in rows),
                "late_zero_queen_damage": sum(row["raw_review_flags"]["late_zero_queen_damage"] for row in rows),
                "passive_raid_avoidance": sum(row["raw_review_flags"]["passive_raid_avoidance"] for row in rows),
                "economic_stress": sum(row["raw_review_flags"]["economic_stress"] for row in rows),
                "food_stockpile": sum(row["raw_review_flags"]["food_stockpile"] for row in rows),
            }
            digest_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                digest_rows[row["paired_trajectory_block_sha256"]].append(row)
            unique_blocks = []
            for digest, block_rows in sorted(digest_rows.items()):
                unique_blocks.append({
                    "paired_trajectory_block_sha256": digest,
                    "scheduled_seed_blocks": sorted({row["seed"] for row in block_rows}),
                    "paired_seat_scores_x2": _score_pair(block_rows),
                })
            matchup_map.append({
                "map_id": map_id, "matchup": key[1],
                "scheduled_episodes": len(rows), "completed_episodes": sum(row["status"] == "complete" for row in rows),
                "terminal_reasons_raw": dict(sorted(Counter(row["terminal"]["reason"] for row in rows).items())),
                "timeout_draws_raw": numerators["timeout_concentration"],
                "raw_schedule_fraction": {"numerator": numerators["timeout_concentration"], "denominator": len(rows)},
                "distinct_paired_trajectory_blocks": len(unique_blocks),
                "duplicate_seed_blocks": len(rows) // 2 - len(unique_blocks),
                "flags": {
                    name: _review_flag(numerators[name])
                    for name in flag_names
                },
                "unique_paired_blocks": unique_blocks,
            })

    unique_contexts: dict[str, dict[str, Any]] = {}
    probe_cells: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in probes:
        probe_cells[(row["map_id"], row["matchup"], row["target_policy"], row["checkpoint_tick"])].append(row)
        digest = row.get("unique_context_sha256", "not_measured")
        if digest == "not_measured" or row["eligibility_status"] != "eligible":
            continue
        old = unique_contexts.get(digest)
        if old is not None and (
            old["label"], old["terminal"].get("reason"), old["terminal"].get("target_score_x2"),
            old["target_score_delta_x2"]
        ) != (
            row["label"], row["terminal"].get("reason"), row["terminal"].get("target_score_x2"),
            row["target_score_delta_x2"]
        ):
            raise Gate1StallingError("mirror/equivalent probe context has conflicting outcomes")
        unique_contexts[digest] = row
    probe_groups = []
    for key in sorted(probe_cells):
        rows = probe_cells[key]
        cell_contexts = {
            row["unique_context_sha256"]: row for row in rows
            if row["eligibility_status"] == "eligible" and row["unique_context_sha256"] != "not_measured"
        }
        raw_labels = Counter(row["label"] for row in rows)
        unique_labels = Counter(row["label"] for row in cell_contexts.values())
        probe_groups.append({
            "map_id": key[0], "matchup": key[1], "target_policy": key[2], "checkpoint_tick": key[3],
            "scheduled_probes": len(rows),
            "eligible_probes_raw": sum(row["eligibility_status"] == "eligible" for row in rows),
            "unique_eligible_contexts": len(cell_contexts),
            "raw_schedule_labels": dict(sorted(raw_labels.items())),
            "unique_context_labels": dict(sorted(unique_labels.items())),
            "unit": "unique_baseline_checkpoint_contexts_for_unique_counts; raw_rows_are_descriptive",
        })

    reasons = Counter(row["terminal"]["reason"] for row in baselines)
    labels_raw = Counter(row["label"] for row in probes)
    labels_unique = Counter(row["label"] for row in unique_contexts.values())
    return {
        "schema": SCHEMA_SUMMARY,
        "status": "complete" if len(baselines) == EXPECTED_BASELINES and len(probes) == EXPECTED_PROBES
        and all(row["status"] == "complete" for row in baselines) else "incomplete",
        "protocol_id": PROTOCOL_ID, "protocol_sha256": _protocol_sha256(),
        "manifest_identity_sha256": manifest_identity_sha,
        "scheduled_baselines": len(baselines), "scheduled_probes": len(probes),
        "baseline_terminal_reasons_raw": dict(sorted(reasons.items())),
        "baseline_timeout_draws_raw": reasons.get("timeout", 0),
        "baseline_timeout_raw_fraction": {"numerator": reasons.get("timeout", 0), "denominator": len(baselines)},
        "distinct_per_match_policy_relative_trajectories": len({row["policy_relative_trajectory_sha256"] for row in baselines}),
        "distinct_random_policy_stream_trajectories": len({
            row["random_policy_stream_trajectory_sha256"] for row in baselines
            if row["random_policy_stream_trajectory_sha256"] != "not_applicable"
        }),
        "distinct_paired_trajectory_blocks_total_by_cell": sum(cell["distinct_paired_trajectory_blocks"] for cell in matchup_map),
        "independence": "pair two seat rows within each map/matchup/seed; collapse equal policy-relative composite digests within map/matchup; random RNG trajectories remain in the trace; no CI/p-values",
        "matchup_map_diagnostics": matchup_map,
        "probe_raw_labels": dict(sorted(labels_raw.items())),
        "probe_unique_context_labels": dict(sorted(labels_unique.items())),
        "unique_eligible_probe_contexts": len(unique_contexts),
        "probe_group_diagnostics": probe_groups,
        "step_tick_calls": {
            "baseline": baseline_calls, "eligible_probe_continuations": probe_calls,
            "study_total": baseline_calls + probe_calls, "budget": STEP_BUDGET,
            "verification_replay": "not_measured",
        },
        "artifact_sha256": {"baselines.jsonl": baselines_sha, "probes.jsonl": probes_sha},
    }


def _rows_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical_dumps(row) + b"\n" for row in rows)


def _read_rows(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise Gate1StallingError(f"{path.name} must end with newline")
    rows = [canonical_loads(line) for line in data.splitlines() if line]
    if any(not isinstance(row, dict) for row in rows):
        raise Gate1StallingError(f"{path.name} rows must be canonical objects")
    return data, rows


def _manifest(source_commit: str, baseline_calls: int = 0, probe_calls: int = 0,
              artifacts: dict[str, str] | None = None) -> dict[str, Any]:
    configs = _build_configs()
    descriptors = _map_descriptors(configs)
    expected_config_hashes = {
        "R0-T0": "3f6aaa36a0d3d15888fa42f142df1151376794e62b553ab8e5f7659c9794145d",
        "R0-T1": "78e9c6979a2fb3973ae9a275f79ad5b62cb4b6ed3893a8f8469e4cc599e5fb34",
        "R0-T2": "b9e43b2aec34f303f371a9f9e5a2bcd8b5c5d894e608a3ca9e71386339f632f1",
        "R1-T0": "407bb3907099533adfe10b7c300a480b45b5d604dc019693b1ac99f47502303f",
        "R1-T1": "3bf8dabf6011ed88275c3fcfec3875c4da46ef849e7da7b0ac17bb2cd0e9e605",
        "R1-T2": "1b6f1f999246711919fb4dee9238e5fed31c5baac0dca779ea76c47afe167ceb",
    }
    actual = {item["id"]: item["config_sha256"] for item in descriptors}
    if actual != expected_config_hashes:
        raise Gate1StallingError("canonical FOR-22 map config hashes changed")
    policies = {name: _policy_descriptor(name) for name in ("random", "turtle", "rush", "balanced", "raider")}
    for name, descriptor in policies.items():
        descriptor["stream_ids"] = [
            policy_stream_id(baseline_spec(name).policy_id, replicate) for replicate in (0, 1)
        ]
    manifest = {
        "schema": SCHEMA_MANIFEST, "protocol_id": PROTOCOL_ID,
        "protocol_sha256": _protocol_sha256(), "base_commit_sha": "9b09e7e7e7229868a7c861aeafebbc69bbb322a3",
        "source_commit_sha": source_commit, "source_fingerprint_sha256": _source_fingerprint(),
        "base_source_fingerprint_sha256": "5843c847756e556755bb753278e0f1d58e2e15afaad9e9139106cde9a91f0900",
        "python_version": platform.python_version(), "python_executable": sys.executable,
        "versions": _versions(), "maps": descriptors, "policies": policies,
        "study": {
            "map_ids": list(MAPS), "policy_pairs": [{"left": a, "right": b} for a, b in PAIRS],
            "open_seeds": list(SEEDS), "seed_schedule_sha256": "ed35d5d2f19af9a36ae72dd6afc33709fd5d43240e626679144b6ee8f3c5b18a",
            "seat_assignments": ["first_A", "first_B"],
            "seat_schedule_sha256": "edbd1306cfd9beea32c7703bb5e77cbb6a89277ef33d6bf999dddc46d6c93252",
            "policy_interval_k": 30, "max_ticks_per_episode": 3000,
            "profile_ticks": list(PROFILE_TICKS), "probe_ticks": list(PROBE_TICKS),
            "interventions": list(INTERVENTIONS), "expected_baselines": EXPECTED_BASELINES,
            "expected_probes": EXPECTED_PROBES, "baseline_budget": BASELINE_BUDGET,
            "probe_budget": PROBE_BUDGET, "step_tick_call_budget": STEP_BUDGET,
            "score_encoding": "score_x2 in {0,1,2}; paired scores use exact Fraction",
            "random_seed_in_trajectory_digest": False,
            "independence": "paired seat rows then policy-relative trajectory block digest within map and matchup",
            "counterfactual_control": "checkpoint/state/RNG/input/action hashes match baseline; verifier replays all baseline suffixes and eligible continuations",
        },
        "step_tick_calls": {"baseline": baseline_calls, "eligible_probe_continuations": probe_calls,
                             "study_total": baseline_calls + probe_calls, "budget": STEP_BUDGET},
    }
    if artifacts is not None:
        manifest["artifact_sha256"] = artifacts
    return manifest


def _manifest_identity(manifest: dict[str, Any]) -> str:
    identity = {k: v for k, v in manifest.items() if k not in ("artifact_sha256", "step_tick_calls")}
    return _sha(canonical_dumps(identity))


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_dumps(_safe(value)) + b"\n")


def _study_run(out_dir: Path, *, verification: bool = False, source_commit: str,
               source_fingerprint: str, progress: dict[str, Any] | None = None
               ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int, int]:
    configs = _build_configs()
    baselines: list[dict[str, Any]] = []
    checkpoint_index: dict[tuple[str, int, str, str], _Checkpoint] = {}
    physical_index: dict[str, dict[str, dict[int, str]]] = {}
    baseline_budget = _RunBudget(BASELINE_BUDGET)
    for map_id, left, right, seed, swapped in _schedule():
        baseline_id = _baseline_id(map_id, left, right, seed, swapped)
        if progress is not None:
            progress.update({"phase": "baseline", "current_row": baseline_id,
                             "completed_baseline_rows": len(baselines),
                             "baseline_step_tick_calls": baseline_budget.calls})
        try:
            row, checkpoints, physical_hashes = _run_baseline(map_id, configs[map_id], left, right, seed, swapped, baseline_budget)
        except BaseException:
            if progress is not None:
                progress["baseline_step_tick_calls"] = baseline_budget.calls
            raise
        row["protocol_id"] = PROTOCOL_ID
        row["protocol_sha256"] = _protocol_sha256()
        row["source_commit_sha"] = source_commit
        row["source_fingerprint_sha256"] = source_fingerprint
        baselines.append(row)
        baseline_id = row["baseline_id"]
        if map_id in {"R0-T0", "R1-T0"} and (left, right) in {("turtle", "balanced"), ("turtle", "raider")}:
            physical_index[baseline_id] = physical_hashes
        for key, checkpoint in checkpoints.items():
            tick, policy, intervention = key
            checkpoint_index[(baseline_id, tick, policy, intervention)] = checkpoint
        if row["status"] != "complete":
            raise Gate1StallingError(f"baseline did not reach a true engine terminal: {row['baseline_id']}")
        if progress is not None:
            progress.update({"completed_baseline_rows": len(baselines),
                             "baseline_step_tick_calls": baseline_budget.calls})
    _attach_block_digests(baselines)
    by_id = {row["baseline_id"]: row for row in baselines}
    control_calls = 0
    if verification:
        control_budget = _RunBudget(BASELINE_BUDGET)
        for map_id, left, right, seed, swapped in _schedule():
            baseline_id = _baseline_id(map_id, left, right, seed, swapped)
            if map_id not in {"R0-T0", "R1-T0"} or (left, right) not in {("turtle", "balanced"), ("turtle", "raider")}:
                continue
            for tick in PROBE_TICKS:
                checkpoint = checkpoint_index.get((baseline_id, tick, left, "raid_once"))
                if checkpoint is not None:
                    control_calls += _verify_control_suffix(checkpoint, left, right, swapped,
                                                            by_id[baseline_id], control_budget)
    probe_budget = _RunBudget(PROBE_BUDGET)
    probes = []
    for descriptor, tick, target, intervention in _probe_schedule():
        baseline_id = descriptor["baseline_id"]
        checkpoint = checkpoint_index.get((baseline_id, tick, target, intervention))
        if progress is not None:
            progress.update({"phase": "counterfactual", "current_row": _probe_id(descriptor, tick, target, intervention),
                             "completed_probe_rows": len(probes),
                             "baseline_step_tick_calls": baseline_budget.calls,
                             "probe_step_tick_calls": probe_budget.calls})
        try:
            probe = _run_probe(descriptor, by_id[baseline_id], checkpoint, tick, target,
                               intervention, probe_budget, physical_index.get(baseline_id, {}))
        except BaseException:
            if progress is not None:
                progress["probe_step_tick_calls"] = probe_budget.calls
            raise
        probe["protocol_id"] = PROTOCOL_ID
        probe["protocol_sha256"] = _protocol_sha256()
        probe["source_commit_sha"] = source_commit
        probe["source_fingerprint_sha256"] = source_fingerprint
        if probe["label"] == "INCOMPLETE":
            raise Gate1StallingError(f"counterfactual did not reach a real engine terminal: {probe['probe_id']}")
        probes.append(probe)
        if progress is not None:
            progress.update({"completed_probe_rows": len(probes),
                             "probe_step_tick_calls": probe_budget.calls})
    if len(baselines) != EXPECTED_BASELINES or len(probes) != EXPECTED_PROBES:
        raise Gate1StallingError("fixed schedule cardinality differs from the preregistration")
    if baseline_budget.calls > BASELINE_BUDGET or probe_budget.calls > PROBE_BUDGET:
        raise Gate1StallingError("study phase exceeded its preregistered call budget")
    return baselines, probes, baseline_budget.calls, probe_budget.calls, control_calls


def run(out: str | Path) -> dict[str, Any]:
    out_dir = Path(out)
    if not out_dir.is_absolute():
        out_dir = _ROOT / out_dir
    if out_dir.exists():
        raise Gate1StallingError(f"output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    try:
        source_commit = _git_head()
        source_fingerprint = _source_fingerprint()
        progress = {
            "phase": "starting", "current_row": "not_started",
            "completed_baseline_rows": 0, "completed_probe_rows": 0,
            "baseline_step_tick_calls": 0, "probe_step_tick_calls": 0,
        }
        baselines, probes, baseline_calls, probe_calls, _control_calls = _study_run(
            out_dir, source_commit=source_commit, source_fingerprint=source_fingerprint,
            progress=progress,
        )
        baselines_bytes = _rows_bytes(baselines)
        probes_bytes = _rows_bytes(probes)
        baselines_sha, probes_sha = _sha(baselines_bytes), _sha(probes_bytes)
        manifest = _manifest(source_commit, baseline_calls, probe_calls)
        identity_sha = _manifest_identity(manifest)
        summary = _build_summary(baselines, probes, baseline_calls, probe_calls,
                                 baselines_sha, probes_sha, identity_sha)
        summary_bytes = canonical_dumps(summary) + b"\n"
        artifacts = {
            "baselines.jsonl": baselines_sha,
            "probes.jsonl": probes_sha,
            "summary.json": _sha(summary_bytes),
        }
        manifest = _manifest(source_commit, baseline_calls, probe_calls, artifacts)
        manifest_bytes = canonical_dumps(manifest) + b"\n"
        (out_dir / "baselines.jsonl").write_bytes(baselines_bytes)
        (out_dir / "probes.jsonl").write_bytes(probes_bytes)
        (out_dir / "summary.json").write_bytes(summary_bytes)
        (out_dir / "manifest.json").write_bytes(manifest_bytes)
        return {"status": summary["status"], "scheduled_baselines": len(baselines),
                "scheduled_probes": len(probes), "baseline_calls": baseline_calls,
                "probe_calls": probe_calls, "manifest_sha256": _sha(manifest_bytes),
                "summary_sha256": _sha(summary_bytes)}
    except BaseException as exc:
        failure = {
            "schema": "formicmind.gate1-stalling-invalid-attempt/v1",
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": _protocol_sha256(),
            "source_commit_sha": _git_head() if out_dir.exists() else "not_measured",
            "failure_type": type(exc).__name__, "failure": str(exc), "status": "invalid_incomplete",
            "progress": progress if "progress" in locals() else {"phase": "setup", "calls": 0},
        }
        _write_json(out_dir / "invalid_attempt.json", failure)
        raise


def _expected_baseline_ids() -> list[str]:
    return [_baseline_id(*entry) for entry in _schedule()]


def verify(input_path: str | Path) -> dict[str, Any]:
    source = Path(input_path)
    if not source.is_absolute():
        source = _ROOT / source
    required = {"manifest.json", "baselines.jsonl", "probes.jsonl", "summary.json"}
    if not source.is_dir() or {p.name for p in source.iterdir()} != required:
        raise Gate1StallingError("verify input must contain exactly the four complete study artifacts")
    manifest = canonical_loads((source / "manifest.json").read_bytes().rstrip(b"\n"))
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA_MANIFEST:
        raise Gate1StallingError("manifest schema mismatch")
    if manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("protocol_sha256") != _protocol_sha256():
        raise Gate1StallingError("protocol ID or frozen protocol hash changed")
    if manifest.get("base_commit_sha") != "9b09e7e7e7229868a7c861aeafebbc69bbb322a3":
        raise Gate1StallingError("manifest base SHA differs from the preregistered origin/main")
    if manifest.get("base_source_fingerprint_sha256") != "5843c847756e556755bb753278e0f1d58e2e15afaad9e9139106cde9a91f0900":
        raise Gate1StallingError("manifest base Python source fingerprint differs from the preregistration")
    current_source = _source_fingerprint()
    if manifest.get("source_fingerprint_sha256") != current_source:
        raise Gate1StallingError("recursive Python source fingerprint changed")
    try:
        _git("merge-base", "--is-ancestor", manifest["source_commit_sha"], "HEAD")
    except subprocess.CalledProcessError as exc:
        raise Gate1StallingError("source commit is not an ancestor of verification HEAD") from exc
    artifacts = {name: _sha((source / name).read_bytes()) for name in required - {"manifest.json"}}
    if manifest.get("artifact_sha256") != {name: artifacts[name] for name in sorted(artifacts)}:
        raise Gate1StallingError("saved artifact SHA-256 mismatch")
    try:
        expected_manifest = _manifest(
            manifest["source_commit_sha"],
            manifest["step_tick_calls"]["baseline"],
            manifest["step_tick_calls"]["eligible_probe_continuations"],
            manifest["artifact_sha256"],
        )
    except (KeyError, TypeError) as exc:
        raise Gate1StallingError("manifest is missing fixed provenance fields") from exc
    if canonical_dumps(expected_manifest) != canonical_dumps(manifest):
        raise Gate1StallingError("manifest config/policy/schedule/version/provenance mismatch")
    baseline_bytes, saved_baselines = _read_rows(source / "baselines.jsonl")
    probes_bytes, saved_probes = _read_rows(source / "probes.jsonl")
    saved_summary = canonical_loads((source / "summary.json").read_bytes().rstrip(b"\n"))
    if [row.get("baseline_id") for row in saved_baselines] != _expected_baseline_ids():
        raise Gate1StallingError("baseline rows are missing, reordered, duplicated, or outside the fixed schedule")
    expected_probe_ids = [
        _probe_id(desc, tick, target, intervention)
        for desc, tick, target, intervention in _probe_schedule()
    ]
    if [row.get("probe_id") for row in saved_probes] != expected_probe_ids:
        raise Gate1StallingError("probe rows are missing, reordered, duplicated, or outside the fixed plan")
    if len(saved_baselines) != EXPECTED_BASELINES or len(saved_probes) != EXPECTED_PROBES:
        raise Gate1StallingError("saved raw row count differs from fixed schedule")
    replay_dir = source.parent / f".{source.name}.verify-replay"
    baselines, probes, baseline_calls, probe_calls, control_calls = _study_run(
        replay_dir, verification=True, source_commit=manifest["source_commit_sha"],
        source_fingerprint=current_source,
    )
    if manifest["step_tick_calls"] != {
        "baseline": baseline_calls,
        "eligible_probe_continuations": probe_calls,
        "study_total": baseline_calls + probe_calls,
        "budget": STEP_BUDGET,
    }:
        raise Gate1StallingError("manifest phase call counters differ from strict replay")
    replay_baselines, replay_probes = _rows_bytes(baselines), _rows_bytes(probes)
    if baseline_bytes != replay_baselines:
        index = next((i for i, (a, b) in enumerate(zip(saved_baselines, baselines)) if a != b), -1)
        raise Gate1StallingError(f"baseline strict replay mismatch at row {index}")
    if probes_bytes != replay_probes:
        index = next((i for i, (a, b) in enumerate(zip(saved_probes, probes)) if a != b), -1)
        raise Gate1StallingError(f"counterfactual strict replay mismatch at row {index}")
    identity_sha = _manifest_identity({k: v for k, v in manifest.items() if k != "artifact_sha256"})
    expected_summary = _build_summary(baselines, probes, baseline_calls, probe_calls,
                                      _sha(replay_baselines), _sha(replay_probes), identity_sha)
    if canonical_dumps(saved_summary) != canonical_dumps(expected_summary):
        raise Gate1StallingError("strictly recomputed summary differs from saved summary")
    return {
        "status": "verified", "verified_baselines": len(baselines), "verified_planned_probes": len(probes),
        "eligible_probe_replays": sum(row["eligibility_status"] == "eligible" for row in probes),
        "verification_step_tick_calls": baseline_calls + probe_calls,
        "verification_control_suffix_calls": control_calls,
        "source_fingerprint_sha256": current_source,
        "baselines_sha256": _sha(replay_baselines), "probes_sha256": _sha(replay_probes),
        "summary_sha256": _sha(canonical_dumps(expected_summary) + b"\n"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m formicmind.gate1_stalling", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="run fixed 288-baseline/384-probe study")
    runner.add_argument("--out", required=True, help="new, not-yet-existing output directory")
    verifier = commands.add_parser("verify", help="strictly replay all baselines and eligible probes")
    verifier.add_argument("--input", required=True, help="saved study output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        value = run(args.out) if args.command == "run" else verify(args.input)
    except (Gate1StallingError, OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"gate1_stalling: error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))
    return 0 if value.get("status") in {"complete", "verified"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

