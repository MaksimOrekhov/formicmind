"""Preregistered Gate 1A causal probes over real L0 matches.

This module is an experiment harness only. Every branch advances through the
existing ``step_tick`` and uses the unchanged FOR-19 policies and action
contracts. The protocol is ``docs/gate1_causal_protocol.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

from .actions import ActionMaskV2, ActionV2, ValidatedActionV2, flatten_mask_v2, get_action_mask, validate_action
from .baselines import baseline_spec, make_baseline
from .canonical import canonical_dumps, canonical_loads
from .config import SimConfig
from .events import Event, OrderEvent, events_hash
from .observation import ObservationV2, build_observation, flatten_v2
from .policy import decide
from .reset import reset
from .rng import Pcg32, policy_stream_id
from .rules import ARMY_RAID_ENEMY_HOME, MIRROR, Q_TOTAL, SIDES, SOLDIER, WORKER, default_config
from .runstate import RunState
from .serialization import (
    checkpoint_hash,
    config_hash,
    config_to_canonical,
    state_to_canonical,
    state_hash,
)
from .snapshot import PolicySnapshot, begin_policy_phase
from .tick import Controller, EpisodeFinishedError, PolicyController, TickResult, step_tick
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

SCHEMA_MANIFEST = "formicmind.gate1-causal-manifest/v1"
SCHEMA_PROBES = "formicmind.gate1-causal-probes/v1"
SCHEMA_SUMMARY = "formicmind.gate1-causal-summary/v1"
PROTOCOL_ID = "formicmind.gate1-causal/v1"
PROTOCOL_RELATIVE = Path("docs/gate1_causal_protocol.md")
SEEDS = tuple(range(20261201, 20261209))
CHECKPOINT_TICKS = (30, 300, 900)
HORIZON_OFFSETS = (30, 150)
MATCHUPS = (
    ("balanced-rush", "balanced", "rush"),
    ("turtle-raider", "turtle", "raider"),
    ("random-balanced", "random", "balanced"),
)
AXES = ("lay_order", "forage_share_q", "army_choice")
MDE_MILLIPOINTS = 100
BOOTSTRAP_REPLICATES = 10_000
STEP_BUDGET = 3_000_000
_ROOT = Path(__file__).resolve().parents[2]


class Gate1CausalError(ValueError):
    """Invalid study input, incompatible provenance, or a failed integrity check."""


class _ArrayDigest:
    """Incremental SHA-256 of the canonical JSON encoding of an array."""

    def __init__(self) -> None:
        self._hash = hashlib.sha256(b"[")
        self._count = 0

    def add(self, value: Any) -> None:
        if self._count:
            self._hash.update(b",")
        self._hash.update(canonical_dumps(value))
        self._count += 1

    @property
    def count(self) -> int:
        return self._count

    def hexdigest(self) -> str:
        digest = self._hash.copy()
        digest.update(b"]")
        return digest.hexdigest()


class _MirrorNormalizer:
    """Canonicalize a trajectory in the target-side frame for mirror grouping.

    Entity ids are assigned persistent, side-local ordinals in allocation order.
    This keeps identity stable across movement, hatching, and deaths while removing
    the world-side-specific id ranges used by reset().
    """

    def __init__(self, target_side: int) -> None:
        self.target_side = target_side
        self._entity_ids: dict[int, dict[int, int]] = {0: {}, 1: {}}
        self._next_entity_id = {0: 0, 1: 0}

    def _register_entities(self, state: Any) -> None:
        for side, colony in enumerate(state.colonies):
            known = self._entity_ids[side]
            current = sorted({item.unit_id for item in (*colony.units, *colony.brood)})
            for unit_id in current:
                if unit_id not in known:
                    known[unit_id] = self._next_entity_id[side]
                    self._next_entity_id[side] += 1

    def _node(self, node: int) -> int:
        if 0 <= node < len(MIRROR):
            return node if self.target_side == 0 else MIRROR[node]
        return node

    def _relative_side(self, side: int) -> int:
        if side not in (0, 1):
            return side
        return side if self.target_side == 0 else 1 - side

    def state_object(self, state: Any) -> dict[str, Any]:
        self._register_entities(state)
        raw = state_to_canonical(state)
        local_to_world = tuple(range(len(MIRROR))) if self.target_side == 0 else MIRROR
        result = dict(raw)
        result["node_food"] = [state.node_food[world_node] for world_node in local_to_world]
        colonies = []
        for relative_side, world_side in enumerate((self.target_side, 1 - self.target_side)):
            colony = dict(raw["colonies"][world_side])
            colony["side"] = relative_side
            entity_ids = self._entity_ids[world_side]
            colony["units"] = [
                {
                    **unit,
                    "unit_id": entity_ids[unit["unit_id"]],
                    "node": self._node(unit["node"]),
                    "from_node": self._node(unit["from_node"]),
                    "to_node": self._node(unit["to_node"]),
                    "target_node": self._node(unit["target_node"]),
                }
                for unit in colony["units"]
            ]
            colony["brood"] = [
                {**brood, "unit_id": entity_ids[brood["unit_id"]]}
                for brood in colony["brood"]
            ]
            intel = dict(colony["intel"])
            intel["nodes"] = [
                colony["intel"]["nodes"][world_node] for world_node in local_to_world
            ]
            colony["intel"] = intel
            colonies.append(colony)
        result["colonies"] = colonies
        return result

    def state_sha256(self, state: Any) -> str:
        return _sha256(canonical_dumps(self.state_object(state)))

    def event_object(self, event: Event) -> dict[str, Any]:
        value = event.to_canonical()
        for key in ("side", "attacker_side", "defender_side"):
            if key in value:
                value[key] = self._relative_side(value[key])
        if "sides" in value:
            value["sides"] = sorted(self._relative_side(side) for side in value["sides"])
        if "winner" in value and value["winner"] in ("A", "B"):
            winner_side = 0 if value["winner"] == "A" else 1
            value["winner"] = "target" if winner_side == self.target_side else "opponent"
        if "queen_hp" in value:
            value["queen_hp"] = [value["queen_hp"][self.target_side], value["queen_hp"][1 - self.target_side]]
        for key in (
            "army_target_node", "old_target", "new_target", "from_node", "to_node",
            "target_node", "node",
        ):
            if key in value:
                value[key] = self._node(value[key])
        for key in ("unit_id",):
            if key in value and value.get("side") in (0, 1):
                # side fields above are already target-relative, recover the world side.
                relative_side = value["side"]
                world_side = relative_side if self.target_side == 0 else 1 - relative_side
                value[key] = self._entity_ids[world_side].get(value[key], value[key])
        if "unit_ids" in value and value.get("side") in (0, 1):
            relative_side = value["side"]
            world_side = relative_side if self.target_side == 0 else 1 - relative_side
            value["unit_ids"] = [
                self._entity_ids[world_side].get(unit_id, unit_id) for unit_id in value["unit_ids"]
            ]
        return value


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _protocol_sha256() -> str:
    try:
        return _sha256((_ROOT / PROTOCOL_RELATIVE).read_bytes())
    except OSError as exc:
        raise Gate1CausalError(f"cannot read preregistered protocol: {exc}") from exc


def _source_fingerprint() -> str:
    package = _ROOT / "src" / "formicmind"
    entries = []
    for path in sorted(package.rglob("*.py")):
        entries.append({"path": path.relative_to(_ROOT).as_posix(), "sha256": _sha256(path.read_bytes())})
    return _sha256(canonical_dumps(entries))


def _git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Gate1CausalError(f"cannot determine source commit: {exc}") from exc
    value = result.stdout.strip()
    if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise Gate1CausalError("git rev-parse returned an invalid commit SHA")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return format(value, ".17g")
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    return value


def _policy_descriptor(name: str) -> dict[str, Any]:
    spec = baseline_spec(name)
    return {
        "name": name,
        "id": spec.full_id,
        "stream_id": policy_stream_id(spec.policy_id, 0),
        "parameters": _json_safe(spec.parameter_object()),
        "parameters_sha256": spec.digest(),
        "stochastic": name == "random",
    }


def _version_manifest() -> dict[str, str]:
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


def _make_manifest(config: SimConfig, source_commit: str) -> dict[str, Any]:
    names = sorted({name for _, left, right in MATCHUPS for name in (left, right)})
    policy_descriptors = [_policy_descriptor(name) for name in names]
    return {
        "schema": SCHEMA_MANIFEST,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": _protocol_sha256(),
        "post_review_reporting_amendment": {
            "version": "FOR-21-review-2026-11-03/v1",
            "trajectory_clusters": "mirror_trajectory_sha256 from target-relative per-tick canonical state, normalized events, and role-keyed random-policy RNG snapshots; raw seat-specific digests remain preserved",
            "intervention_physical_effect": "physical_trace_sha256 over every post-tick physical state and non-order event; standing orders and retarget metadata are excluded",
        },
        "source_commit": source_commit,
        "source_fingerprint_sha256": _source_fingerprint(),
        "config": config_to_canonical(config),
        "config_sha256": config_hash(config),
        "versions": _version_manifest(),
        "policies": policy_descriptors,
        "study": {
            "matchups": [
                {"id": label, "target_policy": left, "opponent_policy": right}
                for label, left, right in MATCHUPS
            ],
            "open_seeds": list(SEEDS),
            "seat_assignments": ["target_on_A", "target_on_B"],
            "checkpoint_start_ticks": list(CHECKPOINT_TICKS),
            "horizon_completed_tick_offsets": list(HORIZON_OFFSETS),
            "axes": list(AXES),
            "intervention_budget_step_tick_calls": STEP_BUDGET,
            "mde_millipoints": MDE_MILLIPOINTS,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed_derivation": "PCG32 initstate/initseq: little-endian u64 from SHA256(protocol_id|bootstrap|axis) bytes 0:8 and 8:16",
            "bootstrap_minimums": {
                "unique_eligible_trajectory_digests": 10,
                "distinct_eligible_pre_state_hashes": 12,
                "unique_trajectories_per_matchup": 2,
            },
            "engine": "reset -> full step_tick; baseline policy continuation after one intervention decision",
        },
    }


def _action_object(action: ActionV2) -> dict[str, Any]:
    return {
        "lay_order": action.lay_order,
        "forage_share_q": list(action.forage_share_q),
        "army_choice": action.army_choice,
        "avoid_center": action.avoid_center,
    }


def _action_from_object(value: dict[str, Any]) -> ActionV2:
    return ActionV2(
        lay_order=value["lay_order"], forage_share_q=tuple(value["forage_share_q"]),
        army_choice=value["army_choice"], avoid_center=value["avoid_center"],
    )


def _action_diff(left: ActionV2, right: ActionV2) -> list[str]:
    return [
        field for field in ("lay_order", "forage_share_q", "army_choice", "avoid_center")
        if getattr(left, field) != getattr(right, field)
    ]


def _mask_sha256(mask: ActionMaskV2) -> str:
    return _sha256(canonical_dumps(list(flatten_mask_v2(mask))))


def _observation_sha256(observation: ObservationV2) -> str:
    return _sha256(flatten_v2(observation))


def _candidate(
    axis: str, control: ActionV2, mask: ActionMaskV2, *, queen_alive: bool,
    worker_count: int, soldier_count: int,
) -> tuple[ActionV2 | None, str | None]:
    if axis == "lay_order":
        if not queen_alive:
            return None, "queen_not_alive"
        new_value = 0 if control.lay_order == 1 else 1
        action = ActionV2(new_value, control.forage_share_q, control.army_choice, control.avoid_center)
    elif axis == "forage_share_q":
        if worker_count == 0:
            return None, "no_adult_workers"
        shares = (Q_TOTAL, 0, 0, 0, 0, 0)
        if control.forage_share_q == shares:
            return None, "already_home_only"
        action = ActionV2(control.lay_order, shares, control.army_choice, control.avoid_center)
    elif axis == "army_choice":
        if soldier_count == 0:
            return None, "no_adult_soldiers"
        if control.army_choice == ARMY_RAID_ENEMY_HOME:
            return None, "already_raid_enemy_home"
        if not mask.army_choice[ARMY_RAID_ENEMY_HOME]:
            return None, "enemy_home_unreachable"
        if not mask.army_avoid_center[ARMY_RAID_ENEMY_HOME][int(control.avoid_center)]:
            return None, "control_avoid_center_illegal_for_raid"
        action = ActionV2(
            control.lay_order, control.forage_share_q, ARMY_RAID_ENEMY_HOME, control.avoid_center,
        )
    else:
        raise Gate1CausalError(f"unknown intervention axis: {axis}")
    if _action_diff(control, action) != [axis]:
        raise Gate1CausalError(f"candidate for {axis} changes undeclared action fields")
    return action, None


@dataclass(slots=True)
class _TickBudget:
    maximum: int = STEP_BUDGET
    calls: int = 0

    def step(self, state, run, controllers) -> TickResult:
        if self.calls >= self.maximum:
            raise Gate1CausalError(f"step_tick call budget exceeded ({self.maximum})")
        result = step_tick(state, run, controllers)
        self.calls += 1
        return result


@dataclass(slots=True)
class _Checkpoint:
    state: Any
    run: RunState
    tick: int
    target_side: int
    target_policy: str
    opponent_policy: str
    matchup: str
    seed: int
    swapped: bool
    pre_state_hash: str
    mirror_pre_state_hash: str
    pre_checkpoint_hash: str
    observation_sha256: str
    mask_sha256: str
    control_action: ActionV2
    candidate_actions: dict[str, ActionV2 | None]
    ineligible_reasons: dict[str, str | None]


def _make_controllers(left: str, right: str, swapped: bool = False) -> tuple[Controller, Controller]:
    policy_names = (right, left) if swapped else (left, right)
    return tuple(
        PolicyController(make_baseline(name), policy_stream_id(baseline_spec(name).policy_id, 0))
        for name in policy_names
    )  # type: ignore[return-value]


def _capture_checkpoint(
    state: Any, run: RunState, *, tick: int, target_side: int, target_policy: str,
    opponent_policy: str, matchup: str, seed: int, swapped: bool,
    mirror_normalizer: _MirrorNormalizer,
) -> _Checkpoint:
    state_copy, run_copy = state.clone(), run.clone()
    before_state_hash = state_hash(state_copy)
    before_checkpoint_hash = checkpoint_hash(state_copy, run_copy)
    snapshot = begin_policy_phase(state_copy, (run_copy.windows[0], run_copy.windows[1]))
    observation = build_observation(snapshot, target_side)
    mask = get_action_mask(observation)
    policy = make_baseline(target_policy)
    stream_id = policy_stream_id(baseline_spec(target_policy).policy_id, 0)
    control = decide(policy, observation, run_copy.stream(stream_id))
    control_action = validate_action(observation, mask, control.action).action
    own_view = snapshot.view(target_side)
    worker_count = sum(unit.caste == WORKER for unit in own_view.units)
    soldier_count = sum(unit.caste == SOLDIER for unit in own_view.units)
    queen_alive = own_view.queen_hp > 0
    candidates, reasons = {}, {}
    for axis in AXES:
        candidate, reason = _candidate(
            axis, control_action, mask, queen_alive=queen_alive,
            worker_count=worker_count, soldier_count=soldier_count,
        )
        if candidate is not None:
            validate_action(observation, mask, candidate)
        candidates[axis] = candidate
        reasons[axis] = reason
    return _Checkpoint(
        state=state.clone(), run=run.clone(), tick=tick, target_side=target_side,
        target_policy=target_policy, opponent_policy=opponent_policy,
        matchup=matchup, seed=seed, swapped=swapped,
        pre_state_hash=before_state_hash, pre_checkpoint_hash=before_checkpoint_hash,
        mirror_pre_state_hash=mirror_normalizer.state_sha256(state_copy),
        observation_sha256=_observation_sha256(observation), mask_sha256=_mask_sha256(mask),
        control_action=control_action, candidate_actions=candidates, ineligible_reasons=reasons,
    )


class _InterventionController:
    """Calls the original policy at every decision and replaces at most one action."""

    def __init__(self, policy_name: str, stream_id: str, checkpoint: _Checkpoint,
                 axis: str, treatment: bool) -> None:
        self.policy_name = policy_name
        self.policy = make_baseline(policy_name)
        self.stream_id = stream_id
        self.checkpoint = checkpoint
        self.axis = axis
        self.treatment = treatment

    def act(self, tick: int, observation: ObservationV2, run: RunState) -> ValidatedActionV2:
        normal = decide(self.policy, observation, run.stream(self.stream_id))
        if tick != self.checkpoint.tick:
            return normal
        if _observation_sha256(observation) != self.checkpoint.observation_sha256:
            raise Gate1CausalError("intervention observation differs from checkpoint preflight")
        mask = get_action_mask(observation)
        if _mask_sha256(mask) != self.checkpoint.mask_sha256:
            raise Gate1CausalError("intervention mask differs from checkpoint preflight")
        if normal.action != self.checkpoint.control_action:
            raise Gate1CausalError("baseline control action differs from checkpoint preflight")
        chosen = normal.action
        if self.treatment:
            candidate = self.checkpoint.candidate_actions[self.axis]
            if candidate is None:
                raise Gate1CausalError("ineligible candidate was scheduled as treatment")
            chosen = candidate
        return validate_action(observation, mask, chosen)


def _stream_snapshot(run: RunState, matchup_names: tuple[str, str]) -> Any:
    if "random" not in matchup_names:
        return None
    stream_id = policy_stream_id(baseline_spec("random").policy_id, 0)
    return run.stream(stream_id).to_canonical()


def _mirror_stream_snapshot(
    run: RunState, target_policy: str, opponent_policy: str,
) -> dict[str, Any]:
    result = {}
    for role, name in (("target", target_policy), ("opponent", opponent_policy)):
        if name == "random":
            stream_id = policy_stream_id(baseline_spec(name).policy_id, 0)
            result[role] = run.stream(stream_id).to_canonical()
    return result


def _score_x2(outcome: Any, side: int) -> int | None:
    return None if outcome is None else int(outcome.score[side] * 2)


def _baseline_metrics(state: Any, run: RunState, target_side: int) -> dict[str, Any]:
    outcome = run.outcome
    return {
        "winner": "unknown_censored" if outcome is None else outcome.winner.value,
        "reason": "unknown_censored" if outcome is None else outcome.reason.value,
        "terminal_tick": -1 if outcome is None else outcome.terminal_tick,
        "tick_completed": False if outcome is None else outcome.tick_completed,
        "target_score_x2": "unknown_censored" if outcome is None else _score_x2(outcome, target_side),
        "state_tick": state.tick,
        "terminal": outcome is not None,
        "queen_hp": [colony.queen_hp for colony in state.colonies],
        "food": [colony.food for colony in state.colonies],
        "workers": [sum(unit.caste == WORKER for unit in colony.units) for colony in state.colonies],
        "soldiers": [sum(unit.caste == SOLDIER for unit in colony.units) for colony in state.colonies],
        "brood": [len(colony.brood) for colony in state.colonies],
    }


def _run_baseline(
    config: SimConfig, matchup: str, left: str, right: str, seed: int,
    swapped: bool, budget: _TickBudget,
) -> tuple[dict[str, Any], list[_Checkpoint]]:
    controllers = _make_controllers(left, right, swapped)
    stream_ids = tuple(controller.stream_id for controller in controllers if isinstance(controller, PolicyController))
    state, run = reset(config, seed, stream_ids)
    target_policy, opponent_policy = left, right
    target_side = 1 if swapped else 0
    random_names = (left, right)
    checkpoint_map: list[_Checkpoint] = []
    trajectory = _ArrayDigest()
    mirror_trajectory = _ArrayDigest()
    mirror_normalizer = _MirrorNormalizer(target_side)
    event_digest = _EventDigest()
    trace = _ArrayDigest()
    while run.outcome is None and state.tick < config.t_max:
        tick = state.tick
        if tick in CHECKPOINT_TICKS:
            checkpoint_map.append(_capture_checkpoint(
                state, run, tick=tick, target_side=target_side, target_policy=target_policy,
                opponent_policy=opponent_policy, matchup=matchup, seed=seed, swapped=swapped,
                mirror_normalizer=mirror_normalizer,
            ))
        start_state_hash = state_hash(state)
        mirror_start_state = mirror_normalizer.state_object(state)
        random_before = _stream_snapshot(run, random_names) if tick % config.policy_interval_k == 0 else None
        mirror_random_before = (
            _mirror_stream_snapshot(run, target_policy, opponent_policy)
            if tick % config.policy_interval_k == 0 else None
        )
        result = budget.step(state, run, controllers)
        end_state_hash = state_hash(state)
        end_checkpoint_hash = checkpoint_hash(state, run)
        canonical_events = [event.to_canonical() for event in result.events]
        event_digest.add_all(canonical_events)
        mirror_end_state = mirror_normalizer.state_object(state)
        mirror_events = [mirror_normalizer.event_object(event) for event in result.events]
        mirror_events.sort(key=canonical_dumps)
        trace.add({
            "tick": tick, "state_hash": end_state_hash,
            "checkpoint_hash": end_checkpoint_hash,
            "events_sha256": events_hash(list(result.events)),
            "tick_completed": state.tick == tick + 1,
        })
        random_after = _stream_snapshot(run, random_names) if tick % config.policy_interval_k == 0 else None
        trajectory_row = {
            "tick": tick, "start_state_hash": start_state_hash, "events": canonical_events,
            "end_state_hash": end_state_hash,
        }
        if random_before is not None:
            trajectory_row["random_stream_before"] = random_before
            trajectory_row["random_stream_after"] = random_after
        trajectory.add(trajectory_row)
        mirror_row = {
            "tick": tick,
            "start_state": mirror_start_state,
            "events": mirror_events,
            "end_state": mirror_end_state,
        }
        if mirror_random_before is not None:
            mirror_row["random_stream_before"] = mirror_random_before
            mirror_row["random_stream_after"] = _mirror_stream_snapshot(run, target_policy, opponent_policy)
        mirror_trajectory.add(mirror_row)
    trajectory_id = trajectory.hexdigest()
    outcome = run.outcome
    baseline = {
        "schema": "formicmind.gate1-causal-baseline/v1",
        "matchup": matchup,
        "seed": seed,
        "seat_assignment": "target_on_B" if swapped else "target_on_A",
        "target_policy": target_policy,
        "opponent_policy": opponent_policy,
        "target_side": target_side,
        "trajectory_sha256": trajectory_id,
        "mirror_trajectory_sha256": mirror_trajectory.hexdigest(),
        "ticks_executed": trace.count,
        "per_tick_trace_sha256": trace.hexdigest(),
        "events_count": event_digest.count,
        "events_sha256": event_digest.hexdigest(),
        "metrics": _baseline_metrics(state, run, target_side),
        "final_state_sha256": state_hash(state),
        "final_checkpoint_sha256": checkpoint_hash(state, run),
    }
    return baseline, checkpoint_map


class _EventDigest:
    """Incremental equivalent of the event array's canonical SHA-256."""

    def __init__(self) -> None:
        from .versions import HASH_DOMAIN_EVENTS
        self._hash = hashlib.sha256(HASH_DOMAIN_EVENTS.encode("ascii") + b"\x00[")
        self.count = 0

    def add_all(self, events: Iterable[dict[str, Any]]) -> None:
        for event in events:
            if self.count:
                self._hash.update(b",")
            self._hash.update(canonical_dumps(event))
            self.count += 1

    def hexdigest(self) -> str:
        result = self._hash.copy()
        result.update(b"]")
        return result.hexdigest()


def _local_node(side: int, world_node: int) -> int:
    return world_node if side == 0 else MIRROR[world_node]


def _world_metrics(state: Any, target_side: int) -> dict[str, Any]:
    opponent_side = 1 - target_side
    target = state.colonies[target_side]
    opponent = state.colonies[opponent_side]
    resource_food = []
    for local_node in range(6):
        world_node = local_node if target_side == 0 else MIRROR[local_node]
        resource_food.append(state.node_food[world_node])
    unit_rows = []
    for world_node in range(7):
        counts = []
        for side in SIDES:
            for caste in (WORKER, SOLDIER):
                counts.append(sum(unit.caste == caste and unit.node == world_node for unit in state.colonies[side].units))
        unit_rows.append(counts)
    target_units = target.units
    return {
        "food": target.food,
        "opponent_food": opponent.food,
        "queen_hp": target.queen_hp,
        "opponent_queen_hp": opponent.queen_hp,
        "workers": sum(unit.caste == WORKER for unit in target_units),
        "soldiers": sum(unit.caste == SOLDIER for unit in target_units),
        "opponent_workers": sum(unit.caste == WORKER for unit in opponent.units),
        "opponent_soldiers": sum(unit.caste == SOLDIER for unit in opponent.units),
        "brood": len(target.brood),
        "opponent_brood": len(opponent.brood),
        "starvation_debt": target.starvation_debt,
        "upkeep_remainder10": target.upkeep_remainder10,
        "resource_food_target_local": resource_food,
        "units_by_world_node_ab_ws": unit_rows,
        "target_transit_by_caste": [
            sum(unit.caste == caste and unit.node < 0 for unit in target_units)
            for caste in (WORKER, SOLDIER)
        ],
        "target_units_at_center_by_caste": [
            sum(unit.caste == caste and unit.node == 3 for unit in target_units)
            for caste in (WORKER, SOLDIER)
        ],
        "target_soldiers_at_enemy_home": sum(
            unit.caste == SOLDIER and unit.node == (6 if target_side == 0 else 0) for unit in target_units
        ),
    }


def _event_totals(events: Iterable[Event], target_side: int) -> tuple[dict[str, int], dict[str, int]]:
    counts: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    for event in events:
        counts[event.kind] += 1
        if getattr(event, "side", None) == target_side:
            if event.kind == "harvest":
                totals["harvested_food"] += event.harvested
            elif event.kind == "upkeep":
                totals["upkeep_paid"] += event.paid
            elif event.kind == "lay" and event.laid:
                totals["egg_spend"] += event.total_cost
                totals["eggs_laid"] += event.eggs
            elif event.kind == "hatch":
                totals["eggs_hatched"] += 1
            elif event.kind == "starvation":
                totals["starvation_deaths"] += 1
            elif event.kind == "death":
                totals["combat_deaths"] += 1
            elif event.kind == "queen_damage":
                totals["queen_damage_taken"] += event.damage
        if event.kind == "combat":
            if event.attacker_side == target_side:
                totals["combat_damage_dealt"] += event.applied
            elif event.defender_side == target_side:
                totals["combat_damage_received"] += event.applied
    return dict(sorted(counts.items())), dict(sorted(totals.items()))


def _outcome_object(run: RunState) -> dict[str, Any]:
    if run.outcome is None:
        return {"status": "unknown_censored"}
    outcome = run.outcome
    return {
        "winner": outcome.winner.value,
        "reason": outcome.reason.value,
        "terminal_tick": outcome.terminal_tick,
        "terminal_phase": int(outcome.terminal_phase),
        "tick_completed": outcome.tick_completed,
        "score_x2": [int(value * 2) for value in outcome.score],
    }


def _arm(
    checkpoint: _Checkpoint, axis: str, *, treatment: bool, budget: _TickBudget,
) -> dict[str, Any]:
    state, run = checkpoint.state.clone(), checkpoint.run.clone()
    start_state_hash = state_hash(state)
    start_checkpoint_hash = checkpoint_hash(state, run)
    if (start_state_hash, start_checkpoint_hash) != (
        checkpoint.pre_state_hash, checkpoint.pre_checkpoint_hash,
    ):
        raise Gate1CausalError("arm clone does not match the preregistered pre-intervention checkpoint")
    names = (checkpoint.target_policy, checkpoint.opponent_policy)
    stream_ids = [policy_stream_id(baseline_spec(name).policy_id, 0) for name in names]
    target_stream = stream_ids[0]
    target_controller = _InterventionController(
        checkpoint.target_policy, target_stream, checkpoint, axis, treatment,
    )
    opponent_controller = PolicyController(make_baseline(checkpoint.opponent_policy), stream_ids[1])
    controllers: tuple[Controller, Controller] = (
        (opponent_controller, target_controller) if checkpoint.swapped else (target_controller, opponent_controller)
    )
    if len(controllers) != 2:
        raise Gate1CausalError("bad branch controller assignment")

    event_digest = _EventDigest()
    event_counts: Counter[str] = Counter()
    event_totals: Counter[str] = Counter()
    trace = _ArrayDigest()
    physical_trace = _ArrayDigest()
    horizon_results: dict[str, Any] = {str(offset): None for offset in HORIZON_OFFSETS}
    horizon_ticks = {checkpoint.tick + offset: str(offset) for offset in HORIZON_OFFSETS}
    order_event: dict[str, Any] | None = None
    executed = 0
    while run.outcome is None and state.tick < state.config.t_max:
        tick = state.tick
        try:
            result = budget.step(state, run, controllers)
        except EpisodeFinishedError as exc:
            raise Gate1CausalError(f"engine ended before a full episode result: {exc}") from exc
        executed += 1
        canonical_events = [event.to_canonical() for event in result.events]
        event_digest.add_all(canonical_events)
        physical_events = [
            event for event in canonical_events if event["kind"] not in ("order", "retarget")
        ]
        event_counts.update(event.kind for event in result.events)
        target_side = checkpoint.target_side
        for event in result.events:
            if getattr(event, "side", None) == target_side:
                if event.kind == "harvest":
                    event_totals["harvested_food"] += event.harvested
                elif event.kind == "upkeep":
                    event_totals["upkeep_paid"] += event.paid
                elif event.kind == "lay" and event.laid:
                    event_totals["egg_spend"] += event.total_cost
                    event_totals["eggs_laid"] += event.eggs
                elif event.kind == "hatch":
                    event_totals["eggs_hatched"] += 1
                elif event.kind == "starvation":
                    event_totals["starvation_deaths"] += 1
                elif event.kind == "death":
                    event_totals["combat_deaths"] += 1
                elif event.kind == "queen_damage":
                    event_totals["queen_damage_taken"] += event.damage
            if event.kind == "combat":
                if event.attacker_side == target_side:
                    event_totals["combat_damage_dealt"] += event.applied
                elif event.defender_side == target_side:
                    event_totals["combat_damage_received"] += event.applied
            if tick == checkpoint.tick and isinstance(event, OrderEvent) and event.side == target_side:
                order_event = event.to_canonical()
        current_state_hash = state_hash(state)
        current_checkpoint_hash = checkpoint_hash(state, run)
        trace.add({
            "tick": tick, "state_hash": current_state_hash,
            "checkpoint_hash": current_checkpoint_hash,
            "events_sha256": events_hash(list(result.events)),
            "tick_completed": state.tick == tick + 1,
        })
        physical_trace.add({
            "tick": tick,
            "state": _physical_state_object(state, target_side),
            "events": physical_events,
        })
        if state.tick in horizon_ticks:
            offset = horizon_ticks[state.tick]
            horizon_results[offset] = {
                "state_tick": state.tick,
                "state_sha256": current_state_hash,
                "checkpoint_sha256": current_checkpoint_hash,
                "metrics": _world_metrics(state, target_side),
                "terminal": run.outcome is not None,
            }
    finished = run.outcome is not None
    outcome = _outcome_object(run)
    for offset, endpoint in tuple(horizon_results.items()):
        if endpoint is None:
            horizon_results[offset] = {
                "status": "censored",
                "reason": "terminal_before_horizon" if finished else "study_cutoff_before_horizon",
            }
    terminal_metrics = _world_metrics(state, checkpoint.target_side) if finished else {"status": "unknown_censored"}
    if order_event is None:
        raise Gate1CausalError("no target-side OrderEvent was emitted at the intervention checkpoint")
    return {
        "start_state_sha256": start_state_hash,
        "start_checkpoint_sha256": start_checkpoint_hash,
        "ticks_executed": executed,
        "per_tick_trace_sha256": trace.hexdigest(),
        "physical_trace_ticks": physical_trace.count,
        "physical_trace_sha256": physical_trace.hexdigest(),
        "events_count": event_digest.count,
        "events_sha256": event_digest.hexdigest(),
        "event_counts": dict(sorted(event_counts.items())),
        "event_totals_target": dict(sorted(event_totals.items())),
        "intervention_order_event": order_event,
        "horizons": horizon_results,
        "terminal": outcome,
        "terminal_metrics": terminal_metrics,
        "final_state_sha256": state_hash(state),
        "final_checkpoint_sha256": checkpoint_hash(state, run),
    }


def _physical_state_object(state: Any, target_side: int) -> dict[str, Any]:
    local_to_world = tuple(range(len(MIRROR))) if target_side == 0 else MIRROR
    colonies = []
    for relative_side, world_side in enumerate((target_side, 1 - target_side)):
        colony = state.colonies[world_side]
        units = [
            list(unit_row) for unit_row in sorted(
                (
                    unit.caste, unit.hp,
                    unit.node if unit.node < 0 or target_side == 0 else MIRROR[unit.node],
                    unit.from_node if unit.from_node < 0 or target_side == 0 else MIRROR[unit.from_node],
                    unit.to_node if unit.to_node < 0 or target_side == 0 else MIRROR[unit.to_node],
                    unit.arrival_tick,
                )
                for unit in colony.units
            )
        ]
        brood = [
            list(row) for row in sorted(
                (record.caste, record.laid_tick, record.hatch_tick) for record in colony.brood
            )
        ]
        colonies.append({
            "side": relative_side,
            "food": colony.food,
            "queen_hp": colony.queen_hp,
            "next_lay_tick": colony.next_lay_tick,
            "upkeep_remainder10": colony.upkeep_remainder10,
            "starvation_debt": colony.starvation_debt,
            "units": units,
            "brood": brood,
        })
    return {
        "tick": state.tick,
        "node_food": [state.node_food[node] for node in local_to_world],
        "colonies": colonies,
    }


def _sampled_metric_physical_delta(control: dict[str, Any], treatment: dict[str, Any]) -> bool:
    endpoints: list[tuple[Any, Any]] = []
    for offset in ("30", "150"):
        a, b = control["horizons"][offset], treatment["horizons"][offset]
        if "metrics" in a and "metrics" in b:
            endpoints.append((a["metrics"], b["metrics"]))
    if "food" in control["terminal_metrics"] and "food" in treatment["terminal_metrics"]:
        endpoints.append((control["terminal_metrics"], treatment["terminal_metrics"]))
    return any(left != right for left, right in endpoints)


def _full_episode_physical_delta(control: dict[str, Any], treatment: dict[str, Any]) -> bool:
    return control["physical_trace_sha256"] != treatment["physical_trace_sha256"]


def _endpoint_deltas(control: dict[str, Any], treatment: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for offset in ("30", "150"):
        left, right = control["horizons"][offset], treatment["horizons"][offset]
        if "metrics" not in left or "metrics" not in right:
            result[offset] = {"status": "censored", "reason": "episode_ended_before_common_horizon"}
        else:
            result[offset] = {
                key: right["metrics"][key] if isinstance(right["metrics"][key], list) else right["metrics"][key] - left["metrics"][key]
                for key in left["metrics"]
                if not isinstance(left["metrics"][key], list)
            }
            result[offset]["resource_food_target_local"] = [
                b - a for a, b in zip(
                    left["metrics"]["resource_food_target_local"],
                    right["metrics"]["resource_food_target_local"],
                )
            ]
            result[offset]["units_by_world_node_ab_ws"] = [
                [b - a for a, b in zip(a_row, b_row)]
                for a_row, b_row in zip(
                    left["metrics"]["units_by_world_node_ab_ws"],
                    right["metrics"]["units_by_world_node_ab_ws"],
                )
            ]
            result[offset]["target_transit_by_caste"] = [
                b - a for a, b in zip(
                    left["metrics"]["target_transit_by_caste"],
                    right["metrics"]["target_transit_by_caste"],
                )
            ]
            result[offset]["target_units_at_center_by_caste"] = [
                b - a for a, b in zip(
                    left["metrics"]["target_units_at_center_by_caste"],
                    right["metrics"]["target_units_at_center_by_caste"],
                )
            ]
    if "food" not in control["terminal_metrics"] or "food" not in treatment["terminal_metrics"]:
        result["terminal"] = {"status": "unknown_censored"}
    else:
        left, right = control["terminal_metrics"], treatment["terminal_metrics"]
        result["terminal"] = {key: right[key] - value for key, value in left.items() if type(value) is int}
        result["terminal"]["resource_food_target_local"] = [
            b - a for a, b in zip(left["resource_food_target_local"], right["resource_food_target_local"])
        ]
        result["terminal"]["units_by_world_node_ab_ws"] = [
            [b - a for a, b in zip(a_row, b_row)]
            for a_row, b_row in zip(left["units_by_world_node_ab_ws"], right["units_by_world_node_ab_ws"])
        ]
        result["terminal"]["target_transit_by_caste"] = [
            b - a for a, b in zip(left["target_transit_by_caste"], right["target_transit_by_caste"])
        ]
        result["terminal"]["target_units_at_center_by_caste"] = [
            b - a for a, b in zip(left["target_units_at_center_by_caste"], right["target_units_at_center_by_caste"])
        ]
    return result


def _probe_pair(checkpoint: _Checkpoint, axis: str, budget: _TickBudget,
                baseline: dict[str, Any], manifest_sha256: str,
                source_commit: str, source_fingerprint: str, config_sha256: str,
                policy_sha256s: dict[str, str]) -> dict[str, Any]:
    candidate = checkpoint.candidate_actions[axis]
    common = {
        "schema": SCHEMA_PROBES,
        "manifest_sha256": manifest_sha256,
        "protocol_sha256": _protocol_sha256(),
        "source_commit": source_commit,
        "source_fingerprint_sha256": source_fingerprint,
        "config_sha256": config_sha256,
        "policies_sha256": policy_sha256s[checkpoint.target_policy] + ":" + policy_sha256s[checkpoint.opponent_policy],
        "probe_id": f"{checkpoint.matchup}|{checkpoint.seed}|{'swap' if checkpoint.swapped else 'base'}|{checkpoint.tick}|{axis}",
        "matchup": checkpoint.matchup,
        "seed": checkpoint.seed,
        "seat_assignment": "target_on_B" if checkpoint.swapped else "target_on_A",
        "target_policy": checkpoint.target_policy,
        "opponent_policy": checkpoint.opponent_policy,
        "target_side": checkpoint.target_side,
        "checkpoint_tick": checkpoint.tick,
        "baseline_trajectory_sha256": baseline["trajectory_sha256"],
        "mirror_baseline_trajectory_sha256": baseline["mirror_trajectory_sha256"],
        "pre_state_sha256": checkpoint.pre_state_hash,
        "mirror_pre_state_sha256": checkpoint.mirror_pre_state_hash,
        "pre_checkpoint_sha256": checkpoint.pre_checkpoint_hash,
        "observation_sha256": checkpoint.observation_sha256,
        "mask_sha256": checkpoint.mask_sha256,
        "axis": axis,
        "control_action": _action_object(checkpoint.control_action),
        "candidate_action": {"status": "ineligible"} if candidate is None else _action_object(candidate),
        "action_diff_fields": [] if candidate is None else _action_diff(checkpoint.control_action, candidate),
        "eligible": candidate is not None,
        "ineligible_reason": checkpoint.ineligible_reasons[axis] or "none",
    }
    if candidate is None:
        return common
    control = _arm(checkpoint, axis, treatment=False, budget=budget)
    treatment = _arm(checkpoint, axis, treatment=True, budget=budget)
    expected_fields = [axis]
    if common["action_diff_fields"] != expected_fields:
        raise Gate1CausalError(f"{axis} candidate changes fields {common['action_diff_fields']}")
    for arm, treatment_flag in ((control, False), (treatment, True)):
        event = arm["intervention_order_event"]
        requested = candidate if treatment_flag else checkpoint.control_action
        if event["lay_order"] != requested.lay_order or tuple(event["forage_share_q"]) != requested.forage_share_q or event["army_choice"] != requested.army_choice:
            raise Gate1CausalError("intervention OrderEvent does not match its validated ActionV2")
    target_score_control = control["terminal"].get("score_x2", ["unknown_censored"] * 2)[checkpoint.target_side]
    target_score_treatment = treatment["terminal"].get("score_x2", ["unknown_censored"] * 2)[checkpoint.target_side]
    delta = (
        (target_score_treatment - target_score_control) * 500
        if type(target_score_control) is int and type(target_score_treatment) is int
        else {"status": "unknown_censored"}
    )
    common.update({
        "control": control,
        "treatment": treatment,
        "primary": {
            "control_target_score_x2": target_score_control,
            "treatment_target_score_x2": target_score_treatment,
            "delta_score_millipoints": delta,
        },
        "endpoint_deltas_treatment_minus_control": _endpoint_deltas(control, treatment),
        "order_event_changed": control["intervention_order_event"] != treatment["intervention_order_event"],
        "sampled_horizon_physical_effect": _sampled_metric_physical_delta(control, treatment),
        "physical_effect": _full_episode_physical_delta(control, treatment),
        "empty_action": (
            control["intervention_order_event"] != treatment["intervention_order_event"]
            and not _full_episode_physical_delta(control, treatment)
            and "score_x2" in control["terminal"] and "score_x2" in treatment["terminal"]
        ),
        "unresolved_censored": "score_x2" not in control["terminal"] or "score_x2" not in treatment["terminal"],
    })
    return common


def _run_study(config: SimConfig, manifest: dict[str, Any], manifest_sha256: str,
               source_commit: str, source_fingerprint: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    policy_sha256s = {item["name"]: item["parameters_sha256"] for item in manifest["policies"]}
    budget = _TickBudget()
    baseline_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    for matchup, left, right in MATCHUPS:
        for seed in SEEDS:
            for swapped in (False, True):
                baseline, checkpoints = _run_baseline(config, matchup, left, right, seed, swapped, budget)
                baseline["manifest_sha256"] = manifest_sha256
                baseline["protocol_sha256"] = manifest["protocol_sha256"]
                baseline["source_commit"] = source_commit
                baseline["source_fingerprint_sha256"] = source_fingerprint
                baseline["config_sha256"] = manifest["config_sha256"]
                baseline["policy_sha256s"] = [policy_sha256s[left], policy_sha256s[right]]
                baseline_rows.append(baseline)
                for checkpoint in checkpoints:
                    for axis in AXES:
                        probe_rows.append(_probe_pair(
                            checkpoint, axis, budget, baseline, manifest_sha256, source_commit,
                            source_fingerprint, manifest["config_sha256"], policy_sha256s,
                        ))
    if budget.calls > STEP_BUDGET:
        raise Gate1CausalError("experiment exceeded the preregistered total tick budget")
    baseline_rows.sort(key=lambda row: (
        next(index for index, item in enumerate(MATCHUPS) if item[0] == row["matchup"]),
        row["seed"], row["seat_assignment"],
    ))
    probe_rows.sort(key=lambda row: (
        next(index for index, item in enumerate(MATCHUPS) if item[0] == row["matchup"]),
        row["seed"], row["seat_assignment"], row["checkpoint_tick"], AXES.index(row["axis"]),
    ))
    return baseline_rows, probe_rows


def _rational(value: Fraction | None) -> Any:
    if value is None:
        return {"status": "not_estimable"}
    return {"numerator": value.numerator, "denominator": value.denominator}


def _unique_clusters(rows: list[dict[str, Any]], axis: str) -> dict[str, dict[str, list[tuple[str, int]]]]:
    clusters: dict[str, dict[str, list[tuple[str, int]]]] = defaultdict(lambda: defaultdict(list))
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        if row["axis"] != axis or not row["eligible"] or type(row["primary"]["delta_score_millipoints"]) is not int:
            continue
        key = (
            row["matchup"], row["mirror_baseline_trajectory_sha256"],
            row["mirror_pre_state_sha256"],
        )
        if key in seen:
            continue
        seen.add(key)
        clusters[row["matchup"]][row["mirror_baseline_trajectory_sha256"]].append(
            (row["mirror_pre_state_sha256"], row["primary"]["delta_score_millipoints"])
        )
    return clusters


def _cluster_values(clusters: dict[str, dict[str, list[tuple[str, int]]]]) -> dict[str, dict[str, Fraction]]:
    return {
        matchup: {
            trajectory: sum((value for _, value in values), Fraction(0)) / len(values)
            for trajectory, values in sorted(trajectories.items())
        }
        for matchup, trajectories in sorted(clusters.items())
    }


def _bootstrap_interval(axis: str, grouped: dict[str, dict[str, Fraction]]) -> tuple[Fraction, Fraction, int, int]:
    seed_bytes = hashlib.sha256(f"{PROTOCOL_ID}|bootstrap|{axis}".encode("utf-8")).digest()
    initstate = int.from_bytes(seed_bytes[0:8], "little")
    initseq = int.from_bytes(seed_bytes[8:16], "little")
    rng = Pcg32.seeded(initstate, initseq)
    labels = [matchup for matchup, _, _ in MATCHUPS]
    ordered = {label: sorted(grouped[label].items()) for label in labels}
    means: list[Fraction] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        matchup_means = []
        for matchup in labels:
            clusters = ordered[matchup]
            sampled = [clusters[rng.below(len(clusters))][1] for _ in range(len(clusters))]
            matchup_means.append(sum(sampled, Fraction(0)) / len(sampled))
        means.append(sum(matchup_means, Fraction(0)) / len(matchup_means))
    means.sort()
    lower_rank = (BOOTSTRAP_REPLICATES * 25 + 999) // 1000
    upper_rank = (BOOTSTRAP_REPLICATES * 975 + 999) // 1000
    return means[lower_rank - 1], means[upper_rank - 1], initstate, initseq


def _has_treatment_loss(row: dict[str, Any]) -> bool:
    deltas = row.get("endpoint_deltas_treatment_minus_control") or {}
    for value in deltas.values():
        if "status" in value:
            continue
        for key in ("food", "queen_hp", "workers", "soldiers"):
            if value.get(key, 0) < 0:
                return True
        if value.get("starvation_debt", 0) > 0 or value.get("target_food", 0) < 0:
            return True
        if value.get("resource_food_target_local") and sum(value["resource_food_target_local"]) < 0:
            return True
    if row.get("control") and row.get("treatment"):
        if row["treatment"]["event_totals_target"].get("starvation_deaths", 0) > row["control"]["event_totals_target"].get("starvation_deaths", 0):
            return True
        if row["treatment"]["event_totals_target"].get("combat_deaths", 0) > row["control"]["event_totals_target"].get("combat_deaths", 0):
            return True
    return False


def _summarize(probe_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]], budget_calls: int) -> dict[str, Any]:
    axis_summaries = {}
    for axis in AXES:
        axis_rows = [row for row in probe_rows if row["axis"] == axis]
        eligible = [row for row in axis_rows if row["eligible"]]
        ineligible_counts = dict(sorted(Counter(row["ineligible_reason"] for row in axis_rows if not row["eligible"]).items()))
        groups = _unique_clusters(axis_rows, axis)
        grouped = _cluster_values(groups)
        trajectory_count = sum(len(values) for values in grouped.values())
        unique_states = len({row["mirror_pre_state_sha256"] for row in eligible})
        per_matchup_counts = {label: len(grouped.get(label, {})) for label, _, _ in MATCHUPS}
        enough = (
            trajectory_count >= 10 and unique_states >= 12
            and all(count >= 2 for count in per_matchup_counts.values())
        )
        if all(grouped.get(label) for label, _, _ in MATCHUPS):
            matchup_means = {
                label: sum(grouped[label].values(), Fraction(0)) / len(grouped[label])
                for label, _, _ in MATCHUPS
            }
            point = sum(matchup_means.values(), Fraction(0)) / len(MATCHUPS)
        else:
            matchup_means, point = {}, None
        interval = {"status": "not_estimable", "reason": "minimum_unique_counts_not_met"}
        bootstrap = {"status": "not_run"}
        classification = "INCONCLUSIVE"
        if enough and point is not None:
            lower, upper, initstate, initseq = _bootstrap_interval(axis, grouped)
            interval = {"lower_millipoints": _rational(lower), "upper_millipoints": _rational(upper)}
            bootstrap = {
                "replicates": BOOTSTRAP_REPLICATES,
                "initstate_u64": str(initstate), "initseq_u64": str(initseq),
                "seed_sha256": hashlib.sha256(f"{PROTOCOL_ID}|bootstrap|{axis}".encode("utf-8")).hexdigest(),
            }
            if lower > MDE_MILLIPOINTS:
                classification = "FAVORABLE_SIGNAL"
            elif upper < -MDE_MILLIPOINTS:
                classification = "HARMFUL_SIGNAL"
            elif lower >= -MDE_MILLIPOINTS and upper <= MDE_MILLIPOINTS:
                classification = "PRACTICALLY_SMALL"
        deltas = [row["primary"]["delta_score_millipoints"] for row in eligible
                  if type(row["primary"]["delta_score_millipoints"]) is int]
        possible_dominant = bool(
            classification == "FAVORABLE_SIGNAL" and deltas
            and all(value >= 0 for value in deltas)
            and not any(_has_treatment_loss(row) for row in eligible)
        )
        axis_summaries[axis] = {
            "eligible_pairs": len(eligible),
            "ineligible_pairs": len(axis_rows) - len(eligible),
            "ineligible_reasons": ineligible_counts,
            "unique_pre_state_hashes": unique_states,
            "unique_trajectory_digests": trajectory_count,
            "unique_raw_trajectory_digests": len({row["baseline_trajectory_sha256"] for row in eligible}),
            "unique_mirror_trajectory_clusters": trajectory_count,
            "unique_trajectories_by_matchup": per_matchup_counts,
            "changed_order_events": sum(row.get("order_event_changed", False) for row in eligible),
            "physical_effect_pairs": sum(row.get("physical_effect", False) for row in eligible),
            "sampled_horizon_physical_effect_pairs": sum(
                row.get("sampled_horizon_physical_effect", False) for row in eligible
            ),
            "empty_action_pairs": sum(row.get("empty_action", False) for row in eligible),
            "unresolved_censored_pairs": sum(row.get("unresolved_censored", False) for row in eligible),
            "primary_delta_millipoints": sorted(deltas),
            "matchup_mean_delta_millipoints": {key: _rational(value) for key, value in matchup_means.items()},
            "mean_delta_millipoints": _rational(point),
            "confidence_interval_95": interval,
            "bootstrap": bootstrap,
            "classification": classification,
            "possible_dominant_or_free_action_flag": possible_dominant,
            "minimum_counts_met": enough,
        }
    unique_baselines = {}
    for label, _, _ in MATCHUPS:
        rows = [row for row in baseline_rows if row["matchup"] == label]
        unique_baselines[label] = {
            "scheduled_rows": len(rows),
            "unique_trajectory_digests": len({row["trajectory_sha256"] for row in rows}),
            "unique_mirror_trajectory_clusters": len({row["mirror_trajectory_sha256"] for row in rows}),
            "terminal_reasons": dict(sorted(Counter(row["metrics"]["reason"] for row in rows).items())),
            "terminal_scores_x2_target": sorted(
                row["metrics"]["target_score_x2"] for row in rows
                if type(row["metrics"]["target_score_x2"]) is int
            ),
        }
    return {
        "schema": SCHEMA_SUMMARY,
        "study_status": "OPEN_DIAGNOSTIC_NOT_GATE_PASS",
        "baseline_runs": len(baseline_rows),
        "probe_rows": len(probe_rows),
        "total_step_tick_calls": budget_calls,
        "unique_baselines_by_matchup": unique_baselines,
        "axes": axis_summaries,
    }


def _rows_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(canonical_dumps(row) + b"\n" for row in rows)


def _write_study(out: Path, manifest: dict[str, Any], baseline_rows: list[dict[str, Any]],
                 probe_rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    if out.exists():
        raise Gate1CausalError(f"output already exists; refusing to overwrite immutable study: {out}")
    out.mkdir(parents=True)
    (out / "manifest.json").write_bytes(canonical_dumps(manifest) + b"\n")
    (out / "baselines.jsonl").write_bytes(_rows_bytes(baseline_rows))
    (out / "probes.jsonl").write_bytes(_rows_bytes(probe_rows))
    (out / "summary.json").write_bytes(canonical_dumps(summary) + b"\n")


def run(out: str | Path) -> dict[str, Any]:
    config = default_config()
    source_commit = _git_head()
    manifest = _make_manifest(config, source_commit)
    manifest_sha256 = _sha256(canonical_dumps(manifest))
    source_fingerprint = manifest["source_fingerprint_sha256"]
    baseline_rows, probe_rows = _run_study(
        config, manifest, manifest_sha256, source_commit, source_fingerprint,
    )
    total_calls = sum(row["ticks_executed"] for row in baseline_rows)
    total_calls += sum(
        sum(arm["ticks_executed"] for arm in (row.get("control"), row.get("treatment") ) if arm is not None)
        for row in probe_rows
    )
    summary = _summarize(probe_rows, baseline_rows, total_calls)
    _write_study(Path(out), manifest, baseline_rows, probe_rows, summary)
    return {
        "manifest_sha256": manifest_sha256,
        "out": str(out),
        "baseline_rows": len(baseline_rows),
        "probe_rows": len(probe_rows),
        "step_tick_calls": total_calls,
        "axes": {
            axis: {
                "eligible_pairs": item["eligible_pairs"],
                "unique_pre_state_hashes": item["unique_pre_state_hashes"],
                "unique_trajectory_digests": item["unique_trajectory_digests"],
                "classification": item["classification"],
            }
            for axis, item in summary["axes"].items()
        },
    }


def _read_canonical(path: Path) -> Any:
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise Gate1CausalError(f"{path.name}: missing final newline")
    content = raw[:-1]
    value = canonical_loads(content)
    if canonical_dumps(value) != content:
        raise Gate1CausalError(f"{path.name}: noncanonical JSON")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise Gate1CausalError(f"{path.name}: missing final newline")
    rows = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        value = canonical_loads(line)
        if type(value) is not dict or canonical_dumps(value) != line:
            raise Gate1CausalError(f"{path.name}:{line_number}: expected canonical JSON object")
        rows.append(value)
    return rows


def verify(input_path: str | Path) -> dict[str, Any]:
    root = Path(input_path)
    manifest = _read_canonical(root / "manifest.json")
    if type(manifest) is not dict or manifest.get("schema") != SCHEMA_MANIFEST:
        raise Gate1CausalError("unsupported Gate 1A manifest schema")
    config = default_config()
    expected = _make_manifest(config, manifest.get("source_commit", ""))
    if canonical_dumps(expected) != canonical_dumps(manifest):
        raise Gate1CausalError("manifest provenance/config/policy/protocol/source does not match this verifier")
    manifest_sha256 = _sha256(canonical_dumps(manifest))
    baseline_rows = _read_jsonl(root / "baselines.jsonl")
    probe_rows = _read_jsonl(root / "probes.jsonl")
    saved_summary = _read_canonical(root / "summary.json")
    source_commit = manifest["source_commit"]
    source_fingerprint = manifest["source_fingerprint_sha256"]
    expected_baselines, expected_probes = _run_study(
        config, manifest, manifest_sha256, source_commit, source_fingerprint,
    )
    if _rows_bytes(baseline_rows) != _rows_bytes(expected_baselines):
        raise Gate1CausalError("baseline runs do not reproduce exactly")
    if _rows_bytes(probe_rows) != _rows_bytes(expected_probes):
        raise Gate1CausalError("causal probe arms/actions/horizons/hashes do not reproduce exactly")
    total_calls = sum(row["ticks_executed"] for row in expected_baselines)
    total_calls += sum(
        sum(arm["ticks_executed"] for arm in (row.get("control"), row.get("treatment")) if arm is not None)
        for row in expected_probes
    )
    computed_summary = _summarize(expected_probes, expected_baselines, total_calls)
    if canonical_dumps(saved_summary) != canonical_dumps(computed_summary):
        raise Gate1CausalError("aggregate summary does not reproduce from verified rows")
    return {
        "verified": True,
        "manifest_sha256": manifest_sha256,
        "baseline_rows": len(baseline_rows),
        "probe_rows": len(probe_rows),
        "step_tick_calls": total_calls,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m formicmind.gate1_causal", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run", help="run the preregistered open diagnostic matrix")
    runner.add_argument("--out", required=True, help="new immutable output directory")
    verifier = sub.add_parser("verify", help="rerun and verify a saved experiment")
    verifier.add_argument("--input", required=True, help="study output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args.out) if args.command == "run" else verify(args.input)
    except (Gate1CausalError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"gate1_causal: error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
