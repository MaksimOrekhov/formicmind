"""Preregistered Gate 1B cross-map context study over the unchanged L0 engine."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

from .canonical import canonical_dumps, canonical_loads
from .config import EdgeSpec, NodeResource, SimConfig, Topology
from .events import (
    CombatEvent,
    HarvestEvent,
    LayEvent,
    OrderEvent,
    QueenDamageEvent,
    RetargetEvent,
    StarvationEvent,
    UpkeepEvent,
    events_hash,
)
from .evaluation import controller_pair
from .gate1_causal import _MirrorNormalizer
from .invariants import validate_config
from .observation import check_observation_config
from .reset import reset
from .rules import (
    MIRROR,
    NODE_C,
    NODE_N1,
    NODE_N2,
    NODE_N4,
    NODE_N5,
    Q_TOTAL,
    SOLDIER,
    WORKER,
    default_config,
    local_to_world,
)
from .serialization import (
    checkpoint_hash,
    config_hash,
    config_to_canonical,
    state_hash,
)
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

PROTOCOL_ID = "formicmind.gate1-context/v1"
PROTOCOL_PATH = Path("docs/gate1_context_protocol.md")
OUT_RELATIVE = Path("results/gate1_context")
SCHEMA_MANIFEST = "formicmind.gate1-context-manifest/v1"
SCHEMA_MATCH = "formicmind.gate1-context-match/v1"
SCHEMA_SUMMARY = "formicmind.gate1-context-summary/v1"
POLICIES = ("random", "turtle", "rush", "balanced", "raider")
SEEDS = (20261221, 20261222, 20261223, 20261224, 20261225, 20261226)
TERRAINS = ("T0", "T1", "T2")
FAMILIES = ("R0", "R1")
SEATS = (False, True)  # first-listed policy in A, then in B
SCORE_SCALE = 2
MDE = Fraction(15, 100)
REVERSAL_MARGIN = Fraction(10, 100)
STEP_BUDGET = 2_160_000
EXPECTED_MATCHES = 720
_ROOT = Path(__file__).resolve().parents[2]

_RESOURCE_VALUES = {
    "R0": {
        NODE_N1: (60, 80, 8, 8),
        NODE_N2: (40, 60, 8, 8),
        NODE_C: (80, 120, 12, 12),
        NODE_N4: (60, 80, 8, 8),
        NODE_N5: (40, 60, 8, 8),
    },
    "R1": {
        NODE_N1: (70, 80, 10, 10),
        NODE_N2: (50, 60, 10, 10),
        NODE_C: (40, 120, 4, 4),
        NODE_N4: (70, 80, 10, 10),
        NODE_N5: (50, 60, 10, 10),
    },
}


class Gate1ContextError(ValueError):
    """Invalid study input, incompatible provenance, or failed strict verification."""


class _Digest:
    def __init__(self) -> None:
        self._hash = hashlib.sha256(b"[")
        self._count = 0

    def add(self, value: Any) -> None:
        if self._count:
            self._hash.update(b",")
        self._hash.update(canonical_dumps(value))
        self._count += 1

    def hexdigest(self) -> str:
        digest = self._hash.copy()
        digest.update(b"]")
        return digest.hexdigest()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_protocol_sha() -> str:
    try:
        return _sha((_ROOT / PROTOCOL_PATH).read_bytes())
    except OSError as exc:
        raise Gate1ContextError(f"cannot read frozen protocol: {exc}") from exc


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
        raise Gate1ContextError(f"git {' '.join(args)} failed: {exc}") from exc
    return result.stdout.strip()


def _git_head() -> str:
    value = _git_value("rev-parse", "HEAD")
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise Gate1ContextError("git HEAD is not a lowercase 40-character SHA")
    return value


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


def _fraction(value: Fraction | int) -> dict[str, int]:
    value = Fraction(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


def _policy_descriptor(name: str) -> dict[str, Any]:
    from .baselines import baseline_spec
    from .rng import policy_stream_id

    spec = baseline_spec(name)
    return {
        "name": name,
        "id": spec.full_id,
        "policy_id": spec.policy_id,
        "parameters": _safe(spec.parameter_object()),
        "parameters_sha256": spec.digest(),
        "stream_id": policy_stream_id(spec.policy_id, 0),
        "stochastic": name == "random",
    }


def _safe(value: Any) -> Any:
    if isinstance(value, float):
        return format(value, ".17g")
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _resource_records(family: str, default: SimConfig) -> tuple[NodeResource, ...]:
    table = _RESOURCE_VALUES[family]
    return tuple(
        replace(
            resource,
            start_food=table[resource.node_id][0],
            capacity=table[resource.node_id][1],
            regen=table[resource.node_id][2],
            max_harvest=table[resource.node_id][3],
        ) if resource.node_id in table else resource
        for resource in default.balance.resources
    )


def _terrain_topology(terrain: str, default: SimConfig) -> Topology:
    changes = {
        "T0": {},
        "T1": {
            (NODE_N1, NODE_C): 1,
            (NODE_C, NODE_N4): 1,
            (NODE_N2, NODE_C): 2,
            (NODE_C, NODE_N5): 2,
        },
        "T2": {
            (NODE_N1, NODE_C): 3,
            (NODE_C, NODE_N4): 3,
            (NODE_N2, NODE_C): 4,
            (NODE_C, NODE_N5): 4,
        },
    }[terrain]
    edges = tuple(
        replace(edge, length=changes[(edge.a, edge.b)])
        if (edge.a, edge.b) in changes else edge
        for edge in default.topology.edges
    )
    return replace(default.topology, edges=edges)


def _connected(config: SimConfig) -> bool:
    graph = {node.node_id: set() for node in config.topology.nodes}
    for edge in config.topology.edges:
        graph[edge.a].add(edge.b)
        graph[edge.b].add(edge.a)
    seen, pending = set(), [0]
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        pending.extend(graph[node] - seen)
    return len(seen) == 7


def _validate_experimental_config(
    config: SimConfig, family: str, terrain: str, default: SimConfig,
) -> None:
    validate_config(config)
    check_observation_config(config)
    if config.rules_version != default.rules_version or config.balance_version != default.balance_version:
        raise Gate1ContextError("experimental config changed a rules or balance version")
    if config.policy_interval_k != 30 or config.t_max != 3000:
        raise Gate1ContextError("experimental config changed policy cadence or T_max")
    if config.topology.nodes != default.topology.nodes or not _connected(config):
        raise Gate1ContextError("experimental topology violates the seven-node connected contract")
    if {(edge.a, edge.b) for edge in config.topology.edges} != {
        (edge.a, edge.b) for edge in default.topology.edges
    }:
        raise Gate1ContextError("experimental topology changed the edge set")
    expected_topology = _terrain_topology(terrain, default)
    if config.topology != expected_topology:
        raise Gate1ContextError(f"{family}-{terrain} has unexpected route geometry")
    if replace(default.balance, resources=config.balance.resources) != config.balance:
        raise Gate1ContextError("experimental map changed a non-resource balance value")
    if config.balance.resources != _resource_records(family, default):
        raise Gate1ContextError(f"{family}-{terrain} has unexpected resource values")
    mirror = MIRROR
    for node_id, resource in enumerate(config.balance.resources):
        other = config.balance.resources[mirror[node_id]]
        if (
            resource.start_food, resource.capacity, resource.regen, resource.max_harvest
        ) != (other.start_food, other.capacity, other.regen, other.max_harvest):
            raise Gate1ContextError("resource map is not mirror-symmetric")
    sources = [resource for resource in config.balance.resources if resource.capacity]
    totals = (
        sum(resource.start_food for resource in sources),
        sum(resource.regen for resource in sources),
        sum(resource.max_harvest for resource in sources),
        sum(resource.capacity for resource in sources),
    )
    if totals != (280, 44, 44, 400):
        raise Gate1ContextError(f"resource totals differ from protocol: {totals}")


def _build_configs() -> dict[str, SimConfig]:
    default = default_config()
    default_canonical = canonical_dumps(config_to_canonical(default))
    configs = {}
    for family in FAMILIES:
        for terrain in TERRAINS:
            key = f"{family}-{terrain}"
            configs[key] = replace(
                default,
                topology=_terrain_topology(terrain, default),
                balance=replace(default.balance, resources=_resource_records(family, default)),
            )
            _validate_experimental_config(configs[key], family, terrain, default)
    if canonical_dumps(config_to_canonical(default_config())) != default_canonical:
        raise Gate1ContextError("default_config changed while generating experimental maps")
    return configs


def _map_descriptors(configs: dict[str, SimConfig]) -> list[dict[str, Any]]:
    result = []
    for name, config in configs.items():
        family, terrain = name.split("-")
        result.append({
            "id": name,
            "resource_family": family,
            "terrain_profile": terrain,
            "config": config_to_canonical(config),
            "config_sha256": config_hash(config),
        })
    return result


def _expected_matchups() -> list[tuple[str, str]]:
    return list(itertools.combinations(POLICIES, 2))


def _manifest(source_commit: str, origin_main_sha: str) -> dict[str, Any]:
    configs = _build_configs()
    matchups = _expected_matchups()
    return {
        "schema": SCHEMA_MANIFEST,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": _read_protocol_sha(),
        "source_commit_sha": source_commit,
        "source_fingerprint_sha256": _source_fingerprint(),
        "origin_main_sha": origin_main_sha,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "pythonpath_environment": os.environ.get("PYTHONPATH", ""),
        "command": "PYTHONPATH=src python -m formicmind.gate1_context run --out results/gate1_context",
        "versions": _versions(),
        "maps": _map_descriptors(configs),
        "policies": [_policy_descriptor(name) for name in POLICIES],
        "study": {
            "policy_order": list(POLICIES),
            "matchups": [{"first": left, "second": right} for left, right in matchups],
            "open_seeds": list(SEEDS),
            "seat_assignments": ["first_policy_on_A", "first_policy_on_B"],
            "resource_families": list(FAMILIES),
            "terrain_profiles": list(TERRAINS),
            "expected_matches": EXPECTED_MATCHES,
            "step_tick_call_budget": STEP_BUDGET,
            "score_encoding": "score_x2 in {0,1,2}; exact Fraction for averages",
            "pairing": "average both seats within each resource family, then R1 minus R0",
            "independence": (
                "one composite paired block across both families and seats; policy-relative "
                "normalized trajectory digests; repeated digests count once"
            ),
            "confidence_intervals": "none; maps are fixed and deterministic repeats are deduplicated",
        },
    }


def _local_node(side: int, world_node: int) -> int:
    return world_node if side == 0 else MIRROR[world_node]


def _colony_metrics(state: Any, side: int) -> dict[str, Any]:
    colony = state.colonies[side]
    adults = {
        caste: [0] * 7 for caste in (WORKER, SOLDIER)
    }
    in_transit = {WORKER: 0, SOLDIER: 0}
    transit_destinations = {WORKER: [0] * 7, SOLDIER: [0] * 7}
    for unit in colony.units:
        if unit.node >= 0:
            adults[unit.caste][_local_node(side, unit.node)] += 1
        else:
            in_transit[unit.caste] += 1
            if unit.to_node >= 0:
                transit_destinations[unit.caste][_local_node(side, unit.to_node)] += 1
    brood = {
        "workers": sum(record.caste == WORKER for record in colony.brood),
        "soldiers": sum(record.caste == SOLDIER for record in colony.brood),
    }
    local_food = [
        state.node_food[local_to_world(side, local)] for local in range(7)
    ]
    return {
        "food": colony.food,
        "queen_hp": colony.queen_hp,
        "workers": sum(unit.caste == WORKER for unit in colony.units),
        "soldiers": sum(unit.caste == SOLDIER for unit in colony.units),
        "adults_stationed_by_local_node": {
            "workers": adults[WORKER],
            "soldiers": adults[SOLDIER],
        },
        "adults_in_transit": {
            "workers": in_transit[WORKER],
            "soldiers": in_transit[SOLDIER],
        },
        "transit_destinations_by_local_node": {
            "workers": transit_destinations[WORKER],
            "soldiers": transit_destinations[SOLDIER],
        },
        "brood": brood,
        "starvation_debt": colony.starvation_debt,
        "upkeep_remainder10": colony.upkeep_remainder10,
        "node_food_local": local_food,
    }


def _new_policy_metrics() -> dict[str, Any]:
    return {
        "order_count": 0,
        "lay_none_orders": 0,
        "home_only_forage_orders": 0,
        "army_choice_counts": {},
        "forage_vector_counts": {},
        "strategy_order_counts": {},
        "lay_attempts": 0,
        "lay_successes": 0,
        "lay_food_rejections": 0,
        "egg_food_spent": 0,
        "harvested": 0,
        "upkeep_due": 0,
        "upkeep_paid": 0,
        "upkeep_missing": 0,
        "starvation_deaths": 0,
        "combat_attacks": 0,
        "combat_damage": 0,
        "queen_damage": 0,
        "retargets": 0,
    }


def _record_event(metrics: dict[str, Any], policy_by_side: tuple[str, str], event: Any) -> None:
    side = getattr(event, "side", None)
    policy = policy_by_side[side] if side in (0, 1) else None
    if isinstance(event, OrderEvent) and policy is not None:
        item = metrics[policy]
        item["order_count"] += 1
        item["lay_none_orders"] += int(event.lay_order == 0)
        item["home_only_forage_orders"] += int(
            tuple(event.forage_share_q) == (Q_TOTAL, 0, 0, 0, 0, 0)
        )
        army = str(event.army_choice)
        item["army_choice_counts"][army] = item["army_choice_counts"].get(army, 0) + 1
        forage = canonical_dumps(list(event.forage_share_q)).decode("ascii")
        item["forage_vector_counts"][forage] = item["forage_vector_counts"].get(forage, 0) + 1
        strategy = canonical_dumps({
            "lay_order": event.lay_order,
            "forage_share_q": list(event.forage_share_q),
            "army_choice": event.army_choice,
            "avoid_center": event.avoid_center,
        }).decode("ascii")
        item["strategy_order_counts"][strategy] = item["strategy_order_counts"].get(strategy, 0) + 1
    elif isinstance(event, LayEvent) and policy is not None:
        item = metrics[policy]
        item["lay_attempts"] += 1
        item["lay_successes"] += int(event.laid)
        item["lay_food_rejections"] += int(not event.laid)
        item["egg_food_spent"] += event.total_cost if event.laid else 0
    elif isinstance(event, HarvestEvent) and policy is not None:
        metrics[policy]["harvested"] += event.harvested
    elif isinstance(event, UpkeepEvent) and policy is not None:
        item = metrics[policy]
        item["upkeep_due"] += event.due
        item["upkeep_paid"] += event.paid
        item["upkeep_missing"] += event.missing
    elif isinstance(event, StarvationEvent) and policy is not None:
        metrics[policy]["starvation_deaths"] += 1
    elif isinstance(event, CombatEvent):
        attacker = policy_by_side[event.attacker_side]
        metrics[attacker]["combat_attacks"] += 1
        metrics[attacker]["combat_damage"] += event.applied
    elif isinstance(event, QueenDamageEvent):
        metrics[policy_by_side[event.side]]["queen_damage"] += event.damage
    elif isinstance(event, RetargetEvent) and policy is not None:
        metrics[policy]["retargets"] += 1


def _finish_policy_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for name, raw in metrics.items():
        orders = raw["order_count"]
        army_counts = raw["army_choice_counts"]
        forage_counts = raw["forage_vector_counts"]
        strategy_counts = raw["strategy_order_counts"]
        highest_army = max(army_counts.values(), default=0)
        highest_forage = max(forage_counts.values(), default=0)
        result[name] = {
            **raw,
            "army_choice_counts": dict(army_counts),
            "forage_vector_counts": dict(forage_counts),
            "strategy_order_counts": dict(strategy_counts),
            "lay_none_fraction": (
                _fraction(Fraction(raw["lay_none_orders"], orders)) if orders else None
            ),
            "home_only_forage_fraction": (
                _fraction(Fraction(raw["home_only_forage_orders"], orders)) if orders else None
            ),
            "army_choice_diversity": len(army_counts),
            "forage_vector_diversity": len(forage_counts),
            "strategy_order_diversity": len(strategy_counts),
            "army_garrison_fraction": (
                _fraction(Fraction(army_counts.get("0", 0), orders)) if orders else None
            ),
            "army_defend_fraction": (
                _fraction(Fraction(sum(army_counts.get(str(choice), 0)
                                       for choice in range(1, 6)), orders))
                if orders else None
            ),
            "army_raid_fraction": (
                _fraction(Fraction(army_counts.get("6", 0), orders)) if orders else None
            ),
            "repetitive_orders_flag": bool(
                orders >= 10 and (
                    highest_forage * 10 >= orders * 9
                    or highest_army * 10 >= orders * 9
                    or raw["lay_none_orders"] * 4 >= orders * 3
                )
            ),
        }
    return result


def _window_object(run: Any, target_side: int) -> list[dict[str, int]]:
    ordered = (run.windows[target_side], run.windows[1 - target_side])
    return [
        {
            "harvest": window.harvest,
            "upkeep_paid": window.upkeep_paid,
            "egg_spend": window.egg_spend,
            "starvation_debt_delta": window.starvation_debt_delta,
            "window_ticks": window.window_ticks,
        }
        for window in ordered
    ]


def _random_stream_state(run: Any, policy_names: tuple[str, str]) -> dict[str, Any]:
    from .baselines import baseline_spec
    from .rng import policy_stream_id

    result = {}
    for name in policy_names:
        if name == "random":
            stream_id = policy_stream_id(baseline_spec(name).policy_id, 0)
            result[name] = run.stream(stream_id).to_canonical()
    return result


def _normalized_checkpoint_sha(
    normalizer: _MirrorNormalizer,
    state: Any,
    run: Any,
    policy_names: tuple[str, str],
) -> str:
    outcome = None if run.outcome is None else run.outcome.to_canonical()
    if outcome is not None:
        winner = outcome["winner"]
        if winner in ("A", "B"):
            winner_side = 0 if winner == "A" else 1
            outcome["winner"] = "target" if winner_side == normalizer.target_side else "opponent"
        outcome["sides"] = sorted(
            0 if side == normalizer.target_side else 1 for side in outcome["sides"]
        )
    value = {
        "state": normalizer.state_object(state),
        "windows": _window_object(run, normalizer.target_side),
        "random_policy_streams": _random_stream_state(run, policy_names),
    }
    if outcome is not None:
        value["outcome"] = outcome
    return _sha(canonical_dumps(value))


def _execute_match(
    map_id: str,
    config: SimConfig,
    first: str,
    second: str,
    seed: int,
    swapped: bool,
    call_counter: list[int],
) -> dict[str, Any]:
    pair = controller_pair(None, a=first, b=second, swapped=swapped)
    controllers = pair.controllers
    if any(not isinstance(controller, PolicyController) for controller in controllers):
        raise Gate1ContextError("the fixed tournament must use only PolicyController")
    stream_ids = tuple(controller.stream_id for controller in controllers)
    state, run = reset(config, seed, stream_ids)
    policy_by_side = (second, first) if swapped else (first, second)
    target_side = 1 if swapped else 0
    normalizer = _MirrorNormalizer(target_side)
    metrics = {first: _new_policy_metrics(), second: _new_policy_metrics()}
    event_counts: dict[str, int] = {}
    event_digest = _Digest()
    trace_digest = _Digest()
    trajectory_digest = _Digest()
    random_trajectory_digest = _Digest()
    horizons = {str(horizon): None for horizon in (100, 300, 1000)}
    ticks_executed = 0

    while run.outcome is None and state.tick < config.t_max:
        if call_counter[0] >= STEP_BUDGET:
            raise Gate1ContextError(f"step_tick budget exceeded ({STEP_BUDGET})")
        tick = state.tick
        start_state_sha = state_hash(state)
        start_checkpoint_sha = checkpoint_hash(state, run)
        normalized_start_state = normalizer.state_object(state)
        normalized_start_checkpoint_sha = _normalized_checkpoint_sha(
            normalizer, state, run, (first, second),
        )
        rng_before = (
            _random_stream_state(run, (first, second))
            if tick % config.policy_interval_k == 0 else None
        )
        result = step_tick(state, run, controllers)
        ticks_executed += 1
        call_counter[0] += 1
        end_state_sha = state_hash(state)
        end_checkpoint_sha = checkpoint_hash(state, run)
        end_normalized_state = normalizer.state_object(state)
        end_normalized_checkpoint_sha = _normalized_checkpoint_sha(
            normalizer, state, run, (first, second),
        )
        tick_events = list(result.events)
        tick_event_sha = events_hash(tick_events)
        event_digest.add({"tick": tick, "events_sha256": tick_event_sha})
        normalized_events = [normalizer.event_object(event) for event in tick_events]
        normalized_events.sort(key=canonical_dumps)
        random_after = (
            _random_stream_state(run, (first, second))
            if tick % config.policy_interval_k == 0 else None
        )
        if rng_before is not None and "random" in rng_before:
            random_trajectory_digest.add({
                "tick": tick,
                "before": rng_before["random"],
                "after": random_after["random"],
            })
        for event in tick_events:
            kind = event.kind
            event_counts[kind] = event_counts.get(kind, 0) + 1
            _record_event(metrics, policy_by_side, event)
        trace_digest.add({
            "tick": tick,
            "start_state_sha256": start_state_sha,
            "start_checkpoint_sha256": start_checkpoint_sha,
            "end_state_sha256": end_state_sha,
            "end_checkpoint_sha256": end_checkpoint_sha,
            "events_sha256": tick_event_sha,
            "tick_completed": state.tick == tick + 1,
        })
        trajectory_tick: dict[str, Any] = {
            "tick": tick,
            "start_state": normalized_start_state,
            "start_checkpoint_sha256": normalized_start_checkpoint_sha,
            "events": normalized_events,
            "end_state": end_normalized_state,
            "end_checkpoint_sha256": end_normalized_checkpoint_sha,
        }
        if rng_before is not None:
            trajectory_tick["random_streams_before"] = rng_before
            trajectory_tick["random_streams_after"] = random_after
        trajectory_digest.add(trajectory_tick)
        if run.outcome is None and state.tick in {100, 300, 1000}:
            horizons[str(state.tick)] = {
                "state_tick": state.tick,
                "policies": {
                    name: _colony_metrics(state, 1 if swapped and name == first
                                          or not swapped and name == second else 0)
                    for name in (first, second)
                },
                "cumulative_events": _finish_policy_metrics(metrics),
            }

    outcome = run.outcome
    if outcome is None:
        status = "censored_incomplete"
        terminal = None
        scores_x2 = None
    else:
        status = "complete"
        terminal = {
            **outcome.to_canonical(),
            "state_tick": state.tick,
            "tick_completed": outcome.tick_completed,
        }
        scores_x2 = {
            first: int(outcome.score[1 if swapped else 0] * SCORE_SCALE),
            second: int(outcome.score[0 if swapped else 1] * SCORE_SCALE),
        }
    for horizon in (100, 300, 1000):
        snapshot = horizons[str(horizon)]
        if snapshot is None:
            horizons[str(horizon)] = {
                "status": "not_measured",
                "requested_completed_ticks": horizon,
                "reason": (
                    "episode_ended_before_live_horizon"
                    if outcome is not None else "engine_cutoff_without_terminal"
                ),
                "terminal_tick": (
                    "not_measured" if outcome is None else outcome.terminal_tick
                ),
                "terminal_reason": (
                    "not_measured" if outcome is None else outcome.reason.value
                ),
            }
        else:
            horizons[str(horizon)] = {
                **snapshot,
                "status": "reached",
                "requested_completed_ticks": horizon,
            }
    terminal_metrics = {
        name: _colony_metrics(state, 1 if (swapped == (name == first)) else 0)
        for name in (first, second)
    }
    return {
        "schema": SCHEMA_MATCH,
        "status": status,
        "map_id": map_id,
        "resource_family": map_id.split("-")[0],
        "terrain_profile": map_id.split("-")[1],
        "config_sha256": config_hash(config),
        "matchup": f"{first}-{second}",
        "policies": [first, second],
        "policy_descriptors_sha256": {
            first: _policy_descriptor(first)["parameters_sha256"],
            second: _policy_descriptor(second)["parameters_sha256"],
        },
        "stream_ids": {
            "A": controllers[0].stream_id,
            "B": controllers[1].stream_id,
        },
        "seed": seed,
        "seat_assignment": {
            "A": policy_by_side[0],
            "B": policy_by_side[1],
        },
        "ticks_executed": ticks_executed,
        "terminal": terminal,
        "score_x2": scores_x2,
        "terminal_metrics": terminal_metrics,
        "horizons": horizons,
        "event_counts": dict(sorted(event_counts.items())),
        "policy_event_metrics": _finish_policy_metrics(metrics),
        "events_trace_sha256": event_digest.hexdigest(),
        "integrity_trace_sha256": trace_digest.hexdigest(),
        "trajectory_sha256": trajectory_digest.hexdigest(),
        "random_stream_trajectory_sha256": (
            random_trajectory_digest.hexdigest()
            if first == "random" or second == "random" else None
        ),
        "final_state_sha256": state_hash(state),
        "final_checkpoint_sha256": checkpoint_hash(state, run),
        "paired_trajectory_block_sha256": None,
    }


def _schedule() -> list[tuple[str, str, str, int, bool]]:
    return [
        (map_id, first, second, seed, swapped)
        for map_id in (f"{family}-{terrain}" for family in FAMILIES for terrain in TERRAINS)
        for first, second in _expected_matchups()
        for seed in SEEDS
        for swapped in SEATS
    ]


def _attach_paired_block_digests(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, int], dict[tuple[str, str], dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "complete":
            continue
        key = (row["matchup"], row["terrain_profile"], row["seed"])
        group = grouped.setdefault(key, {})
        group[(row["resource_family"], row["seat_assignment"]["A"] == row["policies"][0])] = row
    for (matchup, terrain, seed), group in grouped.items():
        components = []
        for family in FAMILIES:
            for first_on_a in (True, False):
                row = group.get((family, first_on_a))
                components.append({
                    "family": family,
                    "first_policy_on_A": first_on_a,
                    "trajectory_sha256": None if row is None else row["trajectory_sha256"],
                    "config_sha256": None if row is None else row["config_sha256"],
                })
        digest = _sha(canonical_dumps({
            "matchup": matchup,
            "terrain_profile": terrain,
            "components": components,
        }))
        for row in group.values():
            row["paired_trajectory_block_sha256"] = digest


def _write_rows(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical_dumps(_canonical_missing(row)) + b"\n" for row in rows)


def _canonical_missing(value: Any) -> Any:
    """Encode absent measurements with the canonical JSON-safe sentinel."""
    if value is None:
        return "not_measured"
    if isinstance(value, dict):
        return {str(key): _canonical_missing(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical_missing(item) for item in value]
    return value


def _read_json(path: Path) -> Any:
    raw = path.read_bytes()
    if raw.endswith(b"\n"):
        raw = raw[:-1]
    return canonical_loads(raw)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    result = []
    for line_number, line in enumerate(path.read_bytes().splitlines(), start=1):
        if not line:
            raise Gate1ContextError(f"{path.name}:{line_number}: empty JSONL row")
        value = canonical_loads(line)
        if not isinstance(value, dict):
            raise Gate1ContextError(f"{path.name}:{line_number}: row is not an object")
        result.append(value)
    return result


def _mean(values: list[Fraction]) -> Fraction | None:
    return None if not values else sum(values, Fraction()) / len(values)


def _row_score(row: dict[str, Any], policy: str) -> Fraction | None:
    value = row.get("score_x2")
    return None if not isinstance(value, dict) else Fraction(value[policy], SCORE_SCALE)


def _paired_seat_score(rows: list[dict[str, Any]], policy: str) -> Fraction | None:
    scores = [_row_score(row, policy) for row in rows]
    if len(scores) != 2 or any(score is None for score in scores):
        return None
    return sum(scores, Fraction()) / 2


def _context_label(
    mean_delta: Fraction | None,
    terrain_means: list[Fraction | None],
    *,
    adequate: bool,
    complete: bool,
) -> str:
    if not adequate or not complete or mean_delta is None or len(terrain_means) != 3:
        return "INCONCLUSIVE"
    direction = 1 if mean_delta > 0 else -1 if mean_delta < 0 else 0
    concordant = sum(
        value is not None and direction != 0 and value * direction > 0
        for value in terrain_means
    )
    return (
        "CONTEXT_SENSITIVITY_CANDIDATE"
        if direction != 0 and abs(mean_delta) >= MDE and concordant >= 2
        else "NO_CONTEXT_SIGNAL_IN_SCHEDULE"
    )


def _rank_reversal_label(
    margins: dict[str, Fraction | None], *, adequate: bool, complete: bool,
) -> str:
    if not adequate or not complete or any(margins.get(family) is None for family in FAMILIES):
        return "INCONCLUSIVE"
    r0, r1 = margins["R0"], margins["R1"]
    if (
        (r0 >= REVERSAL_MARGIN and r1 <= -REVERSAL_MARGIN)
        or (r1 >= REVERSAL_MARGIN and r0 <= -REVERSAL_MARGIN)
    ):
        return "PAIRWISE_RANK_REVERSAL_CANDIDATE"
    return "NO_CONTEXT_SIGNAL_IN_SCHEDULE"


def _groups(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    result: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "complete":
            continue
        key = (row["matchup"], row["terrain_profile"], row["seed"])
        result.setdefault(key, []).append(row)
    return result


def _build_summary(rows: list[dict[str, Any]], calls: int) -> dict[str, Any]:
    schedule_complete = (
        len(rows) == EXPECTED_MATCHES
        and all(row.get("status") == "complete" for row in rows)
    )
    if schedule_complete:
        _attach_paired_block_digests(rows)
    block_groups = _groups(rows)
    pairs = _expected_matchups()
    map_scores = []

    for family in FAMILIES:
        for terrain in TERRAINS:
            map_id = f"{family}-{terrain}"
            for first, second in pairs:
                matchup = f"{first}-{second}"
                scheduled = [
                    row for row in rows
                    if row.get("map_id") == map_id and row.get("matchup") == matchup
                ]
                dedup: dict[str, list[Fraction]] = {}
                wins = draws = losses = timeouts = 0
                for row in scheduled:
                    score = _row_score(row, first)
                    if score is None:
                        continue
                    digest = row["paired_trajectory_block_sha256"]
                    dedup.setdefault(digest, []).append(score)
                    wins += int(score == 1)
                    draws += int(score == Fraction(1, 2))
                    losses += int(score == 0)
                    timeouts += int(row["terminal"]["reason"] == "timeout")
                unique_scores = [
                    sum(values, Fraction()) / len(values) for values in dedup.values()
                ]
                first_mean = _mean(unique_scores)
                map_scores.append({
                    "map_id": map_id,
                    "matchup": matchup,
                    "scheduled_matches": len(scheduled),
                    "completed_matches": len(scheduled) if all(
                        row.get("status") == "complete" for row in scheduled
                    ) else sum(row.get("status") == "complete" for row in scheduled),
                    "first_policy_wins": wins,
                    "draws": draws,
                    "first_policy_losses": losses,
                    "engine_timeouts": timeouts,
                    "unique_paired_blocks": len(dedup),
                    "score_by_policy_unique": None if first_mean is None else {
                        first: _fraction(first_mean),
                        second: _fraction(1 - first_mean),
                    },
                })

    matchup_scores = {}
    for item in map_scores:
        score = item["score_by_policy_unique"]
        if score is not None:
            matchup_scores[(item["map_id"], item["matchup"])] = {
                name: Fraction(value["numerator"], value["denominator"])
                for name, value in score.items()
            }
    policy_map_scores = []
    for family in FAMILIES:
        for terrain in TERRAINS:
            map_id = f"{family}-{terrain}"
            for policy in POLICIES:
                opponents = {}
                values = []
                for opponent in POLICIES:
                    if opponent == policy:
                        continue
                    first, second = next(
                        pair for pair in pairs if policy in pair and opponent in pair
                    )
                    scores = matchup_scores.get((map_id, f"{first}-{second}"))
                    if scores is None:
                        continue
                    value = scores[policy]
                    opponents[opponent] = _fraction(value)
                    values.append(value)
                policy_map_scores.append({
                    "map_id": map_id,
                    "policy": policy,
                    "opponent_scores_equal_weight": opponents,
                    "mean_score_equal_weight_opponents": (
                        None if len(values) != 4 else _fraction(_mean(values))
                    ),
                })
    policy_family_rankings = []
    for family in FAMILIES:
        family_rows = []
        for policy in POLICIES:
            opponent_means = {}
            terrain_means = {}
            for opponent in POLICIES:
                if opponent == policy:
                    continue
                by_terrain = []
                for terrain in TERRAINS:
                    first, second = next(
                        pair for pair in pairs if policy in pair and opponent in pair
                    )
                    scores = matchup_scores.get((f"{family}-{terrain}", f"{first}-{second}"))
                    if scores is not None:
                        by_terrain.append(scores[policy])
                if len(by_terrain) == len(TERRAINS):
                    opponent_means[opponent] = _mean(by_terrain)
            for terrain in TERRAINS:
                per_opponent = []
                for opponent in POLICIES:
                    if opponent == policy:
                        continue
                    first, second = next(
                        pair for pair in pairs if policy in pair and opponent in pair
                    )
                    scores = matchup_scores.get((f"{family}-{terrain}", f"{first}-{second}"))
                    if scores is not None:
                        per_opponent.append(scores[policy])
                if len(per_opponent) == 4:
                    terrain_means[terrain] = _mean(per_opponent)
            overall = (
                _mean(list(terrain_means.values()))
                if len(terrain_means) == len(TERRAINS) else None
            )
            family_rows.append({
                "policy": policy,
                "mean_score_by_opponent_equal_weight": {
                    opponent: None if value is None else _fraction(value)
                    for opponent, value in opponent_means.items()
                },
                "mean_score_by_terrain_equal_weight": {
                    terrain: _fraction(value) for terrain, value in terrain_means.items()
                },
                "family_mean_score": None if overall is None else _fraction(overall),
                "_score": overall,
            })
        for row in family_rows:
            value = row["_score"]
            row["rank"] = (
                None if value is None else 1 + sum(
                    other["_score"] is not None and other["_score"] > value
                    for other in family_rows
                )
            )
        for row in family_rows:
            del row["_score"]
        policy_family_rankings.append({"resource_family": family, "policies": family_rows})

    pair_summaries = []
    for first, second in pairs:
        matchup = f"{first}-{second}"
        terrain_summaries = []
        all_adequate = True
        overall_deltas = []
        family_margins = {family: [] for family in FAMILIES}
        for terrain in TERRAINS:
            block_rows: dict[str, dict[str, Any]] = {}
            scheduled_for_profile = [
                group for key, group in block_groups.items()
                if key[0] == matchup and key[1] == terrain
            ]
            for group in scheduled_for_profile:
                digest = group[0]["paired_trajectory_block_sha256"]
                block_rows[digest] = {"rows": group}
            deltas_by_digest: dict[str, Fraction] = {}
            margins_by_family: dict[str, dict[str, Fraction]] = {f: {} for f in FAMILIES}
            for digest, item in block_rows.items():
                group = item["rows"]
                family_scores = {}
                for family in FAMILIES:
                    selected = [
                        row for row in group if row["resource_family"] == family
                    ]
                    mean_score = _paired_seat_score(selected, first)
                    if mean_score is None:
                        continue
                    family_scores[family] = mean_score
                    margins_by_family[family][digest] = (
                        family_scores[family] - (1 - family_scores[family])
                    )
                if len(family_scores) == 2:
                    deltas_by_digest[digest] = family_scores["R1"] - family_scores["R0"]
            distinct = len(deltas_by_digest)
            complete_scheduled = sum(
                len(group) == 4 and all(row.get("status") == "complete" for row in group)
                for group in scheduled_for_profile
            )
            adequate = distinct >= 2 and complete_scheduled == len(SEEDS)
            all_adequate = all_adequate and adequate
            terrain_mean = _mean(list(deltas_by_digest.values()))
            if terrain_mean is not None:
                overall_deltas.append(terrain_mean)
            for family in FAMILIES:
                margin = _mean(list(margins_by_family[family].values()))
                if margin is not None:
                    family_margins[family].append(margin)
            terrain_summaries.append({
                "terrain_profile": terrain,
                "scheduled_seed_blocks": len(scheduled_for_profile),
                "distinct_paired_trajectory_blocks": distinct,
                "complete_seed_blocks": complete_scheduled,
                "independent_unit_minimum_met": adequate,
                "R0_first_policy_mean_score": (
                    None if not margins_by_family["R0"] else _fraction(
                        _mean([
                            (value + 1) / 2 for value in margins_by_family["R0"].values()
                        ])
                    )
                ),
                "R1_first_policy_mean_score": (
                    None if not margins_by_family["R1"] else _fraction(
                        _mean([
                            (value + 1) / 2 for value in margins_by_family["R1"].values()
                        ])
                    )
                ),
                "R1_minus_R0_score_delta": (
                    None if terrain_mean is None else _fraction(terrain_mean)
                ),
            })
        mean_delta = _mean(overall_deltas) if len(overall_deltas) == len(TERRAINS) else None
        margins = {
            family: _mean(values) if len(values) == len(TERRAINS) else None
            for family, values in family_margins.items()
        }
        terrain_means = [
            None if item["R1_minus_R0_score_delta"] is None
            else Fraction(
                item["R1_minus_R0_score_delta"]["numerator"],
                item["R1_minus_R0_score_delta"]["denominator"],
            )
            for item in terrain_summaries
        ]
        context_label = _context_label(
            mean_delta, terrain_means,
            adequate=all_adequate, complete=schedule_complete,
        )
        reversal_label = _rank_reversal_label(
            margins, adequate=all_adequate, complete=schedule_complete,
        )
        pair_summaries.append({
            "matchup": matchup,
            "first_policy": first,
            "second_policy": second,
            "profiles": terrain_summaries,
            "equally_weighted_context_delta": (
                None if mean_delta is None else _fraction(mean_delta)
            ),
            "context_sensitivity_label": context_label,
            "family_head_to_head_margins_first_minus_second": {
                family: None if value is None else _fraction(value)
                for family, value in margins.items()
            },
            "rank_reversal_label": reversal_label,
        })

    degeneracy = _degeneracy_summary(rows)
    candidate_count = sum(
        item["context_sensitivity_label"] == "CONTEXT_SENSITIVITY_CANDIDATE"
        or item["rank_reversal_label"] == "PAIRWISE_RANK_REVERSAL_CANDIDATE"
        for item in pair_summaries
    )
    evaluated_count = sum(
        item["context_sensitivity_label"] != "INCONCLUSIVE"
        and item["rank_reversal_label"] != "INCONCLUSIVE"
        for item in pair_summaries
    )
    study_label = (
        "INCONCLUSIVE" if not schedule_complete or evaluated_count == 0
        else "DESCRIPTIVE_CANDIDATES_PRESENT" if candidate_count
        else "NO_CONTEXT_SIGNAL_IN_SCHEDULE"
    )
    return {
        "schema": SCHEMA_SUMMARY,
        "status": "complete" if schedule_complete else "incomplete",
        "study_label": study_label,
        "protocol_id": PROTOCOL_ID,
        "scheduled_matches": EXPECTED_MATCHES,
        "completed_matches": sum(row.get("status") == "complete" for row in rows),
        "scheduled_paired_seed_blocks": 180,
        "step_tick_calls": calls,
        "step_tick_budget": STEP_BUDGET,
        "independence_counts": {
            "distinct_config_trajectory_pairs": len({
                (row.get("config_sha256"), row.get("trajectory_sha256")) for row in rows
                if row.get("trajectory_sha256") is not None
            }),
            "distinct_paired_trajectory_blocks": len({
                row.get("paired_trajectory_block_sha256") for row in rows
                if row.get("paired_trajectory_block_sha256") is not None
            }),
            "duplicate_scheduled_paired_blocks": (
                max(0, len(block_groups) - len({
                    row.get("paired_trajectory_block_sha256") for row in rows
                    if row.get("paired_trajectory_block_sha256") is not None
                }))
            ),
            "distinct_random_stream_trajectories": len({
                row["random_stream_trajectory_sha256"] for row in rows
                if row.get("random_stream_trajectory_sha256") is not None
            }),
        },
        "map_matchup_score_matrix": map_scores,
        "policy_map_score_matrix": policy_map_scores,
        "policy_family_rankings": policy_family_rankings,
        "pairwise_context_analysis": pair_summaries,
        "anti_degeneracy_review": degeneracy,
        "confidence_intervals": None,
        "p_values": None,
        "interpretation": (
            "Open diagnostic on six fixed maps and five frozen policies. No inference "
            "to a map population and no Gate 0 or Gate 1 pass."
        ),
    }


def _degeneracy_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for map_id in (f"{family}-{terrain}" for family in FAMILIES for terrain in TERRAINS):
        for first, second in _expected_matchups():
            group = [
                row for row in rows
                if row.get("map_id") == map_id and row.get("matchup") == f"{first}-{second}"
            ]
            timeouts = sum(
                isinstance(row.get("terminal"), dict)
                and row["terminal"].get("reason") == "timeout" for row in group
            )
            zero_combat = sum(
                row.get("status") == "complete"
                and isinstance(row.get("terminal"), dict)
                and row["terminal"].get("state_tick", 0) >= 1000
                and row.get("event_counts", {}).get("combat", 0) == 0
                for row in group
            )
            stockpile = 0
            for row in group:
                terminal = row.get("terminal")
                if not isinstance(terminal, dict) or terminal.get("reason") != "timeout":
                    continue
                if all(
                    metrics["queen_hp"] > 0
                    and metrics["soldiers"] >= 1
                    and metrics["food"] >= 300
                    for metrics in row.get("terminal_metrics", {}).values()
                ):
                    stockpile += 1
            repetitive = sum(
                any(metrics.get("repetitive_orders_flag", False)
                    for metrics in row.get("policy_event_metrics", {}).values())
                for row in group
            )
            result.append({
                "map_id": map_id,
                "matchup": f"{first}-{second}",
                "scheduled_denominator": 12,
                "rows_present": len(group),
                "timeout_count": timeouts,
                "timeout_fraction": _fraction(Fraction(timeouts, 12)),
                "timeout_concentration_flag": timeouts >= 3,
                "extended_zero_combat_count": zero_combat,
                "extended_zero_combat_fraction": _fraction(Fraction(zero_combat, 12)),
                "extended_zero_combat_flag": zero_combat >= 3,
                "food_stockpile_live_army_timeout_count": stockpile,
                "food_stockpile_live_army_timeout_fraction": _fraction(Fraction(stockpile, 12)),
                "food_stockpile_live_army_timeout_flag": stockpile >= 3,
                "repetitive_policy_match_count": repetitive,
                "repetitive_policy_match_fraction": _fraction(Fraction(repetitive, 12)),
                "repetitive_policy_order_flag": repetitive >= 3,
            })
    return result


def _summary_bytes(summary: dict[str, Any], rows_sha256: str) -> bytes:
    value = dict(summary)
    value["matches_jsonl_sha256"] = rows_sha256
    return canonical_dumps(_canonical_missing(value)) + b"\n"


def run(out: str | Path = OUT_RELATIVE) -> dict[str, Any]:
    destination = Path(out)
    if not destination.is_absolute():
        destination = _ROOT / destination
    if destination.exists():
        raise Gate1ContextError(f"output directory already exists: {destination}")
    if Path(out) != OUT_RELATIVE and str(out) != str(OUT_RELATIVE):
        command = f"python -m formicmind.gate1_context run --out {out}"
    else:
        command = "python -m formicmind.gate1_context run --out results/gate1_context"
    configs = _build_configs()
    source_commit = _git_head()
    origin_main_sha = _git_value("rev-parse", "origin/main")
    manifest = _manifest(source_commit, origin_main_sha)
    manifest["command"] = command
    rows: list[dict[str, Any]] = []
    calls = [0]
    error = None
    for map_id, first, second, seed, swapped in _schedule():
        try:
            row = _execute_match(map_id, configs[map_id], first, second, seed, swapped, calls)
        except Exception as exc:  # preserve the fixed failing scheduled row and stop
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "match": {
                    "map_id": map_id, "first_policy": first, "second_policy": second,
                    "seed": seed, "first_policy_on_A": not swapped,
                },
            }
            rows.append({
                "schema": SCHEMA_MATCH,
                "status": "technical_error",
                **error["match"],
                "error": {"type": error["type"], "message": error["message"]},
            })
            break
        rows.append(row)
        if row["status"] != "complete":
            error = {
                "type": "censored_incomplete",
                "message": "engine failed to produce its terminal outcome at T_max",
                "match": {
                    "map_id": map_id, "first_policy": first, "second_policy": second,
                    "seed": seed, "first_policy_on_A": not swapped,
                },
            }
            break
    _attach_paired_block_digests(rows)
    summary = _build_summary(rows, calls[0])
    if error is not None:
        summary["error"] = error
    rows_bytes = _write_rows(rows)
    summary_bytes = _summary_bytes(summary, _sha(rows_bytes))
    manifest["status"] = summary["status"]
    manifest["actual_matches"] = len(rows)
    manifest["step_tick_calls"] = calls[0]
    manifest["error"] = "none" if error is None else error
    manifest["artifact_sha256"] = {
        "matches.jsonl": _sha(rows_bytes),
        "summary.json": _sha(summary_bytes),
    }
    destination.mkdir(parents=True)
    (destination / "matches.jsonl").write_bytes(rows_bytes)
    (destination / "summary.json").write_bytes(summary_bytes)
    (destination / "manifest.json").write_bytes(canonical_dumps(_canonical_missing(manifest)) + b"\n")
    return summary


def _assert_manifest(manifest: Any) -> None:
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA_MANIFEST:
        raise Gate1ContextError("manifest schema is invalid")
    if manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("protocol_sha256") != _read_protocol_sha():
        raise Gate1ContextError("protocol provenance differs from the frozen protocol")
    if manifest.get("source_fingerprint_sha256") != _source_fingerprint():
        raise Gate1ContextError("Python source fingerprint changed; strict cross-commit verify is incompatible")
    if manifest.get("versions") != _versions():
        raise Gate1ContextError("version descriptors changed")
    expected_maps = _map_descriptors(_build_configs())
    if manifest.get("maps") != expected_maps:
        raise Gate1ContextError("canonical map configs or config hashes differ")
    expected_policies = [_policy_descriptor(name) for name in POLICIES]
    if manifest.get("policies") != expected_policies:
        raise Gate1ContextError("policy IDs, parameters, or streams differ")
    if manifest.get("study") != _manifest(
        manifest.get("source_commit_sha", ""), manifest.get("origin_main_sha", ""),
    )["study"]:
        raise Gate1ContextError("fixed schedule or methodology differs")
    sha = manifest.get("source_commit_sha")
    if not isinstance(sha, str) or len(sha) != 40:
        raise Gate1ContextError("source commit SHA is invalid")
    try:
        _git_value("merge-base", "--is-ancestor", sha, "HEAD")
    except Gate1ContextError as exc:
        raise Gate1ContextError("recorded source commit is not an ancestor of verification HEAD") from exc


def verify(input_path: str | Path = OUT_RELATIVE) -> dict[str, Any]:
    source = Path(input_path)
    if not source.is_absolute():
        source = _ROOT / source
    required = {"manifest.json", "matches.jsonl", "summary.json"}
    if not source.is_dir() or {path.name for path in source.iterdir()} != required:
        raise Gate1ContextError("input must contain exactly manifest.json, matches.jsonl, and summary.json")
    manifest = _read_json(source / "manifest.json")
    _assert_manifest(manifest)
    rows_bytes = (source / "matches.jsonl").read_bytes()
    summary_bytes = (source / "summary.json").read_bytes()
    if manifest.get("artifact_sha256") != {
        "matches.jsonl": _sha(rows_bytes),
        "summary.json": _sha(summary_bytes),
    }:
        raise Gate1ContextError("saved artifact hash mismatch")
    stored_rows = _read_rows(source / "matches.jsonl")
    if manifest.get("status") != "complete" or len(stored_rows) != EXPECTED_MATCHES:
        raise Gate1ContextError("cannot strictly verify an incomplete/failed matrix")
    if [tuple((
        row.get("map_id"), row.get("matchup"), row.get("seed"),
        row.get("seat_assignment", {}).get("A"), row.get("seat_assignment", {}).get("B"),
    )) for row in stored_rows] != [
        (
            map_id, f"{first}-{second}", seed,
            second if swapped else first, first if swapped else second,
        )
        for map_id, first, second, seed, swapped in _schedule()
    ]:
        raise Gate1ContextError("saved rows are missing, extra, reordered, or outside the fixed schedule")
    configs = _build_configs()
    calls = [0]
    replayed = [
        _execute_match(map_id, configs[map_id], first, second, seed, swapped, calls)
        for map_id, first, second, seed, swapped in _schedule()
    ]
    _attach_paired_block_digests(replayed)
    replayed_bytes = _write_rows(replayed)
    if replayed_bytes != rows_bytes:
        for index, (expected, actual) in enumerate(zip(stored_rows, replayed)):
            if expected != actual:
                raise Gate1ContextError(f"strict replay differs at scheduled row {index}: {expected.get('map_id')}")
        raise Gate1ContextError("strict replay row bytes differ")
    summary = _build_summary(replayed, calls[0])
    expected_summary_bytes = _summary_bytes(summary, _sha(replayed_bytes))
    if expected_summary_bytes != summary_bytes:
        raise Gate1ContextError("recomputed aggregate differs from saved summary")
    return {
        "status": "verified",
        "verified_matches": len(replayed),
        "step_tick_calls": calls[0],
        "source_fingerprint_sha256": _source_fingerprint(),
        "matches_jsonl_sha256": _sha(rows_bytes),
        "summary_sha256": _sha(summary_bytes),
        "study_label": summary["study_label"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m formicmind.gate1_context", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="run the fixed 720-match study")
    runner.add_argument("--out", required=True, help="new, not-yet-existing output directory")
    verifier = commands.add_parser("verify", help="replay the full matrix and strictly verify artifacts")
    verifier.add_argument("--input", required=True, help="saved study output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        value = run(args.out) if args.command == "run" else verify(args.input)
    except (Gate1ContextError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"gate1_context: error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))
    return 0 if value.get("status") in {"complete", "verified"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
