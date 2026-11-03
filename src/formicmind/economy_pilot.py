"""Reproducible FOR-20 economic feasibility pilot on the existing L0 engine.

This module is an experiment harness only. It derives explicit immutable configs,
uses ``reset`` and the ordinary full ``step_tick`` path, and never changes engine
rules or candidate defaults.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from collections import Counter
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any

from .actions import ActionV2
from .config import SimConfig
from .events import Event, HatchEvent, HarvestEvent, LayEvent, StarvationEvent, UpkeepEvent
from .replay.format import EventsHasher
from .reset import reset
from .rules import (
    ARMY_GARRISON_HOME,
    LAY_NONE,
    LAY_SOLDIER,
    LAY_WORKER,
    NODE_NAMES,
    Q_TOTAL,
    SOLDIER,
    WORKER,
    default_config,
    local_to_world,
)
from .serialization import checkpoint_hash, config_hash, config_to_bytes, state_hash
from .tick import ScriptedController, TickMode, step_tick
from .versions import RULES_VERSION

SCHEMA_MANIFEST = "formicmind.economy-pilot-manifest/v1"
SCHEMA_RUNS = "formicmind.economy-pilot-runs/v1"
SCHEMA_SUMMARY = "formicmind.economy-pilot-summary/v1"
EGGS_PER_LAY = (1, 2, 4)
LAY_REGIMES = ("no-lay", "worker-first", "soldier-focused", "alternating")
FORAGE_REGIMES = ("single-local", "split-local")
ENVIRONMENT_PROFILES = ("default", "scarcity")
OPEN_SEEDS = (20261111, 20261112, 20261113)
PILOT_HORIZON = 1000
CHECKPOINT_INTERVAL = 100
PRIMARY_CHECKPOINTS = (100, 300, 1000)
DEFAULT_TOPOLOGY = "default"
LOCAL_BARRIER_TOPOLOGY = "local-barrier"
PROTOCOL_PATH = Path(__file__).resolve().parents[2] / "docs" / "economy_pilot_protocol.md"
LOCAL_NODES = {0: (0, 1, 2), 1: (4, 5, 6)}
RESOURCE_NODES = (1, 2, 3, 4, 5)
COMBAT_EVENT_KINDS = frozenset({"combat", "wound", "death", "queen_damage"})
METRIC_COUNTERS = (
    "harvested_food", "upkeep_paid", "upkeep_missing", "lay_none_ticks", "cooldown_ticks",
    "eligible_lay_attempts", "laid_batches", "food_rejected_batches", "eggs_laid", "egg_spend",
    "hatched_workers", "hatched_soldiers", "starved_workers", "starved_soldiers",
    "starvation_deaths",
)


class PilotError(ValueError):
    """The locked protocol or its outputs could not be validated."""


class IsolationViolation(PilotError):
    """A run left its mirrored local territory or executed combat/queen damage."""

    def __init__(self, run_key: str, tick: int, details: dict[str, Any]):
        self.run_key = run_key
        self.tick = tick
        self.details = details
        super().__init__(f"isolation failed in {run_key} at tick {tick}: {details}")


def protocol_sha256() -> str:
    try:
        payload = PROTOCOL_PATH.read_bytes()
    except OSError as exc:
        raise PilotError(f"cannot read frozen protocol {PROTOCOL_PATH}: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def _git_value(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True, cwd=PROTOCOL_PATH.parents[1]
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PilotError(f"cannot read Git metadata ({' '.join(args)}): {exc}") from exc


def _resource_profile(config: SimConfig, environment: str) -> SimConfig:
    if environment not in ENVIRONMENT_PROFILES:
        raise PilotError(f"unknown environment profile {environment!r}")
    if environment == "default":
        return config
    resources = tuple(
        replace(resource, start_food=resource.start_food // 2, regen=resource.regen // 2)
        for resource in config.balance.resources
    )
    return replace(config, balance=replace(config.balance, resources=resources))


def _topology_profile(config: SimConfig, topology_mode: str) -> SimConfig:
    if topology_mode == DEFAULT_TOPOLOGY:
        return config
    if topology_mode != LOCAL_BARRIER_TOPOLOGY:
        raise PilotError(f"unknown isolation topology {topology_mode!r}")
    allowed_edges = {(0, 1), (0, 2), (4, 6), (5, 6)}
    edges = tuple(edge for edge in config.topology.edges if (edge.a, edge.b) in allowed_edges)
    return replace(config, topology=replace(config.topology, edges=edges))


def build_config(eggs_per_lay: int, environment: str, topology_mode: str = DEFAULT_TOPOLOGY) -> SimConfig:
    if eggs_per_lay not in EGGS_PER_LAY:
        raise PilotError(f"eggs_per_lay must be one of {EGGS_PER_LAY}")
    config = default_config()
    config = replace(config, balance=replace(config.balance, eggs_per_lay=eggs_per_lay))
    config = _resource_profile(config, environment)
    config = _topology_profile(config, topology_mode)
    return config


def matrix_design(topology_mode: str = DEFAULT_TOPOLOGY) -> list[dict[str, Any]]:
    """Return the full, lexicographically declared 144-row design."""
    rows = []
    for eggs, lay, forage, environment, seed in product(
        EGGS_PER_LAY, LAY_REGIMES, FORAGE_REGIMES, ENVIRONMENT_PROFILES, OPEN_SEEDS
    ):
        config = build_config(eggs, environment, topology_mode)
        rows.append({
            "run_key": f"eggs-{eggs}|{lay}|{forage}|{environment}|{seed}",
            "eggs_per_lay": eggs,
            "lay_regime": lay,
            "forage_regime": forage,
            "environment_profile": environment,
            "seed": seed,
            "topology_mode": topology_mode,
            "config_hash": config_hash(config),
        })
    if len(rows) != 144 or len({row["run_key"] for row in rows}) != 144:
        raise PilotError("locked FOR-20 matrix must contain 144 distinct run keys")
    return rows


def lay_order_for(regime: str, tick: int) -> int:
    if regime == "no-lay":
        return LAY_NONE
    if regime == "worker-first":
        return LAY_WORKER
    if regime == "soldier-focused":
        return LAY_SOLDIER
    if regime == "alternating":
        return LAY_WORKER if (tick // 120) % 2 == 0 else LAY_SOLDIER
    raise PilotError(f"unknown lay regime {regime!r}")


def forage_shares_for(regime: str) -> tuple[int, ...]:
    if regime == "single-local":
        return (0, Q_TOTAL, 0, 0, 0, 0)
    if regime == "split-local":
        return (0, 32768, 32767, 0, 0, 0)
    raise PilotError(f"unknown forage regime {regime!r}")


def _action(lay_order: int, shares: tuple[int, ...]) -> ActionV2:
    return ActionV2(lay_order, shares, ARMY_GARRISON_HOME, False)


def _controllers(config: SimConfig, lay_regime: str, forage_regime: str) -> tuple[ScriptedController, ScriptedController]:
    shares = forage_shares_for(forage_regime)
    script = {
        tick: _action(lay_order_for(lay_regime, tick), shares)
        for tick in range(0, config.t_max, config.policy_interval_k)
    }
    return ScriptedController(script), ScriptedController(script)


def _unit_location(unit: Any) -> dict[str, int]:
    if unit.node >= 0:
        return {"node": unit.node}
    return {"from_node": unit.from_node, "to_node": unit.to_node, "arrival_tick": unit.arrival_tick}


def _isolation_violations(state: Any, tick: int, events: tuple[Event, ...]) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []
    node_occupants: dict[int, list[int]] = {}
    edge_occupants: dict[tuple[int, int], list[int]] = {}
    for side, colony in enumerate(state.colonies):
        allowed = frozenset(LOCAL_NODES[side])
        for unit in colony.units:
            if unit.node >= 0:
                if unit.node not in allowed:
                    problems.append({"type": "unit_outside_local_half", "side": side,
                                     "unit_id": unit.unit_id, "location": _unit_location(unit)})
                node_occupants.setdefault(unit.node, []).append(side)
            else:
                endpoints = {unit.from_node, unit.to_node}
                if not endpoints.issubset(allowed):
                    problems.append({"type": "transit_outside_local_half", "side": side,
                                     "unit_id": unit.unit_id, "location": _unit_location(unit)})
                edge = tuple(sorted((unit.from_node, unit.to_node)))
                edge_occupants.setdefault(edge, []).append(side)
        if colony.queen_hp != state.config.balance.queen_hp:
            problems.append({"type": "queen_hp_changed", "side": side, "queen_hp": colony.queen_hp})
    for node, sides in sorted(node_occupants.items()):
        if len(set(sides)) > 1:
            problems.append({"type": "intercolonial_node_contact", "node": node, "sides": sorted(set(sides))})
    for edge, sides in sorted(edge_occupants.items()):
        if len(set(sides)) > 1:
            problems.append({"type": "intercolonial_edge_contact", "edge": list(edge), "sides": sorted(set(sides))})
    forbidden_events = [event.kind for event in events if event.kind in COMBAT_EVENT_KINDS]
    if forbidden_events:
        problems.append({"type": "combat_or_queen_damage_event", "event_kinds": forbidden_events})
    return problems


def _side_signature(state: Any, side: int) -> tuple[Any, ...]:
    colony = state.colonies[side]
    units = []
    for unit in colony.units:
        if unit.node >= 0:
            location = ("node", local_to_world(side, unit.node))
        else:
            location = ("edge", local_to_world(side, unit.from_node), local_to_world(side, unit.to_node), unit.arrival_tick)
        units.append((unit.caste, unit.hp, location, local_to_world(side, unit.target_node)))
    units.sort()
    brood = sorted((item.caste, item.laid_tick, item.hatch_tick) for item in colony.brood)
    local_stocks = tuple(state.node_food[local_to_world(side, category)] for category in (1, 2, 3, 4, 5))
    return (
        colony.food, colony.queen_hp, colony.next_lay_tick, colony.upkeep_remainder10,
        colony.starvation_debt, colony.lay_order, tuple(colony.forage_q), colony.army_choice,
        colony.avoid_center, tuple(units), tuple(brood), local_stocks,
    )


def _new_counters() -> dict[str, int]:
    return {name: 0 for name in METRIC_COUNTERS}


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 6)


def _adult_counts(colony: Any) -> tuple[int, int]:
    workers = sum(unit.caste == WORKER for unit in colony.units)
    soldiers = sum(unit.caste == SOLDIER for unit in colony.units)
    return workers, soldiers


def _brood_counts(colony: Any) -> tuple[int, int]:
    workers = sum(record.caste == WORKER for record in colony.brood)
    soldiers = sum(record.caste == SOLDIER for record in colony.brood)
    return workers, soldiers


def _side_checkpoint(state: Any, side: int, totals: dict[str, int], debt_peak: int,
                     debt_ticks: int, previous: dict[str, Any] | None, mirror_equal: bool) -> dict[str, Any]:
    colony = state.colonies[side]
    workers, soldiers = _adult_counts(colony)
    brood_workers, brood_soldiers = _brood_counts(colony)
    local_sources = {
        f"N{category}": state.node_food[local_to_world(side, category)] for category in (1, 2)
    }
    all_sources = {NODE_NAMES[node]: state.node_food[node] for node in RESOURCE_NODES}
    local_food = sum(local_sources.values())
    point = {
        "tick": state.tick,
        "adult_workers": workers,
        "adult_soldiers": soldiers,
        "brood_workers": brood_workers,
        "brood_soldiers": brood_soldiers,
        "queen_hp": colony.queen_hp,
        "warehouse_food": colony.food,
        "local_source_stocks": local_sources,
        "all_source_stocks": all_sources,
        "accessible_food_total": colony.food + local_food,
        "starvation_debt": colony.starvation_debt,
        "starvation_debt_peak": debt_peak,
        "starvation_debt_ticks": debt_ticks,
        **totals,
        "egg_spend_per_harvested_food": _ratio(totals["egg_spend"], totals["harvested_food"]),
        "egg_spend_per_initial_plus_harvested_food": _ratio(
            totals["egg_spend"], state.config.balance.start_food + totals["harvested_food"]
        ),
        "mirror_state_equal": mirror_equal,
    }
    if previous is None:
        point["window_delta"] = None
    else:
        point["window_delta"] = {
            name: point[name] - previous[name] for name in
            ("adult_workers", "adult_soldiers", "brood_workers", "brood_soldiers", "warehouse_food",
             "harvested_food", "upkeep_paid", "upkeep_missing", "egg_spend", "eggs_laid",
             "hatched_workers", "hatched_soldiers", "starvation_deaths")
        }
    return point


def _plateau_tick(points: list[dict[str, Any]]) -> int | None:
    """First endpoint of three full consecutive 100-tick low-change windows."""
    if not points or points[0]["tick"] != 0:
        return None
    for end_index in range(3, len(points)):
        window_points = points[end_index - 3:end_index + 1]
        if any(right["tick"] - left["tick"] != 100 for left, right in zip(window_points, window_points[1:])):
            continue
        stable = True
        for start, end in zip(window_points, window_points[1:]):
            workers = start["adult_workers"]
            if workers <= 0:
                stable = False
                break
            threshold = max(1, math.ceil(0.05 * workers))
            if abs(end["adult_workers"] - workers) > threshold:
                stable = False
                break
        if stable:
            return window_points[-1]["tick"]
    return None


def _update_event_totals(event: Event, totals: list[dict[str, int]], event_counts: Counter[str]) -> None:
    event_counts[event.kind] += 1
    if isinstance(event, HarvestEvent):
        totals[event.side]["harvested_food"] += event.harvested
    elif isinstance(event, UpkeepEvent):
        totals[event.side]["upkeep_paid"] += event.paid
        totals[event.side]["upkeep_missing"] += event.missing
    elif isinstance(event, LayEvent):
        if event.laid:
            totals[event.side]["laid_batches"] += 1
            totals[event.side]["eggs_laid"] += event.eggs
            totals[event.side]["egg_spend"] += event.total_cost
        else:
            totals[event.side]["food_rejected_batches"] += 1
    elif isinstance(event, HatchEvent):
        name = "hatched_workers" if event.caste == WORKER else "hatched_soldiers"
        totals[event.side][name] += 1
    elif isinstance(event, StarvationEvent):
        name = "starved_workers" if event.caste == WORKER else "starved_soldiers"
        totals[event.side][name] += 1
        totals[event.side]["starvation_deaths"] += 1


def _classify_lay_opportunities(
    tick: int,
    orders: tuple[int, int],
    next_lay: tuple[int, int],
    queen_hp: tuple[int, int],
    events: tuple[Event, ...],
) -> None:
    lays = [event for event in events if isinstance(event, LayEvent)]
    for side in (0, 1):
        order = orders[side]
        if order == LAY_NONE:
            continue  # counted from the tick state below
        if queen_hp[side] <= 0:
            continue
        if tick < next_lay[side]:
            continue  # explicit cooldown, with no expected LayEvent
        side_lays = [event for event in lays if event.side == side]
        terminal_before_lay = any(getattr(event, "kind", "") == "terminal" and event.phase < 7 for event in events)
        if terminal_before_lay:
            continue
        if len(side_lays) != 1:
            raise PilotError(f"eligible lay at tick {tick}, side {side}, emitted {len(side_lays)} LayEvents")


def _run_one(design: dict[str, Any], *, topology_mode: str, horizon: int = PILOT_HORIZON) -> dict[str, Any]:
    config = build_config(design["eggs_per_lay"], design["environment_profile"], topology_mode)
    if config_hash(config) != design["config_hash"]:
        raise PilotError(f"config hash differs from locked matrix row {design['run_key']}")
    controllers = _controllers(config, design["lay_regime"], design["forage_regime"])
    state, run = reset(config, design["seed"])
    counters = [_new_counters(), _new_counters()]
    event_counts: Counter[str] = Counter()
    events_hasher = EventsHasher()
    debt_peak = [colony.starvation_debt for colony in state.colonies]
    debt_ticks = [0, 0]
    points: list[list[dict[str, Any]]] = [[], []]
    previous_points: list[dict[str, Any] | None] = [None, None]
    mirror_equal = True
    initial_workers = tuple(_adult_counts(colony)[0] for colony in state.colonies)
    initial_state_hash = state_hash(state)
    executed = 0

    while state.tick < horizon and run.outcome is None:
        tick = state.tick
        pre_lay_order = tuple(colony.lay_order for colony in state.colonies)
        pre_next_lay = tuple(colony.next_lay_tick for colony in state.colonies)
        pre_queen = tuple(colony.queen_hp for colony in state.colonies)
        scheduled_orders = tuple(
            lay_order_for(design["lay_regime"], tick) if tick % config.policy_interval_k == 0
            else pre_lay_order[side]
            for side in (0, 1)
        )
        result = step_tick(state, run, controllers, check_invariants=True)
        executed += 1
        if result.mode is not TickMode.FULL or not result.physics_complete or result.missing_phases:
            raise PilotError(f"incomplete physics on {design['run_key']} tick {tick}")
        _classify_lay_opportunities(tick, scheduled_orders, pre_next_lay, pre_queen, result.events)
        for side in (0, 1):
            if scheduled_orders[side] == LAY_NONE:
                counters[side]["lay_none_ticks"] += 1
            elif tick < pre_next_lay[side]:
                counters[side]["cooldown_ticks"] += 1
            elif pre_queen[side] > 0 and not any(
                getattr(event, "kind", "") == "terminal" and event.phase < 7 for event in result.events
            ):
                counters[side]["eligible_lay_attempts"] += 1
        for event in result.events:
            events_hasher.add(event.to_canonical())
            _update_event_totals(event, counters, event_counts)

        violations = _isolation_violations(state, tick, result.events)
        if violations:
            raise IsolationViolation(design["run_key"], tick, violations[0])
        mirror_equal = mirror_equal and (_side_signature(state, 0) == _side_signature(state, 1))
        if not mirror_equal:
            raise PilotError(f"mirrored colonies diverged in {design['run_key']} at tick {tick}")
        for side in (0, 1):
            debt_peak[side] = max(debt_peak[side], state.colonies[side].starvation_debt)
            debt_ticks[side] += state.colonies[side].starvation_debt

        if state.tick != tick and state.tick % CHECKPOINT_INTERVAL == 0 and state.tick <= horizon:
            for side in (0, 1):
                point = _side_checkpoint(state, side, counters[side], debt_peak[side], debt_ticks[side],
                                         previous_points[side], mirror_equal)
                point["state_hash"] = state_hash(state)
                point["checkpoint_hash"] = checkpoint_hash(state, run)
                points[side].append(point)
                previous_points[side] = point

    observed_by_side = [
        {point["tick"]: point for point in side_points} for side_points in points
    ]
    checkpoints = []
    for target in range(CHECKPOINT_INTERVAL, horizon + 1, CHECKPOINT_INTERVAL):
        for side in (0, 1):
            point = observed_by_side[side].get(target)
            if point is None:
                checkpoint = {"tick": target, "side": side, "status": "missing",
                              "missing_reason": "episode_terminated_before_checkpoint" if run.outcome else "pilot_horizon"}
            else:
                checkpoint = {"side": side, "status": "observed", **point}
            checkpoints.append(checkpoint)

    final = {
        "state_tick": state.tick,
        "queen_hp": [colony.queen_hp for colony in state.colonies],
        "colony_food": [colony.food for colony in state.colonies],
        "adult_workers": [_adult_counts(colony)[0] for colony in state.colonies],
        "adult_soldiers": [_adult_counts(colony)[1] for colony in state.colonies],
        "brood_workers": [_brood_counts(colony)[0] for colony in state.colonies],
        "brood_soldiers": [_brood_counts(colony)[1] for colony in state.colonies],
        "starvation_debt": [colony.starvation_debt for colony in state.colonies],
        "state_hash": state_hash(state),
        "checkpoint_hash": checkpoint_hash(state, run),
    }
    outcome = run.outcome
    result = {
        **design,
        "topology_mode": topology_mode,
        "ticks_executed": executed,
        "pilot_horizon": horizon,
        "pilot_cutoff": outcome is None and state.tick == horizon,
        "terminal_reason": None if outcome is None else outcome.reason.value,
        "terminal_tick": None if outcome is None else outcome.terminal_tick,
        "terminal_tick_completed": None if outcome is None else outcome.tick_completed,
        "initial_workers": list(initial_workers),
        "initial_state_hash": initial_state_hash,
        "event_hash": events_hasher.hexdigest(),
        "event_count": events_hasher.count,
        "event_counts": {name: event_counts[name] for name in sorted(event_counts)},
        "mirror_state_equal_every_tick": mirror_equal,
        "isolation_passed": True,
        "isolation_check_count": executed,
        "data_quality_status": "valid",
        "checkpoints": checkpoints,
        "final_state": final,
        "side_totals": counters,
        "debt_peak": debt_peak,
        "debt_ticks": debt_ticks,
    }
    result["trajectory_signature"] = _trajectory_signature(result)
    result["side_summaries"] = _side_summaries(result)
    result["red_flags"] = _red_flags(result)
    result["viability"] = [_viability_label(result, side) for side in (0, 1)]
    return result


def _trajectory_signature(run: dict[str, Any]) -> str:
    state_hashes = [
        checkpoint["state_hash"] for checkpoint in run["checkpoints"]
        if checkpoint["status"] == "observed" and checkpoint["side"] == 0
    ]
    payload = (run["event_hash"] + "|" + "|".join(state_hashes)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _side_summaries(run: dict[str, Any]) -> list[dict[str, Any]]:
    config = build_config(run["eggs_per_lay"], run["environment_profile"], run["topology_mode"])
    if run["config_hash"] != config_hash(config):
        raise PilotError("config hash mismatch while making a side summary")
    summaries = []
    for side in (0, 1):
        points = [point for point in run["checkpoints"] if point["side"] == side and point["status"] == "observed"]
        history = [{"tick": 0, "adult_workers": run["initial_workers"][side]}]
        history.extend({"tick": point["tick"], "adult_workers": point["adult_workers"]} for point in points)
        by_tick = {point["tick"]: point for point in points}
        cp1000 = by_tick.get(1000)
        window_900 = None if cp1000 is None else cp1000["window_delta"]
        summary = {
            "side": side,
            "first_worker_plateau_tick": _plateau_tick(history),
            "reproductive_loop": _reproductive_loop(run, side),
            "red_flags": [],
        }
        if cp1000 is not None:
            summary["survives_horizon"] = cp1000["queen_hp"] > 0 and cp1000["adult_workers"] >= 1
            summary["worker_reduction_at_1000"] = run["initial_workers"][side] - cp1000["adult_workers"]
            summary["worker_count_at_1000"] = cp1000["adult_workers"]
            summary["starvation_debt_at_1000"] = cp1000["starvation_debt"]
            summary["accessible_food_at_1000"] = cp1000["accessible_food_total"]
            summary["window_900_1000"] = window_900
        else:
            summary["survives_horizon"] = False
            summary["missing_horizon_reason"] = "episode_terminated_before_tick_1000" if run["terminal_reason"] else "not_run_to_horizon"
        summaries.append(summary)
    return summaries


def _reproductive_loop(run: dict[str, Any], side: int) -> dict[str, bool]:
    totals = run["side_totals"][side]
    laid = totals["laid_batches"] > 0
    worker = laid and totals["hatched_workers"] > 0
    soldier = laid and totals["hatched_soldiers"] > 0
    if run["lay_regime"] == "worker-first":
        return {"worker": worker, "soldier": False, "any": worker}
    if run["lay_regime"] == "soldier-focused":
        return {"worker": False, "soldier": soldier, "any": soldier}
    if run["lay_regime"] == "alternating":
        return {"worker": worker, "soldier": soldier, "any": worker and soldier}
    return {"worker": False, "soldier": False, "any": False}


def _red_flags(run: dict[str, Any]) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for side in (0, 1):
        totals = run["side_totals"][side]
        attempts = totals["eligible_lay_attempts"]
        if attempts >= 3 and totals["food_rejected_batches"] / attempts >= 0.25:
            flags.append({"side": side, "code": "food_rejected_lays", "eligible_attempts": attempts,
                          "rejected": totals["food_rejected_batches"]})
        cp1000 = next((cp for cp in run["checkpoints"] if cp["side"] == side and cp["tick"] == 1000
                       and cp["status"] == "observed"), None)
        if cp1000 is None:
            continue
        window = cp1000["window_delta"] or {}
        if window.get("starvation_deaths", 0) > 0 or cp1000["starvation_debt"] > 0:
            flags.append({"side": side, "code": "late_starvation", "deaths_900_999": window.get("starvation_deaths", 0),
                          "debt_at_1000": cp1000["starvation_debt"]})
        if cp1000["adult_workers"] <= run["initial_workers"][side] // 2:
            flags.append({"side": side, "code": "adult_worker_reduction_50pct",
                          "workers_initial": run["initial_workers"][side], "workers_at_1000": cp1000["adult_workers"]})
        no_recent_harvest = window.get("harvested_food", 0) == 0
        if cp1000["accessible_food_total"] == 0 and no_recent_harvest:
            flags.append({"side": side, "code": "empty_food_and_no_recent_harvest",
                          "accessible_food_at_1000": 0, "harvested_ticks_900_999": 0})
    return flags


def _viability_label(run: dict[str, Any], side: int) -> dict[str, Any]:
    side_summary = run["side_summaries"][side]
    if run["lay_regime"] == "no-lay":
        return {"side": side, "screen": "survival_only", "survives_horizon": side_summary["survives_horizon"]}
    cp1000 = next((cp for cp in run["checkpoints"] if cp["side"] == side and cp["tick"] == 1000
                   and cp["status"] == "observed"), None)
    if cp1000 is None:
        return {"side": side, "engineering_viability_candidate": False,
                "reason": "episode_terminated_before_tick_1000" if run["terminal_reason"] else "horizon_missing"}
    window = cp1000["window_delta"] or {}
    loops = side_summary["reproductive_loop"]
    loop_ok = loops["any"]
    if run["lay_regime"] == "alternating":
        loop_ok = loops["worker"] and loops["soldier"]
    checks = {
        "survives_horizon": side_summary["survives_horizon"],
        "adult_worker_present": cp1000["adult_workers"] >= 1,
        "zero_starvation_debt": cp1000["starvation_debt"] == 0,
        "no_starvation_deaths_900_999": window.get("starvation_deaths", 0) == 0,
        "positive_accessible_food": cp1000["accessible_food_total"] > 0,
        "reproductive_loop_complete": loop_ok,
    }
    return {"side": side, "engineering_viability_candidate": all(checks.values()), "checks": checks}


def _matrix_configs(topology_mode: str) -> list[dict[str, Any]]:
    configs = []
    seen: set[str] = set()
    for eggs, environment in product(EGGS_PER_LAY, ENVIRONMENT_PROFILES):
        config = build_config(eggs, environment, topology_mode)
        digest = config_hash(config)
        if digest not in seen:
            seen.add(digest)
            configs.append({"eggs_per_lay": eggs, "environment_profile": environment,
                            "config_hash": digest,
                            "config": json.loads(config_to_bytes(config).decode("utf-8"))})
    return configs


def _run_matrix(topology_mode: str) -> list[dict[str, Any]]:
    rows = []
    for design in matrix_design(topology_mode):
        rows.append(_run_one(design, topology_mode=topology_mode))
    if len(rows) != 144:
        raise PilotError(f"incomplete matrix: expected 144 runs, got {len(rows)}")
    return rows


def _condition_key(run: dict[str, Any]) -> tuple[Any, ...]:
    return (run["eggs_per_lay"], run["lay_regime"], run["forage_regime"], run["environment_profile"])


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(_condition_key(run), []).append(run)
    conditions = []
    for key in sorted(grouped):
        group = grouped[key]
        trajectories: dict[str, list[dict[str, Any]]] = {}
        for run in group:
            trajectories.setdefault(run["trajectory_signature"], []).append(run)
        unique = []
        for signature, trajectory_runs in sorted(trajectories.items()):
            unique.append({
                "trajectory_signature": signature,
                "seeds": sorted(run["seed"] for run in trajectory_runs),
                "duplicate_seed_rows": len(trajectory_runs) - 1,
                "representative_run_key": trajectory_runs[0]["run_key"],
                "side_summaries": trajectory_runs[0]["side_summaries"],
                "viability": trajectory_runs[0]["viability"],
                "red_flags": trajectory_runs[0]["red_flags"],
                "checkpoint_metrics": [
                    {"tick": cp["tick"], "side": cp["side"], "status": cp["status"],
                     **({"adult_workers": cp["adult_workers"], "adult_soldiers": cp["adult_soldiers"],
                         "brood_workers": cp["brood_workers"], "brood_soldiers": cp["brood_soldiers"],
                         "warehouse_food": cp["warehouse_food"], "local_source_stocks": cp["local_source_stocks"],
                         "harvested_food": cp["harvested_food"], "upkeep_paid": cp["upkeep_paid"],
                         "upkeep_missing": cp["upkeep_missing"], "egg_spend": cp["egg_spend"],
                         "eggs_laid": cp["eggs_laid"], "laid_batches": cp["laid_batches"],
                         "food_rejected_batches": cp["food_rejected_batches"],
                         "starvation_deaths": cp["starvation_deaths"],
                         "starvation_debt": cp["starvation_debt"], "starvation_debt_peak": cp["starvation_debt_peak"],
                         "starvation_debt_ticks": cp["starvation_debt_ticks"], "window_delta": cp["window_delta"]}
                        if cp["status"] == "observed" else {"missing_reason": cp["missing_reason"]})}
                    for cp in trajectory_runs[0]["checkpoints"]
                ],
            })
        conditions.append({
            "eggs_per_lay": key[0], "lay_regime": key[1], "forage_regime": key[2],
            "environment_profile": key[3], "scheduled_seed_runs": len(group),
            "unique_trajectory_count": len(unique), "side_mirror_equal": all(
                run["mirror_state_equal_every_tick"] for run in group
            ), "unique_trajectories": unique,
        })
    if len(conditions) != 48:
        raise PilotError(f"summary must contain 48 conditions, got {len(conditions)}")
    red_counts = Counter(flag["code"] for run in runs for flag in run["red_flags"])
    return {
        "schema": SCHEMA_SUMMARY,
        "run_count": len(runs),
        "condition_count": len(conditions),
        "seed_rows": len(OPEN_SEEDS),
        "unique_trajectory_count": sum(c["unique_trajectory_count"] for c in conditions),
        "scheduled_duplicate_rows": sum(
            max(0, c["scheduled_seed_runs"] - c["unique_trajectory_count"]) for c in conditions
        ),
        "red_flag_counts": {name: red_counts[name] for name in sorted(red_counts)},
        "conditions": conditions,
    }


def _manifest(topology_mode: str, *, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    protocol_commit = _git_value("log", "-1", "--format=%H", "--", str(PROTOCOL_PATH.relative_to(PROTOCOL_PATH.parents[1])))
    return {
        "schema": SCHEMA_MANIFEST,
        "protocol_path": "docs/economy_pilot_protocol.md",
        "protocol_commit": protocol_commit,
        "protocol_sha256": protocol_sha256(),
        "source_git_sha": _git_value("rev-parse", "HEAD"),
        "command": ["python", "-m", "formicmind.economy_pilot", "run", "--out", "results/economy_pilot"],
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "rules_version": RULES_VERSION,
        "topology_mode": topology_mode,
        "isolation_fallback": fallback,
        "eggs_per_lay": list(EGGS_PER_LAY),
        "lay_regimes": list(LAY_REGIMES),
        "forage_regimes": list(FORAGE_REGIMES),
        "environment_profiles": list(ENVIRONMENT_PROFILES),
        "open_seeds": list(OPEN_SEEDS),
        "pilot_horizon": PILOT_HORIZON,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "configs": _matrix_configs(topology_mode),
    }


def _canonical_json(value: Any) -> bytes:
    try:
        return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PilotError(f"result is not strict JSON: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    payload = _canonical_json(value)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def run_pilot(out: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if out.exists() and any(out.iterdir()):
        raise PilotError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    fallback = None
    topology_mode = DEFAULT_TOPOLOGY
    try:
        runs = _run_matrix(topology_mode)
    except IsolationViolation as exc:
        # No partial default-topology output or outcome summaries are written.
        fallback = {"failed_run_key": exc.run_key, "tick": exc.tick,
                    "violation": exc.details, "selected_topology": LOCAL_BARRIER_TOPOLOGY}
        topology_mode = LOCAL_BARRIER_TOPOLOGY
        runs = _run_matrix(topology_mode)

    # The locked confirmation reruns all rows and compares the complete event/state signatures.
    confirmation = _run_matrix(topology_mode)
    first = [run["run_key"] for run in runs]
    second = [run["run_key"] for run in confirmation]
    if first != second or runs != confirmation:
        raise PilotError("matrix replay did not reproduce exact per-run results and hashes")

    manifest = _manifest(topology_mode, fallback=fallback)
    raw = {"schema": SCHEMA_RUNS, "runs": runs}
    summary = aggregate_runs(runs)
    _write_json(out / "manifest.json", manifest)
    _write_json(out / "runs.json", raw)
    _write_json(out / "summary.json", summary)
    return manifest, summary


def verify_pilot(input_dir: Path) -> dict[str, Any]:
    try:
        manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
        raw = json.loads((input_dir / "runs.json").read_text(encoding="utf-8"))
        saved_summary = json.loads((input_dir / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotError(f"cannot read pilot artifacts from {input_dir}: {exc}") from exc
    if manifest.get("schema") != SCHEMA_MANIFEST or raw.get("schema") != SCHEMA_RUNS:
        raise PilotError("unsupported pilot artifact schema")
    if manifest.get("protocol_sha256") != protocol_sha256():
        raise PilotError("the current protocol digest differs from the one in the manifest")
    if manifest.get("topology_mode") not in (DEFAULT_TOPOLOGY, LOCAL_BARRIER_TOPOLOGY):
        raise PilotError("unknown topology mode in manifest")
    if not isinstance(raw.get("runs"), list) or len(raw["runs"]) != 144:
        raise PilotError("runs.json does not contain the complete 144-row matrix")
    recomputed_summary = aggregate_runs(raw["runs"])
    if recomputed_summary != saved_summary:
        raise PilotError("summary.json does not match deterministic aggregation of runs.json")
    replay = _run_matrix(manifest["topology_mode"])
    if replay != raw["runs"]:
        raise PilotError("replayed matrix differs from runs.json")
    return {"verified": True, "run_count": len(replay), "condition_count": len(recomputed_summary["conditions"]),
            "unique_trajectory_count": recomputed_summary["unique_trajectory_count"],
            "topology_mode": manifest["topology_mode"]}


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Run or verify the preregistered FOR-20 economy pilot.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="execute the locked 144-run matrix and deterministic confirmation")
    run_parser.add_argument("--out", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify", help="replay the matrix and verify saved artifacts")
    verify_parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "run":
            _, summary = run_pilot(args.out)
            print(json.dumps({"runs": summary["run_count"], "conditions": summary["condition_count"],
                              "unique_trajectories": summary["unique_trajectory_count"],
                              "red_flag_counts": summary["red_flag_counts"]}, sort_keys=True))
        else:
            print(json.dumps(verify_pilot(args.input), sort_keys=True))
    except PilotError as exc:
        parser.exit(2, f"economy_pilot: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
