"""Headless E1-b runs, time-series summaries, map-distance diagnostics and calibration sweeps."""

from __future__ import annotations

import statistics
import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable

from .codec import to_canonical
from .config import ConfigE1, default_config
from .engine import RouteTable, TickRecord, step_world_tick
from .events import EventE1, EventLog, format_event
from .fixtures import fixture_controller, normalize_fixture
from .initial import build_world
from .model import MapE1, RunStateE1, WorldStateE1
from .serialization import checkpoint_hash, state_hash


@dataclass(slots=True)
class SimulationResult:
    report: dict
    state: WorldStateE1
    run: RunStateE1
    events: list[EventE1] | None = None


def _interaction_summary(log: EventLog, state: WorldStateE1) -> dict:
    """What the colonies did to each other, from the event log only (no claims about war that the events do not show)."""
    first_contact: dict[tuple[int, int], int] = {}
    attacks = []
    queen_deaths, extinctions, collapses = [], [], []
    combat_ticks: set[int] = set()
    kills = {"workers": 0, "soldiers": 0}
    killed_by_colony: dict[int, dict[str, int]] = {}
    for item in log.events:
        if item.kind == "contact":
            first_contact.setdefault((min(item.colony, item.other), max(item.colony, item.other)), item.tick)
        elif item.kind == "dispatch" and item.a == 3:
            attacks.append({"tick": item.tick, "attacker": item.colony, "target": item.other, "soldiers": item.b})
        elif item.kind == "combat":
            combat_ticks.add(item.tick)
        elif item.kind == "death":
            caste = "soldiers" if item.a else "workers"
            kills[caste] += item.b
            killed_by_colony.setdefault(item.colony, {"workers": 0, "soldiers": 0})[caste] += item.b
        elif item.kind == "queen_dead":
            queen_deaths.append({"colony": item.colony, "tick": item.tick, "killer": item.other})
        elif item.kind == "collapse":
            collapses.append({"colony": item.colony, "tick": item.tick})
        elif item.kind == "extinct":
            extinctions.append({"colony": item.colony, "tick": item.tick})
    counts = log.counts()
    return {
        "events_hash": log.head,
        "event_count": len(log.events),
        "events_by_kind": counts,
        "colonies_explored_nodes": [sum(known.visited for known in colony.memory.nodes) for colony in state.colonies],
        "first_contact_tick": min(first_contact.values()) if first_contact else None,
        "colony_pairs_in_contact": len(first_contact),
        "first_contact_by_pair": [[a, b, tick] for (a, b), tick in sorted(first_contact.items())],
        "nests_found": counts["nest_found"],
        "attacks_dispatched": attacks,
        "combat_ticks": len(combat_ticks),
        "first_combat_tick": min(combat_ticks) if combat_ticks else None,
        "combat_deaths": kills,
        "combat_deaths_by_colony": {str(k): v for k, v in sorted(killed_by_colony.items())},
        "queen_deaths": queen_deaths,
        "collapses": collapses,
        "extinctions": extinctions,
        "colony_outcomes": [{"colony": i, "status": colony.status, "reason": state.outcomes[i].reason,
                             "terminal_tick": state.outcomes[i].terminal_tick}
                            for i, colony in enumerate(state.colonies)],
    }


def _counts(state: WorldStateE1) -> list[dict]:
    rows = []
    for colony in state.colonies:
        workers = sum(c.count for c in state.cohorts if c.colony_id == colony.colony_id and c.caste == "worker")
        soldiers = sum(c.count for c in state.cohorts if c.colony_id == colony.colony_id and c.caste == "soldier")
        rows.append({
            "colony_id": colony.colony_id,
            "status": colony.status,
            "workers": workers,
            "soldiers": soldiers,
            "adults": workers + soldiers,
            "maintenance_we": workers + 2 * soldiers,
            "brood": len(colony.brood),
            "food": colony.food,
            "starvation_debt": colony.starvation_debt,
        })
    return rows


def _snapshot(state: WorldStateE1) -> dict:
    rows = _counts(state)
    workers = sum(row["workers"] for row in rows)
    soldiers = sum(row["soldiers"] for row in rows)
    return {
        "tick": state.tick,
        "workers": workers,
        "soldiers": soldiers,
        "adults": workers + soldiers,
        "maintenance_we": workers + 2 * soldiers,
        "food_in_warehouses": sum(colony.food for colony in state.colonies),
        "food_on_patches": sum(patch.stock for patch in state.patches),
        "food_in_piles": sum(pile.stock for pile in state.piles),
        "food_in_transit": sum(cohort.load for cohort in state.cohorts),
        "colonies": rows,
    }


def map_distance_diagnostics(world_map: MapE1, config: ConfigE1, routes: RouteTable) -> dict:
    """Measure actual patch distances beside the former liquid-model assumptions."""
    expected_ranks = {"near": 1, "border": 2, "far": 3}
    calculator_ticks = {"near": [3], "border": [7, 7], "far": [10, 10, 10]}
    zones: dict[str, list[list[int]]] = {"near": [], "border": [], "far": []}
    economic_supply_by_colony_x1000 = [0] * config.n_colonies
    for node in world_map.nodes:
        if node.patch:
            distances = [routes.distance(nest, node.node_id) for nest in world_map.nests]
            zones[node.patch[0].zone].append(sorted(distances))
            accessible = [colony_id for colony_id, distance in enumerate(distances)
                          if distance <= config.map_params.economic_radius]
            if accessible:
                contribution = node.patch[0].regen * 1000 // len(accessible)
                for colony_id in accessible:
                    economic_supply_by_colony_x1000[colony_id] += contribution
    output = {}
    for zone, rows in zones.items():
        rank_rows = []
        for rank in range(expected_ranks[zone]):
            distances = [row[rank] for row in rows]
            reference = calculator_ticks[zone][rank]
            rank_rows.append({
                "rank": rank + 1,
                "mean_ticks_x100": (sum(distances) * 100 // len(distances)) if distances else 0,
                "min_ticks": min(distances) if distances else 0,
                "max_ticks": max(distances) if distances else 0,
                "calculator_ticks": reference,
                "mean_liquid_food_per_worker_tick_x1000": (
                    sum(16_000 // (2 * distance + 8) for distance in distances) // len(distances)
                    if distances else 0
                ),
                "calculator_liquid_food_per_worker_tick_x1000": 16_000 // (2 * reference + 8),
            })
        contest_counts = [sum(distance <= config.map_params.contest_radius for distance in row) for row in rows]
        economic_counts = [sum(distance <= config.map_params.economic_radius for distance in row) for row in rows]
        output[zone] = {
            "patches": len(rows),
            "calculator_distance_assumptions_ticks": calculator_ticks[zone],
            "actual_rank_distances": rank_rows,
            "mean_colonies_within_contest_radius_x100": (
                sum(contest_counts) * 100 // len(contest_counts) if contest_counts else 0
            ),
            "mean_colonies_within_economic_radius_x100": (
                sum(economic_counts) * 100 // len(economic_counts) if economic_counts else 0
            ),
            "economic_radius_ticks": config.map_params.economic_radius,
            "contest_radius_ticks": config.map_params.contest_radius,
        }
    total_regen_x1000 = sum(node.patch[0].regen * 1000 for node in world_map.nodes if node.patch)
    output["economic_access"] = {
        "radius_ticks": config.map_params.economic_radius,
        "mean_fair_share_food_per_colony_x1000": (
            sum(economic_supply_by_colony_x1000) // config.n_colonies
        ),
        "min_colony_fair_share_food_x1000": min(economic_supply_by_colony_x1000),
        "max_colony_fair_share_food_x1000": max(economic_supply_by_colony_x1000),
        "calculator_assumption_food_per_colony_x1000": total_regen_x1000 // config.n_colonies,
    }
    return output


def run_simulation(config: ConfigE1, seed: int, fixture: str = "peaceful_fair", *, ticks: int | None = None,
                   sample_interval: int = 100, include_map: bool = True,
                   collect_timeline: bool = True, roles: dict[int, str] | None = None,
                   include_events: bool = False) -> SimulationResult:
    """Run a real generated graph with a diagnostic fixture; no food is teleported to a nest."""
    fixture = normalize_fixture(fixture)
    duration = config.horizon if ticks is None else ticks
    if type(duration) is not int or duration < 0 or duration > config.horizon:
        raise ValueError("ticks must be an integer between zero and the configured horizon")
    if sample_interval < 1:
        raise ValueError("sample_interval must be positive")
    world = build_world(config, seed)
    routes = RouteTable(world.world_map)
    initial_food = (sum(patch.stock for patch in world.state.patches)
                    + sum(colony.food for colony in world.state.colonies)
                    + sum(cohort.load for cohort in world.state.cohorts))
    controller = fixture_controller(fixture, roles)
    log = EventLog()
    start = time.perf_counter()
    records = {
        "harvested": 0, "delivered": 0, "regenerated": 0, "upkeep_due": 0, "upkeep_paid": 0,
        "upkeep_missing": 0, "egg_spend": 0, "hatched_workers": 0, "hatched_soldiers": 0,
        "laid_workers": 0, "laid_soldiers": 0, "old_age_workers": 0, "old_age_soldiers": 0,
        "starvation_workers": 0, "starvation_soldiers": 0,
        "combat_losses_workers": 0, "combat_losses_soldiers": 0,
    }
    harvested_by_zone = {zone: 0 for zone in ("near", "border", "far", "pile")}
    delivered_by_colony = [0] * config.n_colonies
    first: dict[str, int | None] = {
        "contention_tick": None,
        "patch_depletion_tick": None,
        "upkeep_shortfall_tick": None,
        "starvation_death_tick": None,
        "first_delivery_tick": None,
        "first_lay_tick": None,
    }
    shared_patches = []
    for node in world.world_map.nodes:
        if node.patch and sum(routes.distance(nest, node.node_id) <= config.map_params.contest_radius
                              for nest in world.world_map.nests) > 1:
            shared_patches.append(node.node_id)
    patch_specs = {node.node_id: node.patch[0] for node in world.world_map.nodes if node.patch}
    shared_capacity = sum(patch_specs[node_id].capacity for node_id in shared_patches)
    shared_stock_sum = 0
    shared_sample_count = 0
    population_sum = 0
    population_peak = 0
    population_min = sum(cohort.count for cohort in world.state.cohorts)
    timeline = [_snapshot(world.state)]
    checkpoint_hashes = {"tick_0": checkpoint_hash(world.state, world.run)}
    midpoint = duration // 2
    if midpoint == 0:
        checkpoint_hashes["midpoint"] = checkpoint_hash(world.state, world.run)

    for _ in range(duration):
        tick_record: TickRecord = step_world_tick(
            world.state, world.run, config, world.world_map, controller, routes=routes,
            validate_input=world.state.tick == 0,
        )
        for key in records:
            records[key] += getattr(tick_record, key)
        log.extend(tick_record.events)
        for zone, amount in tick_record.harvested_by_zone.items():
            harvested_by_zone[zone] += amount
        for colony_id, amount in tick_record.delivered_by_colony.items():
            delivered_by_colony[colony_id] += amount
        if first["contention_tick"] is None and tick_record.contention_nodes:
            first["contention_tick"] = tick_record.tick
        if first["patch_depletion_tick"] is None and tick_record.depleted_nodes:
            first["patch_depletion_tick"] = tick_record.tick
        if first["upkeep_shortfall_tick"] is None and tick_record.upkeep_missing:
            first["upkeep_shortfall_tick"] = tick_record.tick
        if first["starvation_death_tick"] is None and (
            tick_record.starvation_workers or tick_record.starvation_soldiers
        ):
            first["starvation_death_tick"] = tick_record.tick
        if first["first_delivery_tick"] is None and tick_record.delivered:
            first["first_delivery_tick"] = tick_record.tick
        if first["first_lay_tick"] is None and (tick_record.laid_workers or tick_record.laid_soldiers):
            first["first_lay_tick"] = tick_record.tick
        population = sum(cohort.count for cohort in world.state.cohorts)
        population_sum += population
        population_peak = max(population_peak, population)
        population_min = min(population_min, population)
        if shared_capacity:
            stock_by_node = {patch.node_id: patch.stock for patch in world.state.patches}
            shared_stock_sum += sum(stock_by_node[node_id] for node_id in shared_patches)
            shared_sample_count += 1
        if collect_timeline and (world.state.tick % sample_interval == 0 or world.state.tick == duration
                                 or world.state.world_outcome.finished):
            timeline.append(_snapshot(world.state))
        if world.state.tick == midpoint:
            checkpoint_hashes["midpoint"] = checkpoint_hash(world.state, world.run)
        if world.state.tick == config.horizon:
            checkpoint_hashes["horizon"] = checkpoint_hash(world.state, world.run)
        if world.state.world_outcome.finished:
            break
    elapsed = time.perf_counter() - start
    final = _snapshot(world.state)
    ending_food = (final["food_in_warehouses"] + final["food_on_patches"] + final["food_in_piles"]
                   + final["food_in_transit"])
    expected_food = initial_food + records["regenerated"] - records["upkeep_paid"] - records["egg_spend"]
    if ending_food != expected_food:
        raise AssertionError(f"food conservation failed: {ending_food} != {expected_food}")
    report = {
        "schema": "formicmind.eco.simulation/1",
        "config": to_canonical(config),
        "map_seed": str(seed),
        "map_hash": world.world_map.map_hash,
        "fixture": fixture,
        "roles": {str(colony): role for colony, role in sorted(controller.roles.items())},
        "ticks_requested": duration,
        "ticks_completed": world.state.tick,
        "outcome": {"finished": world.state.world_outcome.finished,
                    "reason": world.state.world_outcome.reason,
                    "terminal_tick": world.state.world_outcome.terminal_tick},
        "map_distance_diagnostics": map_distance_diagnostics(world.world_map, config, routes),
        "first_signals": first,
        "interaction": _interaction_summary(log, world.state),
        "totals": {
            **records,
            "harvested_by_zone": harvested_by_zone,
            "delivered_by_colony": delivered_by_colony,
            "mean_shared_patch_fill_permille": (
                shared_stock_sum * 1000 // (shared_capacity * shared_sample_count)
                if shared_capacity and shared_sample_count else None
            ),
            "mean_adults": population_sum / max(1, world.state.tick),
            "min_adults": population_min,
            "peak_adults": population_peak,
        },
        "final": final,
        "food_ledger": {"start": initial_food, "end": ending_food,
                        "regenerated": records["regenerated"], "upkeep_paid": records["upkeep_paid"],
                        "egg_spend": records["egg_spend"], "difference": ending_food - expected_food},
        "hashes": {"state": state_hash(world.state), "checkpoint": checkpoint_hash(world.state, world.run),
                   **checkpoint_hashes},
        "performance": {"wall_seconds": round(elapsed, 6),
                        "ticks_per_second": round(world.state.tick / elapsed, 1) if elapsed else 0.0},
    }
    if collect_timeline:
        report["time_series"] = timeline
    if include_map:
        report["map"] = to_canonical(world.world_map)
    if include_events:
        report["event_timeline"] = [format_event(item) for item in log.events]
    return SimulationResult(report, world.state, world.run, log.events)


def _stats(values: list[int | float | None]) -> dict:
    present = [float(value) for value in values if value is not None]
    return {
        "n": len(present),
        "never": len(values) - len(present),
        "median": round(statistics.median(present), 2) if present else None,
        "mean": round(sum(present) / len(present), 2) if present else None,
        "min": min(present) if present else None,
        "max": max(present) if present else None,
    }


def _distance_summary(run_rows: list[dict]) -> dict:
    output: dict = {"zones": {}}
    for zone in ("near", "border", "far"):
        diagnostics = [row["map_distance_diagnostics"][zone] for row in run_rows]
        ranks = []
        for rank in range(1, len(diagnostics[0]["actual_rank_distances"]) + 1):
            rows = [item["actual_rank_distances"][rank - 1] for item in diagnostics]
            ranks.append({
                "rank": rank,
                "mean_ticks_x100": _stats([item["mean_ticks_x100"] for item in rows]),
                "mean_liquid_food_per_worker_tick_x1000": _stats(
                    [item["mean_liquid_food_per_worker_tick_x1000"] for item in rows]
                ),
                "calculator_ticks": rows[0]["calculator_ticks"],
            })
        output["zones"][zone] = {
            "patches_per_map": diagnostics[0]["patches"],
            "actual_rank_distances": ranks,
            "colonies_within_economic_radius_x100": _stats(
                [item["mean_colonies_within_economic_radius_x100"] for item in diagnostics]
            ),
            "colonies_within_contest_radius_x100": _stats(
                [item["mean_colonies_within_contest_radius_x100"] for item in diagnostics]
            ),
        }
    accesses = [row["map_distance_diagnostics"]["economic_access"] for row in run_rows]
    output["economic_access"] = {
        "radius_ticks": accesses[0]["radius_ticks"],
        "mean_fair_share_food_per_colony_x1000": _stats(
            [item["mean_fair_share_food_per_colony_x1000"] for item in accesses]
        ),
        "min_colony_fair_share_food_x1000": _stats(
            [item["min_colony_fair_share_food_x1000"] for item in accesses]
        ),
        "max_colony_fair_share_food_x1000": _stats(
            [item["max_colony_fair_share_food_x1000"] for item in accesses]
        ),
        "calculator_assumption_food_per_colony_x1000": accesses[0]["calculator_assumption_food_per_colony_x1000"],
    }
    return output


def sweep_simulations(*, sizes: tuple[int, ...] = (6, 7, 8),
                      scenarios: tuple[str, ...] = ("abundance", "moderate", "severe"),
                      fixtures: tuple[str, ...] = ("peaceful_fair", "greedy_scarcity"),
                      seeds: int = 30, seed0: int = 50000, ticks: int = 6000,
                      progress: Callable[[int, int, int, str, str], None] | None = None) -> dict:
    """Run the real-map calibration matrix and keep one compact result row per map/fixture."""
    if seeds < 1 or seed0 < 0 or seed0 + seeds >= 1 << 64:
        raise ValueError("seeds must be positive and the seed range must fit u64")
    normalized_fixtures = tuple(normalize_fixture(name) for name in fixtures)
    configurations = len(sizes) * len(scenarios) * len(normalized_fixtures)
    start = time.perf_counter()
    rows = []
    completed_configurations = 0
    for size in sizes:
        for scenario in scenarios:
            config = default_config(size, scenario)
            for fixture in normalized_fixtures:
                run_rows = []
                for seed in range(seed0, seed0 + seeds):
                    result = run_simulation(config, seed, fixture, ticks=ticks, sample_interval=500,
                                             include_map=False, collect_timeline=True)
                    report = result.report
                    milestones = {str(row["tick"]): {"adults": row["adults"],
                                                       "maintenance_we": row["maintenance_we"],
                                                       "food_in_warehouses": row["food_in_warehouses"]}
                                                       for row in report["time_series"] if row["tick"] in (0, 500, 1000, 3000, 6000)}
                    run_rows.append({
                        "seed": seed,
                        "map_hash": report["map_hash"],
                        "first_signals": report["first_signals"],
                        "map_distance_diagnostics": report["map_distance_diagnostics"],
                        "surviving_colonies": sum(row["status"] == "alive" for row in report["final"]["colonies"]),
                        "final_adults": report["final"]["adults"],
                        "adult_change": report["final"]["adults"] - report["time_series"][0]["adults"],
                        "final_maintenance_we": report["final"]["maintenance_we"],
                        "final_warehouse_food": report["final"]["food_in_warehouses"],
                        "final_patch_stock": report["final"]["food_on_patches"],
                        "delivered_food": report["totals"]["delivered"],
                        "first_delivery_tick": report["first_signals"]["first_delivery_tick"],
                        "first_lay_tick": report["first_signals"]["first_lay_tick"],
                        "runtime_seconds": report["performance"]["wall_seconds"],
                        "ticks_per_second": report["performance"]["ticks_per_second"],
                        "starvation_deaths": report["totals"]["starvation_workers"] + report["totals"]["starvation_soldiers"],
                        "old_age_deaths": report["totals"]["old_age_workers"] + report["totals"]["old_age_soldiers"],
                        "shared_patch_fill_permille": report["totals"]["mean_shared_patch_fill_permille"],
                        "milestones": milestones,
                        "state_hash": report["hashes"]["state"],
                    })
                rows.append({
                    "colonies": size,
                    "scenario": scenario,
                    "fixture": fixture,
                    "seeds": seeds,
                    "seed0": seed0,
                    "surviving_colonies": _stats([row["surviving_colonies"] for row in run_rows]),
                    "final_adults": _stats([row["final_adults"] for row in run_rows]),
                    "adult_change": _stats([row["adult_change"] for row in run_rows]),
                    "final_maintenance_we": _stats([row["final_maintenance_we"] for row in run_rows]),
                    "final_warehouse_food": _stats([row["final_warehouse_food"] for row in run_rows]),
                    "first_contention_tick": _stats([row["first_signals"]["contention_tick"] for row in run_rows]),
                    "first_patch_depletion_tick": _stats([row["first_signals"]["patch_depletion_tick"] for row in run_rows]),
                    "first_upkeep_shortfall_tick": _stats([row["first_signals"]["upkeep_shortfall_tick"] for row in run_rows]),
                    "first_starvation_death_tick": _stats([row["first_signals"]["starvation_death_tick"] for row in run_rows]),
                    "starvation_deaths": _stats([row["starvation_deaths"] for row in run_rows]),
                    "old_age_deaths": _stats([row["old_age_deaths"] for row in run_rows]),
                    "delivered_food": _stats([row["delivered_food"] for row in run_rows]),
                    "first_delivery_tick": _stats([row["first_delivery_tick"] for row in run_rows]),
                    "first_lay_tick": _stats([row["first_lay_tick"] for row in run_rows]),
                    "runtime_seconds": _stats([row["runtime_seconds"] for row in run_rows]),
                    "ticks_per_second": _stats([row["ticks_per_second"] for row in run_rows]),
                    "shared_patch_fill_permille": _stats([row["shared_patch_fill_permille"] for row in run_rows]),
                    "map_distance_summary": _distance_summary(run_rows),
                    "runs": run_rows,
                })
                completed_configurations += 1
                if progress is not None:
                    progress(completed_configurations, configurations, size, scenario, fixture)
    elapsed = time.perf_counter() - start
    return {
        "schema": "formicmind.eco.simulation-sweep/1",
        "runs_per_configuration": seeds,
        "ticks_per_run": ticks,
        "total_runs": len(rows) * seeds,
        "seed0": seed0,
        "performance": {"wall_seconds": round(elapsed, 3),
                        "worlds_per_second": round(len(rows) * seeds / elapsed, 3) if elapsed else 0.0,
                        "simulated_ticks_per_second": round(len(rows) * seeds * ticks / elapsed, 1) if elapsed else 0.0},
        "results": rows,
    }


def benchmark_simulation(config: ConfigE1, seed0: int, fixture: str = "peaceful_fair", *, runs: int = 3,
                         ticks: int = 6000) -> dict:
    if runs < 1:
        raise ValueError("runs must be positive")
    seconds = []
    tps = []
    for offset in range(runs):
        result = run_simulation(config, seed0 + offset, fixture, ticks=ticks, include_map=False,
                                collect_timeline=False)
        wall = result.report["performance"]["wall_seconds"]
        seconds.append(wall)
        tps.append(ticks / wall if wall else 0.0)
    tracemalloc.start()
    run_simulation(config, seed0, fixture, ticks=ticks, include_map=False, collect_timeline=False)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "schema": "formicmind.eco.simulation-benchmark/1",
        "colonies": config.n_colonies,
        "scenario": config.scenario,
        "fixture": normalize_fixture(fixture),
        "ticks": ticks,
        "runs": runs,
        "wall_seconds": {"median": round(statistics.median(seconds), 6),
                         "mean": round(sum(seconds) / len(seconds), 6),
                         "min": min(seconds), "max": max(seconds)},
        "ticks_per_second": {"median": round(statistics.median(tps), 1),
                             "mean": round(sum(tps) / len(tps), 1),
                             "min": round(min(tps), 1), "max": round(max(tps), 1)},
        "peak_traced_memory_bytes": peak,
        "memory_measurement_runs": 1,
    }


INTERACTION_SCENARIOS: dict[str, dict[int, str]] = {
    "control_fair": {},
    "mixed_roles": {0: "raider", 2: "turtle", 4: "retreat", 6: "scout"},
    "four_raiders": {0: "raider", 2: "raider", 4: "raider", 6: "raider"},
}


def interaction_sweep(*, colonies: int = 8, scenario: str = "moderate", seeds: int = 6, seed0: int = 50000, ticks: int = 6000,
                      scenarios: dict[str, dict[int, str]] | None = None,
                      progress: Callable[[int, int, str, int], None] | None = None) -> dict:
    """Run named role assignments on the same maps and report only what the event logs show (no claim that war is inevitable)."""
    if seeds < 1 or seed0 < 0 or seed0 + seeds >= 1 << 64:
        raise ValueError("seeds must be positive and the seed range must fit u64")
    scenarios = INTERACTION_SCENARIOS if scenarios is None else scenarios
    config = default_config(colonies, scenario)
    start = time.perf_counter()
    rows, summary = [], {}
    total = len(scenarios) * seeds
    done = 0
    for name, roles in scenarios.items():
        runs = []
        for seed in range(seed0, seed0 + seeds):
            result = run_simulation(config, seed, "peaceful_fair", ticks=ticks, include_map=False, collect_timeline=False,
                                    roles=roles)
            report, inter = result.report, result.report["interaction"]
            final = report["final"]
            run = {
                "scenario": name, "seed": seed,
                "survivors": sum(c["status"] == "alive" for c in final["colonies"]),
                "orphans": sum(c["status"] == "orphan" for c in final["colonies"]),
                "extinct": sum(c["status"] == "extinct" for c in final["colonies"]),
                "adults": final["adults"], "maintenance_we": final["maintenance_we"],
                "delivered_food": report["totals"]["delivered"], "egg_spend": report["totals"]["egg_spend"],
                "laid_soldiers": report["totals"]["laid_soldiers"],
                "starvation_deaths": report["totals"]["starvation_workers"] + report["totals"]["starvation_soldiers"],
                "colony_pairs_in_contact": inter["colony_pairs_in_contact"], "first_contact_tick": inter["first_contact_tick"],
                "nests_found": inter["nests_found"], "attacks": len(inter["attacks_dispatched"]),
                "combat_ticks": inter["combat_ticks"], "combat_deaths": inter["combat_deaths"],
                "queen_deaths": inter["queen_deaths"], "collapses": inter["collapses"], "extinctions": inter["extinctions"],
                "events_hash": inter["events_hash"], "state_hash": report["hashes"]["state"],
            }
            runs.append(run)
            rows.append(run)
            done += 1
            if progress is not None:
                progress(done, total, name, seed)
        def count(predicate) -> int:
            return sum(1 for run in runs if predicate(run))
        summary[name] = {
            "roles": {str(k): v for k, v in sorted(roles.items())},
            "worlds": len(runs),
            "worlds_with_contact": count(lambda r: r["colony_pairs_in_contact"] > 0),
            "worlds_with_nest_found": count(lambda r: r["nests_found"] > 0),
            "worlds_with_attack": count(lambda r: r["attacks"] > 0),
            "worlds_with_combat": count(lambda r: r["combat_ticks"] > 0),
            "worlds_with_queen_death": count(lambda r: bool(r["queen_deaths"])),
            "total_attacks": sum(r["attacks"] for r in runs),
            "total_combat_deaths_workers": sum(r["combat_deaths"]["workers"] for r in runs),
            "total_combat_deaths_soldiers": sum(r["combat_deaths"]["soldiers"] for r in runs),
            "total_queen_deaths": sum(len(r["queen_deaths"]) for r in runs),
            "total_extinctions": sum(len(r["extinctions"]) for r in runs),
            "mean_survivors": round(statistics.mean(r["survivors"] for r in runs), 2),
            "mean_adults": round(statistics.mean(r["adults"] for r in runs), 1),
            "mean_delivered_food": round(statistics.mean(r["delivered_food"] for r in runs)),
            "mean_egg_spend": round(statistics.mean(r["egg_spend"] for r in runs)),
            "mean_laid_soldiers": round(statistics.mean(r["laid_soldiers"] for r in runs), 1),
            "mean_starvation_deaths": round(statistics.mean(r["starvation_deaths"] for r in runs)),
        }
    elapsed = time.perf_counter() - start
    return {
        "schema": "formicmind.eco.interaction-sweep/1",
        "colonies": colonies, "scenario": scenario, "seeds": seeds, "seed0": seed0, "ticks": ticks,
        "summary": summary, "runs": rows,
        "performance": {"wall_seconds": round(elapsed, 3),
                        "simulated_ticks_per_second": round(total * ticks / elapsed, 1) if elapsed else 0.0},
    }
