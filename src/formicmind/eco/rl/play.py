"""Play one real E1 world with any controller and measure it the same way for every policy: the colonies' true viability reward, survival, size,
delivered food and what they did (exploration, contacts, attacks). Torch-free, so scripted baselines are scored by exactly the code that scores
the learned policy. The measures use the omniscient world only for *reporting*; none of them reaches a policy.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import ConfigE1
from ..engine import RouteTable, step_world_tick
from ..events import EventLog
from ..initial import build_world
from ..viability import DEATH_PENALTY, Q, ViabilityTracker


@dataclass(slots=True)
class PlayResult:
    config: ConfigE1
    seed: int
    state: object
    tracker: ViabilityTracker
    events: EventLog
    delivered: list[int]
    ticks: int
    wall_seconds: float


def play_world(config: ConfigE1, seed: int, controller, ticks: int) -> PlayResult:
    """Run ``ticks`` ticks (or until the world ends) with the tracker watching from outside. ``controller`` is any engine controller."""
    import time
    started = time.perf_counter()
    world = build_world(config, seed)
    routes = RouteTable(world.world_map)
    tracker = ViabilityTracker(world.state)
    events = EventLog()
    delivered = [0] * config.n_colonies
    for _ in range(ticks):
        record = step_world_tick(world.state, world.run, config, world.world_map, controller, routes=routes,
                                 validate_input=world.state.tick == 0)
        events.extend(record.events)
        tracker.update(world.state, record)
        for colony, amount in record.delivered_by_colony.items():
            delivered[colony] += amount
        if world.state.world_outcome.finished:
            break
    return PlayResult(config, seed, world.state, tracker, events, delivered, world.state.tick, time.perf_counter() - started)


def colony_metrics(result: PlayResult) -> list[dict]:
    """Per-colony report of a finished play (true values, for evaluation only)."""
    config, state, tracker = result.config, result.state, result.tracker
    adults = [[0, 0] for _ in state.colonies]
    for cohort in state.cohorts:
        adults[cohort.colony_id][0 if cohort.caste == "worker" else 1] += cohort.count
    per_colony_events = [{"discover": 0, "contact": 0, "nest_found": 0, "dispatch_attack": 0} for _ in state.colonies]
    for item in result.events.events:
        if item.kind in ("discover", "contact", "nest_found"):
            per_colony_events[item.colony][item.kind] += 1
        elif item.kind == "dispatch" and item.a == 3:    # MISSION_CODES["attack"]
            per_colony_events[item.colony]["dispatch_attack"] += 1
    out = []
    for colony in state.colonies:
        trace = tracker.traces[colony.colony_id]
        dead = trace.death_tick is not None
        total = sum(trace.v_q)
        out.append({
            "colony": colony.colony_id,
            "reward": total / (Q * config.horizon) - (DEATH_PENALTY if dead else 0.0),
            "mean_viability": total / (Q * max(1, result.ticks)),     # zeros after death count: a dead colony is not viable
            "viability_end": (trace.v_q[-1] if trace.v_q else 0) / Q,
            "survived": colony.status == "alive",
            "death_tick": None if trace.death_tick is None else trace.death_tick - 1,
            "hollow": colony.status == "alive" and not any(adults[colony.colony_id]),   # alive on paper: a queen with no workers or soldiers
            "workers_end": adults[colony.colony_id][0],
            "soldiers_end": adults[colony.colony_id][1],
            "delivered": result.delivered[colony.colony_id],
            "food_end": colony.food,
            **per_colony_events[colony.colony_id],
        })
    return out


def summarize(runs: list[list[dict]]) -> dict:
    """Mean of the per-colony metrics over all colonies of all runs."""
    rows = [row for run in runs for row in run]
    keys = ("reward", "mean_viability", "viability_end", "workers_end", "soldiers_end", "delivered", "food_end", "discover", "contact",
            "nest_found", "dispatch_attack")
    summary = {key: sum(row[key] for row in rows) / len(rows) for key in keys}
    summary["survival"] = sum(1 for row in rows if row["survived"]) / len(rows)
    summary["hollow"] = sum(1 for row in rows if row["hollow"]) / len(rows)      # share of colonies that "survive" with nobody left
    summary["colonies"] = len(rows)
    return summary
