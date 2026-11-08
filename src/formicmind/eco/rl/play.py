"""Play one real E1 world with any controller and measure it the same way for every policy: the colonies' true viability reward, survival, size,
delivered food and what they did (exploration, contacts, attacks). Torch-free, so scripted baselines are scored by exactly the code that scores
the learned policy. The measures use the omniscient world only for *reporting*; none of them reaches a policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import ConfigE1
from ..engine import RouteTable, step_world_tick
from ..events import EventLog
from ..initial import build_world
from ..viability import DEATH_PENALTY, Q, REWARD_VERSION, ViabilityTracker


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
    eggs_laid: list[int] = field(default_factory=list)       # per colony: eggs laid over the whole play
    worker_sum: list[int] = field(default_factory=list)      # per colony: workers summed over the sampled ticks (every ``SAMPLE_EVERY``)
    worker_peak: list[int] = field(default_factory=list)
    samples: int = 0
    tracker_v1: ViabilityTracker | None = None                # the first pilot's reward, kept only to re-score runs against it


SAMPLE_EVERY = 10   # ticks between the worker-count samples of the report


def play_world(config: ConfigE1, seed: int, controller, ticks: int, *, reward_version: int = REWARD_VERSION) -> PlayResult:
    """Run ``ticks`` ticks (or until the world ends) with the tracker watching from outside. ``controller`` is any engine controller.

    ``reward_version`` picks the reward of ``reward``; the first pilot's reward is always tracked as well (``reward_v1``) so that old and new
    scores of the same run can be compared."""
    import time
    started = time.perf_counter()
    world = build_world(config, seed)
    routes = RouteTable(world.world_map)
    tracker = ViabilityTracker(world.state, reward_version)
    tracker_v1 = tracker if reward_version == 1 else ViabilityTracker(world.state, 1)
    events = EventLog()
    n = config.n_colonies
    delivered, eggs, worker_sum, worker_peak = [0] * n, [0] * n, [0] * n, [0] * n
    samples = 0
    for _ in range(ticks):
        record = step_world_tick(world.state, world.run, config, world.world_map, controller, routes=routes,
                                 validate_input=world.state.tick == 0)
        events.extend(record.events)
        tracker.update(world.state, record)
        if tracker_v1 is not tracker:
            tracker_v1.update(world.state, record)
        for colony, amount in record.delivered_by_colony.items():
            delivered[colony] += amount
        for colony in world.state.colonies:
            eggs[colony.colony_id] += sum(1 for brood in colony.brood if brood.laid_tick == record.tick)
        if record.tick % SAMPLE_EVERY == 0:
            samples += 1
            workers = [0] * n
            for cohort in world.state.cohorts:
                if cohort.caste == "worker":
                    workers[cohort.colony_id] += cohort.count
            for i in range(n):
                worker_sum[i] += workers[i]
                worker_peak[i] = max(worker_peak[i], workers[i])
        if world.state.world_outcome.finished:
            break
    return PlayResult(config, seed, world.state, tracker, events, delivered, world.state.tick, time.perf_counter() - started,
                      eggs, worker_sum, worker_peak, samples, tracker_v1)


def _reward(trace, horizon: int) -> float:
    return sum(trace.v_q) / (Q * horizon) - (DEATH_PENALTY if trace.death_tick is not None else 0.0)


def colony_metrics(result: PlayResult) -> list[dict]:
    """Per-colony report of a finished play (true values, for evaluation only)."""
    config, state, tracker = result.config, result.state, result.tracker
    adults = [[0, 0] for _ in state.colonies]
    for cohort in state.cohorts:
        adults[cohort.colony_id][0 if cohort.caste == "worker" else 1] += cohort.count
    per_colony_events = [{"discover": 0, "contact": 0, "nest_found": 0, "dispatch_attack": 0, "combat": 0, "queen_damage": 0} for _ in state.colonies]
    for item in result.events.events:
        if item.kind in ("discover", "contact", "nest_found"):
            per_colony_events[item.colony][item.kind] += 1
        elif item.kind == "dispatch" and item.a == 3:    # MISSION_CODES["attack"]
            per_colony_events[item.colony]["dispatch_attack"] += 1
        elif item.kind == "combat":                      # a fight the colony took part in, on either side
            per_colony_events[item.colony]["combat"] += 1
            if 0 <= item.other < len(per_colony_events) and item.other != item.colony:
                per_colony_events[item.other]["combat"] += 1
        elif item.kind == "queen_damage":
            per_colony_events[item.colony]["queen_damage"] += 1
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
            "reward_v1": _reward(result.tracker_v1.traces[colony.colony_id], config.horizon) if result.tracker_v1 else None,
            "survived": colony.status == "alive",
            "death_tick": None if trace.death_tick is None else trace.death_tick - 1,
            "hollow": colony.status == "alive" and not any(adults[colony.colony_id]),   # alive on paper: a queen with no workers or soldiers
            "workers_end": adults[colony.colony_id][0],
            "workers_mean": result.worker_sum[colony.colony_id] / max(1, result.samples) if result.worker_sum else 0.0,
            "workers_peak": result.worker_peak[colony.colony_id] if result.worker_peak else 0,
            "brood_end": len(colony.brood),
            "eggs_laid": result.eggs_laid[colony.colony_id] if result.eggs_laid else 0,
            "soldiers_end": adults[colony.colony_id][1],
            "delivered": result.delivered[colony.colony_id],
            "food_end": colony.food,
            **per_colony_events[colony.colony_id],
        })
    return out


def summarize(runs: list[list[dict]]) -> dict:
    """Mean of the per-colony metrics over all colonies of all runs."""
    rows = [row for run in runs for row in run]
    keys = ("reward", "reward_v1", "mean_viability", "viability_end", "workers_end", "workers_mean", "workers_peak", "soldiers_end", "brood_end",
            "eggs_laid", "delivered", "food_end", "discover", "contact", "nest_found", "dispatch_attack", "combat", "queen_damage")
    summary = {key: sum(row[key] for row in rows) / len(rows) for key in keys}
    summary["survival"] = sum(1 for row in rows if row["survived"]) / len(rows)
    summary["hollow"] = sum(1 for row in rows if row["hollow"]) / len(rows)      # share of colonies that "survive" with nobody left
    summary["colonies"] = len(rows)
    return summary
