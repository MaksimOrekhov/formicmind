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


SAMPLE_EVERY = 10   # ticks between the adult-count samples of the report
LEGACY_VERSIONS = (1, 2)   # older rewards that are tracked beside the current one only to re-score a run against them (reports label them)


class PlayStats:
    """Reporting-only counters of one running world, shared by evaluation (``play_world``) and training rollouts: food delivered into each nest,
    eggs laid, and workers / soldiers sampled every ``SAMPLE_EVERY`` ticks (their mean and the peak of the workers). Reads the omniscient world; nothing
    here reaches a policy."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.delivered, self.eggs_laid = [0] * n, [0] * n
        self.worker_eggs, self.soldier_eggs = [0] * n, [0] * n      # eggs_laid split by caste (FOR-50: the nest gate counts worker eggs)
        self.worker_sum, self.soldier_sum, self.worker_peak = [0] * n, [0] * n, [0] * n
        self.scout_orders, self.scouts_sent, self.scout_arrivals = [0] * n, [0] * n, [0] * n      # FOR-52: scout sorties actually applied / sent / arrived
        self.found = {source: [0] * n for source in ("scout", "forage", "mixed", "other")}         # first visits by the kind of cohort that made them
        self.scout_sum = [0] * n                                                                    # scout workers sampled every SAMPLE_EVERY ticks
        self.samples = 0

    def tick(self, state, record) -> None:
        for item in record.events:
            if item.kind == "dispatch" and item.a == 1:     # MISSION_CODES["scout"]: the order was applied and idle workers left the nest
                self.scout_orders[item.colony] += 1
                self.scouts_sent[item.colony] += item.b
        for colony, _node, count in record.scout_arrivals:
            self.scout_arrivals[colony] += count
        for colony, _node, source in record.discoveries:
            self.found[source][colony] += 1
        for colony, amount in record.delivered_by_colony.items():
            self.delivered[colony] += amount
        for colony in state.colonies:
            for brood in colony.brood:
                if brood.laid_tick == record.tick:
                    self.eggs_laid[colony.colony_id] += 1
                    (self.worker_eggs if brood.caste == "worker" else self.soldier_eggs)[colony.colony_id] += 1
        if record.tick % SAMPLE_EVERY == 0:
            self.samples += 1
            workers, soldiers, scouts = [0] * self.n, [0] * self.n, [0] * self.n
            for cohort in state.cohorts:
                (workers if cohort.caste == "worker" else soldiers)[cohort.colony_id] += cohort.count
                if cohort.caste == "worker" and cohort.mission.kind == "scout":
                    scouts[cohort.colony_id] += cohort.count
            for i in range(self.n):
                self.scout_sum[i] += scouts[i]
                self.worker_sum[i] += workers[i]
                self.soldier_sum[i] += soldiers[i]
                self.worker_peak[i] = max(self.worker_peak[i], workers[i])


@dataclass(slots=True)
class PlayResult:
    config: ConfigE1
    seed: int
    state: object
    tracker: ViabilityTracker
    events: EventLog
    stats: PlayStats
    ticks: int
    wall_seconds: float
    legacy: dict = field(default_factory=dict)      # reward version -> tracker of an older reward over the same run (reports only)

    @property
    def delivered(self) -> list[int]:
        return self.stats.delivered


def play_world(config: ConfigE1, seed: int, controller, ticks: int, *, reward_version: int = REWARD_VERSION) -> PlayResult:
    """Run ``ticks`` ticks (or until the world ends) with the trackers watching from outside. ``controller`` is any engine controller.

    The world is the natural one: nothing is retired when a colony's agent episode ends (reward v3 ``functional_extinction``), the policy keeps
    playing its colony, and the terminal only shows in the scores. ``reward`` is the reward of ``reward_version``; the older rewards are tracked beside
    it (``reward_v1``, ``reward_v2``) so that old and new scores of the same run can be compared - but never as numbers on one scale."""
    import time
    started = time.perf_counter()
    world = build_world(config, seed)
    routes = RouteTable(world.world_map)
    tracker = ViabilityTracker(world.state, reward_version)
    legacy = {v: tracker if v == reward_version else ViabilityTracker(world.state, v) for v in LEGACY_VERSIONS}
    events = EventLog()
    stats = PlayStats(config.n_colonies)
    for _ in range(ticks):
        record = step_world_tick(world.state, world.run, config, world.world_map, controller, routes=routes,
                                 validate_input=world.state.tick == 0)
        events.extend(record.events)
        tracker.update(world.state, record)
        for other in legacy.values():
            if other is not tracker:
                other.update(world.state, record)
        stats.tick(world.state, record)
        if world.state.world_outcome.finished:
            break
    return PlayResult(config, seed, world.state, tracker, events, stats, world.state.tick, time.perf_counter() - started, legacy)


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
            **{f"reward_v{v}": _reward(other.traces[colony.colony_id], config.horizon) for v, other in sorted(result.legacy.items())},
            "survived": colony.status == "alive",                    # physical survival: queen alive and the colony not extinct in the world
            "functional_survived": not dead,                         # the agent episode did not end (v3: also not functionally extinct)
            "terminal_reason": trace.reason,
            "death_tick": None if trace.death_tick is None else trace.death_tick - 1,
            "hollow": colony.status == "alive" and not any(adults[colony.colony_id]),   # alive on paper: a queen with no workers or soldiers
            "workers_end": adults[colony.colony_id][0],
            "adults_end": sum(adults[colony.colony_id]),
            "workers_mean": result.stats.worker_sum[colony.colony_id] / max(1, result.stats.samples),   # mean over the samples every SAMPLE_EVERY ticks
            "adults_mean": (result.stats.worker_sum[colony.colony_id] + result.stats.soldier_sum[colony.colony_id]) / max(1, result.stats.samples),
            "workers_peak": result.stats.worker_peak[colony.colony_id],
            "brood_end": len(colony.brood),
            "eggs_laid": result.stats.eggs_laid[colony.colony_id],
            "worker_eggs_laid": result.stats.worker_eggs[colony.colony_id],
            "soldier_eggs_laid": result.stats.soldier_eggs[colony.colony_id],
            "soldiers_end": adults[colony.colony_id][1],
            # FOR-52 scouting, true values for reporting only. ``discovered_by_scout`` counts first visits made while only scout cohorts of the colony stood
            # on the node, ``discovered_by_forage`` only foragers; ``discovered_mixed`` (both there in the same tick) and ``discovered_other`` are kept apart, so the
            # four add up to ``discover`` and no node is counted twice. A scouting colony is one that sent scouts and has a first visit credited to them alone.
            "scout_orders": result.stats.scout_orders[colony.colony_id],
            "scouts_sent": result.stats.scouts_sent[colony.colony_id],
            "scout_arrivals": result.stats.scout_arrivals[colony.colony_id],
            "discovered_by_scout": result.stats.found["scout"][colony.colony_id],
            "discovered_by_forage": result.stats.found["forage"][colony.colony_id],
            "discovered_mixed": result.stats.found["mixed"][colony.colony_id],
            "discovered_other": result.stats.found["other"][colony.colony_id],
            "scouts_mean": result.stats.scout_sum[colony.colony_id] / max(1, result.stats.samples),
            "scout_share": result.stats.scout_sum[colony.colony_id] / max(1, result.stats.worker_sum[colony.colony_id]),
            "non_scout_workers_mean": (result.stats.worker_sum[colony.colony_id] - result.stats.scout_sum[colony.colony_id]) / max(1, result.stats.samples),
            "delivered": result.delivered[colony.colony_id],
            "food_end": colony.food,
            **per_colony_events[colony.colony_id],
        })
    return out


def summarize(runs: list[list[dict]]) -> dict:
    """Mean of the per-colony metrics over all colonies of all runs."""
    rows = [row for run in runs for row in run]
    keys = ("reward", "reward_v1", "reward_v2", "mean_viability", "viability_end", "workers_end", "adults_end", "workers_mean", "adults_mean",
            "workers_peak", "soldiers_end", "brood_end", "eggs_laid", "worker_eggs_laid", "soldier_eggs_laid", "delivered", "food_end", "discover", "contact", "nest_found", "dispatch_attack",
            "combat", "queen_damage", "scout_orders", "scouts_sent", "scout_arrivals", "discovered_by_scout", "discovered_by_forage", "discovered_mixed",
            "discovered_other", "scouts_mean", "scout_share", "non_scout_workers_mean")
    summary = {key: sum(row[key] for row in rows) / len(rows) for key in keys}
    summary["survival"] = sum(1 for row in rows if row["survived"]) / len(rows)                 # = physical_survival (kept for older reports)
    summary["physical_survival"] = summary["survival"]
    summary["functional_survival"] = sum(1 for row in rows if row["functional_survived"]) / len(rows)
    summary["share_worker_eggs_ge_50"] = sum(1 for row in rows if row["worker_eggs_laid"] >= 50) / len(rows)       # FOR-50 gate, per colony
    summary["share_sent_scout"] = sum(1 for row in rows if row["scout_orders"] > 0) / len(rows)                    # FOR-52 gate, per colony
    summary["share_scouted_new_node"] = sum(1 for row in rows if row["scout_orders"] > 0 and row["discovered_by_scout"] > 0) / len(rows)
    summary["share_workers_end_ge_5"] = sum(1 for row in rows if row["workers_end"] >= 5) / len(rows)
    reasons: dict[str, int] = {}
    for row in rows:
        if row["terminal_reason"]:
            reasons[row["terminal_reason"]] = reasons.get(row["terminal_reason"], 0) + 1
    summary["terminal_reasons"] = dict(sorted(reasons.items()))
    summary["functional_extinction"] = reasons.get("functional_extinction", 0) / len(rows)
    summary["hollow"] = sum(1 for row in rows if row["hollow"]) / len(rows)      # share of colonies that "survive" with nobody left
    summary["colonies"] = len(rows)
    return summary
