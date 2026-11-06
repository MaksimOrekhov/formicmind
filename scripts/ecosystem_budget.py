"""FOR-43 design aid: deterministic *fluid* budget calculator for Ecosystem v1 (E1).

This is NOT the Ecosystem engine and NOT a balance validation. It is a back-of-the-envelope model used to
derive the numeric scenarios in ``docs/ecosystem_v1_economy.md`` (initial food surplus, growing consumption,
depletion / regeneration, carrying capacity, finite lifespan, when competition starts, when a war pays). It reads and writes
nothing except stdout, uses no RNG and no third-party packages, and does not import any ``formicmind`` module,
so the frozen L0 engine and every historical artifact are untouched.

Everything here is a *proposal* (E1 candidates), explicitly separate from the L0 facts it borrows
(upkeep 0.4/0.8 per tick, egg cost 10/20, hatch delay 20/30, soldier damage 2, home bonus 5/4, HP 20/30/240).

Usage::

    python scripts/ecosystem_budget.py            # all tables, as in the economy document
    python scripts/ecosystem_budget.py --json     # machine-readable dump of the same numbers
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field, replace

# ----------------------------------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Econ:
    """Per-unit economy. L0 facts: upkeep, egg costs, hatch delays. E1 proposals: lay interval, carry model."""

    upkeep_worker: float = 0.4  # L0 fact: 4 tenths / tick
    upkeep_soldier: float = 0.8  # L0 fact: 8 tenths / tick
    egg_worker: int = 10  # L0 fact
    egg_soldier: int = 20  # L0 fact
    hatch_worker: int = 20  # L0 fact
    hatch_soldier: int = 30  # L0 fact
    lay_interval: int = 20  # E1 proposal (L0: 10): slower, "gradual" population growth
    load: int = 16  # E1 proposal: food units a worker carries per trip
    harvest_rate: int = 2  # E1 proposal: units a worker takes per tick while standing on a patch
    start_workers: int = 10  # E1 proposal (L0: 12)
    start_soldiers: int = 2  # E1 proposal (L0: 4)
    start_food: int = 80  # E1 proposal (L0: 60)
    storage_cap: int = 600  # E1 proposal: nest warehouse capacity (L0 has none); foragers stop when it is full
    decision_k: int = 30  # same cadence as L0
    horizon: int = 6000  # E1 proposal (L0: 3000)
    starvation_unit_cost: int = 5  # L0 fact: unpaid food units per adult lost
    lifespan_worker: int = 2000  # E1 proposal: adult lifespan in ticks after hatching; 0 = immortal (L0 has no ageing)
    lifespan_soldier: int = 2000  # E1 proposal
    start_age_spread: float = 0.5  # start adults are aged evenly over [0, spread * lifespan): no demographic cliff
    age_bucket: int = 100  # ledger granularity in ticks


@dataclass(frozen=True, slots=True)
class PatchClass:
    name: str
    count_per_colony: float  # patches of this class reachable by one colony (incidence)
    capacity: int
    regen: int  # R: food / tick at the source
    site_cap: int  # Q: maximum food / tick that can be taken from the patch by everybody together
    distance: int  # one-way path length in ticks from each colony that can reach it
    sharers: int  # how many colonies reach it
    known_from: int  # tick from which a colony knows (has scouted) this class


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    sigma: float  # multiplier of every regen / capacity / site cap (1.0 = "moderate")
    colonies: int = 8
    classes: tuple[PatchClass, ...] = field(default_factory=tuple)


HOME = PatchClass("home", 2, 100, 5, 10, 3, 1, 0)
BORDER = PatchClass("border", 2, 120, 6, 12, 7, 2, 150)
FAR = PatchClass("far", 3, 160, 8, 16, 10, 3, 400)


def scenario(name: str, sigma: float, colonies: int = 8) -> Scenario:
    def scaled(p: PatchClass) -> PatchClass:
        return replace(p, capacity=round(p.capacity * sigma), regen=max(1, round(p.regen * sigma)),
                       site_cap=max(1, round(p.site_cap * sigma)))

    return Scenario(name, sigma, colonies, tuple(scaled(p) for p in (HOME, BORDER, FAR)))


SCENARIOS = (scenario("abundance", 1.5), scenario("moderate", 1.0), scenario("severe", 0.6))


def forager_rate(d: int, e: Econ) -> float:
    """Average deposit rate of one forager on a patch ``d`` ticks away: ``load / (2d + load / harvest_rate)``."""
    return e.load / (2 * d + e.load / e.harvest_rate)


def net_per_forager(d: int, e: Econ) -> float:
    return forager_rate(d, e) - e.upkeep_worker


def break_even_distance(e: Econ) -> float:
    """Largest one-way distance at which a forager still pays for its own upkeep."""
    return (e.load / e.upkeep_worker - e.load / e.harvest_rate) / 2


# ----------------------------------------------------------------------------------------------------------------------
# Static budget (no dynamics)
# ----------------------------------------------------------------------------------------------------------------------


def static_budget(sc: Scenario, e: Econ) -> dict:
    """Supply per colony by patch class, and the population (worker-equivalents, WE) that supply can feed."""
    rows = []
    cumulative_supply = 0.0
    for cls in sc.classes:
        # Fair-share supply per colony: n patches * R / sharers. "Reach" is gated by profitability of the trip.
        per_colony = cls.count_per_colony * cls.regen / cls.sharers
        phi = forager_rate(cls.distance, e)
        foragers_needed = per_colony / phi
        cumulative_supply += per_colony
        rows.append({
            "class": cls.name,
            "distance": cls.distance,
            "rate_per_forager": round(phi, 3),
            "net_per_forager": round(phi - e.upkeep_worker, 3),
            "fair_supply_per_colony": round(per_colony, 2),
            "foragers_for_fair_supply": round(foragers_needed, 1),
            "cumulative_supply": round(cumulative_supply, 2),
            "cumulative_WE_fed": round(cumulative_supply / e.upkeep_worker, 1),
            "known_from": cls.known_from,
        })
    start_we = e.start_workers + 2 * e.start_soldiers
    start_upkeep = e.upkeep_worker * e.start_workers + e.upkeep_soldier * e.start_soldiers
    home_supply = rows[0]["fair_supply_per_colony"]
    return {
        "scenario": sc.name,
        "sigma": sc.sigma,
        "start_upkeep": round(start_upkeep, 2),
        "start_WE": start_we,
        "home_supply": home_supply,
        "start_surplus": round(home_supply - start_upkeep, 2),
        "start_utilisation": round(start_upkeep / home_supply, 3),
        "world_regen": round(sum(c.regen * c.count_per_colony * sc.colonies / c.sharers for c in sc.classes), 1),
        "classes": rows,
    }


# ----------------------------------------------------------------------------------------------------------------------
# Fluid simulation of 8 colonies on the ring-8 abstraction
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(slots=True)
class _Colony:
    workers: int
    soldiers: int
    food: float
    brood: list  # [(hatch_tick, caste)]
    debt: float = 0.0
    next_lay: int = 0
    lay_acc: float = 0.0
    in_transit: list = field(default_factory=list)  # [(arrive_tick, amount)]
    income_ema: float = 0.0
    upkeep_ema: float = 0.0
    alive: bool = True
    first_starve: int = -1
    paid: float = 0.0
    spent_eggs: float = 0.0
    harvested: float = 0.0
    assigned: dict = field(default_factory=dict)  # patch key -> foragers
    idle: int = 0
    queen_alive: bool = True
    ages_w: dict = field(default_factory=dict)  # birth bucket start -> workers
    ages_s: dict = field(default_factory=dict)  # birth bucket start -> soldiers
    extinct_tick: int = -1
    old_age_deaths: int = 0


def _patch_keys(sc: Scenario) -> list[tuple]:
    """Ring-8 abstraction. Patch ``(class, index)``; ``reach[(class, index)]`` = colonies that can use it."""
    n = sc.colonies
    keys = []
    for cls in sc.classes:
        total = round(cls.count_per_colony * n / cls.sharers)
        for k in range(total):
            keys.append((cls.name, k))
    return keys


def _reach(sc: Scenario, key: tuple) -> list[int]:
    n = sc.colonies
    cls = next(c for c in sc.classes if c.name == key[0])
    k = key[1]
    if cls.sharers == 1:
        return [k // round(cls.count_per_colony)]  # "home": two patches per colony
    if cls.sharers == 2:
        return [k % n, (k + 1) % n]  # "border": between colony k and k+1
    return [(k - 1) % n, k % n, (k + 1) % n]  # "far": colonies k-1, k, k+1


def _book_deaths(ledger: dict, n: int) -> None:
    """Book ``n`` non-age deaths against the age ledger proportionally (largest remainder, ties -> older bucket)."""
    total = sum(ledger.values())
    n = min(n, total)
    if n <= 0:
        return
    keys = sorted(ledger)
    exact = {k: n * ledger[k] / total for k in keys}
    take = {k: min(ledger[k], math.floor(exact[k])) for k in keys}
    left = n - sum(take.values())
    for k in sorted(keys, key=lambda k: (-(exact[k] - math.floor(exact[k])), k)):
        if left <= 0:
            break
        if take[k] < ledger[k]:
            take[k] += 1
            left -= 1
    for k in keys:
        ledger[k] -= take[k]
        if ledger[k] <= 0:
            del ledger[k]


def _seed_ages(count: int, lifespan: int, e: Econ) -> dict:
    ledger: dict[int, int] = {}
    for i in range(count):
        age = int(i * e.start_age_spread * lifespan / max(1, count)) if lifespan else 0
        bucket = -((age + e.age_bucket - 1) // e.age_bucket) * e.age_bucket  # bucket start (<= 0)
        ledger[bucket] = ledger.get(bucket, 0) + 1
    return ledger


def simulate(sc: Scenario, e: Econ, *, policy: str = "fair", soldier_share: float = 0.0,
             ticks: int | None = None, queen_death: dict | None = None, trace: bool = False,
             soldier_floor: int = 2) -> dict:
    """Deterministic fluid run. ``policy``: ``fair`` (take only the fair share of a shared patch) or ``greedy``
    (aim at the whole site throughput). Both are calibration *fixtures*, not agents. Returns checkpoints and events."""
    ticks = e.horizon if ticks is None else ticks
    n = sc.colonies
    classes = {c.name: c for c in sc.classes}
    keys = _patch_keys(sc)
    reach = {k: _reach(sc, k) for k in keys}
    stock = {k: float(classes[k[0]].capacity) for k in keys}
    colonies = [_Colony(e.start_workers, e.start_soldiers, float(e.start_food), []) for _ in range(n)]
    for c in colonies:
        c.ages_w = _seed_ages(e.start_workers, e.lifespan_worker, e)
        c.ages_s = _seed_ages(e.start_soldiers, e.lifespan_soldier, e)
    queen_death = queen_death or {}
    series: list[tuple] = []

    ledger_start = sum(stock.values()) + sum(c.food for c in colonies)
    regen_added = upkeep_paid = egg_spent = 0.0
    events = {"first_contention": -1, "first_famine": -1, "first_shared_depletion": -1,
              "first_idle_workers": -1}
    checkpoints = []
    dist = lambda key: classes[key[0]].distance  # noqa: E731

    def assign(c: _Colony, i: int, t: int) -> None:
        avail = c.workers
        c.assigned = {}
        usable = []
        for key in keys:
            if i not in reach[key]:
                continue
            cls = classes[key[0]]
            if t < cls.known_from or net_per_forager(cls.distance, e) <= 0:
                continue
            usable.append(key)
        usable.sort(key=lambda key: (dist(key), key))
        for key in usable:
            cls = classes[key[0]]
            phi = forager_rate(cls.distance, e)
            target_supply = cls.site_cap if policy == "greedy" else cls.regen / len(reach[key])
            want = math.ceil(target_supply / phi)
            take = min(avail, want)
            if take > 0:
                c.assigned[key] = take
                avail -= take
        c.idle = avail

    for t in range(ticks):
        # decision step: re-assign labour and (below) laying policy
        if t % e.decision_k == 0:
            for i, c in enumerate(colonies):
                if c.alive:
                    assign(c, i, t)

        # hatch
        for c in colonies:
            if not c.alive:
                continue
            due = [b for b in c.brood if b[0] <= t]
            c.brood = [b for b in c.brood if b[0] > t]
            bucket = (t // e.age_bucket) * e.age_bucket
            for _, caste in due:
                if caste == "w":
                    c.workers += 1
                    c.ages_w[bucket] = c.ages_w.get(bucket, 0) + 1
                else:
                    c.soldiers += 1
                    c.ages_s[bucket] = c.ages_s.get(bucket, 0) + 1

        # queen death (test hook) and ageing: a bucket dies when its start + lifespan has passed
        for i, c in enumerate(colonies):
            if c.alive and c.queen_alive and queen_death.get(i, -1) == t:
                c.queen_alive = False
            if not c.alive:
                continue
            for ledger, lifespan, attr in ((c.ages_w, e.lifespan_worker, "workers"),
                                           (c.ages_s, e.lifespan_soldier, "soldiers")):
                if not lifespan:
                    continue
                for b in [b for b in sorted(ledger) if b + lifespan <= t]:
                    k = ledger.pop(b)
                    setattr(c, attr, max(0, getattr(c, attr) - k))
                    c.old_age_deaths += k
            if c.workers + c.soldiers == 0 and not c.brood and c.alive:
                c.alive = False
                c.extinct_tick = t

        # deliveries reaching the nest
        for c in colonies:
            keep = []
            for arrive, amount in c.in_transit:
                if arrive <= t:
                    c.food += amount
                else:
                    keep.append((arrive, amount))
            c.in_transit = keep

        # harvest (before regeneration, as in L0): proportional to labour demand, remainder stays on the patch
        demand_by_patch: dict[tuple, dict[int, float]] = {k: {} for k in keys}
        for i, c in enumerate(colonies):
            if not c.alive or c.food >= e.storage_cap:
                continue  # a full warehouse: foragers stay home, the food is left on the patch for others
            for key, count in c.assigned.items():
                demand_by_patch[key][i] = count * forager_rate(classes[key[0]].distance, e)
        for key in keys:
            demands = demand_by_patch[key]
            total = sum(demands.values())
            if total <= 0:
                continue
            cls = classes[key[0]]
            supply = min(float(cls.site_cap), stock[key])
            served = min(total, supply)
            if cls.sharers > 1 and total > supply + 1e-9 and events["first_contention"] < 0 and len(demands) > 1:
                events["first_contention"] = t
            for i, d_i in demands.items():
                got = served * d_i / total
                stock[key] -= got
                colonies[i].harvested += got
                colonies[i].in_transit.append((t + cls.distance, got))

        # upkeep and starvation
        for c in colonies:
            if not c.alive:
                continue
            due = e.upkeep_worker * c.workers + e.upkeep_soldier * c.soldiers
            pay = min(due, max(c.food, 0.0))
            c.food -= pay
            upkeep_paid += pay
            c.paid += pay
            c.upkeep_ema += (due - c.upkeep_ema) / 100
            shortfall = due - pay
            c.debt += shortfall
            while c.debt >= e.starvation_unit_cost and (c.workers + c.soldiers) > 0:
                if c.soldiers > 0:
                    c.soldiers -= 1
                    _book_deaths(c.ages_s, 1)
                else:
                    c.workers -= 1
                    _book_deaths(c.ages_w, 1)
                c.debt -= e.starvation_unit_cost
                if c.first_starve < 0:
                    c.first_starve = t
                if events["first_famine"] < 0:
                    events["first_famine"] = t
            if shortfall == 0 and due > 0:
                c.debt = 0.0
            if c.workers + c.soldiers == 0 and not c.brood:
                c.alive = False
                c.extinct_tick = t

        # lay (policy fixture): keep growing while foragers are useful, hold a reserve, soldiers by share
        for i, c in enumerate(colonies):
            if not c.alive or not c.queen_alive or t < c.next_lay:
                continue
            c.next_lay = t + e.lay_interval
            reserve = 25 * (e.upkeep_worker * c.workers + e.upkeep_soldier * c.soldiers) / 2
            useful_workers = c.idle <= 1
            c.lay_acc += soldier_share
            soldier_turn = c.lay_acc >= 1.0
            if soldier_turn:
                c.lay_acc -= 1.0
            elif c.soldiers + sum(1 for _, caste in c.brood if caste == "s") < soldier_floor:
                soldier_turn = True  # fixture keeps a minimal home guard, replacing soldiers lost to age
            cost = e.egg_soldier if soldier_turn else e.egg_worker
            if soldier_turn or useful_workers:
                if c.food - cost >= reserve:
                    c.food -= cost
                    egg_spent += cost
                    c.spent_eggs += cost
                    delay = e.hatch_soldier if soldier_turn else e.hatch_worker
                    c.brood.append((t + delay, "s" if soldier_turn else "w"))

        # regeneration (after harvest and lay)
        for key in keys:
            cls = classes[key[0]]
            before = stock[key]
            stock[key] = min(float(cls.capacity), stock[key] + cls.regen)
            regen_added += stock[key] - before
            if cls.sharers > 1 and stock[key] < 1e-9 + cls.regen and events["first_shared_depletion"] < 0 \
                    and t > cls.known_from + 10 and sum(demand_by_patch[key].values()) > cls.regen:
                events["first_shared_depletion"] = t

        if events["first_idle_workers"] < 0 and colonies[0].idle >= 2 and t >= e.decision_k:
            events["first_idle_workers"] = t

        if trace and t % 50 == 0:
            series.append((t, [(c.workers, c.soldiers, c.alive) for c in colonies]))
        if (t + 1) % 500 == 0 or t == 0:
            alive = [c for c in colonies if c.alive]
            checkpoints.append({
                "tick": t + 1,
                "alive": len(alive),
                "workers_mean": round(sum(c.workers for c in alive) / max(1, len(alive)), 1),
                "soldiers_mean": round(sum(c.soldiers for c in alive) / max(1, len(alive)), 1),
                "WE_mean": round(sum(c.workers + 2 * c.soldiers for c in alive) / max(1, len(alive)), 1),
                "food_mean": round(sum(c.food for c in alive) / max(1, len(alive)), 1),
                "shared_stock_frac": round(
                    sum(stock[k] for k in keys if classes[k[0]].sharers > 1)
                    / max(1.0, sum(classes[k[0]].capacity for k in keys if classes[k[0]].sharers > 1)), 3),
                "idle_mean": round(sum(c.idle for c in alive) / max(1, len(alive)), 1),
            })

    # ledger: patches + warehouses + in transit == start + regen - upkeep - eggs
    total_now = sum(stock.values()) + sum(c.food + sum(a for _, a in c.in_transit) for c in colonies)
    ledger_expected = ledger_start + regen_added - upkeep_paid - egg_spent
    return {
        "scenario": sc.name,
        "policy": policy,
        "soldier_share": soldier_share,
        "events": events,
        "checkpoints": checkpoints,
        "ledger_residual": round(total_now - ledger_expected, 9),
        "final_alive": sum(1 for c in colonies if c.alive),
        "final_WE_mean": round(sum(c.workers + 2 * c.soldiers for c in colonies if c.alive)
                               / max(1, sum(1 for c in colonies if c.alive)), 1),
        "starved_colonies": sum(1 for c in colonies if c.first_starve >= 0),
        "egg_spent": round(egg_spent, 1),
        "harvested": round(sum(c.harvested for c in colonies), 1),
        "old_age_deaths": sum(c.old_age_deaths for c in colonies),
        "extinct_ticks": [c.extinct_tick for c in colonies],
        "series": series,
    }


# ----------------------------------------------------------------------------------------------------------------------
# Combat and war profitability (L0 combat rule generalised to a cohort fight at one node)
# ----------------------------------------------------------------------------------------------------------------------

SOLDIER_HP = 30
WORKER_HP = 20
SOLDIER_DAMAGE = 2


def _apply(damage: int, soldiers_hp: int, workers_hp: int) -> tuple[int, int]:
    """Damage goes to soldiers first, then to workers; the remainder is wasted (L0 rule)."""
    take = min(damage, soldiers_hp)
    soldiers_hp -= take
    damage -= take
    workers_hp -= min(damage, workers_hp)
    return soldiers_hp, workers_hp


def engagement(att_soldiers: int, def_soldiers: int, def_workers: int, *, defender_home: bool = False,
               max_ticks: int = 400) -> dict:
    """Simultaneous per-tick damage 2/soldier (home defenders x5/4 floored), HP pools instead of unit objects.

    Returns survivors and the tick count. ``hp pool`` representation: alive = ceil(hp / max_hp)."""
    a_hp = att_soldiers * SOLDIER_HP
    ds_hp = def_soldiers * SOLDIER_HP
    dw_hp = def_workers * WORKER_HP
    t = 0
    while t < max_ticks and a_hp > 0 and (ds_hp > 0 or dw_hp > 0):
        a_alive = math.ceil(a_hp / SOLDIER_HP)
        d_alive = math.ceil(ds_hp / SOLDIER_HP)
        att_dmg = a_alive * SOLDIER_DAMAGE
        def_base = d_alive * SOLDIER_DAMAGE
        def_dmg = def_base * 5 // 4 if defender_home else def_base
        ds_hp, dw_hp = _apply(att_dmg, ds_hp, dw_hp)
        a_hp = max(0, a_hp - def_dmg)
        t += 1
        if def_dmg == 0 and ds_hp == 0 and dw_hp == 0:
            break
    return {
        "ticks": t,
        "attacker_survivors": math.ceil(a_hp / SOLDIER_HP),
        "defender_soldier_survivors": math.ceil(ds_hp / SOLDIER_HP),
        "defender_worker_survivors": math.ceil(dw_hp / WORKER_HP),
        "attacker_lost": att_soldiers - math.ceil(a_hp / SOLDIER_HP),
        "defender_soldiers_lost": def_soldiers - math.ceil(ds_hp / SOLDIER_HP),
        "defender_workers_lost": def_workers - math.ceil(dw_hp / WORKER_HP),
    }


def share_gain(supply: float, my_demand: float, rival_demands: list[float], removed: int) -> float:
    """Increase of my food rate from a shared patch after rival ``removed`` leaves (proportional labour rule)."""
    def mine(demands: list[float]) -> float:
        total = my_demand + sum(demands)
        if total <= supply:
            return my_demand
        return supply * my_demand / total

    before = mine(rival_demands)
    after = mine([d for k, d in enumerate(rival_demands) if k != removed])
    return after - before


# ----------------------------------------------------------------------------------------------------------------------
# Competition scenarios on one shared border patch (steady state rates + one-off costs)
# ----------------------------------------------------------------------------------------------------------------------

STRATEGIES = ("peace_fair", "match_greedy", "guarded_fair", "attack", "attack_weak")
POP_CAP = 60.0  # the reward saturates here (contracts section 9): extra population beyond it is worth nothing
ARMY_TAIL = 600  # a built army that is no longer needed is still fed for this many ticks (design assumption)


def _patch_for(sc: Scenario) -> PatchClass:
    return next(c for c in sc.classes if c.name == "border")


def duel(sc: Scenario, e: Econ, *, b_behavior: str, third: int, t_remaining: int, ratio: float = 2.0,
         garrison: int = 2, rival_soldiers: int = 4, army: str = "existing", cap: float = POP_CAP) -> dict:
    """Value of colony A's options on a border patch it shares with B (+ ``third`` more colonies behaving like B).

    The colony is fed by its own home patches plus whatever it takes at the border patch. Sustainable workers
    ``W = (D_home + D_border - 0.8 * dedicated_soldiers) / 0.4`` (every worker, forager or not, costs 0.4); foragers
    are limited by the workers available. Value = integral of ``min(cap, W(t) + 0.5 * soldiers)`` in worker-ticks,
    where extra workers appear only as fast as the queen lays (``1 / lay_interval`` per tick, after the hatch delay),
    minus one-off costs converted to worker-ticks (``food / 0.4``).

    Strategies: ``peace_fair`` (fair share), ``match_greedy`` (aim at the whole site throughput), ``guarded_fair``
    (fair share + standing garrison), ``attack`` (beat B's soldiers with ``ratio`` x force, then hold with
    ``garrison``), ``attack_weak`` (equal force: mutual annihilation, B's workers stay). Not attacking = retreat.
    ``army``: ``existing`` (the soldiers are already there; cost = losses + garrison) or ``built`` (they have to be
    raised first: egg + hatch upkeep, and survivors beyond the garrison are fed for ``ARMY_TAIL`` more ticks)."""
    cls = _patch_for(sc)
    home = next(c for c in sc.classes if c.name == "home")
    d_home = home.count_per_colony * home.regen
    supply = float(cls.regen)
    site = float(cls.site_cap)
    phi = forager_rate(cls.distance, e)
    phi_home = forager_rate(home.distance, e)
    n_home = math.ceil(d_home / phi_home - 1e-9)
    sharers = 2 + third

    def foragers(demand: float) -> int:
        return max(1, math.ceil(demand / phi - 1e-9))

    fair_n = foragers(supply / sharers)
    greedy_n = foragers(site)
    rival_n = fair_n if b_behavior == "fair" else greedy_n
    rival_d = rival_n * phi

    def share(a_n: int, rivals: int) -> float:
        a_d = a_n * phi
        total = a_d + rivals * rival_d
        return a_d if total <= supply + 1e-9 else supply * a_d / total

    def workers(n_b: int, rivals: int, g: int) -> tuple[float, float]:
        """Sustainable workers and border delivery, with the foragers limited by the workers that exist."""
        n = n_b
        for _ in range(6):
            delivered = share(n, rivals)
            w = (d_home + delivered - e.upkeep_soldier * g) / e.upkeep_worker
            limit = max(0, math.floor(w - n_home))
            if n <= limit:
                break
            n = limit
        return w, share(n, rivals) if n else 0.0

    after_n = foragers(supply / (1 + third)) if third else foragers(supply)
    plans: dict[str, dict] = {}
    plans["peace_fair"] = dict(n=fair_n, rivals=1 + third, g=0, one_off=0.0, start=0, tail=0.0)
    plans["match_greedy"] = dict(n=greedy_n, rivals=1 + third, g=0, one_off=0.0, start=0, tail=0.0)
    plans["guarded_fair"] = dict(n=fair_n, rivals=1 + third, g=garrison, one_off=0.0, start=0, tail=0.0)
    for name, attackers in (("attack", max(1, round(ratio * rival_soldiers))), ("attack_weak", rival_soldiers)):
        fight = engagement(attackers, rival_soldiers, rival_n)
        won = fight["defender_soldier_survivors"] == 0 and fight["attacker_survivors"] > 0
        losses = fight["attacker_lost"] * (e.egg_soldier + e.upkeep_soldier * e.hatch_soldier)
        build = attackers * (e.egg_soldier + e.upkeep_soldier * e.hatch_soldier) if army == "built" else 0.0
        surplus = max(0, fight["attacker_survivors"] - garrison) if won else fight["attacker_survivors"]
        tail = surplus * e.upkeep_soldier * min(t_remaining, ARMY_TAIL) if army == "built" else 0.0
        start = fight["ticks"] + cls.distance
        if won:
            plans[name] = dict(n=after_n, rivals=third, g=garrison, one_off=losses + build + tail, start=start, tail=tail)
        else:
            plans[name] = dict(n=fair_n, rivals=1 + third, g=0, one_off=losses + build + tail, start=start, tail=tail)

    w_peace, _ = workers(fair_n, 1 + third, 0)
    out = {}
    for name, plan in plans.items():
        w_target, delivered = workers(plan["n"], plan["rivals"], plan["g"])
        extra = w_target - w_peace
        egg_food = max(0.0, extra) * (e.egg_worker + e.upkeep_worker * e.hatch_worker)
        total = 0.0
        for t in range(t_remaining):
            if t < plan["start"]:
                w = w_peace
            elif extra >= 0:
                w = w_peace + min(extra, max(0.0, (t - plan["start"] - e.hatch_worker) / e.lay_interval))
            else:
                w = w_target
            total += min(cap, w + 0.5 * plan["g"])
        value = total - (plan["one_off"] + egg_food) / e.upkeep_worker
        out[name] = {"workers_target": round(w_target, 1), "border_delivered": round(delivered, 2),
                     "one_off_food": round(plan["one_off"] + egg_food, 1), "value": round(value, 1)}
    baseline = out["peace_fair"]["value"]
    for name in out:
        out[name]["vs_peace"] = round(out[name]["value"] - baseline, 1)
    best = max(STRATEGIES, key=lambda k: (out[k]["value"], -STRATEGIES.index(k)))
    gain_w = out["attack"]["workers_target"] - out["peace_fair"]["workers_target"]
    payback = None
    if gain_w > 0:
        payback = round(plans["attack"]["start"] + (out["attack"]["one_off_food"] / e.upkeep_worker) / gain_w
                        + e.hatch_worker, 1)
    return {
        "scenario": sc.name, "b_behavior": b_behavior, "third": third, "t_remaining": t_remaining, "army": army,
        "values": out, "best": best, "attack_break_even_ticks": payback,
    }


def guard_break_even_raids(e: Econ, *, garrison: int = 2, foragers: int = 5, downtime: int = 200,
                           net_rate: float = 1.0) -> dict:
    """How often must an unguarded expansion be raided before a standing garrison pays for itself?"""
    loss_per_raid = foragers * (e.egg_worker + e.upkeep_worker * e.hatch_worker) + net_rate * downtime
    cost_per_tick = garrison * e.upkeep_soldier
    return {"loss_per_raid": round(loss_per_raid, 1), "garrison_cost_per_tick": round(cost_per_tick, 2),
            "break_even_one_raid_per_ticks": round(loss_per_raid / cost_per_tick, 1)}


def duel_grid(e: Econ) -> list[dict]:
    rows = []
    for sc in SCENARIOS:
        for b in ("fair", "greedy"):
            for third in (0, 1):
                for t_rem in (600, 3000):
                    r = duel(sc, e, b_behavior=b, third=third, t_remaining=t_rem, army="existing")
                    r["best_if_army_built"] = duel(sc, e, b_behavior=b, third=third, t_remaining=t_rem,
                                                   army="built")["best"]
                    rows.append(r)
    return rows


def lifespan_sensitivity(e: Econ) -> list[dict]:
    """Effect of finite adult lifespan on the peaceful (fair) and unchecked-growth (greedy) fixtures."""
    rows = []
    for life in (0, 1000, 2000, 4000):
        ee = replace(e, lifespan_worker=life, lifespan_soldier=life)
        for sc in SCENARIOS:
            for policy in ("fair", "greedy"):
                r = simulate(sc, ee, policy=policy, ticks=6000)
                cp = {c["tick"]: c for c in r["checkpoints"]}
                rows.append({
                    "lifespan": life or "inf", "scenario": sc.name, "policy": policy,
                    "WE_1000": cp[1000]["WE_mean"], "WE_3000": cp[3000]["WE_mean"], "WE_6000": cp[6000]["WE_mean"],
                    "eggs_per_harvest": round(r["egg_spent"] / max(1.0, r["harvested"]), 3),
                    "old_age_deaths": r["old_age_deaths"], "first_famine": r["events"]["first_famine"],
                    "alive": r["final_alive"],
                })
    return rows


def orphan_runs(e: Econ) -> list[dict]:
    """Colony 0 loses its queen at tick 1500 (moderate, fair); the others are untouched. Standing orders continue."""
    sc = scenario("moderate", 1.0)
    rows = []
    for life in (1000, 2000, 4000):
        ee = replace(e, lifespan_worker=life, lifespan_soldier=life)
        r = simulate(sc, ee, policy="fair", ticks=6000, queen_death={0: 1500}, trace=True)
        ser = {t: v[0] for t, v in r["series"]}
        rows.append({
            "lifespan": life, "extinct_tick": r["extinct_ticks"][0],
            "ticks_after_queen_death": r["extinct_ticks"][0] - 1500 if r["extinct_ticks"][0] >= 0 else None,
            "adults_at": {t: ser[t][0] + ser[t][1] for t in (1500, 2000, 2500, 3000, 4000)},
            "others_alive": sum(1 for x in r["extinct_ticks"][1:] if x < 0),
        })
    return rows


# ----------------------------------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------------------------------


def report() -> dict:
    e = Econ()
    out: dict = {
        "econ": {k: getattr(e, k) for k in e.__dataclass_fields__},
        "break_even_distance": round(break_even_distance(e), 2),
        "forager_table": [
            {"d": d, "rate": round(forager_rate(d, e), 3), "net": round(net_per_forager(d, e), 3)}
            for d in (2, 3, 4, 5, 7, 10, 13, 16)
        ],
        "static": [static_budget(sc, e) for sc in SCENARIOS],
        "dynamic": [],
        "engagements": [],
    }
    for sc in SCENARIOS:
        out["dynamic"].append(simulate(sc, e, policy="fair", soldier_share=0.0))
        out["dynamic"].append(simulate(sc, e, policy="greedy", soldier_share=0.0))
        out["dynamic"].append(simulate(sc, e, policy="greedy", soldier_share=0.3))
    out["lifespan"] = lifespan_sensitivity(e)
    out["orphan"] = orphan_runs(e)
    out["duel"] = duel_grid(e)
    out["guard_break_even"] = guard_break_even_raids(e)
    for att, ds, dw, home in ((4, 4, 12, False), (6, 4, 12, False), (8, 4, 12, False), (8, 4, 12, True),
                              (8, 8, 12, False), (12, 8, 12, False), (12, 8, 12, True), (16, 8, 12, False)):
        r = engagement(att, ds, dw, defender_home=home)
        out["engagements"].append({"attackers": att, "def_soldiers": ds, "def_workers": dw, "home": home, **r})
    return out


def _fmt_table(rows: list[dict], cols: list[str]) -> str:
    widths = [max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols]
    lines = ["  ".join(c.ljust(w) for c, w in zip(cols, widths))]
    for r in rows:
        lines.append("  ".join(str(r.get(c, "")).ljust(w) for c, w in zip(cols, widths)))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="dump the numbers as canonical JSON")
    args = parser.parse_args(argv)
    data = report()
    if args.json:
        json.dump(data, sys.stdout, ensure_ascii=False, sort_keys=True, indent=2, default=str)
        sys.stdout.write("\n")
        return 0
    print("== forager rate (load=16, h=2/tick): rate = 16/(2d+8); upkeep 0.4")
    print(_fmt_table(data["forager_table"], ["d", "rate", "net"]))
    print(f"break-even one-way distance: {data['break_even_distance']} ticks")
    for s in data["static"]:
        print(f"\n== static budget: {s['scenario']} (sigma={s['sigma']}) start upkeep {s['start_upkeep']}, "
              f"home supply {s['home_supply']}, utilisation {s['start_utilisation']}")
        print(_fmt_table(s["classes"], ["class", "distance", "rate_per_forager", "net_per_forager",
                                        "fair_supply_per_colony", "foragers_for_fair_supply",
                                        "cumulative_supply", "cumulative_WE_fed", "known_from"]))
    for d in data["dynamic"]:
        print(f"\n== dynamic: {d['scenario']} policy={d['policy']} soldier_share={d['soldier_share']} "
              f"events={d['events']} ledger_residual={d['ledger_residual']} alive={d['final_alive']} "
              f"starved_colonies={d['starved_colonies']}")
        print(_fmt_table(d["checkpoints"], ["tick", "alive", "workers_mean", "soldiers_mean", "WE_mean",
                                            "food_mean", "shared_stock_frac", "idle_mean"]))
    print("\n== lifespan sensitivity")
    print(_fmt_table(data["lifespan"], ["lifespan", "scenario", "policy", "WE_1000", "WE_3000", "WE_6000",
                                        "eggs_per_harvest", "old_age_deaths", "first_famine", "alive"]))
    print("\n== orphan (queen of colony 0 dies at tick 1500)")
    print(_fmt_table(data["orphan"], ["lifespan", "extinct_tick", "ticks_after_queen_death", "adults_at", "others_alive"]))
    print("\n== competition scenarios on a border patch (A vs B; value in worker-ticks relative to peace)")
    flat = []
    for d in data["duel"]:
        flat.append({"scenario": d["scenario"], "B": d["b_behavior"], "third": d["third"], "T_rem": d["t_remaining"],
                     **{f"{k}": v["vs_peace"] for k, v in d["values"].items() if k != "peace_fair"},
                     "best": d["best"], "best_if_army_built": d["best_if_army_built"],
                     "attack_payback": d["attack_break_even_ticks"]})
    print(_fmt_table(flat, list(flat[0])))
    print("guard break-even:", data["guard_break_even"])
    print("\n== engagements (L0 combat rule at one node)")
    print(_fmt_table(data["engagements"], ["attackers", "def_soldiers", "def_workers", "home", "ticks", "attacker_lost",
                                           "defender_soldiers_lost", "defender_workers_lost",
                                           "attacker_survivors"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
