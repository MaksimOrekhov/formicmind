"""Economy and brood phases (FOR-12; FOR-1 v0.2.1 sections 3, 4, 8 and errata B, C, D, E, F).

Phases (called by ``tick``): ``Completion -> Harvest -> Upkeep/Starvation -> Lay -> Regeneration``.
Everything is integer arithmetic; food is a non-negative ``int`` and upkeep is kept in tenths
with the carried ``upkeep_remainder10``. No phase reads or writes an RNG, a float, or the
iteration order of a ``set``/``dict``; side loops are over ``(0, 1)`` and every result that
could depend on that order (the contested split, the food ledger) is computed symmetrically.

Decisions that are interpretations of the specification (also listed in ``docs/economy.md``):

* **Contested harvest (errata B, C).** A side claims ``min(workers stationed at the node,
  max_harvest)``; ``max_harvest`` is a limit *per colony* (FOR-1 section 3: "Максимальный сбор одной
  колонией"), not a node-wide cap, so two colonies can take 2 * max_harvest from a rich node. A side whose node holds a stationed enemy soldier (own soldiers do not help)
  claims nothing. If both claims fit the stock both are paid in full; otherwise the stock is
  split as ``floor(stock * claim_i / total)`` and the (at most one) leftover unit goes to the side
  with the strictly larger fractional part, and **stays in the node** on an exact tie.
* **Starvation (errata D).** ``debt += due - paid`` on an underpaid tick; ``while debt >=
  starvation_unit_cost and adults remain: kill one, debt -= cost``; a *fully paid, non-zero*
  upkeep (``due > 0`` and ``missing == 0``) resets whatever debt is left to 0. A tick with
  ``due == 0`` pays nothing and therefore neither charges nor clears the debt. Victim order:
  soldiers before workers; inside a caste *home before other nodes before transit* (FOR-1 v0.2
  section 3.2, "сначала дома, затем узлы, потом отряды в пути"), then ascending ``unit_id``.
* **Hatching (errata E, F).** Newborns appear at home with full HP and their brood ``unit_id``.
  Soldiers take the colony's ``army_choice`` target. A worker takes the forage category with the
  largest positive deficit between the largest-remainder quotas for ``N + 1`` workers and the
  workers' current ``target_node`` counts; equal deficits go to the **side-local** category order
  ``home, N1, N2, C, N4, N5`` (the R1 convention, not the world node id, so the rule is mirror
  symmetric). Invalid weights leave the worker at home. ``AutomaticDepart`` then moves them.
* **Lay (errata F).** One paid batch of ``eggs_per_lay`` eggs, atomically: all or nothing.
"""

from __future__ import annotations

from .events import (
    HarvestEvent,
    HatchEvent,
    LayEvent,
    Phase,
    RegenerateEvent,
    StarvationEvent,
    UpkeepEvent,
)
from .intel import home_node
from .model import NOWHERE, BroodRecord, ColonyState, SimState, Unit
from .movement import enemy_soldier_nodes
from .orders import worker_quotas
from .rules import (
    FORAGE_CATEGORIES,
    LAY_NONE,
    LAY_WORKER,
    NODE_COUNT,
    SIDES,
    SOLDIER,
    WORKER,
    local_to_world,
)
from .runstate import RunState
from .snapshot import WindowStats


def _add_window(run: RunState, side: int, *, harvest: int = 0, upkeep_paid: int = 0,
                egg_spend: int = 0, debt_delta: int = 0) -> None:
    w = run.windows[side]
    run.windows[side] = WindowStats(
        w.harvest + harvest, w.upkeep_paid + upkeep_paid, w.egg_spend + egg_spend,
        w.starvation_debt_delta + debt_delta, w.window_ticks,
    )


def _adults(colony: ColonyState, caste: int) -> int:
    return sum(1 for u in colony.units if u.caste == caste)


# --- Completion ------------------------------------------------------------------------------

def newborn_worker_target(state: SimState, colony: ColonyState) -> int:
    """World node for one new worker under the current ``forage_q`` (errata E)."""
    side = colony.side
    home = home_node(state, side)
    workers = [u for u in colony.units if u.caste == WORKER]
    try:
        desired = worker_quotas(len(workers) + 1, colony.forage_q)
    except ValueError:  # malformed weights: no invented scouting, stay home
        return home
    current = [0] * FORAGE_CATEGORIES
    for unit in workers:
        category = local_to_world(side, unit.target_node)  # involution: world -> local
        if 0 <= category < FORAGE_CATEGORIES:
            current[category] += 1
    best, best_deficit = -1, 0
    for category in range(FORAGE_CATEGORIES):  # side-local order breaks ties
        deficit = desired[category] - current[category]
        if deficit > best_deficit:
            best, best_deficit = category, deficit
    return home if best < 0 else local_to_world(side, best)


def complete_hatching(state: SimState, phase: int = Phase.COMPLETION) -> list[HatchEvent]:
    tick = state.tick
    for colony in state.colonies:  # check everything before mutating anything
        for b in colony.brood:
            if b.hatch_tick < tick:
                raise ValueError(f"brood {b.unit_id} missed its hatch tick {b.hatch_tick}")
    bal = state.config.balance
    hp = {WORKER: bal.worker_hp, SOLDIER: bal.soldier_hp}
    events: list[HatchEvent] = []
    for side in SIDES:
        colony = state.colonies[side]
        home = home_node(state, side)
        ripe = [b for b in colony.brood if b.hatch_tick == tick]  # brood is sorted by unit_id
        colony.brood[:] = [b for b in colony.brood if b.hatch_tick != tick]
        for b in ripe:
            if b.caste == SOLDIER:
                target = local_to_world(side, colony.army_choice)
            else:
                target = newborn_worker_target(state, colony)
            unit = Unit(
                unit_id=b.unit_id, caste=b.caste, hp=hp[b.caste], node=home,
                from_node=NOWHERE, to_node=NOWHERE, arrival_tick=NOWHERE, target_node=target,
            )
            colony.units.append(unit)
            colony.units.sort(key=lambda u: u.unit_id)  # later workers see earlier newborns
            events.append(HatchEvent(
                tick=tick, phase=int(phase), side=side, unit_id=unit.unit_id, caste=unit.caste,
                node=home, hp=unit.hp, laid_tick=b.laid_tick, target_node=target))
    events.sort(key=lambda e: (e.unit_id, e.side))
    return events


# --- Harvest ---------------------------------------------------------------------------------

def split_contested(stock: int, claims: tuple[int, int]) -> tuple[int, int]:
    """Integer split of a scarce stock between two claims (errata C), symmetric under a swap.

    ``stock >= total claim``: both are paid in full. Otherwise ``floor(stock * c_i / total)``
    each; with two claimants at most one unit is left over, which goes to the strictly larger
    fractional part and stays in the node when the parts are equal. No float, no side priority.
    """
    if stock < 0 or min(claims) < 0:
        raise ValueError("stock and claims must be non-negative")
    total = claims[0] + claims[1]
    if total <= stock:
        return claims
    shares = [stock * c // total for c in claims]
    parts = [stock * c % total for c in claims]
    left = stock - sum(shares)
    assert 0 <= left <= 1, "two claimants leave at most one unit"
    if left and parts[0] != parts[1]:
        shares[0 if parts[0] > parts[1] else 1] += 1
    return shares[0], shares[1]


def harvest(state: SimState, run: RunState, phase: int = Phase.HARVEST) -> list[HarvestEvent]:
    tick = state.tick
    resources = state.config.balance.resources
    events: list[HarvestEvent] = []
    blocked_at = [enemy_soldier_nodes(state, side) for side in SIDES]
    for node in range(NODE_COUNT):
        max_harvest = resources[node].max_harvest
        if max_harvest == 0:
            continue
        workers = [
            sum(1 for u in state.colonies[side].units if u.caste == WORKER and u.node == node)
            for side in SIDES
        ]
        if not any(workers):
            continue
        demand = [min(w, max_harvest) for w in workers]
        blocked = [node in blocked_at[side] for side in SIDES]
        claim = [0 if blocked[side] else demand[side] for side in SIDES]
        before = state.node_food[node]
        got = split_contested(before, (claim[0], claim[1]))
        state.node_food[node] = before - got[0] - got[1]
        for side in SIDES:
            state.colonies[side].food += got[side]
            _add_window(run, side, harvest=got[side])
            if workers[side]:
                events.append(HarvestEvent(
                    tick=tick, phase=int(phase), side=side, node=node, workers=workers[side],
                    demand=demand[side], blocked=blocked[side], claim=claim[side],
                    harvested=got[side], stock_before=before, stock_after=state.node_food[node]))
    return events


# --- Upkeep / starvation -----------------------------------------------------------------------

def starvation_victim_key(home: int, unit: Unit) -> tuple[int, int, int]:
    """Soldiers first; then home, other nodes, in transit; then ``unit_id``."""
    where = 2 if unit.node == NOWHERE else 0 if unit.node == home else 1
    return (0 if unit.caste == SOLDIER else 1, where, unit.unit_id)


def upkeep(state: SimState, run: RunState, phase: int = Phase.UPKEEP
           ) -> list[UpkeepEvent | StarvationEvent]:
    tick = state.tick
    bal = state.config.balance
    cost = bal.starvation_unit_cost
    upkeep_events: list[UpkeepEvent] = []
    deaths: list[StarvationEvent] = []
    for side in SIDES:
        colony = state.colonies[side]
        workers, soldiers = _adults(colony, WORKER), _adults(colony, SOLDIER)
        remainder_before = colony.upkeep_remainder10
        due, remainder_after = divmod(
            remainder_before + bal.worker_upkeep_tenths * workers
            + bal.soldier_upkeep_tenths * soldiers, 10)
        food_before = colony.food
        paid = min(due, food_before)
        missing = due - paid
        colony.food = food_before - paid
        colony.upkeep_remainder10 = remainder_after
        debt_before = debt = colony.starvation_debt
        reset = False
        victim_count = 0
        if missing:
            debt += missing
            home = home_node(state, side)
            order = sorted(colony.units, key=lambda u: starvation_victim_key(home, u))
            while debt >= cost and order:
                victim = order.pop(0)
                before_pay = debt
                debt -= cost
                colony.units.remove(victim)
                deaths.append(StarvationEvent(
                    tick=tick, phase=int(phase), side=side, seq=victim_count,
                    unit_id=victim.unit_id, caste=victim.caste, hp=victim.hp, node=victim.node,
                    to_node=victim.to_node, debt_before=before_pay, debt_after=debt))
                victim_count += 1
        elif due > 0 and debt:
            debt, reset = 0, True  # a fully paid upkeep ends the old episode of hunger
        colony.starvation_debt = debt
        _add_window(run, side, upkeep_paid=paid, debt_delta=debt - debt_before)
        upkeep_events.append(UpkeepEvent(
            tick=tick, phase=int(phase), side=side, adult_workers=workers, adult_soldiers=soldiers,
            remainder_before=remainder_before, remainder_after=remainder_after, due=due,
            paid=paid, missing=missing, food_before=food_before, food_after=colony.food,
            debt_before=debt_before, debt_after=debt, deaths=victim_count, debt_reset=reset))
    return [*upkeep_events, *deaths]


# --- Lay -----------------------------------------------------------------------------------------

def lay(state: SimState, run: RunState, phase: int = Phase.LAY) -> list[LayEvent]:
    tick = state.tick
    bal = state.config.balance
    events: list[LayEvent] = []
    for side in SIDES:
        colony = state.colonies[side]
        if colony.queen_hp <= 0 or colony.lay_order == LAY_NONE or tick < colony.next_lay_tick:
            continue
        caste = WORKER if colony.lay_order == LAY_WORKER else SOLDIER
        unit_cost = bal.worker_egg_cost if caste == WORKER else bal.soldier_egg_cost
        delay = bal.worker_hatch_delay if caste == WORKER else bal.soldier_hatch_delay
        total = bal.eggs_per_lay * unit_cost
        food_before = colony.food
        if food_before < total:  # the whole batch or nothing; the cooldown is not spent
            events.append(LayEvent(
                tick=tick, phase=int(phase), side=side, laid=False, caste=caste,
                eggs=bal.eggs_per_lay, unit_cost=unit_cost, total_cost=total,
                food_before=food_before, food_after=food_before, unit_ids=(),
                hatch_tick=NOWHERE, next_lay_tick=colony.next_lay_tick))
            continue
        colony.food = food_before - total
        ids = tuple(state.allocate_unit_id() for _ in range(bal.eggs_per_lay))
        for unit_id in ids:  # new ids exceed every existing id, so the brood stays sorted
            colony.brood.append(BroodRecord(unit_id, caste, tick, tick + delay))
        colony.next_lay_tick = tick + bal.lay_interval_ticks
        _add_window(run, side, egg_spend=total)
        events.append(LayEvent(
            tick=tick, phase=int(phase), side=side, laid=True, caste=caste,
            eggs=bal.eggs_per_lay, unit_cost=unit_cost, total_cost=total,
            food_before=food_before, food_after=colony.food, unit_ids=ids,
            hatch_tick=tick + delay, next_lay_tick=colony.next_lay_tick))
    return events


# --- Regeneration ----------------------------------------------------------------------------------

def regenerate(state: SimState, phase: int = Phase.REGENERATION) -> list[RegenerateEvent]:
    events: list[RegenerateEvent] = []
    for node, res in enumerate(state.config.balance.resources):
        if res.capacity == 0:
            continue
        before = state.node_food[node]
        after = min(res.capacity, before + res.regen)
        state.node_food[node] = after
        events.append(RegenerateEvent(
            tick=state.tick, phase=int(phase), node=node, before=before, regen=res.regen,
            added=after - before, after=after, capacity=res.capacity))
    return events
