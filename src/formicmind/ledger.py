"""Explicit food ledger, per-tick transition checks and economy statistics (FOR-12).

The simulator state keeps no accumulators beyond FOR-11's ``WindowStats``; everything here is
derived from the typed event stream, so the omniscient log is enough to audit (and, later, to
replay) the economy without re-implementing it.

Food conservation (the *ledger*), for one tick and for any number of ticks::

    sum(node_food) + sum(colony.food)  ==  initial + regen_added - upkeep_paid - egg_spend

``Harvest`` only moves food from a node to a colony, ``Upkeep`` and ``Lay`` destroy it,
``Regeneration`` creates it (capacity-limited). Nothing else changes any food value.

``check_transition(before, after, events)`` verifies one executed tick against that ledger and
against the unit rules (ids, brood -> adult, no removal except starvation, new adults' place).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .events import Event
from .intel import home_node
from .invariants import InvariantError, validate_state
from .model import NOWHERE, SimState
from .rules import NODE_COUNT, SIDES, SOLDIER, WORKER


def total_food(state: SimState) -> int:
    return sum(state.node_food) + sum(c.food for c in state.colonies)


@dataclass(slots=True)
class SideTotals:
    harvested: int = 0
    harvest_blocked: int = 0  # (node, tick) harvests blocked by enemy soldiers
    upkeep_due: int = 0
    upkeep_paid: int = 0
    upkeep_missing: int = 0
    debt_resets: int = 0
    egg_spend: int = 0
    eggs_laid: int = 0
    lay_batches: int = 0
    lay_blocked_food: int = 0  # due lay attempts refused for lack of food
    hatched: list[int] = field(default_factory=lambda: [0, 0])  # by caste
    starved: list[int] = field(default_factory=lambda: [0, 0])  # by caste


@dataclass(slots=True)
class EconomyTotals:
    sides: list[SideTotals] = field(default_factory=lambda: [SideTotals(), SideTotals()])
    regen_added: int = 0

    def add(self, events: "list[Event] | tuple[Event, ...]") -> None:
        for e in events:
            kind = e.kind
            if kind == "harvest":
                s = self.sides[e.side]
                s.harvested += e.harvested
                s.harvest_blocked += 1 if e.blocked else 0
            elif kind == "upkeep":
                s = self.sides[e.side]
                s.upkeep_due += e.due
                s.upkeep_paid += e.paid
                s.upkeep_missing += e.missing
                s.debt_resets += 1 if e.debt_reset else 0
            elif kind == "lay":
                s = self.sides[e.side]
                if e.laid:
                    s.egg_spend += e.total_cost
                    s.eggs_laid += e.eggs
                    s.lay_batches += 1
                else:
                    s.lay_blocked_food += 1
            elif kind == "hatch":
                self.sides[e.side].hatched[e.caste] += 1
            elif kind == "starvation":
                self.sides[e.side].starved[e.caste] += 1
            elif kind == "regenerate":
                self.regen_added += e.added

    def expected_total_food(self, initial: int) -> int:
        return (initial + self.regen_added
                - sum(s.upkeep_paid + s.egg_spend for s in self.sides))


def economy_totals(events: "list[Event] | tuple[Event, ...]") -> EconomyTotals:
    totals = EconomyTotals()
    totals.add(events)
    return totals


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvariantError(message)


def check_transition(before: SimState, after: SimState, events: "list[Event] | tuple[Event, ...]"
                     ) -> None:
    """Verify one executed tick ``before -> after`` against its events (raises ``InvariantError``)."""
    t = before.tick
    _require(after.tick == t + 1, "a tick advances the clock by exactly one")
    validate_state(after)
    bal = after.config.balance

    totals = economy_totals(events)
    # --- food: global ledger, per colony, per node
    _require(
        total_food(after) == totals.expected_total_food(total_food(before)),
        "food ledger broken: node + warehouse food must equal start + regeneration - upkeep - eggs",
    )
    for side in SIDES:
        s = totals.sides[side]
        _require(
            after.colonies[side].food == before.colonies[side].food + s.harvested - s.upkeep_paid
            - s.egg_spend,
            f"colony {side}: warehouse food does not match harvest - upkeep - egg spend",
        )
    harvested_at = [0] * NODE_COUNT
    regen_at = [0] * NODE_COUNT
    for e in events:
        if e.kind == "harvest":
            harvested_at[e.node] += e.harvested
        elif e.kind == "regenerate":
            regen_at[e.node] += e.added
    for node in range(NODE_COUNT):
        _require(
            after.node_food[node] == before.node_food[node] - harvested_at[node] + regen_at[node],
            f"node {node}: stock does not match harvest and regeneration",
        )

    # --- units: ids, brood -> adult, removal only by starvation
    hatched = {e.unit_id: e for e in events if e.kind == "hatch"}
    starved = {e.unit_id for e in events if e.kind == "starvation"}
    laid = {uid for e in events if e.kind == "lay" and e.laid for uid in e.unit_ids}
    for side in SIDES:
        b, a = before.colonies[side], after.colonies[side]
        ripe = {r.unit_id for r in b.brood if r.hatch_tick == t}
        mine = {uid for uid, e in hatched.items() if e.side == side}
        _require(mine == ripe, f"colony {side}: exactly the ripe brood must hatch")
        adults_before = {u.unit_id for u in b.units}
        adults_after = {u.unit_id for u in a.units}
        gone = {e.unit_id for e in events if e.kind == "starvation" and e.side == side}
        _require(
            adults_after == (adults_before | mine) - gone,
            f"colony {side}: adults may only appear by hatching and vanish by starvation",
        )
        _require(not (gone - (adults_before | mine)), f"colony {side}: starved a unit that did not exist")
        brood_after = {r.unit_id for r in a.brood}
        mine_laid = {uid for e in events if e.kind == "lay" and e.laid and e.side == side
                     for uid in e.unit_ids}
        _require(
            brood_after == ({r.unit_id for r in b.brood} - ripe) | mine_laid,
            f"colony {side}: brood must lose exactly the hatched and gain exactly the laid eggs",
        )
        _require(not (mine_laid & (adults_before | adults_after)), f"colony {side}: egg id reuse")
        home = home_node(after, side)
        survivors = {u.unit_id: u for u in a.units}
        old = {u.unit_id: u for u in b.units}
        for uid, unit in survivors.items():
            if uid in old:
                _require(unit.hp == old[uid].hp and unit.caste == old[uid].caste,
                         f"unit {uid}: hp/caste changed without combat")
        for uid, e in hatched.items():
            if e.side != side or uid in starved:
                continue
            unit = survivors[uid]
            full = bal.worker_hp if unit.caste == WORKER else bal.soldier_hp
            _require(unit.hp == full and unit.caste == e.caste, f"unit {uid}: newborn must have full hp")
            _require(unit.target_node == e.target_node, f"unit {uid}: newborn target changed")
            at_home = unit.node == home
            left_home = unit.node == NOWHERE and unit.from_node == home
            _require(at_home or left_home,
                     f"unit {uid}: newborn must stand at home or have left it this tick")
            if unit.node == home and unit.target_node != home:
                # allowed only when the home node holds a stationed enemy soldier (blocked)
                enemy_here = any(
                    v.node == home and v.caste == SOLDIER for v in after.colonies[1 - side].units)
                _require(enemy_here, f"unit {uid}: newborn with a remote target must depart")
    all_ids = [u.unit_id for c in after.colonies for u in c.units]
    all_ids += [r.unit_id for c in after.colonies for r in c.brood]
    _require(len(all_ids) == len(set(all_ids)), "unit ids must stay unique")
    _require(laid <= set(all_ids), "laid eggs must exist in a brood list")
