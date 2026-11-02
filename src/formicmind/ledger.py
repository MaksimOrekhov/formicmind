"""Explicit food ledger, per-tick transition checks and economy statistics (FOR-12).

The simulator state keeps no accumulators beyond FOR-11's ``WindowStats``; everything here is
derived from the typed event stream, so the omniscient log is enough to audit (and, later, to
replay) the economy without re-implementing it.

Food conservation (the *ledger*), for one tick and for any number of ticks::

    sum(node_food) + sum(colony.food)  ==  initial + regen_added - upkeep_paid - egg_spend

``Harvest`` only moves food from a node to a colony, ``Upkeep`` and ``Lay`` destroy it,
``Regeneration`` creates it (capacity-limited). Nothing else changes any food value.

``check_transition(before, after, events)`` verifies one executed tick against that ledger and
against the unit rules (ids, brood -> adult, new adults' place). Since FOR-13 an adult can also vanish in
Combat, and hp and ``queen_hp`` can fall - but only by *recorded* combat damage:

* every ``wound``/``death`` is exactly ``hp_before - damage == hp_after`` for a unit that really had
  ``hp_before``; a unit's hp after the tick is its hp before minus its recorded damage (newborns start at full hp);
* a ``queen_damage`` is the whole difference of ``queen_hp``, which never grows;
* damage conservation per ``combat`` summary: ``damage == applied + wasted``, ``applied`` equals what the
  unit/queen events of that node and attacker add up to, ``applied == min(damage, target_hp_total)`` and
  damage may be wasted only if the whole target category was exhausted - it is never carried over;
* a combat death is never also a starvation death, and a removal needs one of the two events;
* a tick cut at QueenTerminal keeps ``state.tick == t`` and contains no event of a later phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .events import (
    TARGET_QUEEN,
    Event,
    Phase,
)
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


@dataclass(slots=True)
class CombatSideTotals:
    """What one side did (attacker view) and suffered (defender view) in Combat."""

    attacks: int = 0  # combat summaries as attacker
    damage: int = 0  # computed damage as attacker
    applied: int = 0
    wasted: int = 0
    soldiers_killed: int = 0  # enemy soldiers killed by this side
    workers_killed: int = 0
    wounds_dealt: int = 0
    queen_damage_dealt: int = 0
    lost_soldiers: int = 0  # own units killed
    lost_workers: int = 0
    queen_hp_lost: int = 0


@dataclass(slots=True)
class CombatTotals:
    sides: list[CombatSideTotals] = field(default_factory=lambda: [CombatSideTotals(), CombatSideTotals()])

    def add(self, events: "list[Event] | tuple[Event, ...]") -> None:
        for e in events:
            kind = e.kind
            if kind == "combat":
                s = self.sides[e.attacker_side]
                s.attacks += 1
                s.damage += e.damage
                s.applied += e.applied
                s.wasted += e.wasted
            elif kind == "wound":
                self.sides[e.attacker_side].wounds_dealt += 1
            elif kind == "death":
                killer, victim = self.sides[e.attacker_side], self.sides[e.side]
                if e.caste == SOLDIER:
                    killer.soldiers_killed += 1
                    victim.lost_soldiers += 1
                else:
                    killer.workers_killed += 1
                    victim.lost_workers += 1
            elif kind == "queen_damage":
                self.sides[e.attacker_side].queen_damage_dealt += e.damage
                self.sides[e.side].queen_hp_lost += e.damage


def combat_totals(events: "list[Event] | tuple[Event, ...]") -> CombatTotals:
    totals = CombatTotals()
    totals.add(events)
    return totals


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
    cut = any(e.kind == "terminal" and e.phase == Phase.QUEEN_TERMINAL for e in events)
    _require(after.tick == (t if cut else t + 1),
             "a tick advances the clock by exactly one (a tick cut at QueenTerminal does not advance it)")
    if cut:
        _require(all(int(e.phase) <= Phase.QUEEN_TERMINAL for e in events),
                 "no phase after QueenTerminal may run once a queen died")
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

    # --- combat: damage conservation
    _check_combat(before, after, events)

    # --- units: ids, brood -> adult, removal only by starvation or a recorded combat death
    hatched = {e.unit_id: e for e in events if e.kind == "hatch"}
    starved = {e.unit_id for e in events if e.kind == "starvation"}
    killed = {e.unit_id for e in events if e.kind == "death"}
    damaged: dict[int, int] = {}
    for e in events:
        if e.kind in ("wound", "death"):
            damaged[e.unit_id] = damaged.get(e.unit_id, 0) + e.damage
    _require(not (starved & killed), "a unit cannot die both in combat and of starvation")
    laid = {uid for e in events if e.kind == "lay" and e.laid for uid in e.unit_ids}
    for side in SIDES:
        b, a = before.colonies[side], after.colonies[side]
        ripe = {r.unit_id for r in b.brood if r.hatch_tick == t}
        mine = {uid for uid, e in hatched.items() if e.side == side}
        _require(mine == ripe, f"colony {side}: exactly the ripe brood must hatch")
        adults_before = {u.unit_id for u in b.units}
        adults_after = {u.unit_id for u in a.units}
        gone = {e.unit_id for e in events if e.kind == "starvation" and e.side == side}
        gone |= {e.unit_id for e in events if e.kind == "death" and e.side == side}
        _require(
            adults_after == (adults_before | mine) - gone,
            f"colony {side}: adults may only appear by hatching and vanish by starvation or combat death",
        )
        _require(not (gone - (adults_before | mine)), f"colony {side}: removed a unit that did not exist")
        brood_after = {r.unit_id for r in a.brood}
        mine_laid = {uid for e in events if e.kind == "lay" and e.laid and e.side == side
                     for uid in e.unit_ids}
        _require(
            brood_after == ({r.unit_id for r in b.brood} - ripe) | mine_laid,
            f"colony {side}: brood must lose exactly the hatched and gain exactly the laid eggs",
        )
        _require(not (mine_laid & (adults_before | adults_after)), f"colony {side}: egg id reuse")
        # queen: only recorded combat damage, never healing
        q_loss = sum(e.damage for e in events if e.kind == "queen_damage" and e.side == side)
        _require(a.queen_hp <= b.queen_hp, f"colony {side}: queen_hp may only fall")
        _require(a.queen_hp == b.queen_hp - q_loss, f"colony {side}: queen_hp does not match queen_damage")
        home = home_node(after, side)
        survivors = {u.unit_id: u for u in a.units}
        old = {u.unit_id: u for u in b.units}
        for uid, unit in survivors.items():
            if uid in old:
                _require(unit.caste == old[uid].caste, f"unit {uid}: caste changed")
                _require(unit.hp == old[uid].hp - damaged.get(uid, 0),
                         f"unit {uid}: hp changed without recorded combat damage")
        for uid, e in hatched.items():
            if e.side != side or uid in starved or uid in killed:
                continue
            unit = survivors[uid]
            full = bal.worker_hp if unit.caste == WORKER else bal.soldier_hp
            _require(unit.hp == full - damaged.get(uid, 0) and unit.caste == e.caste,
                     f"unit {uid}: newborn must have full hp minus recorded combat damage")
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


def _check_combat(before: SimState, after: SimState, events: "list[Event] | tuple[Event, ...]") -> None:
    """Per-event arithmetic and damage conservation of the Combat phase (see the module docstring)."""
    bal = after.config.balance
    for e in events:
        if e.kind in ("wound", "death"):
            _require(e.damage > 0 and e.hp_before - e.damage == e.hp_after,
                     f"unit {e.unit_id}: hp_before - damage must equal hp_after")
            _require((e.hp_after == 0) == (e.kind == "death"),
                     f"unit {e.unit_id}: a death is exactly hp_after == 0")
            _require(e.attacker_side == 1 - e.side, f"unit {e.unit_id}: only the enemy can hurt a unit")
        elif e.kind == "queen_damage":
            _require(e.damage > 0 and e.queen_hp_before - e.damage == e.queen_hp_after
                     and e.queen_hp_after >= 0, f"queen {e.side}: inconsistent queen_damage")
            _require(e.node == home_node(after, e.side) and e.attacker_side == 1 - e.side,
                     f"queen {e.side}: she is hurt only in her own home by the enemy")
    for e in events:
        if e.kind != "combat":
            continue
        tag = f"combat at node {e.node} by side {e.attacker_side}"
        _require(e.defender_side == 1 - e.attacker_side, f"{tag}: defender must be the other side")
        base = e.attacker_soldiers * bal.soldier_damage
        _require(e.base_damage == base, f"{tag}: base damage must be soldiers * soldier_damage")
        _require(e.home_bonus == (e.node == home_node(after, e.attacker_side)),
                 f"{tag}: the home bonus belongs to the attackers' own home only")
        expected = base * bal.home_defense_num // bal.home_defense_den if e.home_bonus else base
        _require(e.damage == expected, f"{tag}: damage must follow the exact integer home-defence rule")
        _require(e.applied + e.wasted == e.damage and e.applied >= 0 and e.wasted >= 0,
                 f"{tag}: damage must equal applied + wasted")
        _require(e.applied == min(e.damage, e.target_hp_total), f"{tag}: applied must be min(damage, pool)")
        hurt = [x for x in events if x.kind in ("wound", "death")
                and x.node == e.node and x.attacker_side == e.attacker_side]
        queen = [x for x in events if x.kind == "queen_damage"
                 and x.node == e.node and x.attacker_side == e.attacker_side]
        if e.target == TARGET_QUEEN:
            _require(not hurt and sum(x.damage for x in queen) == e.applied,
                     f"{tag}: queen damage must equal what was applied")
            if e.wasted:
                _require(queen and queen[0].queen_hp_after == 0, f"{tag}: damage wasted on a living queen")
        else:
            _require(not queen and sum(x.damage for x in hurt) == e.applied,
                     f"{tag}: unit damage must equal what was applied")
            _require(all(x.caste == (SOLDIER if e.target == "soldier" else WORKER) for x in hurt),
                     f"{tag}: damage went to another caste than the target category")
            if e.wasted:  # nothing is carried: the whole category must have been killed
                _require(sum(1 for x in hurt if x.kind == "death") == e.target_units,
                         f"{tag}: damage was wasted while part of the category survived")
            # sequential pour in ascending unit_id: only the last damaged unit may be left alive
            ordered = sorted(hurt, key=lambda x: x.unit_id)
            _require(all(x.kind == "death" for x in ordered[:-1]),
                     f"{tag}: a unit before the last damaged one survived (damage is poured in unit_id order)")
