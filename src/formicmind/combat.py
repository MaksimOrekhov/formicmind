"""Simultaneous combat (FOR-1 v0.2.1 section 6; FOR-13). No RNG, no float, no side priority.

Phase ``Combat`` runs after Arrival and the ``POST_ARRIVAL_PRE_COMBAT`` sensor and before QueenTerminal and
Harvest. Combat exists only in a node, between *stationed* units; nothing happens on edges.

For every node and for every side with at least one stationed soldier (counted in the **pre-combat** state
of the tick):

* ``base = soldiers * soldier_damage``; soldiers standing in their *own home* deal
  ``base * home_defense_num // home_defense_den`` (``floor(1.25 * base)`` by default, the total is rounded,
  not each soldier). The bonus belongs to the defender's home only; it never helps an attacker.
* The damage goes to **one** target category of the enemy units stationed in that node, by formal priority:
  enemy soldiers, else enemy workers, else the enemy queen - and the queen only in the enemy's own home and
  only if no enemy soldier or worker stands there. Workers deal no damage but are the last shield of the queen.
* Inside the category the damage is poured over the units in ascending ``unit_id`` order, each unit taking
  at most its current hp. Damage left over when the category is exhausted is **wasted**: it is not carried
  to the next category, to the queen or to the next phase/tick. A unit with ``hp == 0`` is removed.

Both sides' damage is computed from the same pre-combat state and applied afterwards, so the outcome does not
depend on which side is processed first: a soldier killed in this phase still deals its blow. Every damaged
unit leaves one ``wound`` (survived) or ``death`` event, every attack one ``combat`` summary and every blow at
a queen one ``queen_damage`` event (absolute ``queen_hp_after``).
"""

from __future__ import annotations

from .events import (
    TARGET_QUEEN,
    TARGET_SOLDIER,
    TARGET_WORKER,
    CombatEvent,
    DeathEvent,
    Phase,
    QueenDamageEvent,
    WoundEvent,
)
from .intel import home_node
from .model import NOWHERE, SimState, Unit
from .rules import NODE_COUNT, SIDES, SOLDIER

CombatEvents = list[CombatEvent | WoundEvent | DeathEvent | QueenDamageEvent]


def attack_damage(soldiers: int, soldier_damage: int, at_home: bool, num: int, den: int) -> tuple[int, int]:
    """``(base, damage)`` of ``soldiers`` living soldiers; the home bonus rounds the total down."""
    base = soldiers * soldier_damage
    return base, (base * num // den if at_home else base)


def resolve_combat(state: SimState, phase: int = Phase.COMBAT) -> CombatEvents:
    tick = state.tick
    bal = state.config.balance
    homes = [home_node(state, side) for side in SIDES]

    # Pre-combat index of stationed units, ascending unit_id inside every list (units are kept sorted).
    soldiers: list[list[list[Unit]]] = [[[] for _ in range(NODE_COUNT)] for _ in SIDES]
    workers: list[list[list[Unit]]] = [[[] for _ in range(NODE_COUNT)] for _ in SIDES]
    for side in SIDES:
        for unit in state.colonies[side].units:
            if unit.node == NOWHERE:
                continue  # no combat on edges
            (soldiers if unit.caste == SOLDIER else workers)[side][unit.node].append(unit)

    summaries: list[CombatEvent] = []
    hits: list[tuple[Unit, int, int, int]] = []  # (unit, attacker side, node, damage taken)
    queen_hits: list[tuple[int, int, int, int]] = []  # (queen owner, attacker side, node, damage taken)
    for node in range(NODE_COUNT):
        for attacker in SIDES:
            count = len(soldiers[attacker][node])
            if not count:
                continue
            defender = 1 - attacker
            at_home = node == homes[attacker]
            base, damage = attack_damage(
                count, bal.soldier_damage, at_home, bal.home_defense_num, bal.home_defense_den)
            if damage <= 0:
                continue
            if soldiers[defender][node]:
                target, units = TARGET_SOLDIER, soldiers[defender][node]
            elif workers[defender][node]:
                target, units = TARGET_WORKER, workers[defender][node]
            elif node == homes[defender] and state.colonies[defender].queen_hp > 0:
                target, units = TARGET_QUEEN, []
            else:
                continue  # nobody to hit in this node
            if target == TARGET_QUEEN:
                pool = state.colonies[defender].queen_hp
                applied = min(damage, pool)
                queen_hits.append((defender, attacker, node, applied))
                n_targets = 1
            else:
                pool = sum(u.hp for u in units)
                applied = min(damage, pool)
                remaining = applied
                for unit in units:  # ascending unit_id; the remainder is never carried over
                    taken = min(unit.hp, remaining)
                    if taken:
                        hits.append((unit, attacker, node, taken))
                        remaining -= taken
                n_targets = len(units)
            summaries.append(CombatEvent(
                tick=tick, phase=int(phase), node=node, attacker_side=attacker, defender_side=defender,
                attacker_soldiers=count, base_damage=base, home_bonus=at_home, damage=damage,
                target=target, target_units=n_targets, target_hp_total=pool, applied=applied,
                wasted=damage - applied))

    # Apply only now: every number above came from the pre-combat state.
    wounds: list[WoundEvent] = []
    deaths: list[DeathEvent] = []
    for unit, attacker, node, taken in hits:
        owner = 1 - attacker
        before = unit.hp
        unit.hp = before - taken
        if unit.hp == 0:
            deaths.append(DeathEvent(
                tick=tick, phase=int(phase), side=owner, unit_id=unit.unit_id, caste=unit.caste,
                node=node, attacker_side=attacker, hp_before=before, damage=taken, hp_after=0))
        else:
            wounds.append(WoundEvent(
                tick=tick, phase=int(phase), side=owner, unit_id=unit.unit_id, caste=unit.caste,
                node=node, attacker_side=attacker, hp_before=before, damage=taken, hp_after=unit.hp))
    for side in SIDES:
        dead = {e.unit_id for e in deaths if e.side == side}
        if dead:
            colony = state.colonies[side]
            colony.units[:] = [u for u in colony.units if u.unit_id not in dead]
    queen_events: list[QueenDamageEvent] = []
    for owner, attacker, node, taken in queen_hits:
        colony = state.colonies[owner]
        before = colony.queen_hp
        colony.queen_hp = before - taken
        queen_events.append(QueenDamageEvent(
            tick=tick, phase=int(phase), side=owner, node=node, attacker_side=attacker, damage=taken,
            queen_hp_before=before, queen_hp_after=colony.queen_hp))
    wounds.sort(key=lambda e: (e.unit_id, e.side))
    deaths.sort(key=lambda e: (e.unit_id, e.side))
    queen_events.sort(key=lambda e: e.side)
    return [*summaries, *wounds, *deaths, *queen_events]

