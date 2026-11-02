"""Mutable game-model state (``SimState``) and its parts.

``SimState`` holds only game-model state: node food, both colonies (units,
brood, standing orders, private ``IntelMemory``) and the shared unit-id
allocator. Policy RNG streams are *not* part of it; they live in ``RunState``.

Representation choices (engineering choices, not FOR-1 decisions):

* A "group" is not an object. Units are individual records
  ``(unit_id, caste, hp)`` with a location (node, or edge + arrival tick) and a
  target node; groups and transit aggregates are derived views.
* Unit ids are shared and stable across both colonies. An egg carries the id of
  the adult it will become (allocated once, at lay time).
* "Nowhere" is ``-1`` (no ``None`` in canonical data).
* Standing orders are stored exactly as ActionV2 expresses them, i.e. in the
  side's own canonical node frame (see ``rules.local_to_world``).
* ``IntelMemory`` is indexed by world node id, one per side, never shared.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import SimConfig

NOWHERE = -1


@dataclass(slots=True)
class Unit:
    unit_id: int
    caste: int
    hp: int
    node: int  # node id when stationed, NOWHERE when on an edge
    from_node: int  # edge origin when in transit, else NOWHERE
    to_node: int  # edge destination when in transit, else NOWHERE
    arrival_tick: int  # arrival tick when in transit, else NOWHERE
    target_node: int  # world node the unit is assigned to

    def clone(self) -> "Unit":
        return Unit(
            self.unit_id, self.caste, self.hp, self.node,
            self.from_node, self.to_node, self.arrival_tick, self.target_node,
        )


@dataclass(slots=True)
class BroodRecord:
    unit_id: int  # id of the future adult
    caste: int
    laid_tick: int
    hatch_tick: int

    def clone(self) -> "BroodRecord":
        return BroodRecord(self.unit_id, self.caste, self.laid_tick, self.hatch_tick)


@dataclass(slots=True)
class NodeIntel:
    seen_tick: int = NOWHERE  # NOWHERE: never observed (all values 0)
    food_last: int = 0
    enemy_workers_last: int = 0
    enemy_soldiers_last: int = 0

    def clone(self) -> "NodeIntel":
        return NodeIntel(
            self.seen_tick, self.food_last, self.enemy_workers_last, self.enemy_soldiers_last
        )


@dataclass(slots=True)
class IntelMemory:
    nodes: list[NodeIntel]  # by world node id
    enemy_queen_hp_last: int = 0
    enemy_queen_seen_tick: int = NOWHERE

    def clone(self) -> "IntelMemory":
        return IntelMemory(
            [n.clone() for n in self.nodes],
            self.enemy_queen_hp_last,
            self.enemy_queen_seen_tick,
        )


@dataclass(slots=True)
class ColonyState:
    side: int
    food: int
    queen_hp: int
    next_lay_tick: int
    upkeep_remainder10: int
    starvation_debt: int
    # standing orders (ActionV2 form, own-side frame)
    lay_order: int
    forage_q: list[int]  # 6 shares summing to Q_TOTAL: home, N1, N2, C, N4, N5
    army_choice: int
    avoid_center: bool
    units: list[Unit]  # sorted by unit_id
    brood: list[BroodRecord]  # sorted by unit_id
    intel: IntelMemory

    def clone(self) -> "ColonyState":
        return ColonyState(
            side=self.side,
            food=self.food,
            queen_hp=self.queen_hp,
            next_lay_tick=self.next_lay_tick,
            upkeep_remainder10=self.upkeep_remainder10,
            starvation_debt=self.starvation_debt,
            lay_order=self.lay_order,
            forage_q=list(self.forage_q),
            army_choice=self.army_choice,
            avoid_center=self.avoid_center,
            units=[u.clone() for u in self.units],
            brood=[b.clone() for b in self.brood],
            intel=self.intel.clone(),
        )


@dataclass(slots=True)
class SimState:
    config: SimConfig = field(repr=False)
    tick: int
    node_food: list[int]  # by node id
    colonies: list[ColonyState]  # index == side
    next_unit_id: int

    def allocate_unit_id(self) -> int:
        unit_id = self.next_unit_id
        self.next_unit_id += 1
        return unit_id

    def clone(self) -> "SimState":
        return SimState(
            config=self.config,
            tick=self.tick,
            node_food=list(self.node_food),
            colonies=[c.clone() for c in self.colonies],
            next_unit_id=self.next_unit_id,
        )
