"""Typed events: movement (``order``, ``retarget``, ``depart``, ``arrive``, FOR-11), economy
(``hatch``, ``harvest``, ``upkeep``, ``starvation``, ``lay``, ``regenerate``, FOR-12) and combat /
terminal (``combat``, ``wound``, ``death``, ``queen_damage``, ``terminal``, FOR-13).

Boundary and ordering
---------------------
``step_tick`` is called with ``state.tick == t``: the state *at the start of tick t*,
before ``PRE_POLICY``. It executes the phases of tick ``t`` and leaves
``state.tick == t + 1`` (the start of the next tick, i.e. after tick ``t``'s
AutomaticDepart). Every event carries ``tick == t`` (the tick in which it happened)
and the ``Phase`` that produced it. A ``depart`` at tick ``t`` over an edge of length
``d`` carries ``arrival_tick == t + d``; the matching ``arrive`` carries ``tick == t + d``.

Events are returned in execution order, which equals sorting by ``event_sort_key``
(``(phase, kind rank, *event.sort_tail())``): phases in tick order; inside a phase, kinds in
rank order. Movement kinds sort by ``(unit_id, side)``: all ``order`` events (by side), then
``retarget``, ``depart``, ``arrive`` by ascending unit id (unit ids are global and stable).
Economy kinds: ``hatch`` by unit id; ``harvest`` by ``(node, side)``; ``upkeep`` by side;
``starvation`` by ``(side, seq)`` where ``seq`` is the victim order (soldiers first, see
``economy.starvation_victim_key``), which is *not* ascending unit id; ``lay`` by side;
``regenerate`` by node. Combat kinds: ``combat`` (one summary per node and attacking side) by
``(node, attacker_side)``, then ``wound`` and ``death`` by ``(unit_id, side)``, then ``queen_damage`` by side;
``terminal`` closes the episode. Events hold only data the omniscient log may hold; nothing here is
ever shown to a policy.

``to_canonical`` gives plain int/str/bool/list/dict values (canonical-JSON safe); the Replay v1 writer
(``formicmind.replay``, FOR-15, docs/replay_v1.md) records exactly these dicts in this order.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, ClassVar

from .canonical import canonical_dumps, domain_hash
from .versions import HASH_DOMAIN_EVENTS


class Phase(enum.IntEnum):
    """Tick phases in FOR-1 v0.2.1 (errata A) order."""

    POLICY = 0
    COMPLETION = 1
    ARRIVAL = 2
    COMBAT = 3
    QUEEN_TERMINAL = 4
    HARVEST = 5
    UPKEEP = 6
    LAY = 7
    REGENERATION = 8
    AUTOMATIC_DEPART = 9
    TERMINAL_LOG = 10


@dataclass(frozen=True, slots=True)
class OrderEvent:
    kind: ClassVar[str] = "order"
    kind_rank: ClassVar[int] = 0
    tick: int
    phase: int
    side: int
    lay_order: int
    forage_share_q: tuple[int, ...]  # own-side frame, as in ActionV2
    army_choice: int
    avoid_center: bool
    army_target_node: int  # world node id
    worker_quota: tuple[int, ...]  # workers per forage category (own-side frame)

    unit_id: ClassVar[int] = -1

    def sort_tail(self) -> tuple[int, ...]:
        return (self.unit_id, self.side)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "tick": self.tick, "phase": int(self.phase), "side": self.side,
            "lay_order": self.lay_order, "forage_share_q": list(self.forage_share_q),
            "army_choice": self.army_choice, "avoid_center": self.avoid_center,
            "army_target_node": self.army_target_node, "worker_quota": list(self.worker_quota),
        }


@dataclass(frozen=True, slots=True)
class RetargetEvent:
    kind: ClassVar[str] = "retarget"
    kind_rank: ClassVar[int] = 1
    tick: int
    phase: int
    side: int
    unit_id: int
    caste: int
    old_target: int
    new_target: int

    def sort_tail(self) -> tuple[int, ...]:
        return (self.unit_id, self.side)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "tick": self.tick, "phase": int(self.phase), "side": self.side,
            "unit_id": self.unit_id, "caste": self.caste,
            "old_target": self.old_target, "new_target": self.new_target,
        }


@dataclass(frozen=True, slots=True)
class DepartEvent:
    kind: ClassVar[str] = "depart"
    kind_rank: ClassVar[int] = 2
    tick: int
    phase: int
    side: int
    unit_id: int
    caste: int
    from_node: int
    to_node: int
    arrival_tick: int
    target_node: int

    def sort_tail(self) -> tuple[int, ...]:
        return (self.unit_id, self.side)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "tick": self.tick, "phase": int(self.phase), "side": self.side,
            "unit_id": self.unit_id, "caste": self.caste, "from_node": self.from_node,
            "to_node": self.to_node, "arrival_tick": self.arrival_tick,
            "target_node": self.target_node,
        }


@dataclass(frozen=True, slots=True)
class ArriveEvent:
    kind: ClassVar[str] = "arrive"
    kind_rank: ClassVar[int] = 3
    tick: int
    phase: int
    side: int
    unit_id: int
    caste: int
    node: int
    from_node: int
    target_node: int

    def sort_tail(self) -> tuple[int, ...]:
        return (self.unit_id, self.side)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "tick": self.tick, "phase": int(self.phase), "side": self.side,
            "unit_id": self.unit_id, "caste": self.caste, "node": self.node,
            "from_node": self.from_node, "target_node": self.target_node,
        }


@dataclass(frozen=True, slots=True)
class HatchEvent:
    """Completion: a brood record became a full-HP adult stationed at its home node."""

    kind: ClassVar[str] = "hatch"
    kind_rank: ClassVar[int] = 4
    tick: int
    phase: int
    side: int
    unit_id: int
    caste: int
    node: int  # home node where the adult appears
    hp: int
    laid_tick: int
    target_node: int  # standing-order target assigned at hatch (errata E)

    def sort_tail(self) -> tuple[int, ...]:
        return (self.unit_id, self.side)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "tick": self.tick, "phase": int(self.phase), "side": self.side,
            "unit_id": self.unit_id, "caste": self.caste, "node": self.node, "hp": self.hp,
            "laid_tick": self.laid_tick, "target_node": self.target_node,
        }


@dataclass(frozen=True, slots=True)
class HarvestEvent:
    """Harvest of one side at one resource node (emitted for every side with a worker standing there)."""

    kind: ClassVar[str] = "harvest"
    kind_rank: ClassVar[int] = 5
    tick: int
    phase: int
    side: int
    node: int
    workers: int  # workers of this side stationed at the node when the phase ran
    demand: int  # min(workers, max_harvest)
    blocked: bool  # enemy soldiers stand in the node: the side may not harvest (errata B)
    claim: int  # demand actually put into the competition (0 when blocked)
    harvested: int
    stock_before: int  # node stock before the phase (same for both sides)
    stock_after: int  # node stock after both sides harvested

    def sort_tail(self) -> tuple[int, ...]:
        return (self.node, self.side)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "tick": self.tick, "phase": int(self.phase), "side": self.side,
            "node": self.node, "workers": self.workers, "demand": self.demand,
            "blocked": self.blocked, "claim": self.claim, "harvested": self.harvested,
            "stock_before": self.stock_before, "stock_after": self.stock_after,
        }


@dataclass(frozen=True, slots=True)
class UpkeepEvent:
    """Upkeep of one side (every tick, both sides), in tenths of food with an integer remainder."""

    kind: ClassVar[str] = "upkeep"
    kind_rank: ClassVar[int] = 6
    tick: int
    phase: int
    side: int
    adult_workers: int
    adult_soldiers: int
    remainder_before: int  # upkeep_remainder10 before the phase
    remainder_after: int
    due: int  # whole food units owed this tick
    paid: int
    missing: int  # due - paid
    food_before: int
    food_after: int
    debt_before: int
    debt_after: int  # after the deaths and the reset, final value of the tick
    deaths: int
    debt_reset: bool  # a fully paid, non-zero upkeep cleared the old debt (errata D)

    def sort_tail(self) -> tuple[int, ...]:
        return (-1, self.side)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "tick": self.tick, "phase": int(self.phase), "side": self.side,
            "adult_workers": self.adult_workers, "adult_soldiers": self.adult_soldiers,
            "remainder_before": self.remainder_before, "remainder_after": self.remainder_after,
            "due": self.due, "paid": self.paid, "missing": self.missing,
            "food_before": self.food_before, "food_after": self.food_after,
            "debt_before": self.debt_before, "debt_after": self.debt_after,
            "deaths": self.deaths, "debt_reset": self.debt_reset,
        }


@dataclass(frozen=True, slots=True)
class StarvationEvent:
    """One adult removed by starvation (the other way an adult disappears is a combat ``death``)."""

    kind: ClassVar[str] = "starvation"
    kind_rank: ClassVar[int] = 7
    tick: int
    phase: int
    side: int
    seq: int  # 0-based victim order within the side and tick
    unit_id: int
    caste: int
    hp: int
    node: int  # stationed node, or NOWHERE when the victim was in transit
    to_node: int  # edge destination when in transit
    debt_before: int
    debt_after: int  # debt after paying this unit's price

    def sort_tail(self) -> tuple[int, ...]:
        return (self.side, self.seq)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "tick": self.tick, "phase": int(self.phase), "side": self.side,
            "seq": self.seq, "unit_id": self.unit_id, "caste": self.caste, "hp": self.hp,
            "node": self.node, "to_node": self.to_node,
            "debt_before": self.debt_before, "debt_after": self.debt_after,
        }


@dataclass(frozen=True, slots=True)
class LayEvent:
    """A lay attempt that was due: either a whole paid batch, or no egg at all (food short)."""

    kind: ClassVar[str] = "lay"
    kind_rank: ClassVar[int] = 8
    tick: int
    phase: int
    side: int
    laid: bool
    caste: int
    eggs: int  # eggs_per_lay of the batch
    unit_cost: int
    total_cost: int  # eggs * unit_cost, charged once and only when laid
    food_before: int
    food_after: int
    unit_ids: tuple[int, ...]  # empty when not laid
    hatch_tick: int  # NOWHERE (-1) when not laid
    next_lay_tick: int  # after the phase

    def sort_tail(self) -> tuple[int, ...]:
        return (-1, self.side)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "tick": self.tick, "phase": int(self.phase), "side": self.side,
            "laid": self.laid, "caste": self.caste, "eggs": self.eggs,
            "unit_cost": self.unit_cost, "total_cost": self.total_cost,
            "food_before": self.food_before, "food_after": self.food_after,
            "unit_ids": list(self.unit_ids), "hatch_tick": self.hatch_tick,
            "next_lay_tick": self.next_lay_tick,
        }


@dataclass(frozen=True, slots=True)
class RegenerateEvent:
    """Regeneration of one resource node: ``after = min(capacity, before + regen)``."""

    kind: ClassVar[str] = "regenerate"
    kind_rank: ClassVar[int] = 9
    tick: int
    phase: int
    node: int
    before: int  # stock after Harvest
    regen: int
    added: int  # what actually arrived (capacity-limited)
    after: int
    capacity: int

    side: ClassVar[int] = -1

    def sort_tail(self) -> tuple[int, ...]:
        return (self.node, self.side)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "tick": self.tick, "phase": int(self.phase), "node": self.node,
            "before": self.before, "regen": self.regen, "added": self.added, "after": self.after,
            "capacity": self.capacity,
        }


TARGET_SOLDIER = "soldier"
TARGET_WORKER = "worker"
TARGET_QUEEN = "queen"


@dataclass(frozen=True, slots=True)
class CombatEvent:
    """Summary of one attack: the soldiers of ``attacker_side`` standing in ``node`` against one target caste.

    ``damage`` is computed from the pre-combat state of the tick (``attacker_soldiers`` alive when the phase
    began, ``base_damage = soldiers * soldier_damage``, home bonus ``base * num // den`` when the attackers
    stand in their own home). It is applied to exactly one target category (soldiers, else workers, else the
    queen of the defender's own home) in ascending ``unit_id`` order; ``applied`` is what was absorbed and
    ``wasted`` what remained once that category was exhausted. Wasted damage is never carried to the next
    category or to the next tick (FOR-1 section 6).
    """

    kind: ClassVar[str] = "combat"
    kind_rank: ClassVar[int] = 10
    tick: int
    phase: int
    node: int
    attacker_side: int
    defender_side: int
    attacker_soldiers: int
    base_damage: int
    home_bonus: bool  # attackers stand in their own home: floor(base * 5 / 4)
    damage: int
    target: str  # "soldier" | "worker" | "queen"
    target_units: int  # units of the category stationed in the node before the phase (1 for the queen)
    target_hp_total: int  # their summed hp (queen hp for the queen)
    applied: int
    wasted: int

    side: ClassVar[int] = -1

    def sort_tail(self) -> tuple[int, ...]:
        return (self.node, self.attacker_side)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "tick": self.tick, "phase": int(self.phase), "node": self.node,
            "attacker_side": self.attacker_side, "defender_side": self.defender_side,
            "attacker_soldiers": self.attacker_soldiers, "base_damage": self.base_damage,
            "home_bonus": self.home_bonus, "damage": self.damage, "target": self.target,
            "target_units": self.target_units, "target_hp_total": self.target_hp_total,
            "applied": self.applied, "wasted": self.wasted,
        }


@dataclass(frozen=True, slots=True)
class WoundEvent:
    """A unit that took combat damage and survived (``hp_after > 0``). ``side`` owns the wounded unit."""

    kind: ClassVar[str] = "wound"
    kind_rank: ClassVar[int] = 11
    tick: int
    phase: int
    side: int
    unit_id: int
    caste: int
    node: int
    attacker_side: int
    hp_before: int
    damage: int
    hp_after: int

    def sort_tail(self) -> tuple[int, ...]:
        return (self.unit_id, self.side)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "tick": self.tick, "phase": int(self.phase), "side": self.side,
            "unit_id": self.unit_id, "caste": self.caste, "node": self.node,
            "attacker_side": self.attacker_side, "hp_before": self.hp_before,
            "damage": self.damage, "hp_after": self.hp_after,
        }


@dataclass(frozen=True, slots=True)
class DeathEvent:
    """A unit killed in combat (``hp_after == 0``); ``damage == hp_before``. Starvation has its own event."""

    kind: ClassVar[str] = "death"
    kind_rank: ClassVar[int] = 12
    tick: int
    phase: int
    side: int
    unit_id: int
    caste: int
    node: int
    attacker_side: int
    hp_before: int
    damage: int
    hp_after: int

    def sort_tail(self) -> tuple[int, ...]:
        return (self.unit_id, self.side)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "tick": self.tick, "phase": int(self.phase), "side": self.side,
            "unit_id": self.unit_id, "caste": self.caste, "node": self.node,
            "attacker_side": self.attacker_side, "hp_before": self.hp_before,
            "damage": self.damage, "hp_after": self.hp_after,
        }


@dataclass(frozen=True, slots=True)
class QueenDamageEvent:
    """Combat damage to a queen. ``queen_hp_after`` is the absolute value (for a future spectator)."""

    kind: ClassVar[str] = "queen_damage"
    kind_rank: ClassVar[int] = 13
    tick: int
    phase: int
    side: int  # owner of the queen
    node: int  # her home node
    attacker_side: int
    damage: int
    queen_hp_before: int
    queen_hp_after: int

    def sort_tail(self) -> tuple[int, ...]:
        return (-1, self.side)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "tick": self.tick, "phase": int(self.phase), "side": self.side,
            "node": self.node, "attacker_side": self.attacker_side, "damage": self.damage,
            "queen_hp_before": self.queen_hp_before, "queen_hp_after": self.queen_hp_after,
        }


@dataclass(frozen=True, slots=True)
class TerminalEvent:
    """The episode ended. ``phase`` is QueenTerminal (mid-tick) or Terminal/Log (end of a full tick)."""

    kind: ClassVar[str] = "terminal"
    kind_rank: ClassVar[int] = 14
    tick: int
    phase: int
    reason: str  # "queen_death" | "economic_collapse" | "timeout"
    winner: str  # "A" | "B" | "draw"
    sides: tuple[int, ...]  # sides whose queen died / whose economy collapsed (empty for a timeout)
    phases_executed: int  # phases of this tick reached, the terminal one included
    queen_hp: tuple[int, int]  # absolute, by side
    tick_completed: bool  # False: the tick was cut at QueenTerminal and state.tick stays at it

    side: ClassVar[int] = -1

    def sort_tail(self) -> tuple[int, ...]:
        return (-1, -1)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "tick": self.tick, "phase": int(self.phase), "reason": self.reason,
            "winner": self.winner, "sides": list(self.sides), "phases_executed": self.phases_executed,
            "queen_hp": list(self.queen_hp), "tick_completed": self.tick_completed,
        }


Event = (
    OrderEvent | RetargetEvent | DepartEvent | ArriveEvent
    | HatchEvent | HarvestEvent | UpkeepEvent | StarvationEvent | LayEvent | RegenerateEvent
    | CombatEvent | WoundEvent | DeathEvent | QueenDamageEvent | TerminalEvent
)


def event_sort_key(event: Event) -> tuple[int, ...]:
    """Sort key inside one tick; events of the same ``(phase, kind)`` share a tail shape."""
    return (int(event.phase), event.kind_rank, *event.sort_tail())


def events_to_canonical(events: tuple[Event, ...] | list[Event]) -> list[dict[str, Any]]:
    return [e.to_canonical() for e in events]


def events_hash(events: tuple[Event, ...] | list[Event]) -> str:
    return domain_hash(HASH_DOMAIN_EVENTS, canonical_dumps(events_to_canonical(events)))
