"""Movement-level events: ``order``, ``retarget``, ``depart``, ``arrive``.

Boundary and ordering
---------------------
``step_tick`` is called with ``state.tick == t``: the state *at the start of tick t*,
before ``PRE_POLICY``. It executes the phases of tick ``t`` and leaves
``state.tick == t + 1`` (the start of the next tick, i.e. after tick ``t``'s
AutomaticDepart). Every event carries ``tick == t`` (the tick in which it happened)
and the ``Phase`` that produced it. A ``depart`` at tick ``t`` over an edge of length
``d`` carries ``arrival_tick == t + d``; the matching ``arrive`` carries ``tick == t + d``.

Events are returned in execution order, which equals sorting by
``(phase, kind rank, unit_id, side)``: phases in tick order; inside a phase, all
``order`` events (by side), then ``retarget``, then ``depart``, then ``arrive``, the
unit-level ones by ascending unit id (unit ids are global and stable, so side does not
matter for the order). Events hold only data the omniscient log may hold; nothing
here is ever shown to a policy.

``to_canonical`` gives plain int/str/bool/list/dict values (canonical-JSON safe), so a
later replay writer needs no extra conversion. JSONL replay itself is not implemented.
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

    def to_canonical(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "tick": self.tick, "phase": int(self.phase), "side": self.side,
            "unit_id": self.unit_id, "caste": self.caste, "node": self.node,
            "from_node": self.from_node, "target_node": self.target_node,
        }


Event = OrderEvent | RetargetEvent | DepartEvent | ArriveEvent


def event_sort_key(event: Event) -> tuple[int, int, int, int]:
    return (int(event.phase), event.kind_rank, event.unit_id, event.side)


def events_to_canonical(events: tuple[Event, ...] | list[Event]) -> list[dict[str, Any]]:
    return [e.to_canonical() for e in events]


def events_hash(events: tuple[Event, ...] | list[Event]) -> str:
    return domain_hash(HASH_DOMAIN_EVENTS, canonical_dumps(events_to_canonical(events)))
