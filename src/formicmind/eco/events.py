"""Typed world events of E1-c and their hash chain (contracts section 6).

Events are the omniscient log of what physically happened. They are not part of ``WorldStateE1`` (so ``state_hash`` is the rules state
only, as in L0); a run keeps them in an ``EventLog`` whose chain hash ``events_hash`` is stable for equal inputs and for a
checkpoint-resumed run (log of the first part + log of the continuation = log of the continuous run).

Field use by kind (all integers; ``-1`` = not applicable; ``node`` is a world node id; the order inside a tick is the order of the phases):

========== ============================= ============================ =================================== ============================
kind       colony                        other                        a                                   b
========== ============================= ============================ =================================== ============================
discover   colony that explored          -1                           memory slot of the node             0
contact    observer                      colony it met                observer's slot of that colony      0
nest_found observer                      owner of the nest            observer's slot of the node         1 queen alive / 2 queen dead
dispatch   colony that issued the order  target colony / -1           mission code (MISSION_CODES)        units sent
combat     attacking side of the pair    side that takes the damage   damage dealt by ``colony``          part of it applied to ``other``
death      colony that lost units        -1                           caste (0 worker, 1 soldier)         units killed
queen_dmg  colony of the queen           -1                           damage applied                      queen hp afterwards
queen_dead colony of the queen           colony that dealt most dmg   0                                   0
collapse   colony                        -1                           0                                   0
extinct    colony                        -1                           food turned into a pile             0
========== ============================= ============================ =================================== ============================
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..canonical import canonical_dumps, domain_hash
from .codec import to_canonical
from .versions import HASH_DOMAIN_EVENTS

EVENT_KINDS = ("discover", "contact", "nest_found", "dispatch", "combat", "death", "queen_damage", "queen_dead",
               "collapse", "extinct")
MISSION_CODES = {"scout": 1, "guard": 2, "attack": 3, "withdraw": 4}
CASTE_CODES = {"worker": 0, "soldier": 1}


@dataclass(frozen=True, slots=True)
class EventE1:
    tick: int
    kind: str
    colony: int
    other: int
    node: int
    a: int
    b: int


def event(tick: int, kind: str, colony: int, other: int = -1, node: int = -1, a: int = 0, b: int = 0) -> EventE1:
    if kind not in EVENT_KINDS:
        raise ValueError(f"unknown event kind {kind!r}")
    return EventE1(tick, kind, colony, other, node, a, b)


@dataclass(slots=True)
class EventLog:
    """Append-only log with a running chain hash: ``head' = H(domain, canonical({prev: head, event: e}))``."""

    events: list[EventE1] = field(default_factory=list)
    head: str = ""

    def extend(self, new: list[EventE1]) -> None:
        for item in new:
            self.head = domain_hash(HASH_DOMAIN_EVENTS, canonical_dumps({"prev": self.head, "event": to_canonical(item)}))
            self.events.append(item)

    def counts(self) -> dict[str, int]:
        out = {kind: 0 for kind in EVENT_KINDS}
        for item in self.events:
            out[item.kind] += 1
        return out


_TEXT = {
    "discover": "colony {c} explored node {n} (memory slot {a})",
    "contact": "colony {c} met colony {o} at node {n}",
    "nest_found": "colony {c} found the nest of colony {o} at node {n} (queen {q})",
    "dispatch": "colony {c} sent {b} unit(s) on {m} to node {n}{t}",
    "combat": "colony {c} hit colony {o} at node {n}: {a} damage, {b} applied",
    "death": "colony {c} lost {b} {k}(s) at node {n}",
    "queen_damage": "queen of colony {c} took {a} damage at node {n}, {b} hp left",
    "queen_dead": "queen of colony {c} died at node {n}{k}",
    "collapse": "colony {c} collapsed economically",
    "extinct": "colony {c} died out; {a} food became a pile at node {n}",
}
_MISSIONS = {code: name for name, code in MISSION_CODES.items()}


def format_event(item: EventE1) -> str:
    """One compact line for a timeline: ``t=<tick> <kind>: <sentence>``."""
    text = _TEXT[item.kind].format(
        c=item.colony, o=item.other, n=item.node, a=item.a, b=item.b,
        q="alive" if item.b == 1 else "dead", m=_MISSIONS.get(item.a, "?"),
        t=f" against colony {item.other}" if item.other >= 0 else "",
        k=("" if item.kind != "queen_dead" else (f", killed by colony {item.other}" if item.other >= 0 else ""))
        if item.kind == "queen_dead" else ("soldier" if item.a else "worker"),
    )
    return f"t={item.tick} {item.kind}: {text}"
