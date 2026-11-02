"""ViewFrame: the omniscient display state of one boundary, and the per-tick display patches between frames.

A ``ViewFrame`` is a *view*, not a state: it holds what a spectator needs and nothing a rule needs. It is built from the
real ``SimState`` after a tick (never by a second simulator) and contains

* ``tick`` = ``state.tick`` (completed ticks) and ``boundary``: ``"start"`` (the start of tick ``tick``, before its
  Policy phase) or ``"terminal"`` (the final state after the last executed phase);
* ``terminal``: ``{"finished": false}`` or the terminal facts (winner, reason, ``terminal_tick``, ``terminal_phase``,
  ``tick_completed``), so a terminal frame says by itself whether the tick was cut at QueenTerminal;
* ``node_food``: the stock of every node (7 ints);
* per colony: warehouse ``food``, ``queen_hp``, ``stationed`` groups ``(node, caste, count, hp_sum)``, ``transit``
  groups on edges ``(from_node, to_node, arrival_tick, caste, count, hp_sum)`` and ``brood`` ``(caste, hatch_tick,
  count)``. Units are grouped by caste (and by edge/arrival for movers); individual ids are not part of the view:
  this is a display of groups, not an L1 ant simulation.

**Display patches.** The file stores the full frame at t=0, at every policy boundary and at the end, and for every
executed tick a small *patch* that turns the previous frame body into the next one (changed node stocks, changed
scalars, ``set``/``del`` rows of the three group tables). A reader reaches any tick with
``snapshot + at most K-1 patches``; no rule is executed. Choosing patches over a full frame per tick keeps a quiet
3000-tick match well under a megabyte of view data; the price is that a patch is only meaningful on top of its
predecessor, which the hash of every resulting frame and the per-line hash chain make safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..canonical import canonical_dumps, domain_hash
from ..rules import NODE_COUNT
from ..terminal import EpisodeOutcome
from ..versions import HASH_DOMAIN_REPLAY_VIEW
from .format import need, need_int, need_keys

BOUNDARY_START = "start"
BOUNDARY_TERMINAL = "terminal"
BOUNDARIES = (BOUNDARY_START, BOUNDARY_TERMINAL)

# table name -> (key length, value length); a row is key + value
TABLES: dict[str, tuple[int, int]] = {"stationed": (2, 2), "transit": (4, 2), "brood": (2, 1)}
_SCALARS = ("food", "queen_hp")
_TERMINAL_KEYS = {"finished", "winner", "reason", "terminal_tick", "terminal_phase", "tick_completed"}


# --- canonical frame dicts -------------------------------------------------------------------------------------------

def terminal_object(outcome: EpisodeOutcome | None) -> dict[str, Any]:
    if outcome is None:
        return {"finished": False}
    return {"finished": True, "winner": outcome.winner.value, "reason": outcome.reason.value,
            "terminal_tick": outcome.terminal_tick, "terminal_phase": int(outcome.terminal_phase),
            "tick_completed": outcome.tick_completed}


def terminal_object_from_event(event: dict) -> dict[str, Any]:
    """The same object, taken from a recorded ``terminal`` event (what a reader without a result record has)."""
    return {"finished": True, "winner": event["winner"], "reason": event["reason"],
            "terminal_tick": event["tick"], "terminal_phase": event["phase"],
            "tick_completed": event["tick_completed"]}


def build_body(state: Any) -> dict[str, Any]:
    """The patchable part of a frame, read from a ``SimState`` (a pure read: nothing is executed or changed)."""
    colonies = []
    for colony in state.colonies:
        stationed: dict[tuple[int, int], list[int]] = {}
        transit: dict[tuple[int, int, int, int], list[int]] = {}
        for unit in colony.units:
            if unit.node >= 0:
                row = stationed.setdefault((unit.node, unit.caste), [0, 0])
            else:
                row = transit.setdefault((unit.from_node, unit.to_node, unit.arrival_tick, unit.caste), [0, 0])
            row[0] += 1
            row[1] += unit.hp
        brood: dict[tuple[int, int], list[int]] = {}
        for egg in colony.brood:
            brood.setdefault((egg.caste, egg.hatch_tick), [0])[0] += 1
        colonies.append({
            "side": colony.side, "food": colony.food, "queen_hp": colony.queen_hp,
            "stationed": [[*k, *v] for k, v in sorted(stationed.items())],
            "transit": [[*k, *v] for k, v in sorted(transit.items())],
            "brood": [[*k, *v] for k, v in sorted(brood.items())],
        })
    return {"node_food": list(state.node_food), "colonies": colonies}


def frame_dict(tick: int, boundary: str, terminal: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    return {"tick": tick, "boundary": boundary, "terminal": terminal,
            "node_food": body["node_food"], "colonies": body["colonies"]}


def body_of(frame: dict[str, Any]) -> dict[str, Any]:
    return {"node_food": frame["node_food"], "colonies": frame["colonies"]}


def frame_hash(frame: dict[str, Any]) -> str:
    return domain_hash(HASH_DOMAIN_REPLAY_VIEW, canonical_dumps(frame))


def copy_body(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_food": list(body["node_food"]),
        "colonies": [
            {k: ([list(r) for r in v] if k in TABLES else v) for k, v in colony.items()}
            for colony in body["colonies"]
        ],
    }


# --- validation ------------------------------------------------------------------------------------------------------

def _check_rows(rows: object, table: str, what: str) -> None:
    key_len, value_len = TABLES[table]
    need(type(rows) is list, f"{what}.{table}: expected an array")
    previous: list[int] | None = None
    for row in rows:  # type: ignore[union-attr]
        need(type(row) is list and len(row) == key_len + value_len and all(type(v) is int for v in row),
             f"{what}.{table}: malformed row")
        key = row[:key_len]
        need(previous is None or key > previous, f"{what}.{table}: rows must be sorted by unique key")
        need(all(v >= 0 for v in row[key_len:]), f"{what}.{table}: negative value")
        previous = key


def check_body(body: object, what: str = "frame") -> None:
    need_keys(body, {"node_food", "colonies"}, what)
    food = body["node_food"]  # type: ignore[index]
    need(type(food) is list and len(food) == NODE_COUNT and all(type(v) is int and v >= 0 for v in food),
         f"{what}.node_food: expected {NODE_COUNT} non-negative ints")
    colonies = body["colonies"]  # type: ignore[index]
    need(type(colonies) is list and len(colonies) == 2, f"{what}.colonies: expected one entry per side")
    for side, colony in enumerate(colonies):
        need_keys(colony, {"side", "food", "queen_hp", *TABLES}, f"{what}.colonies[{side}]")
        need(colony["side"] == side and type(colony["side"]) is int, f"{what}.colonies[{side}]: wrong side")
        for name in _SCALARS:
            need_int(colony[name], f"{what}.colonies[{side}].{name}", 0)
        for table in TABLES:
            _check_rows(colony[table], table, f"{what}.colonies[{side}]")


def check_terminal(obj: object, what: str = "frame.terminal") -> None:
    need(type(obj) is dict, f"{what}: expected an object")
    if obj == {"finished": False}:
        return
    need_keys(obj, _TERMINAL_KEYS, what)
    d = obj  # type: ignore[assignment]
    need(d["finished"] is True and type(d["winner"]) is str and type(d["reason"]) is str
         and type(d["tick_completed"]) is bool and all(type(d[k]) is int for k in ("terminal_tick", "terminal_phase")),
         f"{what}: malformed terminal object")


def check_frame(frame: object, what: str = "frame") -> None:
    need_keys(frame, {"tick", "boundary", "terminal", "node_food", "colonies"}, what)
    need_int(frame["tick"], f"{what}.tick", 0)  # type: ignore[index]
    need(frame["boundary"] in BOUNDARIES, f"{what}.boundary: unknown boundary")  # type: ignore[index]
    check_terminal(frame["terminal"], f"{what}.terminal")  # type: ignore[index]
    need((frame["boundary"] == BOUNDARY_TERMINAL) == frame["terminal"]["finished"],  # type: ignore[index]
         f"{what}: only a terminal frame carries a finished outcome")
    check_body(body_of(frame), what)  # type: ignore[arg-type]


# --- patches ---------------------------------------------------------------------------------------------------------

def diff_body(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """The minimal patch from ``old`` to ``new`` in canonical form (deterministic, sorted)."""
    patch: dict[str, Any] = {}
    changed = [[i, b] for i, (a, b) in enumerate(zip(old["node_food"], new["node_food"])) if a != b]
    if changed:
        patch["nf"] = changed
    per_colony: list[dict[str, Any]] = []
    for before, after in zip(old["colonies"], new["colonies"]):
        entry: dict[str, Any] = {name: after[name] for name in _SCALARS if before[name] != after[name]}
        for table, (key_len, _) in TABLES.items():
            old_rows = {tuple(r[:key_len]): r for r in before[table]}
            new_rows = {tuple(r[:key_len]): r for r in after[table]}
            put = [r for k, r in new_rows.items() if old_rows.get(k) != r]
            drop = [list(k) for k in old_rows if k not in new_rows]
            if put or drop:
                entry[table] = {"set": sorted(put), "del": sorted(drop)}
        per_colony.append(entry)
    patch["c"] = per_colony
    return patch


def apply_patch(body: dict[str, Any], patch: object, what: str = "patch") -> dict[str, Any]:
    """``body`` + ``patch`` -> a new body. The input is never modified. Strict: a patch that does not fit is an error."""
    need(type(patch) is dict and set(patch) <= {"nf", "c"} and "c" in patch, f"{what}: malformed patch")
    out = copy_body(body)
    changes = patch.get("nf", [])  # type: ignore[union-attr]
    need(type(changes) is list, f"{what}.nf: expected an array")
    last = -1
    for change in changes:
        need(type(change) is list and len(change) == 2 and all(type(v) is int for v in change),
             f"{what}.nf: malformed change")
        index, value = change
        need(last < index < NODE_COUNT and value >= 0, f"{what}.nf: index/value out of range or unsorted")
        out["node_food"][index] = value
        last = index
    entries = patch["c"]  # type: ignore[index]
    need(type(entries) is list and len(entries) == 2, f"{what}.c: expected one entry per side")
    for side, entry in enumerate(entries):
        need(type(entry) is dict and set(entry) <= {*_SCALARS, *TABLES}, f"{what}.c[{side}]: unknown keys")
        colony = out["colonies"][side]
        for name in _SCALARS:
            if name in entry:
                colony[name] = need_int(entry[name], f"{what}.c[{side}].{name}", 0)
        for table, (key_len, _) in TABLES.items():
            if table not in entry:
                continue
            part = entry[table]
            need_keys(part, {"set", "del"}, f"{what}.c[{side}].{table}")
            rows = {tuple(r[:key_len]): r for r in colony[table]}
            _check_rows(part["set"], table, f"{what}.c[{side}].set")
            need(type(part["del"]) is list, f"{what}.c[{side}].{table}.del: expected an array")
            previous: list[int] | None = None
            deleted: set[tuple[int, ...]] = set()
            for key in part["del"]:
                need(type(key) is list and len(key) == key_len and all(type(v) is int for v in key)
                     and (previous is None or key > previous), f"{what}.c[{side}].{table}.del: malformed or unsorted key")
                need(tuple(key) in rows, f"{what}.c[{side}].{table}.del: deleting a row that does not exist")
                del rows[tuple(key)]
                deleted.add(tuple(key))
                previous = key
            for row in part["set"]:
                need(tuple(row[:key_len]) not in deleted, f"{what}: a key is both set and deleted")
                rows[tuple(row[:key_len])] = list(row)
            colony[table] = [rows[k] for k in sorted(rows)]
    check_body(out, what)
    return out


# --- typed read-only view --------------------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TerminalView:
    winner: str  # "A" | "B" | "draw"
    reason: str
    terminal_tick: int
    terminal_phase: int  # Phase.QUEEN_TERMINAL (4) or Phase.TERMINAL_LOG (10)
    tick_completed: bool  # False: the tick was cut at QueenTerminal and ``tick`` stays at ``terminal_tick``


@dataclass(frozen=True, slots=True)
class ColonyView:
    side: int
    food: int
    queen_hp: int
    stationed: tuple[tuple[int, int, int, int], ...]  # (node, caste, count, hp_sum)
    transit: tuple[tuple[int, int, int, int, int, int], ...]  # (from_node, to_node, arrival_tick, caste, count, hp_sum)
    brood: tuple[tuple[int, int, int], ...]  # (caste, hatch_tick, count)


@dataclass(frozen=True, slots=True)
class ViewFrame:
    """An immutable, self-contained omniscient frame (what ``ReplayReader.seek`` returns)."""

    tick: int
    boundary: str
    terminal: TerminalView | None
    node_food: tuple[int, ...]
    colonies: tuple[ColonyView, ColonyView]

    @classmethod
    def from_canonical(cls, frame: dict[str, Any]) -> "ViewFrame":
        check_frame(frame)
        t = frame["terminal"]
        terminal = None if not t["finished"] else TerminalView(
            t["winner"], t["reason"], t["terminal_tick"], t["terminal_phase"], t["tick_completed"])
        colonies = tuple(
            ColonyView(c["side"], c["food"], c["queen_hp"], *(tuple(tuple(r) for r in c[n]) for n in TABLES))
            for c in frame["colonies"]
        )
        return cls(frame["tick"], frame["boundary"], terminal, tuple(frame["node_food"]), colonies)  # type: ignore[arg-type]

    def to_canonical(self) -> dict[str, Any]:
        terminal = {"finished": False} if self.terminal is None else {
            "finished": True, "winner": self.terminal.winner, "reason": self.terminal.reason,
            "terminal_tick": self.terminal.terminal_tick, "terminal_phase": self.terminal.terminal_phase,
            "tick_completed": self.terminal.tick_completed}
        return {
            "tick": self.tick, "boundary": self.boundary, "terminal": terminal,
            "node_food": list(self.node_food),
            "colonies": [
                {"side": c.side, "food": c.food, "queen_hp": c.queen_hp,
                 "stationed": [list(r) for r in c.stationed], "transit": [list(r) for r in c.transit],
                 "brood": [list(r) for r in c.brood]}
                for c in self.colonies
            ],
        }

    def hash(self) -> str:
        return frame_hash(self.to_canonical())


def view_from_state(state: Any, run: Any) -> ViewFrame:
    """The omniscient frame of the engine's current state (``boundary`` follows ``run.outcome``)."""
    boundary = BOUNDARY_START if run.outcome is None else BOUNDARY_TERMINAL
    return ViewFrame.from_canonical(
        frame_dict(state.tick, boundary, terminal_object(run.outcome), build_body(state)))


__all__ = [
    "BOUNDARIES", "BOUNDARY_START", "BOUNDARY_TERMINAL", "ColonyView", "TABLES", "TerminalView", "ViewFrame",
    "apply_patch", "body_of", "build_body", "check_frame", "copy_body", "diff_body", "frame_dict", "frame_hash",
    "terminal_object", "terminal_object_from_event", "view_from_state",
]
