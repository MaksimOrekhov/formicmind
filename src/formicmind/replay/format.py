"""Replay v1 envelope: errors, canonical JSONL lines with a hash chain, event schema, seat descriptors.

File = UTF-8 JSON Lines (in fact ASCII: canonical JSON escapes everything else), ``\\n`` line ends, **one canonical
JSON object per line**, no BOM, no blank lines, no trailing garbage. Canonical means exactly ``canonical_dumps``: sorted
keys, no whitespace, no float/NaN/None; ``canonical_loads`` rejects any other byte spelling of the same value.

Every record carries ``type``, ``seq`` (0-based line number) and ``chain``. The chain is a SHA-256 hash chain::

    chain[i] = SHA-256("formicmind/replay-record/v1" || 0x00 || chain[i-1] || 0x00 || line[i] with its chain field
                       replaced by 64 zeros)            chain[-1] = 64 zeros

so changing, dropping, inserting or reordering any line breaks that line's chain and every later one, and the
``result`` record repeats the head of the chain. A forger who recomputes the chain still has to survive
``verify``, which re-executes the match through the one ``step_tick``.

Errors are separated on purpose: ``ReplayFormatError`` (the file is not a valid v1 replay: version, structure,
hash chain, impossible order) is never mixed with ``ReplayVerifyError`` (a well-formed replay that the engine does
not reproduce) or with technical policy errors, which never occur in a replay at all.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ..canonical import CanonicalError, canonical_dumps, canonical_loads, domain_hash
from ..events import Phase
from ..rng import MASK64, PCG_MULT, check_stream_id
from ..versions import (
    ACTION_VERSION,
    BALANCE_VERSION,
    DECODER_VERSION,
    HASH_DOMAIN_CHECKPOINT,
    HASH_DOMAIN_CONFIG,
    HASH_DOMAIN_EVENTS,
    HASH_DOMAIN_OBSERVATION,
    HASH_DOMAIN_REPLAY_RECORD,
    HASH_DOMAIN_REPLAY_VIEW,
    HASH_DOMAIN_RNG_STREAM,
    HASH_DOMAIN_STATE,
    OBSERVATION_VERSION,
    RNG_VERSION,
    RULES_VERSION,
    SERIALIZATION_VERSION,
    STREAM_DERIVATION_VERSION,
)

ZERO_CHAIN = "0" * 64
HEX64 = re.compile(r"[0-9a-f]{64}")
_CHAIN_FIELD = re.compile(rb'"chain":"([0-9a-f]{64})"')
_ZERO_FIELD = b'"chain":"' + ZERO_CHAIN.encode("ascii") + b'"'


# --- errors -----------------------------------------------------------------------------------------------------

class ReplayError(Exception):
    """Base class of every replay problem."""


class ReplayFormatError(ReplayError):
    """The bytes are not a valid Replay v1 file (structure, hash chain, order, impossible values)."""


class ReplayVersionError(ReplayFormatError):
    """Unknown replay schema or contract versions: the file is refused before anything else is trusted."""


class ReplayVerifyError(ReplayError):
    """A well-formed replay that the engine does not reproduce (hash, event, observation, outcome mismatch)."""


class ReplayWriteError(ReplayError):
    """The recorder was misused or a controller broke the recording contract (never written as a "loss")."""


class SeekError(ReplayError, ValueError):
    """``seek`` was asked for a tick/boundary that does not exist in this replay."""


def bad(message: str) -> ReplayFormatError:
    return ReplayFormatError(message)


# --- versions ---------------------------------------------------------------------------------------------------

def contract_versions() -> dict[str, str]:
    """Everything a replay depends on besides the replay schema itself; the header must repeat it exactly."""
    return {
        "rules_version": RULES_VERSION,
        "balance_version": BALANCE_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "action_version": ACTION_VERSION,
        "decoder_version": DECODER_VERSION,
        "serialization_version": SERIALIZATION_VERSION,
        "rng_version": RNG_VERSION,
        "stream_derivation_version": STREAM_DERIVATION_VERSION,
        "hash_domain_config": HASH_DOMAIN_CONFIG,
        "hash_domain_state": HASH_DOMAIN_STATE,
        "hash_domain_checkpoint": HASH_DOMAIN_CHECKPOINT,
        "hash_domain_events": HASH_DOMAIN_EVENTS,
        "hash_domain_rng_stream": HASH_DOMAIN_RNG_STREAM,
        "hash_domain_observation": HASH_DOMAIN_OBSERVATION,
        "hash_domain_view": HASH_DOMAIN_REPLAY_VIEW,
        "hash_domain_record": HASH_DOMAIN_REPLAY_RECORD,
    }


# --- lines and the hash chain ------------------------------------------------------------------------------------

def _chain_hash(prev: str, line_with_zero_chain: bytes) -> str:
    return domain_hash(HASH_DOMAIN_REPLAY_RECORD, prev.encode("ascii") + b"\x00" + line_with_zero_chain)


def seal_record(record: dict, prev_chain: str) -> tuple[bytes, str]:
    """``record`` (without ``chain``) -> (one canonical line including the newline, its chain value)."""
    if "chain" in record:
        raise ReplayFormatError("a record must not carry its own chain before sealing")
    line = canonical_dumps({**record, "chain": ZERO_CHAIN})
    if len(_CHAIN_FIELD.findall(line)) != 1:
        raise ReplayFormatError("record content collides with the chain field")
    chain = _chain_hash(prev_chain, line)
    return line.replace(_ZERO_FIELD, b'"chain":"' + chain.encode("ascii") + b'"', 1) + b"\n", chain


def open_record(line: bytes, prev_chain: str, number: int) -> tuple[dict, str]:
    """One line (no newline) -> (record without ``chain``, chain value). Strict: canonical bytes and a valid chain."""
    try:
        record = canonical_loads(line)
    except CanonicalError as exc:
        raise bad(f"line {number}: {exc}") from exc
    if type(record) is not dict:
        raise bad(f"line {number}: a record must be a JSON object")
    found = _CHAIN_FIELD.findall(line)
    if len(found) != 1 or record.get("chain") != found[0].decode("ascii"):
        raise bad(f"line {number}: missing or ambiguous chain field")
    chain = found[0].decode("ascii")
    if _chain_hash(prev_chain, _CHAIN_FIELD.sub(_ZERO_FIELD, line, count=1)) != chain:
        raise bad(f"line {number}: hash chain broken (the file was modified, reordered or cut)")
    del record["chain"]
    return record, chain


def split_lines(data: bytes) -> list[bytes]:
    """Split a file into lines. The file must end with ``\\n``; CR, blank lines and an unfinished last line are errors."""
    if not data:
        raise bad("empty file")
    if not data.endswith(b"\n"):
        raise bad("truncated file: the last line is not terminated")
    lines = data[:-1].split(b"\n")
    for number, line in enumerate(lines):
        if not line:
            raise bad(f"line {number}: blank line")
        if b"\r" in line:
            raise bad(f"line {number}: carriage return (a replay uses plain \\n line ends)")
    return lines


# --- small strict helpers ------------------------------------------------------------------------------------------

def need(condition: bool, message: str) -> None:
    if not condition:
        raise bad(message)


def need_keys(obj: object, keys: set[str] | frozenset[str], what: str) -> dict:
    if type(obj) is not dict:
        raise bad(f"{what}: expected an object")
    if set(obj) != set(keys):
        raise bad(f"{what}: missing keys {sorted(set(keys) - set(obj))}, unexpected keys {sorted(set(obj) - set(keys))}")
    return obj


def is_int(value: object, low: int | None = None, high: int | None = None) -> bool:
    return type(value) is int and (low is None or value >= low) and (high is None or value <= high)


def same(value: object, expected: int) -> bool:
    """``value == expected`` for a recorded integer: a bool (``True == 1`` in Python) or a float never matches."""
    return type(value) is int and value == expected


def need_int(value: object, what: str, low: int | None = None, high: int | None = None) -> int:
    if not is_int(value, low, high):
        raise bad(f"{what}: expected an int" + ("" if low is None else f" >= {low}") + ("" if high is None else f" <= {high}"))
    return value  # type: ignore[return-value]


def need_hex64(value: object, what: str) -> str:
    if type(value) is not str or not HEX64.fullmatch(value):
        raise bad(f"{what}: expected a lowercase SHA-256 hex string")
    return value


def need_int_list(value: object, what: str, length: int | None = None) -> list[int]:
    if type(value) is not list or any(type(v) is not int for v in value) or (length is not None and len(value) != length):
        raise bad(f"{what}: expected an int array" + ("" if length is None else f" of length {length}"))
    return value


# --- seats ---------------------------------------------------------------------------------------------------------

_POLICY_ID = re.compile(r"[A-Za-z0-9_.\-]{1,64}")
_REF = re.compile(r"[ -~]{0,200}")
SEAT_KINDS = ("scripted", "policy", "other")


@dataclass(frozen=True, slots=True)
class SeatSpec:
    """A seat's controller as *data*: never an executable object. ``stream_id`` is the RNG stream the controller
    owns (``policy:<policy_id>:<replicate>``), ``""`` for a controller that consumes none. ``ref`` is an opaque
    configuration reference (a name or a hash), informational only."""

    policy_id: str
    kind: str = "other"
    stream_id: str = ""
    ref: str = ""

    def __post_init__(self) -> None:
        ok = (
            type(self.policy_id) is str and _POLICY_ID.fullmatch(self.policy_id) is not None
            and self.kind in SEAT_KINDS
            and type(self.stream_id) is str and (self.stream_id == "" or _stream_ok(self.stream_id))
            and type(self.ref) is str and _REF.fullmatch(self.ref) is not None
        )
        if not ok:
            raise ValueError(f"invalid seat descriptor: {self!r}")

    def to_canonical(self, side: int) -> dict:
        return {"side": side, "policy_id": self.policy_id, "kind": self.kind,
                "stream_id": self.stream_id, "ref": self.ref}


def _stream_ok(stream_id: str) -> bool:
    try:
        check_stream_id(stream_id)
    except ValueError:
        return False
    return stream_id != "world"


def seat_from_canonical(obj: object, side: int) -> SeatSpec:
    need_keys(obj, {"side", "policy_id", "kind", "stream_id", "ref"}, f"seat {side}")
    need(obj["side"] == side and type(obj["side"]) is int, f"seat {side}: wrong side index")  # type: ignore[index]
    try:
        return SeatSpec(obj["policy_id"], obj["kind"], obj["stream_id"], obj["ref"])  # type: ignore[index]
    except ValueError as exc:
        raise bad(f"seat {side}: {exc}") from exc


# --- events --------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EventSchema:
    rank: int  # kind_rank of events.py: order inside one phase
    phases: tuple[int, ...]  # phases that may emit it
    fields: tuple[tuple[str, str], ...]  # (name, type): i int, b bool, s str, I int list, J int list of two


def _schema(rank: int, phases: tuple[int, ...], spec: str) -> EventSchema:
    fields = []
    for token in spec.split():
        name, _, kind = token.partition(":")
        fields.append((name, kind or "i"))
    return EventSchema(rank, tuple(int(p) for p in phases), tuple(fields))


_P = Phase
# Explicit on purpose (a hand-written contract, pinned against ``events.py`` by a test): a new field or kind
# in the engine must not silently become part of replay v1.
EVENT_SCHEMA: dict[str, EventSchema] = {
    "order": _schema(0, (_P.POLICY,), "tick phase side lay_order forage_share_q:I army_choice avoid_center:b "
                                      "army_target_node worker_quota:I"),
    "retarget": _schema(1, (_P.POLICY,), "tick phase side unit_id caste old_target new_target"),
    "depart": _schema(2, (_P.POLICY, _P.AUTOMATIC_DEPART), "tick phase side unit_id caste from_node to_node "
                                                          "arrival_tick target_node"),
    "arrive": _schema(3, (_P.ARRIVAL,), "tick phase side unit_id caste node from_node target_node"),
    "hatch": _schema(4, (_P.COMPLETION,), "tick phase side unit_id caste node hp laid_tick target_node"),
    "harvest": _schema(5, (_P.HARVEST,), "tick phase side node workers demand blocked:b claim harvested "
                                         "stock_before stock_after"),
    "upkeep": _schema(6, (_P.UPKEEP,), "tick phase side adult_workers adult_soldiers remainder_before "
                                       "remainder_after due paid missing food_before food_after debt_before "
                                       "debt_after deaths debt_reset:b"),
    "starvation": _schema(7, (_P.UPKEEP,), "tick phase side seq unit_id caste hp node to_node debt_before "
                                           "debt_after"),
    "lay": _schema(8, (_P.LAY,), "tick phase side laid:b caste eggs unit_cost total_cost food_before food_after "
                                 "unit_ids:I hatch_tick next_lay_tick"),
    "regenerate": _schema(9, (_P.REGENERATION,), "tick phase node before regen added after capacity"),
    "combat": _schema(10, (_P.COMBAT,), "tick phase node attacker_side defender_side attacker_soldiers "
                                        "base_damage home_bonus:b damage target:s target_units target_hp_total "
                                        "applied wasted"),
    "wound": _schema(11, (_P.COMBAT,), "tick phase side unit_id caste node attacker_side hp_before damage hp_after"),
    "death": _schema(12, (_P.COMBAT,), "tick phase side unit_id caste node attacker_side hp_before damage hp_after"),
    "queen_damage": _schema(13, (_P.COMBAT,), "tick phase side node attacker_side damage queen_hp_before "
                                              "queen_hp_after"),
    "terminal": _schema(14, (_P.QUEEN_TERMINAL, _P.TERMINAL_LOG), "tick phase reason:s winner:s sides:I "
                                                                  "phases_executed queen_hp:J tick_completed:b"),
}
# ``sort_tail`` of events.py as field names ("" = the constant -1): inside one (phase, kind) the engine emits events in
# strictly ascending tail order, so a recorded stream must too. ``starvation`` is (side, seq), not unit_id.
_SORT_TAIL: dict[str, tuple[str, str]] = {
    "order": ("", "side"), "retarget": ("unit_id", "side"), "depart": ("unit_id", "side"),
    "arrive": ("unit_id", "side"), "hatch": ("unit_id", "side"), "harvest": ("node", "side"),
    "upkeep": ("", "side"), "starvation": ("side", "seq"), "lay": ("", "side"), "regenerate": ("node", ""),
    "combat": ("node", "attacker_side"), "wound": ("unit_id", "side"), "death": ("unit_id", "side"),
    "queen_damage": ("", "side"), "terminal": ("", ""),
}
assert set(_SORT_TAIL) == set(EVENT_SCHEMA)


def event_sort_key(event: dict) -> tuple[int, ...]:
    """``events.event_sort_key`` computed from a canonical event dict: ``(phase, kind rank, *sort_tail)``."""
    first, second = _SORT_TAIL[event["kind"]]
    return (event["phase"], EVENT_SCHEMA[event["kind"]].rank,
            event[first] if first else -1, event[second] if second else -1)


EVENT_KEYS = {kind: frozenset({"kind", *(name for name, _ in s.fields)}) for kind, s in EVENT_SCHEMA.items()}
_TERMINAL_PHASES = (int(Phase.QUEEN_TERMINAL), int(Phase.TERMINAL_LOG))


def check_event(event: object, tick: int, where: str) -> EventSchema:
    """Strict structural check of one recorded event; returns its schema. Value semantics are ``verify``'s job."""
    if type(event) is not dict or type(event.get("kind")) is not str:
        raise bad(f"{where}: an event must be an object with a 'kind'")
    schema = EVENT_SCHEMA.get(event["kind"])
    if schema is None:
        raise bad(f"{where}: unknown event kind {event['kind']!r}")
    need_keys(event, EVENT_KEYS[event["kind"]], where)
    for name, kind in schema.fields:
        value = event[name]
        if kind == "i":
            ok = type(value) is int
        elif kind == "b":
            ok = type(value) is bool
        elif kind == "s":
            ok = type(value) is str
        elif kind == "I":
            ok = type(value) is list and all(type(v) is int for v in value)
        else:  # "J"
            ok = type(value) is list and len(value) == 2 and all(type(v) is int for v in value)
        need(ok, f"{where}: field {name!r} has the wrong type")
    need(event["tick"] == tick, f"{where}: event tick {event['tick']} outside tick {tick}")
    need(event["phase"] in schema.phases, f"{where}: kind {event['kind']!r} cannot happen in phase {event['phase']}")
    if "side" in event:
        need(event["side"] in (0, 1), f"{where}: bad side")
    if event["kind"] == "terminal":
        need(event["phases_executed"] == event["phase"] + 1, f"{where}: terminal phases_executed")
        need(event["tick_completed"] == (event["phase"] == Phase.TERMINAL_LOG), f"{where}: terminal tick_completed")
        need(event["phase"] in _TERMINAL_PHASES, f"{where}: terminal phase")
        need(all(s in (0, 1) for s in event["sides"]) and event["sides"] == sorted(set(event["sides"])),
             f"{where}: terminal sides")
    return schema


class EventsHasher:
    """Incremental twin of ``events.events_hash``: SHA-256 over ``domain || 0x00 || canonical_json(list)``."""

    def __init__(self) -> None:
        self._hash = hashlib.sha256(HASH_DOMAIN_EVENTS.encode("ascii") + b"\x00[")
        self.count = 0

    def add(self, event: dict) -> None:
        if self.count:
            self._hash.update(b",")
        self._hash.update(canonical_dumps(event))
        self.count += 1

    def hexdigest(self) -> str:
        final = self._hash.copy()
        final.update(b"]")
        return final.hexdigest()


# --- RNG -----------------------------------------------------------------------------------------------------------

def pcg_advance(state: int, inc: int, steps: int) -> int:
    """State of a PCG32 stream after ``steps`` ``next_u32`` calls, in O(log steps) (O'Neill's LCG jump-ahead)."""
    acc_mult, acc_plus = 1, 0
    cur_mult, cur_plus = PCG_MULT, inc
    while steps > 0:
        if steps & 1:
            acc_mult = (acc_mult * cur_mult) & MASK64
            acc_plus = (acc_plus * cur_mult + cur_plus) & MASK64
        cur_plus = ((cur_mult + 1) * cur_plus) & MASK64
        cur_mult = (cur_mult * cur_mult) & MASK64
        steps >>= 1
    return (acc_mult * state + acc_plus) & MASK64
