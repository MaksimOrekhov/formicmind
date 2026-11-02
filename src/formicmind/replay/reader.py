"""Replay v1 reader: strict validation, in-memory index, ``seek`` - and **no simulation**.

``ReplayReader.open`` reads the whole file once and checks everything that can be checked from the file alone:

* bytes: one canonical JSON object per ``\\n``-terminated line, an intact hash chain, a ``result`` record last;
* versions: replay schema and every contract version/hash domain must equal the engine's (``ReplayVersionError``);
* the record grammar (see ``writer``): decisions exactly at ``t % K == 0``, consecutive ticks, snapshots at the
  start, at every policy boundary and at the end, nothing after the terminal tick, nothing after ``result``;
* every event against the explicit schema, its phase, order and tick; the terminal event against the delta and the
  result; global event indices; the incremental ``events_hash``;
* every display patch against its predecessor: the frame after each delta must hash to the recorded ``frame_hash`` and
  every snapshot must equal the frame reached through the patches;
* the RNG stream states recorded after each decision: stream ids against the header, ``inc`` unchanged, and the new
  state reachable from the previous one in exactly ``draws`` steps.

It never calls ``step_tick``, routing, economy, movement or combat; ``seek`` is a snapshot plus at most ``K-1``
patches. Whether the *engine* reproduces the match is ``verify``'s question.
"""

from __future__ import annotations

import copy
import os
from bisect import bisect_right
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..actions import MASK_V2_LENGTH, ActionV2, InvalidAction, check_action_structure
from ..canonical import CanonicalError, canonical_dumps, u64_from_str
from ..config import SimConfig
from ..events import Phase
from ..invariants import InvariantError
from ..rng import WORLD_STREAM_ID, Pcg32, new_stream
from ..serialization import (
    checkpoint_from_bytes,
    checkpoint_hash,
    config_from_canonical,
    config_hash,
)
from ..terminal import EpisodeOutcome, Reason, outcome_from_canonical
from ..versions import REPLAY_FORMAT, REPLAY_SCHEMA_VERSION
from . import frames
from .format import (
    EVENT_SCHEMA,
    ZERO_CHAIN,
    EventsHasher,
    ReplayVersionError,
    SeatSpec,
    SeekError,
    bad,
    check_event,
    contract_versions,
    need,
    need_hex64,
    need_int,
    need_keys,
    open_record,
    pcg_advance,
    same,
    seat_from_canonical,
    split_lines,
)
from .frames import BOUNDARY_START, BOUNDARY_TERMINAL, ViewFrame


@dataclass(frozen=True, slots=True)
class ReplayHeader:
    config: SimConfig
    config_hash: str
    root_seed: int
    streams: tuple[str, ...]  # registered RNG streams besides ``world``
    seats: tuple[SeatSpec, SeatSpec]
    start_kind: str  # "reset" | "checkpoint"
    start_tick: int  # cursor: the first tick of this file
    start_event_index: int  # cursor: index of its first event in the full journal
    start_checkpoint_hash: str  # "" for a reset start
    engine: str
    source: str
    meta: dict[str, str]
    raw: dict[str, Any]  # the canonical header record (without chain)


@dataclass(frozen=True, slots=True)
class DecisionSeat:
    side: int
    obs_hash: str
    mask: str  # 30 characters, "1" = legal
    action: ActionV2
    stream_id: str  # "" when the seat consumes no RNG
    draws: int  # u32 draws consumed during the decision
    state_after: Pcg32 | None  # the stream right after the decision


@dataclass(frozen=True, slots=True)
class Decision:
    tick: int
    seats: tuple[DecisionSeat, DecisionSeat]


@dataclass(frozen=True, slots=True)
class TickRecord:
    tick: int  # the executed tick
    to_tick: int  # state.tick after it
    to_boundary: str
    patch: dict[str, Any]
    frame_hash: str
    first_event: int  # slice of the reader's event list
    end_event: int


@dataclass(frozen=True, slots=True)
class ReplayResult:
    finished: bool  # False: a truncated harness run, ``outcome is None``
    outcome: EpisodeOutcome | None
    tick_completed: bool | None  # None for a truncated run
    state_tick: int
    ticks_executed: int
    decision_count: int
    event_count: int
    events_hash: str
    final_state_hash: str
    final_checkpoint_hash: str
    record_count: int
    chain_head: str
    raw: dict[str, Any]


def _stream_origin(header: dict[str, Any], config: SimConfig) -> tuple[Any, dict[str, Pcg32], str]:
    """Checkpoint (or None) and the initial state of every non-world stream, as the header defines them."""
    seed = u64_from_str(header["root_seed"])
    streams = header["streams"]
    start = header["start"]
    if start == {"kind": "reset"}:
        return None, {sid: new_stream(seed, sid) for sid in streams}, ""
    need_keys(start, {"kind", "checkpoint", "checkpoint_hash"}, "header.start")
    need(start["kind"] == "checkpoint", "header.start: unknown start kind")
    declared = need_hex64(start["checkpoint_hash"], "header.start.checkpoint_hash")
    try:
        state, run = checkpoint_from_bytes(canonical_dumps(start["checkpoint"]), config)
    except (CanonicalError, InvariantError, ValueError, TypeError, KeyError) as exc:
        raise bad(f"header.start.checkpoint: {exc}") from exc
    need(checkpoint_hash(state, run) == declared, "header.start: checkpoint hash mismatch")
    need(run.outcome is None and state.tick < config.t_max, "header.start: the checkpoint is not a running episode")
    need(run.root_seed == seed, "header.start: checkpoint root_seed differs from the header")
    need(sorted(set(run.streams) - {WORLD_STREAM_ID}) == streams and WORLD_STREAM_ID in run.streams,
         "header.start: checkpoint streams differ from the header")
    return (state, run), {sid: run.streams[sid].clone() for sid in streams}, declared


class ReplayReader:
    """A validated, indexed replay. Construct with ``ReplayReader.open(path)`` or ``ReplayReader.from_bytes``."""

    # -- construction -------------------------------------------------------------------------------------------

    @classmethod
    def open(cls, path: str | os.PathLike[str]) -> "ReplayReader":
        return cls.from_bytes(Path(path).read_bytes())

    @classmethod
    def from_bytes(cls, data: bytes) -> "ReplayReader":
        reader = cls()
        reader._load(data)
        return reader

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []
        self._ticks: list[TickRecord] = []
        self._decisions: dict[int, Decision] = {}
        self._start_frames: dict[int, dict[str, Any]] = {}
        self._start_ticks: list[int] = []
        self._terminal_frame: dict[str, Any] | None = None
        self._header: ReplayHeader | None = None
        self._result: ReplayResult | None = None
        self.record_count = 0

    # -- public API ---------------------------------------------------------------------------------------------

    def read_header(self) -> ReplayHeader:
        assert self._header is not None
        return self._header

    def read_result(self) -> ReplayResult:
        assert self._result is not None
        return self._result

    @property
    def decisions(self) -> dict[int, Decision]:
        return dict(self._decisions)

    @property
    def tick_records(self) -> tuple[TickRecord, ...]:
        return tuple(self._ticks)

    def events_of(self, tick: TickRecord) -> list[dict[str, Any]]:
        """The recorded canonical events of one executed tick (internal lists; do not modify)."""
        return self._events[tick.first_event:tick.end_event]

    def first_frame(self) -> dict[str, Any]:
        """The first snapshot (canonical dict, internal; do not modify)."""
        return self._start_frames[self._start_ticks[0]]

    def iter_events(self, *, tick: int | None = None, kind: str | None = None) -> Iterator[dict[str, Any]]:
        """Recorded events in engine order, as independent canonical dicts (optionally one tick and/or kind)."""
        if kind is not None and kind not in EVENT_SCHEMA:
            raise SeekError(f"unknown event kind {kind!r}")
        for event in self._events:
            if (tick is None or event["tick"] == tick) and (kind is None or event["kind"] == kind):
                yield copy.deepcopy(event)

    @property
    def first_tick(self) -> int:
        return self.read_header().start_tick

    @property
    def last_start_tick(self) -> int:
        """The last tick that has a ``"start"`` frame: ``terminal_tick`` of a finished match, else the cutoff tick."""
        result = self.read_result()
        return result.outcome.terminal_tick if result.outcome is not None else result.state_tick

    def seek(self, tick: int, boundary: str = BOUNDARY_START) -> ViewFrame:
        """The omniscient frame at an explicit boundary - a snapshot plus patches, no rules executed.

        * ``"start"``: the state at the start of tick ``tick`` (after ``tick`` completed ticks), before its Policy
          phase; valid for ``first_tick <= tick <= last_start_tick``.
        * ``"terminal"``: the final state after the last executed phase; ``tick`` must be ``result.state_tick``
          (``terminal_tick`` after a queen death cut mid-tick, ``terminal_tick + 1`` after an end-of-tick terminal).
        """
        if type(tick) is not int:
            raise SeekError("tick must be an int")
        if boundary == BOUNDARY_TERMINAL:
            result = self.read_result()
            if self._terminal_frame is None or tick != result.state_tick:
                raise SeekError(f"no terminal frame at tick {tick} (state_tick of this replay: "
                                f"{result.state_tick if self._terminal_frame is not None else 'none: truncated run'})")
            return ViewFrame.from_canonical(self._terminal_frame)
        if boundary != BOUNDARY_START:
            raise SeekError(f"unknown boundary {boundary!r}")
        if not self.first_tick <= tick <= self.last_start_tick:
            raise SeekError(f"no start frame at tick {tick}: this replay holds {self.first_tick}..{self.last_start_tick}")
        base = self._start_ticks[bisect_right(self._start_ticks, tick) - 1]
        body = frames.body_of(self._start_frames[base])
        for t in range(base, tick):
            body = frames.apply_patch(body, self._ticks[t - self.first_tick].patch, f"patch of tick {t}")
        return ViewFrame.from_canonical(
            frames.frame_dict(tick, BOUNDARY_START, {"finished": False}, body))

    # -- loading ------------------------------------------------------------------------------------------------

    def _load(self, data: bytes) -> None:
        lines = split_lines(data)
        chain = ZERO_CHAIN
        records: list[dict[str, Any]] = []
        for number, line in enumerate(lines):
            record, chain_after = open_record(line, chain, number)
            need(same(record.get("seq"), number), f"line {number}: seq {record.get('seq')!r} out of order")
            records.append(record)
            if record.get("type") == "result":
                need(number == len(lines) - 1, f"line {number + 1}: records after the result")
                need(record.get("chain_head") == chain, f"line {number}: chain_head does not match the chain")
            chain = chain_after
        need(records[-1].get("type") == "result", "missing result record: the file is truncated or was never closed")
        self.record_count = len(records)
        self._read_records(records)

    def _read_records(self, records: list[dict[str, Any]]) -> None:
        head, checkpoint, stream_state = self._read_header(records[0])
        config = head.config
        k = config.policy_interval_k
        # Where the grammar is: ``cur`` is the state.tick of the current frame, ``body`` its patchable body.
        first = self._read_snapshot(records[1], "first snapshot", expect_tick=head.start_tick)
        if checkpoint is not None:
            expected = frames.frame_dict(head.start_tick, BOUNDARY_START, {"finished": False},
                                         frames.build_body(checkpoint[0]))
            need(first == expected, "first snapshot does not match the embedded checkpoint")
        self._start_frames[head.start_tick] = first
        self._start_ticks.append(head.start_tick)
        body = frames.body_of(first)
        cur = head.start_tick
        event_index = head.start_event_index
        hasher = EventsHasher()
        last_start = head.start_tick
        need_snapshot: tuple[str, int] | None = None
        terminal_event: dict[str, Any] | None = None
        closing = False
        finished = False
        pos = 2
        total = len(records) - 1  # index of the result record
        while pos < total:
            rec = records[pos]
            rtype = rec.get("type")
            where = f"line {pos}"
            need(not closing, f"{where}: unexpected record after the cutoff snapshot")
            if need_snapshot is not None:
                need(rtype == "snapshot", f"{where}: expected the {need_snapshot[0]} snapshot of tick {need_snapshot[1]}")
                frame = self._read_snapshot(rec, where, expect_tick=None)
                need((frame["boundary"], frame["tick"]) == need_snapshot, f"{where}: snapshot at the wrong boundary")
                is_terminal = need_snapshot[0] == BOUNDARY_TERMINAL
                terminal = frames.terminal_object_from_event(terminal_event) if is_terminal else {"finished": False}  # type: ignore[arg-type]
                need(frame == frames.frame_dict(cur, need_snapshot[0], terminal, body),
                     f"{where}: snapshot does not match the frame reached through the patches")
                if is_terminal:
                    self._terminal_frame = frame
                else:
                    self._start_frames[cur] = frame
                    self._start_ticks.append(cur)
                    last_start = cur
                need_snapshot = None
                pos += 1
                continue
            if rtype == "snapshot":  # only the cutoff frame of a truncated run may stand alone
                frame = self._read_snapshot(rec, where, expect_tick=cur)
                need(not finished and frame["boundary"] == BOUNDARY_START and cur % k != 0 and last_start != cur,
                     f"{where}: unexpected snapshot")
                need(frame == frames.frame_dict(cur, BOUNDARY_START, {"finished": False}, body),
                     f"{where}: snapshot does not match the frame reached through the patches")
                self._start_frames[cur] = frame
                self._start_ticks.append(cur)
                last_start = cur
                closing = True
                pos += 1
                continue
            # ---- one executed tick: [decision] event* delta
            need(not finished, f"{where}: a record after the terminal tick")
            need(cur < config.t_max, f"{where}: a tick beyond the horizon T_max")
            tick = cur
            if tick % k == 0:
                need(rtype == "decision", f"{where}: tick {tick} is a policy tick and needs its decision first")
                self._decisions[tick] = self._read_decision(rec, tick, head, stream_state)
                pos += 1
            else:
                need(rtype != "decision", f"{where}: tick {tick} is not a policy tick but has a decision")
            first_event = len(self._events)
            last_key = (-1, -1)
            while pos < total and records[pos].get("type") == "event":
                er = records[pos]
                where = f"line {pos}"
                need_keys(er, {"type", "seq", "i", "e"}, where)
                need(same(er["i"], event_index), f"{where}: event index {er['i']!r}, expected {event_index}")
                schema = check_event(er["e"], tick, where)
                key = (er["e"]["phase"], schema.rank)
                need(key >= last_key, f"{where}: event out of engine order")
                last_key = key
                need(terminal_event is None, f"{where}: an event after the terminal event")
                if er["e"]["kind"] == "terminal":
                    terminal_event = er["e"]
                hasher.add(er["e"])
                self._events.append(er["e"])
                event_index += 1
                pos += 1
            need(pos < total and records[pos].get("type") == "delta", f"line {pos}: tick {tick} has no delta record")
            dr = records[pos]
            where = f"line {pos}"
            need_keys(dr, {"type", "seq", "tick", "to_tick", "to_boundary", "patch", "frame_hash"}, where)
            need(same(dr["tick"], tick), f"{where}: delta of tick {dr['tick']!r}, expected {tick}")
            ended = terminal_event is not None
            expect_boundary = BOUNDARY_TERMINAL if ended else BOUNDARY_START
            need(dr["to_boundary"] == expect_boundary, f"{where}: boundary does not follow from the terminal event")
            if ended:
                completed = terminal_event["tick_completed"]  # type: ignore[index]
                need(terminal_event["tick"] == tick, f"{where}: terminal event of another tick")  # type: ignore[index]
                need(same(dr["to_tick"], tick + (1 if completed else 0)), f"{where}: terminal state tick")
                if terminal_event["phase"] == Phase.TERMINAL_LOG:  # type: ignore[index]
                    need(terminal_event["reason"] != Reason.QUEEN_DEATH.value, f"{where}: queen death is decided in QueenTerminal")  # type: ignore[index]
                else:
                    need(terminal_event["reason"] == Reason.QUEEN_DEATH.value, f"{where}: only a queen death ends a tick mid-way")  # type: ignore[index]
                if terminal_event["reason"] == Reason.TIMEOUT.value:  # type: ignore[index]
                    need(tick == config.t_max - 1, f"{where}: a timeout only ends the last tick")
            else:
                need(same(dr["to_tick"], tick + 1), f"{where}: a completed tick advances state.tick by one")
            body = frames.apply_patch(body, dr["patch"], f"{where}.patch")
            terminal = frames.terminal_object_from_event(terminal_event) if ended else {"finished": False}  # type: ignore[arg-type]
            after = frames.frame_dict(dr["to_tick"], expect_boundary, terminal, body)
            need(frames.frame_hash(after) == need_hex64(dr["frame_hash"], f"{where}.frame_hash"),
                 f"{where}: the display patch does not reproduce the recorded frame hash")
            self._ticks.append(TickRecord(tick, dr["to_tick"], expect_boundary, dr["patch"], dr["frame_hash"],
                                          first_event, len(self._events)))
            cur = dr["to_tick"]
            pos += 1
            if ended:
                finished = True
                need_snapshot = (BOUNDARY_TERMINAL, cur)
            elif cur % k == 0 and cur < config.t_max:
                need_snapshot = (BOUNDARY_START, cur)
        self._read_result(records[-1], head, hasher, finished, terminal_event, cur, last_start, need_snapshot)
        self._header = head

    # -- records ------------------------------------------------------------------------------------------------

    def _read_header(self, rec: dict[str, Any]) -> tuple[ReplayHeader, Any, dict[str, Pcg32]]:
        need(rec.get("type") == "header", "the first record must be the header")
        if rec.get("format") != REPLAY_FORMAT:
            raise bad("not a FormicMind replay (format tag)")
        if rec.get("replay_schema") != REPLAY_SCHEMA_VERSION:
            raise ReplayVersionError(f"unsupported replay schema {rec.get('replay_schema')!r} (this reader: v1)")
        need_keys(rec, {"type", "seq", "format", "replay_schema", "engine", "contract", "config", "config_hash",
                        "root_seed", "streams", "seats", "start", "cursor", "source", "meta"}, "header")
        mine = contract_versions()
        theirs = rec["contract"]
        if theirs != mine:
            diff = sorted(k for k in set(mine) | set(theirs if type(theirs) is dict else {})
                          if (theirs.get(k) if type(theirs) is dict else None) != mine.get(k))
            raise ReplayVersionError(f"contract versions differ from this engine: {diff}")
        try:
            config = config_from_canonical(rec["config"])
        except (CanonicalError, InvariantError, ValueError, TypeError, KeyError) as exc:
            raise bad(f"header.config: {exc}") from exc
        need(need_hex64(rec["config_hash"], "header.config_hash") == config_hash(config), "header.config_hash mismatch")
        try:
            seed = u64_from_str(rec["root_seed"])
        except CanonicalError as exc:
            raise bad(f"header.root_seed: {exc}") from exc
        streams = rec["streams"]
        need(type(streams) is list and all(type(s) is str for s in streams) and streams == sorted(set(streams))
             and WORLD_STREAM_ID not in streams, "header.streams: expected a sorted list of unique non-world ids")
        seats_raw = rec["seats"]
        need(type(seats_raw) is list and len(seats_raw) == 2, "header.seats: expected two seats")
        seats = (seat_from_canonical(seats_raw[0], 0), seat_from_canonical(seats_raw[1], 1))
        owned = [s.stream_id for s in seats if s.stream_id]
        need(len(set(owned)) == len(owned) and set(owned) <= set(streams), "header.seats: bad or shared RNG streams")
        cursor = need_keys(rec["cursor"], {"tick", "event_index"}, "header.cursor")
        need_int(cursor["tick"], "header.cursor.tick", 0, config.t_max - 1)
        need_int(cursor["event_index"], "header.cursor.event_index", 0)
        kind = rec["start"].get("kind") if type(rec["start"]) is dict else None
        need(kind in ("reset", "checkpoint"), "header.start: unknown start kind")
        checkpoint, stream_state, declared = _stream_origin(rec, config)
        if kind == "reset":
            need(cursor == {"tick": 0, "event_index": 0}, "header.cursor: a reset start begins at tick 0, event 0")
        else:
            need(checkpoint[0].tick == cursor["tick"], "header.cursor.tick does not match the checkpoint")
        need(type(rec["engine"]) is str and type(rec["source"]) is str, "header: engine/source must be strings")
        meta = rec["meta"]
        need(type(meta) is dict and all(type(k) is str and type(v) is str for k, v in meta.items()),
             "header.meta: expected string pairs")
        head = ReplayHeader(config, rec["config_hash"], seed, tuple(streams), seats, kind, cursor["tick"],  # type: ignore[arg-type]
                            cursor["event_index"], declared, rec["engine"], rec["source"], dict(meta), rec)
        return head, checkpoint, stream_state

    def _read_snapshot(self, rec: dict[str, Any], where: str, expect_tick: int | None) -> dict[str, Any]:
        need_keys(rec, {"type", "seq", "frame", "frame_hash"}, where)
        need(rec["type"] == "snapshot", f"{where}: expected a snapshot")
        frames.check_frame(rec["frame"], where)
        need(frames.frame_hash(rec["frame"]) == need_hex64(rec["frame_hash"], f"{where}.frame_hash"),
             f"{where}: snapshot hash mismatch")
        if expect_tick is not None:
            need(rec["frame"]["tick"] == expect_tick and rec["frame"]["boundary"] == BOUNDARY_START,
                 f"{where}: expected the start frame of tick {expect_tick}")
        return rec["frame"]

    def _read_decision(self, rec: dict[str, Any], tick: int, head: ReplayHeader,
                       stream_state: dict[str, Pcg32]) -> Decision:
        where = f"decision of tick {tick}"
        need_keys(rec, {"type", "seq", "tick", "seats"}, where)
        need(same(rec["tick"], tick) and type(rec["seats"]) is list and len(rec["seats"]) == 2, f"{where}: malformed")
        seats = []
        for side, raw in enumerate(rec["seats"]):
            at = f"{where}, seat {side}"
            need_keys(raw, {"side", "obs_hash", "mask", "action", "rng"}, at)
            need(same(raw["side"], side), f"{at}: wrong side")
            need_hex64(raw["obs_hash"], f"{at}.obs_hash")
            mask = raw["mask"]
            need(type(mask) is str and len(mask) == MASK_V2_LENGTH and set(mask) <= {"0", "1"}, f"{at}.mask")
            act = need_keys(raw["action"], {"lay_order", "forage_share_q", "army_choice", "avoid_center"}, f"{at}.action")
            need(type(act["forage_share_q"]) is list, f"{at}.action.forage_share_q")
            action = ActionV2(act["lay_order"], tuple(act["forage_share_q"]), act["army_choice"], act["avoid_center"])
            try:
                check_action_structure(action)
            except InvalidAction as exc:
                raise bad(f"{at}.action: {exc}") from exc
            wanted = head.seats[side].stream_id
            rng = raw["rng"]
            need(type(rng) is list and len(rng) == (1 if wanted else 0), f"{at}.rng: expected {1 if wanted else 0} stream entries")
            stream_id, draws, after = "", 0, None
            if wanted:
                entry = need_keys(rng[0], {"stream_id", "draws", "state_after"}, f"{at}.rng")
                need(entry["stream_id"] == wanted, f"{at}.rng: stream {entry['stream_id']!r} is not the seat's stream {wanted!r}")
                draws = need_int(entry["draws"], f"{at}.rng.draws", 0, 1 << 40)
                try:
                    after = Pcg32.from_canonical(entry["state_after"])
                except CanonicalError as exc:
                    raise bad(f"{at}.rng.state_after: {exc}") from exc
                before = stream_state[wanted]
                need(after.inc == before.inc, f"{at}.rng: the stream identity (inc) changed")
                need(after.state == pcg_advance(before.state, before.inc, draws),
                     f"{at}.rng: state_after is not reachable from the previous state in {draws} draws")
                stream_state[wanted] = after
                stream_id = wanted
            seats.append(DecisionSeat(side, raw["obs_hash"], mask, action, stream_id, draws, after))
        return Decision(tick, (seats[0], seats[1]))

    def _read_result(self, rec: dict[str, Any], head: ReplayHeader, hasher: EventsHasher, finished: bool,
                     terminal_event: dict[str, Any] | None, cur: int, last_start: int,
                     need_snapshot: tuple[str, int] | None) -> None:
        keys = {"type", "seq", "finished", "outcome", "state_tick", "ticks_executed", "decision_count",
                "event_count", "events_hash", "final_state_hash", "final_checkpoint_hash", "record_count",
                "chain_head"}
        need_keys(rec, keys, "result")
        need(need_snapshot is None, "the file ends before the required snapshot")
        need(rec["finished"] is finished, "result.finished disagrees with the recorded ticks")
        need(same(rec["state_tick"], cur) and same(rec["record_count"], rec["seq"]), "result: state_tick/record_count")
        need(same(rec["ticks_executed"], len(self._ticks)), "result.ticks_executed")
        need(same(rec["decision_count"], len(self._decisions)), "result.decision_count")
        need(same(rec["event_count"], hasher.count) and hasher.count == len(self._events), "result.event_count")
        need(rec["events_hash"] == hasher.hexdigest(), "result.events_hash does not match the recorded events")
        need_hex64(rec["final_state_hash"], "result.final_state_hash")
        need_hex64(rec["final_checkpoint_hash"], "result.final_checkpoint_hash")
        outcome: EpisodeOutcome | None = None
        completed: bool | None = None
        if finished:
            raw = rec["outcome"]
            need(type(raw) is dict and raw.get("finished") is True and "tick_completed" in raw, "result.outcome")
            body = {k: v for k, v in raw.items() if k not in ("finished", "tick_completed")}
            try:
                outcome = outcome_from_canonical(body)
            except CanonicalError as exc:
                raise bad(f"result.outcome: {exc}") from exc
            completed = raw["tick_completed"]
            ev = terminal_event
            assert ev is not None
            need(completed is outcome.tick_completed and outcome.state_tick == cur
                 and outcome.terminal_tick == ev["tick"] and int(outcome.terminal_phase) == ev["phase"]
                 and outcome.phases_executed == ev["phases_executed"] and list(outcome.sides) == ev["sides"]
                 and outcome.winner.value == ev["winner"] and outcome.reason.value == ev["reason"],
                 "result.outcome disagrees with the terminal event")
        else:
            need(rec["outcome"] == {"finished": False}, "result.outcome of a truncated run must be {'finished': false}")
            need(cur < head.config.t_max, "a run on the horizon without an outcome is impossible")
            need(last_start == cur, "a truncated replay must end with the start frame of its last state")
        self._result = ReplayResult(
            finished, outcome, completed, cur, len(self._ticks), len(self._decisions), hasher.count,
            rec["events_hash"], rec["final_state_hash"], rec["final_checkpoint_hash"], rec["record_count"],
            rec["chain_head"], rec)


# --- convenience functions on paths ----------------------------------------------------------------------------------

def read_header(path: str | os.PathLike[str]) -> ReplayHeader:
    return ReplayReader.open(path).read_header()


def read_result(path: str | os.PathLike[str]) -> ReplayResult:
    return ReplayReader.open(path).read_result()


def iter_events(path: str | os.PathLike[str], *, tick: int | None = None, kind: str | None = None) -> Iterator[dict[str, Any]]:
    return ReplayReader.open(path).iter_events(tick=tick, kind=kind)


def seek(path: str | os.PathLike[str], tick: int, boundary: str = BOUNDARY_START) -> ViewFrame:
    return ReplayReader.open(path).seek(tick, boundary)


__all__ = [
    "Decision", "DecisionSeat", "ReplayHeader", "ReplayReader", "ReplayResult", "TickRecord",
    "iter_events", "read_header", "read_result", "seek",
]
