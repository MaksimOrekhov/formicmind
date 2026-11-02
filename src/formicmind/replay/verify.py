"""``verify``: re-execute a replay through the one ``step_tick`` and compare everything.

The original policies are **not** called and their classes are not needed. ``ReplayController`` stands in for a seat:

1. the engine builds both observations; the controller compares the observation hash and the legal mask with the
   record, then passes the recorded ``ActionV2`` through the shared ``validate_action`` (a rejected action is a
   mismatch, never a fallback);
2. **post-decision RNG restoration.** A real ``PolicyController`` may consume its own stream during ``act``. The
   record holds the stream state right after that call. The controller sets the live stream to it *inside
   ``act``* - i.e. during the Policy phase, after both observations were built, before any order is applied, in seat
   order - exactly where the original consumption happened. Before it does, it checks that the live stream is the one
   the previous decision left (identity ``inc`` and state) and that the recorded state is ``draws`` steps ahead. The
   restored states are what make ``checkpoint_hash`` equal; ``state_hash`` and the events do not depend on them.
   Nothing is guessed: unknown streams, a changed ``inc`` or an unreachable state fail.

After every tick the events (kind, order, every field), the display patch and the frame hash are compared; at the
end the outcome, ``events_hash``, ``state_hash`` and ``checkpoint_hash``. The checkpoint format and hash formula are
the engine's own, unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ..actions import InvalidAction, ValidatedActionV2, get_action_mask, validate_action
from ..canonical import canonical_dumps
from ..events import events_hash
from ..model import SimState
from ..observation import ObservationV2
from ..reset import reset
from ..runstate import RunState
from ..serialization import checkpoint_from_bytes, checkpoint_hash, state_hash
from ..tick import step_tick
from . import frames
from .format import EventsHasher, ReplayError, ReplayVerifyError, pcg_advance
from .reader import Decision, ReplayReader
from .writer import mask_string, observation_hash, outcome_object


def mismatch(message: str) -> ReplayVerifyError:
    return ReplayVerifyError(message)


class ReplayController:
    """The recorded decisions of one seat as a controller (see the module docstring)."""

    def __init__(self, side: int, decisions: dict[int, Decision], expected: dict[str, tuple[int, int]],
                 calls: list[tuple[int, int]]) -> None:
        self.side = side
        self._decisions = decisions
        self._expected = expected  # stream id -> (state, inc) the previous decision left
        self._calls = calls

    def act(self, tick: int, observation: ObservationV2, run: RunState) -> ValidatedActionV2:
        decision = self._decisions.get(tick)
        if decision is None:
            raise mismatch(f"tick {tick}: the engine asks for a decision that the replay does not hold")
        seat = decision.seats[self.side]
        where = f"tick {tick}, seat {self.side}"
        if observation_hash(observation) != seat.obs_hash:
            raise mismatch(f"{where}: ObservationV2 hash differs from the recorded one")
        mask = get_action_mask(observation)
        if mask_string(observation) != seat.mask:
            raise mismatch(f"{where}: action mask differs from the recorded one")
        try:
            validated = validate_action(observation, mask, seat.action)
        except InvalidAction as exc:
            raise mismatch(f"{where}: the recorded action is rejected by the validator: {exc}") from exc
        if seat.stream_id:
            try:
                stream = run.stream(seat.stream_id)
            except KeyError:
                raise mismatch(f"{where}: RNG stream {seat.stream_id!r} is not registered in the replayed run") from None
            assert seat.state_after is not None
            if (stream.state, stream.inc) != self._expected[seat.stream_id]:
                raise mismatch(f"{where}: the live RNG stream {seat.stream_id!r} is not where the previous decision left it")
            if seat.state_after.inc != stream.inc or pcg_advance(stream.state, stream.inc, seat.draws) != seat.state_after.state:
                raise mismatch(f"{where}: recorded RNG state is not {seat.draws} draws ahead of the live stream")
            stream.state = seat.state_after.state  # the restoration: right here, in the Policy phase
            self._expected[seat.stream_id] = (stream.state, stream.inc)
        self._calls.append((tick, self.side))
        return validated


@dataclass(frozen=True, slots=True)
class VerifyReport:
    ticks: int
    decisions: int
    events: int
    finished: bool
    winner: str  # "" for a truncated run
    reason: str
    terminal_tick: int  # -1 for a truncated run
    terminal_phase: int  # -1 for a truncated run
    tick_completed: bool | None
    state_tick: int
    events_hash: str
    final_state_hash: str
    final_checkpoint_hash: str
    records: int


def _initial(reader: ReplayReader) -> tuple[SimState, RunState]:
    head = reader.read_header()
    if head.start_kind == "reset":
        return reset(head.config, head.root_seed, head.streams)
    state, run = checkpoint_from_bytes(canonical_dumps(head.raw["start"]["checkpoint"]), head.config)
    return state, run


def _first_difference(got: list[dict[str, Any]], recorded: list[dict[str, Any]]) -> str:
    for i, (a, b) in enumerate(zip(got, recorded)):
        if a != b:
            fields = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            return f"event #{i} ({b.get('kind')}) differs in {fields}"
    return f"the engine produced {len(got)} events, the replay holds {len(recorded)}"


def verify(source: ReplayReader | str | os.PathLike[str] | bytes) -> VerifyReport:
    """Verify a replay file, bytes or an opened reader. Raises ``ReplayFormatError`` (not a valid v1 file) or
    ``ReplayVerifyError`` (valid, but the engine does not reproduce it); returns a ``VerifyReport`` on success."""
    if isinstance(source, ReplayReader):
        reader = source
    elif isinstance(source, bytes):
        reader = ReplayReader.from_bytes(source)
    else:
        reader = ReplayReader.open(source)
    head, result = reader.read_header(), reader.read_result()
    state, run = _initial(reader)
    first = frames.view_from_state(state, run)
    if first.to_canonical() != reader.first_frame():
        raise mismatch(f"the first snapshot (tick {state.tick}) is not the frame of the replayed start state")

    expected = {sid: (run.streams[sid].state, run.streams[sid].inc) for sid in head.streams}
    calls: list[tuple[int, int]] = []
    decisions = reader.decisions
    controllers = (ReplayController(0, decisions, expected, calls), ReplayController(1, decisions, expected, calls))
    body = frames.build_body(state)
    hasher = EventsHasher()
    k = head.config.policy_interval_k

    for record in reader.tick_records:
        t = record.tick
        if state.tick != t:
            raise mismatch(f"tick {t}: the replayed state stands at tick {state.tick}")
        calls.clear()
        try:
            tick_result = step_tick(state, run, controllers)
        except ReplayError:
            raise
        except Exception as exc:  # engine or validator refusal: the replay is not reproducible
            raise mismatch(f"tick {t}: the engine refused to execute it: {type(exc).__name__}: {exc}") from exc
        if sorted(calls) != ([(t, 0), (t, 1)] if t % k == 0 else []):
            raise mismatch(f"tick {t}: the decisions asked of the seats differ from the recorded ones")
        got = [e.to_canonical() for e in tick_result.events]
        recorded = reader.events_of(record)
        if got != recorded:
            raise mismatch(f"tick {t}: {_first_difference(got, recorded)}")
        for event in got:
            hasher.add(event)
        new_body = frames.build_body(state)
        boundary = frames.BOUNDARY_START if run.outcome is None else frames.BOUNDARY_TERMINAL
        frame = frames.frame_dict(state.tick, boundary, frames.terminal_object(run.outcome), new_body)
        if (state.tick, boundary) != (record.to_tick, record.to_boundary):
            raise mismatch(f"tick {t}: the engine ended it at ({state.tick}, {boundary}), the replay at "
                           f"({record.to_tick}, {record.to_boundary})")
        if frames.diff_body(body, new_body) != record.patch or frames.frame_hash(frame) != record.frame_hash:
            raise mismatch(f"tick {t}: the display frame after the tick differs from the recorded one")
        body = new_body

    if state.tick != result.state_tick:
        raise mismatch(f"final state.tick {state.tick}, recorded {result.state_tick}")
    if outcome_object(run.outcome) != result.raw["outcome"]:
        raise mismatch("the outcome differs from the recorded one")
    if hasher.hexdigest() != result.events_hash:
        raise mismatch("events_hash differs from the recorded one")
    if state_hash(state) != result.final_state_hash:
        raise mismatch("final state_hash differs from the recorded one")
    if checkpoint_hash(state, run) != result.final_checkpoint_hash:
        raise mismatch("final checkpoint_hash differs from the recorded one (RNG streams, windows or outcome)")
    out = result.outcome
    return VerifyReport(
        ticks=result.ticks_executed, decisions=result.decision_count, events=result.event_count,
        finished=result.finished, winner="" if out is None else out.winner.value,
        reason="" if out is None else out.reason.value, terminal_tick=-1 if out is None else out.terminal_tick,
        terminal_phase=-1 if out is None else int(out.terminal_phase), tick_completed=result.tick_completed,
        state_tick=result.state_tick, events_hash=result.events_hash, final_state_hash=result.final_state_hash,
        final_checkpoint_hash=result.final_checkpoint_hash, records=result.record_count + 1)


__all__ = ["ReplayController", "VerifyReport", "verify"]
