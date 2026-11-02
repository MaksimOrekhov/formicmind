"""Streaming Replay v1 writer: records the real outputs of the one ``step_tick``; it is not a second simulator.

Record order (the whole grammar, checked again by the reader)::

    header
    snapshot            start frame at the first tick (t=0 for a fresh match, before Policy(0))
    -- per executed tick t --
    decision            only if t % K == 0: both ActionV2 (after the shared validator), the observation hashes,
                        masks and the post-decision RNG streams of the seats
    event*              the tick's Event.to_canonical() in engine order, with a global event index
    delta               display patch from the previous frame to the frame after this tick
    snapshot            "start" frame of tick t+1 if (t+1) % K == 0 and the match goes on, or the "terminal"
                        frame if the tick ended the match
    -- end --
    snapshot            "start" cutoff frame, only when a truncated run stops at a non-policy tick
    result

``RecordingController`` wraps a seat's real controller. It never changes what the controller returns; it notes the
action, the observation hash and what the controller did to the RNG streams, and the notes are written only after
``step_tick`` has finished the tick (the Policy phase is all-or-nothing, a failed tick leaves nothing behind).
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from ..actions import ValidatedActionV2, flatten_mask_v2, get_action_mask
from ..canonical import canonical_dumps, canonical_loads, domain_hash, u64_to_str
from ..config import SimConfig
from ..match import EpisodeResult, _register_streams, play, summarize
from ..model import SimState
from ..observation import ObservationV2, flatten_v2
from ..reset import reset
from ..rng import WORLD_STREAM_ID, Pcg32
from ..runstate import RunState
from ..serialization import checkpoint_hash, checkpoint_to_bytes, config_hash, config_to_canonical, state_hash
from ..terminal import EpisodeOutcome
from ..tick import Controller, PolicyController, ScriptedController, TickMode, TickResult
from ..versions import HASH_DOMAIN_OBSERVATION, REPLAY_ENGINE_PIN, REPLAY_FORMAT, REPLAY_SCHEMA_VERSION
from . import frames
from .format import (
    ZERO_CHAIN,
    EventsHasher,
    ReplayWriteError,
    SeatSpec,
    contract_versions,
    seal_record,
)

DRAW_SCAN_LIMIT = 1 << 20  # a decision may consume at most this many u32 draws from its own stream


def observation_hash(observation: ObservationV2) -> str:
    """SHA-256 over the ObservationV2 values as float32 little-endian bytes (the exact bytes a policy received)."""
    return domain_hash(HASH_DOMAIN_OBSERVATION, flatten_v2(observation))


def mask_string(observation: ObservationV2) -> str:
    return "".join("1" if bit else "0" for bit in flatten_mask_v2(get_action_mask(observation)))


def action_object(action: Any) -> dict[str, Any]:
    return {"lay_order": action.lay_order, "forage_share_q": list(action.forage_share_q),
            "army_choice": action.army_choice, "avoid_center": action.avoid_center}


def outcome_object(outcome: EpisodeOutcome | None) -> dict[str, Any]:
    if outcome is None:
        return {"finished": False}
    return {"finished": True, **outcome.to_canonical(), "tick_completed": outcome.tick_completed}


def count_draws(before: Pcg32, after: Pcg32) -> int:
    """How many ``next_u32`` calls turned ``before`` into ``after`` (same stream). Unreachable states are rejected."""
    if before.inc != after.inc:
        raise ReplayWriteError("a controller replaced the identity (inc) of its RNG stream")
    probe = before.clone()
    for draws in range(DRAW_SCAN_LIMIT + 1):
        if probe.state == after.state:
            return draws
        probe.next_u32()
    raise ReplayWriteError(
        f"the RNG stream state after the decision is not reachable within {DRAW_SCAN_LIMIT} draws")


def describe_controller(controller: Controller) -> SeatSpec:
    """Default seat descriptor of a controller (data only). Override it with ``seats=`` for a better name."""
    if isinstance(controller, ScriptedController):
        script = [[t, action_object(a)] for t, a in sorted(controller.script.items())]
        digest = hashlib.sha256(canonical_dumps(script)).hexdigest()
        return SeatSpec("scripted", "scripted", "", f"script-sha256:{digest}")
    stream_id = getattr(controller, "stream_id", None) or ""
    if isinstance(controller, PolicyController) or stream_id:
        parts = stream_id.split(":")
        policy_id = parts[1] if len(parts) == 3 and parts[0] == "policy" else "policy"
        return SeatSpec(policy_id, "policy", stream_id, "")
    return SeatSpec("controller", "other", "", "")


@dataclass(frozen=True, slots=True)
class ReplaySummary:
    """What the writer knows when it closes the file."""

    records: int  # lines including the result
    size_bytes: int
    ticks: int
    decisions: int
    events: int
    finished: bool
    events_hash: str
    final_state_hash: str
    final_checkpoint_hash: str


class RecordingController:
    """Transparent wrapper: forwards ``act`` to the real controller and notes the decision for the writer."""

    def __init__(self, inner: Controller, side: int, writer: "ReplayWriter") -> None:
        self.inner = inner
        self.side = side
        self._writer = writer

    @property
    def stream_id(self) -> str | None:
        return getattr(self.inner, "stream_id", None)

    def act(self, tick: int, observation: ObservationV2, run: RunState) -> ValidatedActionV2:
        before = {sid: stream.clone() for sid, stream in run.streams.items()}
        answer = self.inner.act(tick, observation, run)
        self._writer._note_decision(self.side, tick, observation, answer, before, run)
        return answer


class ReplayWriter:
    def __init__(
        self,
        sink: BinaryIO,
        config: SimConfig,
        seats: Sequence[SeatSpec],
        *,
        source: str = "",
        meta: Mapping[str, str] | None = None,
        event_index_base: int = 0,
        embed_checkpoint: bool = False,
    ) -> None:
        if len(seats) != 2 or not all(isinstance(s, SeatSpec) for s in seats):
            raise ReplayWriteError("two SeatSpec descriptors are required, one per side")
        streams = [s.stream_id for s in seats if s.stream_id]
        if len(set(streams)) != len(streams):
            raise ReplayWriteError("two seats may not share one RNG stream")
        meta = dict(meta or {})
        if any(type(k) is not str or type(v) is not str or k == "chain" for k, v in meta.items()):
            raise ReplayWriteError("meta must map strings to strings (the key 'chain' is reserved)")
        if type(source) is not str or len(source) > 200 or not source.isascii() or not source.isprintable():
            raise ReplayWriteError("source must be a short printable ASCII string")
        if type(event_index_base) is not int or event_index_base < 0:
            raise ReplayWriteError("event_index_base must be a non-negative int")
        self._sink = sink
        self._config = config
        self._seats = tuple(seats)
        self._source, self._meta = source, meta
        self._event_index = self._first_event_index = event_index_base
        self._embed = embed_checkpoint
        self._seq = 0
        self._chain = ZERO_CHAIN
        self._bytes = 0
        self._started = self._finished = False
        self._pending: dict[int, dict[str, Any]] = {}
        self._hasher = EventsHasher()
        self._body: dict[str, Any] | None = None
        self._next_tick = 0
        self._ticks = self._decisions = 0
        self._last_start_snapshot = -1
        self._terminal = False

    # -- public -------------------------------------------------------------------------------------------------

    def wrap(self, controllers: Sequence[Controller]) -> tuple[RecordingController, RecordingController]:
        if len(controllers) != 2:
            raise ReplayWriteError("one controller per side")
        return (RecordingController(controllers[0], 0, self), RecordingController(controllers[1], 1, self))

    def start(self, state: SimState, run: RunState) -> None:
        if self._started:
            raise ReplayWriteError("the replay has already been started")
        if state.config != self._config:
            raise ReplayWriteError("the state belongs to a different config")
        if run.outcome is not None or state.tick >= self._config.t_max:
            raise ReplayWriteError("a replay needs a running episode (no outcome, tick < T_max)")
        streams = sorted(set(run.streams) - {WORLD_STREAM_ID})
        if WORLD_STREAM_ID not in run.streams:
            raise ReplayWriteError("the run has no world stream")
        for seat in self._seats:
            if seat.stream_id and seat.stream_id not in streams:
                raise ReplayWriteError(f"seat stream {seat.stream_id!r} is not registered in the run")
        fresh = reset(self._config, run.root_seed, streams)
        is_reset = (not self._embed and state.tick == 0 and self._event_index == 0
                    and checkpoint_hash(state, run) == checkpoint_hash(*fresh))
        if is_reset:
            start: dict[str, Any] = {"kind": "reset"}
        else:
            start = {"kind": "checkpoint", "checkpoint": canonical_loads(checkpoint_to_bytes(state, run)),
                     "checkpoint_hash": checkpoint_hash(state, run)}
        self._emit({
            "type": "header", "format": REPLAY_FORMAT, "replay_schema": REPLAY_SCHEMA_VERSION,
            "engine": REPLAY_ENGINE_PIN, "contract": contract_versions(),
            "config": config_to_canonical(self._config), "config_hash": config_hash(self._config),
            "root_seed": u64_to_str(run.root_seed), "streams": streams,
            "seats": [seat.to_canonical(i) for i, seat in enumerate(self._seats)],
            "start": start, "cursor": {"tick": state.tick, "event_index": self._event_index},
            "source": self._source, "meta": self._meta,
        })
        self._started = True
        self._next_tick = state.tick
        self._body = frames.build_body(state)
        self._snapshot(state, run)

    def record_tick(self, state: SimState, run: RunState, result: TickResult) -> None:
        """Write one executed tick. Call it with the state *after* ``step_tick`` (``play(on_tick=...)`` does)."""
        self._require_open()
        config = self._config
        pending, self._pending = self._pending, {}
        if self._terminal:
            raise ReplayWriteError("the episode has ended: there is no further tick to record")
        if result.mode is not TickMode.FULL or result.tick != self._next_tick:
            raise ReplayWriteError("only consecutive FULL ticks can be recorded")
        if result.outcome != run.outcome:
            raise ReplayWriteError("the tick result and the run state disagree about the outcome")
        tick = result.tick
        if tick % config.policy_interval_k == 0:
            if sorted(pending) != [0, 1] or any(p["tick"] != tick for p in pending.values()):
                raise ReplayWriteError(
                    f"tick {tick} is a policy tick but the recording controllers did not see both decisions "
                    "(wrap the controllers with ReplayWriter.wrap)")
            self._emit({"type": "decision", "tick": tick, "seats": [
                {k: v for k, v in pending[side].items() if k != "tick"} for side in (0, 1)]})
            self._decisions += 1
        elif pending:
            raise ReplayWriteError(f"tick {tick} is not a policy tick but a decision was recorded")
        for event in result.events:
            canonical = event.to_canonical()
            self._hasher.add(canonical)
            self._emit({"type": "event", "i": self._event_index, "e": canonical})
            self._event_index += 1
        body = frames.build_body(state)
        boundary = frames.BOUNDARY_START if run.outcome is None else frames.BOUNDARY_TERMINAL
        frame = frames.frame_dict(state.tick, boundary, frames.terminal_object(run.outcome), body)
        self._emit({"type": "delta", "tick": tick, "to_tick": state.tick, "to_boundary": boundary,
                    "patch": frames.diff_body(self._body, body), "frame_hash": frames.frame_hash(frame)})
        self._body = body
        self._ticks += 1
        self._next_tick = state.tick
        if run.outcome is not None:
            self._terminal = True
            self._snapshot(state, run)
        elif state.tick % config.policy_interval_k == 0:
            self._snapshot(state, run)

    def finish(self, state: SimState, run: RunState) -> ReplaySummary:
        """Close the file with the result record (and the cutoff frame of a truncated run)."""
        self._require_open()
        if self._pending:
            raise ReplayWriteError("a decision is pending: the last tick was not recorded")
        if state.tick != self._next_tick:
            raise ReplayWriteError("the state is not the one after the last recorded tick")
        if run.outcome is None and self._last_start_snapshot != state.tick:
            self._snapshot(state, run)  # harness cutoff: the last state is a legitimate "start" boundary
        digest = self._hasher.hexdigest()
        result = {
            "type": "result", "finished": run.outcome is not None, "outcome": outcome_object(run.outcome),
            "state_tick": state.tick, "ticks_executed": self._ticks, "decision_count": self._decisions,
            "event_count": self._hasher.count, "events_hash": digest,
            "final_state_hash": state_hash(state), "final_checkpoint_hash": checkpoint_hash(state, run),
            "record_count": self._seq, "chain_head": self._chain,
        }
        self._emit(result)
        self._finished = True
        return ReplaySummary(self._seq, self._bytes, self._ticks, self._decisions, self._hasher.count,
                             run.outcome is not None, digest, result["final_state_hash"],
                             result["final_checkpoint_hash"])

    # -- internals ----------------------------------------------------------------------------------------------

    def _require_open(self) -> None:
        if not self._started or self._finished:
            raise ReplayWriteError("the replay is not open (call start() first, nothing after finish())")

    def _emit(self, record: dict[str, Any]) -> None:
        line, self._chain = seal_record({"seq": self._seq, **record}, self._chain)
        self._sink.write(line)
        self._bytes += len(line)
        self._seq += 1

    def _snapshot(self, state: SimState, run: RunState) -> None:
        boundary = frames.BOUNDARY_START if run.outcome is None else frames.BOUNDARY_TERMINAL
        frame = frames.frame_dict(state.tick, boundary, frames.terminal_object(run.outcome), self._body)  # type: ignore[arg-type]
        self._emit({"type": "snapshot", "frame": frame, "frame_hash": frames.frame_hash(frame)})
        if boundary == frames.BOUNDARY_START:
            self._last_start_snapshot = state.tick

    def _note_decision(self, side: int, tick: int, observation: ObservationV2, answer: object,
                       before: dict[str, Pcg32], run: RunState) -> None:
        if not isinstance(answer, ValidatedActionV2):
            return  # the engine refuses it (technical error); nothing is recorded for a failed tick
        if set(run.streams) != set(before):
            raise ReplayWriteError("a controller added or removed an RNG stream during its decision")
        changed = [sid for sid in sorted(run.streams) if run.streams[sid] != before[sid]]
        seat = self._seats[side]
        if seat.stream_id:
            if any(sid != seat.stream_id for sid in changed):
                raise ReplayWriteError(f"seat {side} consumed an RNG stream other than {seat.stream_id!r}: {changed}")
            post = run.streams[seat.stream_id]
            rng = [{"stream_id": seat.stream_id, "draws": count_draws(before[seat.stream_id], post),
                    "state_after": post.to_canonical()}]
        else:
            if changed:
                raise ReplayWriteError(f"seat {side} declares no RNG stream but consumed {changed}")
            rng = []
        self._pending[side] = {
            "tick": tick, "side": side, "obs_hash": observation_hash(observation),
            "mask": mask_string(observation), "action": action_object(answer.action), "rng": rng,
        }


def _seats_for(controllers: Sequence[Controller], seats: Sequence[SeatSpec] | None) -> tuple[SeatSpec, ...]:
    return tuple(seats) if seats is not None else tuple(describe_controller(c) for c in controllers)


def record_play(
    path: str | os.PathLike[str],
    state: SimState,
    run: RunState,
    controllers: Sequence[Controller],
    max_ticks: int | None = None,
    *,
    seats: Sequence[SeatSpec] | None = None,
    source: str = "",
    meta: Mapping[str, str] | None = None,
    event_index_base: int = 0,
    embed_checkpoint: bool = False,
    check_invariants: bool = False,
    stop_before_tick: int | None = None,
    on_tick: Callable[[SimState, RunState, TickResult], None] | None = None,
) -> tuple[EpisodeResult, ReplaySummary]:
    """Continue ``(state, run)`` with ``play`` and record it. A fresh ``reset`` state starts with ``{"kind": "reset"}``,
    any other state (e.g. restored from a checkpoint) embeds the checkpoint. The file appears under ``path`` only
    when it is complete; on any error the partial file is removed."""
    controllers = tuple(controllers)
    target = Path(path)
    part = target.with_name(target.name + ".part")
    try:
        with open(part, "wb") as sink:
            writer = ReplayWriter(sink, state.config, _seats_for(controllers, seats), source=source, meta=meta,
                                  event_index_base=event_index_base, embed_checkpoint=embed_checkpoint)
            writer.start(state, run)

            def observe(s: SimState, r: RunState, result: TickResult) -> None:
                writer.record_tick(s, r, result)
                if on_tick is not None:
                    on_tick(s, r, result)

            events, executed = play(state, run, writer.wrap(controllers), max_ticks,
                                    check_invariants=check_invariants, stop_before_tick=stop_before_tick,
                                    on_tick=observe)
            summary = writer.finish(state, run)
        os.replace(part, target)
    except BaseException:
        try:
            part.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return summarize(state, run, events, executed), summary


def record_episode(
    path: str | os.PathLike[str],
    config: SimConfig,
    seed: int,
    controller_a: Controller,
    controller_b: Controller,
    max_ticks: int | None = None,
    *,
    seats: Sequence[SeatSpec] | None = None,
    source: str = "",
    meta: Mapping[str, str] | None = None,
    check_invariants: bool = False,
    extra_streams: Sequence[str] = (),
) -> tuple[EpisodeResult, ReplaySummary]:
    """``run_episode`` with a replay: ``reset`` -> ``step_tick`` until terminal (or ``max_ticks``) -> JSONL at ``path``."""
    controllers = (controller_a, controller_b)
    state, run = reset(config, seed)
    _register_streams(run, controllers, extra_streams)
    return record_play(path, state, run, controllers, max_ticks, seats=seats, source=source, meta=meta,
                       check_invariants=check_invariants)


__all__ = [
    "RecordingController", "ReplaySummary", "ReplayWriter", "action_object", "count_draws", "describe_controller",
    "mask_string", "observation_hash", "outcome_object", "record_episode", "record_play",
]
