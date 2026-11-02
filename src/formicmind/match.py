"""Thin episode runner over the single ``step_tick`` (FOR-13). No training, no tournament, no second engine.

``run_episode(config, seed, controller_a, controller_b, max_ticks)`` does ``reset`` and calls
``step_tick`` until the episode has an outcome (queen death, economic collapse or timeout) or ``max_ticks`` ticks
have been executed, and returns a structured ``EpisodeResult`` (winner, reason, terminal tick, score, queens, food,
population, hashes, the omniscient event log). ``play`` continues an existing ``(state, run)`` - for example one
restored from a checkpoint - with the same loop.

Rules for the harness:

* RNG streams belong to the *policy* (``policy:<policy_id>:<replicate>``), never to a seat; controllers that
  expose a ``stream_id`` get it registered, and two seats may not share one stream (that would make one seat's
  samples depend on the other's order). The ``seed`` is a plain engineering number: callers must not pass FOR-4
  sealed seeds for functional runs.
* A controller error or an invalid action is a *technical* failure: the Policy phase rolls the tick back and the
  exception propagates. It is never turned into a loss, a draw or an outcome.
* ``max_ticks`` is a harness cutoff for tests, not a game rule: a run stopped by it has ``outcome is None`` and
  ``truncated is True``. ``max_ticks`` can never exceed ``config.t_max``, so a match always ends by ``T_max``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .config import SimConfig
from .events import Event, events_hash
from .model import SimState
from .reset import reset
from .rules import SOLDIER, WORKER
from .runstate import RunState
from .serialization import checkpoint_hash, config_hash, state_hash
from .terminal import EpisodeOutcome
from .tick import Controller, TickResult, step_tick


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    outcome: EpisodeOutcome | None  # None: stopped by ``max_ticks`` before a terminal outcome
    truncated: bool
    ticks_executed: int  # step_tick calls of this call (an incomplete terminal tick counts)
    final_tick: int  # state.tick of the final state (completed ticks)
    score: tuple[float, float] | None  # FOR-1 section 9: win/draw/loss = 1/0.5/0
    queen_hp: tuple[int, int]
    food: tuple[int, int]
    workers: tuple[int, int]
    soldiers: tuple[int, int]
    brood: tuple[int, int]
    event_counts: tuple[tuple[str, int], ...]
    events: tuple[Event, ...]  # omniscient log of this call, in execution order
    config_hash: str
    events_hash: str
    state_hash: str
    checkpoint_hash: str

    @property
    def winner(self) -> str | None:
        return None if self.outcome is None else self.outcome.winner.value

    @property
    def reason(self) -> str | None:
        return None if self.outcome is None else self.outcome.reason.value

    @property
    def terminal_tick(self) -> int | None:
        return None if self.outcome is None else self.outcome.terminal_tick


def _register_streams(run: RunState, controllers: Sequence[Controller], extra: Sequence[str]) -> None:
    seen: set[str] = set()
    for stream_id in [*(getattr(c, "stream_id", None) for c in controllers), *extra]:
        if stream_id is None:
            continue
        if stream_id in seen:
            raise ValueError(
                f"RNG stream {stream_id!r} is shared by both seats; use different policy ids or replicates")
        seen.add(stream_id)
        if stream_id not in run.streams:
            run.add_stream(stream_id)


def play(
    state: SimState,
    run: RunState,
    controllers: Sequence[Controller],
    max_ticks: int | None = None,
    *,
    check_invariants: bool = False,
    stop_before_tick: int | None = None,
    on_tick: Callable[[SimState, RunState, TickResult], None] | None = None,
) -> tuple[list[Event], int]:
    """Step until an outcome, ``max_ticks`` executed ticks, or ``state.tick == stop_before_tick``.

    Returns ``(events, executed ticks)``. ``stop_before_tick`` stops *before* executing that tick, e.g. to
    checkpoint right before a lethal combat. ``on_tick(state, run, result)`` is called after every executed tick
    with the state *after* it (the replay recorder, FOR-15); it only observes and must not change the state.
    """
    limit = state.config.t_max if max_ticks is None else max_ticks
    if limit < 0 or limit > state.config.t_max:
        raise ValueError(f"max_ticks must be within 0..T_max ({state.config.t_max})")
    events: list[Event] = []
    executed = 0
    while run.outcome is None and executed < limit and state.tick < state.config.t_max:
        if stop_before_tick is not None and state.tick >= stop_before_tick:
            break
        result = step_tick(state, run, controllers, check_invariants=check_invariants)
        events.extend(result.events)
        executed += 1
        if on_tick is not None:
            on_tick(state, run, result)
    return events, executed


def summarize(state: SimState, run: RunState, events: Sequence[Event], executed: int) -> EpisodeResult:
    outcome = run.outcome
    colonies = state.colonies

    def count(caste: int) -> tuple[int, int]:
        return tuple(sum(1 for u in c.units if u.caste == caste) for c in colonies)  # type: ignore[return-value]

    return EpisodeResult(
        outcome=outcome,
        truncated=outcome is None,
        ticks_executed=executed,
        final_tick=state.tick,
        score=None if outcome is None else outcome.score,
        queen_hp=(colonies[0].queen_hp, colonies[1].queen_hp),
        food=(colonies[0].food, colonies[1].food),
        workers=count(WORKER),
        soldiers=count(SOLDIER),
        brood=(len(colonies[0].brood), len(colonies[1].brood)),
        event_counts=tuple(sorted(Counter(e.kind for e in events).items())),
        events=tuple(events),
        config_hash=config_hash(state.config),
        events_hash=events_hash(list(events)),
        state_hash=state_hash(state),
        checkpoint_hash=checkpoint_hash(state, run),
    )


def run_episode(
    config: SimConfig,
    seed: int,
    controller_a: Controller,
    controller_b: Controller,
    max_ticks: int | None = None,
    *,
    check_invariants: bool = False,
    extra_streams: Sequence[str] = (),
) -> EpisodeResult:
    """``reset`` -> ``step_tick`` until terminal (or ``max_ticks``). Seats are A = ``controller_a``, B = ``controller_b``."""
    controllers = (controller_a, controller_b)
    state, run = reset(config, seed)
    _register_streams(run, controllers, extra_streams)
    events, executed = play(state, run, controllers, max_ticks, check_invariants=check_invariants)
    return summarize(state, run, events, executed)


__all__ = ["EpisodeResult", "play", "run_episode", "summarize"]
