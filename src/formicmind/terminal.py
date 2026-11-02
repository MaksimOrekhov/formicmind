"""Episode outcome and terminal rules (FOR-1 v0.2.1 section 9 + audit D3; FOR-13).

Priority is structural: **queen death > economic collapse > timeout**.

* ``QueenTerminal`` runs right after Combat. A queen at ``queen_hp == 0`` ends the episode *in that tick*:
  the other side wins; both queens destroyed by the same simultaneous combat is a draw. The remaining
  phases (Harvest, Upkeep, Lay, Regeneration, AutomaticDepart, Terminal/Log) are **not executed**, so the tick
  is incomplete and ``state.tick`` stays at ``terminal_tick``.
* ``Terminal/Log`` closes a normal tick (after the economy and AutomaticDepart). A colony is *economically
  collapsed* when it has no adult worker, no already paid worker in its brood and
  ``food < eggs_per_lay * worker_egg_cost`` (the cost of a whole worker batch; 10 at the default
  ``eggs_per_lay = 1``, which is the literal FOR-1 rule). One collapsed side loses, both at once is a draw (D3).
  Otherwise a tick with ``tick + 1 == t_max`` ends in a timeout, always a draw (FOR-1: no winner by food or
  "strength"). Ticks are ``0 .. t_max - 1``; a timeout therefore has ``terminal_tick == t_max - 1`` and
  leaves ``state.tick == t_max``.

``state.tick`` counts **completed** ticks: it is ``terminal_tick + 1`` after an end-of-tick terminal and stays
``terminal_tick`` after a queen terminal. The outcome lives in ``RunState`` (and the checkpoint), not in the
game state, so ``state_hash`` is purely the mechanics and ``checkpoint_hash`` knows whether the episode is over.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

from .events import Phase, TerminalEvent
from .invariants import InvariantError
from .model import SimState
from .rules import SIDES, WORKER


class Winner(enum.Enum):
    A = "A"
    B = "B"
    DRAW = "draw"


class Reason(enum.Enum):
    QUEEN_DEATH = "queen_death"
    ECONOMIC_COLLAPSE = "economic_collapse"
    TIMEOUT = "timeout"


SCORE = {Winner.A: (1.0, 0.0), Winner.B: (0.0, 1.0), Winner.DRAW: (0.5, 0.5)}  # FOR-1 section 9


@dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    winner: Winner
    reason: Reason
    terminal_tick: int  # the tick t (0-based) in which the episode ended
    terminal_phase: int  # Phase.QUEEN_TERMINAL or Phase.TERMINAL_LOG
    phases_executed: int  # phases of tick ``terminal_tick`` reached, the terminal one included
    sides: tuple[int, ...]  # sides whose queen died / whose economy collapsed; empty for a timeout

    @property
    def tick_completed(self) -> bool:
        """True when the whole tick ran (end-of-tick terminal); False when it was cut at QueenTerminal."""
        return self.terminal_phase == Phase.TERMINAL_LOG

    @property
    def state_tick(self) -> int:
        """``state.tick`` of the terminal state."""
        return self.terminal_tick + (1 if self.tick_completed else 0)

    @property
    def score(self) -> tuple[float, float]:
        return SCORE[self.winner]

    def to_canonical(self) -> dict[str, Any]:
        return {
            "winner": self.winner.value, "reason": self.reason.value,
            "terminal_tick": self.terminal_tick, "terminal_phase": int(self.terminal_phase),
            "phases_executed": self.phases_executed, "sides": list(self.sides),
        }


def winner_for(sides: tuple[int, ...]) -> Winner:
    """Outcome for the set of losing sides: none (timeout) or both -> draw, one -> the other side wins."""
    if len(sides) == 1:
        return Winner.B if sides[0] == 0 else Winner.A
    return Winner.DRAW


def make_outcome(reason: Reason, sides: tuple[int, ...], tick: int, phase: int) -> EpisodeOutcome:
    sides = tuple(sorted(sides))
    if reason is Reason.TIMEOUT:
        sides = ()
    return EpisodeOutcome(winner_for(sides), reason, tick, int(phase), int(phase) + 1, sides)


def worker_batch_cost(state: SimState) -> int:
    bal = state.config.balance
    return bal.eggs_per_lay * bal.worker_egg_cost


def is_collapsed(state: SimState, side: int) -> bool:
    """No adult worker now, no paid worker in the brood, and food below one whole worker batch."""
    colony = state.colonies[side]
    if any(u.caste == WORKER for u in colony.units):
        return False
    if any(b.caste == WORKER for b in colony.brood):
        return False
    return colony.food < worker_batch_cost(state)


def collapsed_sides(state: SimState) -> tuple[int, ...]:
    return tuple(side for side in SIDES if is_collapsed(state, side))


def dead_queens(state: SimState) -> tuple[int, ...]:
    return tuple(side for side in SIDES if state.colonies[side].queen_hp <= 0)


def evaluate_terminal(state: SimState, tick: int, phase: int) -> EpisodeOutcome | None:
    """The one place that orders the terminal conditions: queen death > economic collapse > timeout.

    ``phase == QUEEN_TERMINAL`` looks at the queens only; ``phase == TERMINAL_LOG`` (end of a full tick)
    checks all three in that order. ``tick`` is the executed tick ``t`` (``state.tick`` may still equal it).
    """
    dead = dead_queens(state)
    if dead:
        return make_outcome(Reason.QUEEN_DEATH, dead, tick, phase)
    if phase != Phase.TERMINAL_LOG:
        return None
    collapsed = collapsed_sides(state)
    if collapsed:
        return make_outcome(Reason.ECONOMIC_COLLAPSE, collapsed, tick, phase)
    if tick + 1 >= state.config.t_max:
        return make_outcome(Reason.TIMEOUT, (), tick, phase)
    return None


def terminal_event(state: SimState, outcome: EpisodeOutcome) -> TerminalEvent:
    return TerminalEvent(
        tick=outcome.terminal_tick, phase=int(outcome.terminal_phase), reason=outcome.reason.value,
        winner=outcome.winner.value, sides=outcome.sides, phases_executed=outcome.phases_executed,
        queen_hp=(state.colonies[0].queen_hp, state.colonies[1].queen_hp),
        tick_completed=outcome.tick_completed)


def validate_outcome(state: SimState, outcome: EpisodeOutcome | None, *, dev_mode: bool = False) -> None:
    """Check that ``outcome`` (``None`` = episode still running) is consistent with ``state``.

    Used when a checkpoint is restored: a terminal flag that disagrees with the mechanics is rejected.

    A running episode (``outcome is None``) must have a live queen on both sides and must not stand on or
    beyond the horizon: a FULL episode always ends in a timeout at the end of tick ``T_max - 1``, so
    ``state.tick >= T_max`` without an outcome is a forged or corrupted "finished: false" checkpoint
    (FOR-14). The only legitimate way to reach ``state.tick == T_max`` without an outcome is an incomplete
    development mode (``movement_only`` / ``partial_economy``), which never decides anything; such a state
    can only be restored with ``dev_mode=True`` (no outcome is invented for it). ``state.tick > T_max`` is
    impossible in every mode (``step_tick`` refuses to run at the horizon) and is always rejected.
    """
    dead = dead_queens(state)
    if outcome is None:
        if dead:
            raise InvariantError("a queen is dead but the run records no terminal outcome")
        t_max = state.config.t_max
        if state.tick > t_max:
            raise InvariantError(f"state.tick {state.tick} is beyond the horizon T_max {t_max}")
        if state.tick == t_max and not dev_mode:
            raise InvariantError(
                f"state.tick == T_max ({t_max}) but the run records no outcome: a full episode ends "
                "with a timeout in tick T_max - 1 (restore a development-mode run with dev_mode=True)")
        return

    def need(condition: bool, message: str) -> None:
        if not condition:
            raise InvariantError(f"inconsistent episode outcome: {message}")

    t_max = state.config.t_max
    need(isinstance(outcome.winner, Winner) and isinstance(outcome.reason, Reason), "bad enum fields")
    need(0 <= outcome.terminal_tick < t_max, "terminal_tick outside 0..t_max-1")
    need(outcome.terminal_phase in (Phase.QUEEN_TERMINAL, Phase.TERMINAL_LOG), "bad terminal phase")
    need(outcome.phases_executed == int(outcome.terminal_phase) + 1, "phases_executed")
    need(state.tick == outcome.state_tick, "state.tick does not match the terminal boundary")
    need(outcome.sides == tuple(sorted(set(outcome.sides))) and set(outcome.sides) <= set(SIDES), "sides")
    need(outcome.winner is winner_for(outcome.sides), "winner does not follow from the losing sides")
    if outcome.reason is Reason.QUEEN_DEATH:
        need(outcome.terminal_phase == Phase.QUEEN_TERMINAL, "queen death ends the tick at QueenTerminal")
        need(outcome.sides == dead and dead, "the losing sides must be exactly the sides with queen_hp == 0")
        return
    need(not dead, "a dead queen outranks every other reason")
    need(outcome.terminal_phase == Phase.TERMINAL_LOG, "collapse/timeout are decided in Terminal/Log")
    if outcome.reason is Reason.ECONOMIC_COLLAPSE:
        need(outcome.sides == collapsed_sides(state) and outcome.sides, "sides must be exactly the collapsed colonies")
    else:
        need(outcome.sides == () and outcome.winner is Winner.DRAW, "a timeout is a draw")
        need(outcome.terminal_tick == t_max - 1, "a timeout happens only in the last tick")
        need(not collapsed_sides(state), "a collapse outranks the timeout")


def outcome_from_canonical(data: Any) -> EpisodeOutcome:
    from .canonical import CanonicalError

    keys = {"winner", "reason", "terminal_tick", "terminal_phase", "phases_executed", "sides"}
    if type(data) is not dict or set(data) != keys:
        raise CanonicalError("outcome: expected an object with exactly " + ", ".join(sorted(keys)))
    ints = ("terminal_tick", "terminal_phase", "phases_executed")
    if any(type(data[k]) is not int for k in ints):
        raise CanonicalError("outcome: terminal_tick/terminal_phase/phases_executed must be ints")
    sides = data["sides"]
    if type(sides) is not list or any(type(s) is not int for s in sides):
        raise CanonicalError("outcome.sides: expected an int array")
    try:
        winner, reason = Winner(data["winner"]), Reason(data["reason"])
        phase = Phase(data["terminal_phase"])
    except ValueError as exc:
        raise CanonicalError(f"outcome: {exc}") from exc
    return EpisodeOutcome(winner, reason, data["terminal_tick"], int(phase), data["phases_executed"], tuple(sides))
