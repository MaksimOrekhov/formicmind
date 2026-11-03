"""Tick orchestrator: the one ordered rule system of L0 (FOR-11 movement, FOR-12 economy, FOR-13 combat).

``Policy -> Completion -> Arrival -> Combat -> QueenTerminal -> Harvest -> Upkeep -> Lay -> Regeneration ->
AutomaticDepart -> Terminal/Log`` (FOR-1 v0.2.1 errata A). There is a single implementation of the tick.

* ``step_tick(state, run, controllers)`` with no flags executes **all** phases (``TickMode.FULL``) and
  returns ``physics_complete=True``. This is the only mode in which a run is a match.
* ``movement_only=True`` and ``partial_economy=True`` are kept as explicitly **incomplete development modes**
  (no Combat, no QueenTerminal, no terminal check; ``physics_complete=False``). They fill fewer entries of the
  same phase table and use the same phase functions; they exist for the FOR-11/FOR-12 comparison tests.

Boundary: ``step_tick`` takes the state *at the start of tick t* (``state.tick == t``, before ``PRE_POLICY``),
executes tick ``t`` and leaves ``state.tick == t + 1`` - ``state.tick`` counts **completed** ticks.

Terminal handling (``terminal.py``): when ``QueenTerminal`` ends the episode the remaining phases of the tick are
not executed, ``state.tick`` stays ``t`` (the tick is incomplete) and ``run.outcome`` is set. When Terminal/Log
ends it (economic collapse, timeout) the tick is complete, ``state.tick == t + 1`` and ``run.outcome`` is set.
After that ``step_tick`` raises ``EpisodeFinishedError``: there is no "next" tick. ``state.tick >= T_max`` is
also refused, so an episode never runs beyond ticks ``0 .. T_max - 1``.

Policy phase (only when ``t % K == 0``): ``PRE_POLICY`` for both sides -> one frozen
``PolicySnapshot`` (windows come from ``RunState``) -> both observations -> both
controllers decide on those observations -> only then orders are applied atomically,
the available units are dispatched, and the policy windows restart. If any controller or
validator fails, the state, the run state and the RNG streams are restored untouched (a technical
failure of a controller is an error, never a loss). Between policy ticks the standing orders persist
and no controller is called.
"""

from __future__ import annotations

import enum
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .actions import ActionV2, ValidatedActionV2, get_action_mask, validate_action
from .combat import resolve_combat
from .economy import complete_hatching, harvest, lay, regenerate, upkeep
from .events import Event, Phase
from .intel import IntelPhase, update_local_intel
from .ledger import check_transition
from .model import SimState
from .movement import arrive_units, dispatch_available
from .observation import ObservationV2, build_observation
from .orders import apply_joint_orders
from .policy import Policy, decide
from .rules import Q_TOTAL, SIDES
from .runstate import RunState
from .snapshot import WindowStats, begin_policy_phase
from .terminal import EpisodeOutcome, evaluate_terminal, terminal_event


class EpisodeFinishedError(RuntimeError):
    """The episode already has an outcome: there is no further tick to execute."""


class TickMode(enum.Enum):
    """``FULL`` is the match; the other two are explicitly incomplete development modes."""

    FULL = "FULL (ALL PHASES)"
    MOVEMENT_ONLY = "MOVEMENT ONLY (NO ECONOMY/COMBAT/TERMINAL)"
    PARTIAL_ECONOMY = "PARTIAL (NO COMBAT/TERMINAL)"


class Controller(Protocol):
    def act(self, tick: int, observation: ObservationV2, run: RunState) -> ValidatedActionV2: ...


@dataclass(frozen=True, slots=True)
class PolicyController:
    """A ``Policy`` decoded and validated by the single shared path (``decide``)."""

    policy: Policy
    stream_id: str  # role-keyed stream ``policy:<policy_id>:<replicate>``

    def act(self, tick: int, observation: ObservationV2, run: RunState) -> ValidatedActionV2:
        return decide(self.policy, observation, run.stream(self.stream_id))


RESET_ACTION = ActionV2(lay_order=0, forage_share_q=(Q_TOTAL, 0, 0, 0, 0, 0),
                        army_choice=0, avoid_center=False)


@dataclass(frozen=True, slots=True)
class ScriptedController:
    """Thin test adapter: the latest scripted action at or before the tick, else the reset orders."""

    script: Mapping[int, ActionV2] = field(default_factory=dict)

    def act(self, tick: int, observation: ObservationV2, run: RunState) -> ValidatedActionV2:
        past = [t for t in self.script if t <= tick]
        action = self.script[max(past)] if past else RESET_ACTION
        return validate_action(observation, get_action_mask(observation), action)


@dataclass(frozen=True, slots=True)
class PhaseRecord:
    phase: Phase
    implemented: bool


@dataclass(frozen=True, slots=True)
class TickResult:
    tick: int  # the tick that was executed
    events: tuple[Event, ...]
    phases: tuple[PhaseRecord, ...]  # phases reached, in order
    terminal: str | None = None  # reason of a terminal outcome (``outcome.reason.value``), else None
    physics_complete: bool = False  # True only for a FULL tick whose reached phases are all implemented
    mode: TickMode | None = None  # the mode this tick was executed in
    outcome: EpisodeOutcome | None = None  # set when this tick ended the episode

    @property
    def missing_phases(self) -> tuple[Phase, ...]:
        """Phases reached that are still placeholders in this mode."""
        return tuple(r.phase for r in self.phases if not r.implemented)


@dataclass(slots=True)
class TickContext:
    state: SimState
    run: RunState
    controllers: Sequence[Controller]
    tick: int
    events: list[Event] = field(default_factory=list)
    terminal: str | None = None
    outcome: EpisodeOutcome | None = None


def _phase_policy(ctx: TickContext) -> None:
    state, run = ctx.state, ctx.run
    if ctx.tick % state.config.policy_interval_k != 0:
        return
    state_backup = state.clone()
    run_backup = run.clone()
    events_mark = len(ctx.events)
    try:
        snapshot = begin_policy_phase(state, (run.windows[0], run.windows[1]))
        observations = [build_observation(snapshot, side) for side in SIDES]  # before any decision
        masks = [get_action_mask(obs) for obs in observations]
        answers = tuple(ctx.controllers[side].act(ctx.tick, observations[side], run) for side in SIDES)
        if not all(isinstance(a, ValidatedActionV2) for a in answers):
            raise TypeError("controllers must return ValidatedActionV2")
        # ValidatedActionV2 can be forged by hand: never trust the wrapper, re-validate each
        # action against its own side's observation and mask before anything is applied.
        actions = tuple(
            validate_action(observations[side], masks[side], answers[side].action) for side in SIDES
        )
        ctx.events.extend(apply_joint_orders(state, actions, Phase.POLICY))
        ctx.events.extend(dispatch_available(state, Phase.POLICY))
        run.windows = [WindowStats(), WindowStats()]  # a new policy window starts
    except BaseException:
        # Whatever failed (controller, validator, commit or dispatch), the policy phase is
        # all-or-nothing: state (incl. intel), RNG streams, windows and events are restored.
        state.restore_from(state_backup)
        run.streams, run.windows = run_backup.streams, run_backup.windows
        del ctx.events[events_mark:]
        raise


def _phase_arrival(ctx: TickContext) -> None:
    ctx.events.extend(arrive_units(ctx.state, Phase.ARRIVAL))
    for side in SIDES:  # sensor sub-phase: after Arrival, before Combat
        update_local_intel(ctx.state, side, IntelPhase.POST_ARRIVAL_PRE_COMBAT)


def _phase_completion(ctx: TickContext) -> None:
    ctx.events.extend(complete_hatching(ctx.state, Phase.COMPLETION))


def _phase_harvest(ctx: TickContext) -> None:
    ctx.events.extend(harvest(ctx.state, ctx.run, Phase.HARVEST))


def _phase_upkeep(ctx: TickContext) -> None:
    ctx.events.extend(upkeep(ctx.state, ctx.run, Phase.UPKEEP))


def _phase_lay(ctx: TickContext) -> None:
    ctx.events.extend(lay(ctx.state, ctx.run, Phase.LAY))


def _phase_regeneration(ctx: TickContext) -> None:
    ctx.events.extend(regenerate(ctx.state, Phase.REGENERATION))


def _phase_automatic_depart(ctx: TickContext) -> None:
    ctx.events.extend(dispatch_available(ctx.state, Phase.AUTOMATIC_DEPART))


def _phase_combat(ctx: TickContext) -> None:
    ctx.events.extend(resolve_combat(ctx.state, Phase.COMBAT))


def _end(ctx: TickContext, outcome: EpisodeOutcome) -> None:
    ctx.outcome = outcome
    ctx.terminal = outcome.reason.value
    ctx.events.append(terminal_event(ctx.state, outcome))


def _phase_queen_terminal(ctx: TickContext) -> None:
    outcome = evaluate_terminal(ctx.state, ctx.tick, Phase.QUEEN_TERMINAL)
    if outcome is not None:
        _end(ctx, outcome)


def _account_window_tick(ctx: TickContext) -> None:
    ctx.run.windows = [
        WindowStats(w.harvest, w.upkeep_paid, w.egg_spend, w.starvation_debt_delta, w.window_ticks + 1)
        for w in ctx.run.windows
    ]


def _phase_terminal_log(ctx: TickContext) -> None:
    # Development modes: window accounting only; the terminal check needs Combat (FULL mode).
    _account_window_tick(ctx)


def _phase_terminal_log_full(ctx: TickContext) -> None:
    _account_window_tick(ctx)  # this tick was executed completely, whatever follows
    outcome = evaluate_terminal(ctx.state, ctx.tick, Phase.TERMINAL_LOG)
    if outcome is not None:
        _end(ctx, outcome)


PhaseFunc = Callable[[TickContext], None]

# Movement-only development mode (FOR-11): the economy and brood phases are placeholders.
MOVEMENT_ONLY_PHASES: dict[Phase, PhaseFunc | None] = {
    Phase.POLICY: _phase_policy,
    Phase.COMPLETION: None,
    Phase.ARRIVAL: _phase_arrival,
    Phase.COMBAT: None,
    Phase.QUEEN_TERMINAL: None,
    Phase.HARVEST: None,
    Phase.UPKEEP: None,
    Phase.LAY: None,
    Phase.REGENERATION: None,
    Phase.AUTOMATIC_DEPART: _phase_automatic_depart,
    Phase.TERMINAL_LOG: _phase_terminal_log,
}
# Partial economy mode (FOR-12): Completion, Harvest, Upkeep, Lay and Regeneration are added; still no
# Combat, QueenTerminal or terminal check.
PARTIAL_ECONOMY_PHASES: dict[Phase, PhaseFunc | None] = {
    **MOVEMENT_ONLY_PHASES,
    Phase.COMPLETION: _phase_completion,
    Phase.HARVEST: _phase_harvest,
    Phase.UPKEEP: _phase_upkeep,
    Phase.LAY: _phase_lay,
    Phase.REGENERATION: _phase_regeneration,
}
# The match (FOR-13): every phase is implemented.
FULL_PHASES: dict[Phase, PhaseFunc | None] = {
    **PARTIAL_ECONOMY_PHASES,
    Phase.COMBAT: _phase_combat,
    Phase.QUEEN_TERMINAL: _phase_queen_terminal,
    Phase.TERMINAL_LOG: _phase_terminal_log_full,
}
PHASE_TABLES: dict[TickMode, dict[Phase, PhaseFunc | None]] = {
    TickMode.FULL: FULL_PHASES,
    TickMode.MOVEMENT_ONLY: MOVEMENT_ONLY_PHASES,
    TickMode.PARTIAL_ECONOMY: PARTIAL_ECONOMY_PHASES,
}

# FOR-11 names, kept for the movement-only comparison mode.
DEFAULT_PHASES = MOVEMENT_ONLY_PHASES
IMPLEMENTED_PHASES = tuple(p for p, f in MOVEMENT_ONLY_PHASES.items() if f is not None)
NOT_IMPLEMENTED_PHASES = tuple(p for p, f in MOVEMENT_ONLY_PHASES.items() if f is None)
ECONOMY_IMPLEMENTED_PHASES = tuple(p for p, f in PARTIAL_ECONOMY_PHASES.items() if f is not None)
ECONOMY_NOT_IMPLEMENTED_PHASES = tuple(p for p, f in PARTIAL_ECONOMY_PHASES.items() if f is None)


def resolve_mode(movement_only: bool, partial_economy: bool) -> TickMode:
    if movement_only and partial_economy:
        raise ValueError("choose one development mode: movement_only or partial_economy")
    if partial_economy:
        return TickMode.PARTIAL_ECONOMY
    if movement_only:
        return TickMode.MOVEMENT_ONLY
    return TickMode.FULL


def step_tick(
    state: SimState,
    run: RunState,
    controllers: Sequence[Controller],
    *,
    movement_only: bool = False,
    partial_economy: bool = False,
    phases: Mapping[Phase, PhaseFunc | None] | None = None,
    check_invariants: bool = False,
    on_phase_timing: Callable[[Phase, int], None] | None = None,
) -> TickResult:
    """Execute tick ``state.tick``. Without flags every phase runs (``TickMode.FULL``) and the tick may end the
    episode (``run.outcome``). ``check_invariants`` re-verifies the food ledger, the unit rules and the combat
    damage conservation of the finished tick against its events (``ledger.check_transition``); it is for tests
    and smoke runs. Optional ``on_phase_timing`` observes each reached phase and receives its elapsed nanoseconds;
    it is intended only for separate diagnostics and must not mutate the simulation."""
    mode = resolve_mode(movement_only, partial_economy)
    if len(controllers) != len(SIDES):
        raise ValueError("one controller per side")
    if run.outcome is not None:
        raise EpisodeFinishedError(
            f"the episode ended ({run.outcome.reason.value}, winner {run.outcome.winner.value}, "
            f"tick {run.outcome.terminal_tick}): there is no next tick")
    if state.tick >= state.config.t_max:
        raise ValueError("the episode horizon T_max has been reached")
    table = PHASE_TABLES[mode] if phases is None else phases
    before = state.clone() if check_invariants else None
    ctx = TickContext(state=state, run=run, controllers=controllers, tick=state.tick)
    reached: list[PhaseRecord] = []
    for phase in Phase:
        func = table[phase]
        reached.append(PhaseRecord(phase, func is not None))
        if on_phase_timing is None:
            if func is not None:
                func(ctx)
        else:
            started_ns = time.perf_counter_ns()
            if func is not None:
                func(ctx)
            on_phase_timing(phase, time.perf_counter_ns() - started_ns)
        if ctx.terminal is not None and phase != Phase.TERMINAL_LOG:
            # Mid-tick terminal: the rest of the tick does not run and the tick is not completed.
            run.outcome = ctx.outcome
            if before is not None and ctx.outcome is not None:
                check_transition(before, state, ctx.events)
            return _result(ctx, reached, mode)
    state.tick = ctx.tick + 1
    run.outcome = ctx.outcome  # Terminal/Log ended the episode (or None: it goes on)
    if before is not None:
        check_transition(before, state, ctx.events)
    return _result(ctx, reached, mode)


def _result(ctx: TickContext, reached: list[PhaseRecord], mode: TickMode) -> TickResult:
    complete = mode is TickMode.FULL and all(r.implemented for r in reached)
    return TickResult(ctx.tick, tuple(ctx.events), tuple(reached), terminal=ctx.terminal,
                      physics_complete=complete, mode=mode, outcome=ctx.outcome)
