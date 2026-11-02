"""Tick orchestrator (partial: movement skeleton FOR-11 + economy and brood FOR-12).

One ordered rule system, ``Policy -> Completion -> Arrival -> Combat -> QueenTerminal ->
Harvest -> Upkeep -> Lay -> Regeneration -> AutomaticDepart -> Terminal/Log``
(FOR-1 v0.2.1 errata A). There is a single implementation of the tick; the two *development
modes* below only choose which entries of the one phase table are filled:

* ``movement_only=True`` (FOR-11): Policy, Arrival (with the post-arrival intel sensor),
  AutomaticDepart and the window accounting of Terminal/Log. No economy, no brood.
* ``partial_economy=True`` (FOR-12, label ``PARTIAL (NO COMBAT/TERMINAL)``): additionally
  Completion, Harvest, Upkeep/Starvation, Lay and Regeneration.

**Combat, QueenTerminal and the Terminal check (queen death, economic collapse, timeout) are not
implemented in any mode**; a run is therefore never a match. ``TickResult.physics_complete`` is
always ``False`` and ``step_tick`` refuses to run unless the caller names a development mode.

Boundary: ``step_tick`` takes the state *at the start of tick t* (``state.tick == t``,
before ``PRE_POLICY``), executes tick ``t`` and leaves ``state.tick == t + 1``.

If a phase reports a terminal outcome (none does yet) the remaining phases of that tick
are not executed and ``state.tick`` is **not** advanced; ``TickResult.terminal`` names the
reason.

Policy phase (only when ``t % K == 0``): ``PRE_POLICY`` for both sides -> one frozen
``PolicySnapshot`` (windows come from ``RunState``) -> both observations -> both
controllers decide on those observations -> only then orders are applied atomically,
the available units are dispatched, and the policy windows restart. If any controller or
validator fails, the state, the run state and the RNG streams are restored untouched.
Between policy ticks the standing orders persist and no controller is called.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .actions import ActionV2, ValidatedActionV2, get_action_mask, validate_action
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


class IncompleteRulesError(RuntimeError):
    """Combat and the terminal check do not exist; name a development mode to acknowledge it."""


class TickMode(enum.Enum):
    """Explicit development modes of the (still incomplete) tick engine."""

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
    terminal: str | None = None
    physics_complete: bool = False  # always False until every phase is implemented
    mode: TickMode | None = None  # the development mode this tick was executed in

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


def _phase_terminal_log(ctx: TickContext) -> None:
    # Window accounting only; the terminal check itself needs combat (not implemented).
    ctx.run.windows = [
        WindowStats(w.harvest, w.upkeep_paid, w.egg_spend, w.starvation_debt_delta, w.window_ticks + 1)
        for w in ctx.run.windows
    ]


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
# Partial economy mode (FOR-12): only Combat, QueenTerminal and the terminal check are missing.
PARTIAL_ECONOMY_PHASES: dict[Phase, PhaseFunc | None] = {
    **MOVEMENT_ONLY_PHASES,
    Phase.COMPLETION: _phase_completion,
    Phase.HARVEST: _phase_harvest,
    Phase.UPKEEP: _phase_upkeep,
    Phase.LAY: _phase_lay,
    Phase.REGENERATION: _phase_regeneration,
}
PHASE_TABLES: dict[TickMode, dict[Phase, PhaseFunc | None]] = {
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
    raise IncompleteRulesError(
        "Combat, QueenTerminal and the Terminal check are not implemented, so a tick is not a match step; "
        "pass partial_economy=True (PARTIAL: NO COMBAT/TERMINAL) or movement_only=True "
        "(movement comparison; also no Completion/Harvest/Upkeep/Lay/Regeneration)"
    )


def step_tick(
    state: SimState,
    run: RunState,
    controllers: Sequence[Controller],
    *,
    movement_only: bool = False,
    partial_economy: bool = False,
    phases: Mapping[Phase, PhaseFunc | None] | None = None,
    check_invariants: bool = False,
) -> TickResult:
    """Execute tick ``state.tick``. ``check_invariants`` re-verifies the food ledger and the unit rules
    of the finished tick against its events (``ledger.check_transition``); it is for tests and smoke runs."""
    mode = resolve_mode(movement_only, partial_economy)
    if len(controllers) != len(SIDES):
        raise ValueError("one controller per side")
    if state.tick >= state.config.t_max:
        raise ValueError("the episode horizon T_max has been reached")
    table = PHASE_TABLES[mode] if phases is None else phases
    before = state.clone() if check_invariants else None
    ctx = TickContext(state=state, run=run, controllers=controllers, tick=state.tick)
    reached: list[PhaseRecord] = []
    for phase in Phase:
        func = table[phase]
        reached.append(PhaseRecord(phase, func is not None))
        if func is not None:
            func(ctx)
            if ctx.terminal is not None:
                return TickResult(ctx.tick, tuple(ctx.events), tuple(reached),
                                  terminal=ctx.terminal, mode=mode)
    state.tick = ctx.tick + 1
    if before is not None:
        check_transition(before, state, ctx.events)
    return TickResult(ctx.tick, tuple(ctx.events), tuple(reached), mode=mode)
