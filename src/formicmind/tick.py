"""Tick orchestrator (movement-only skeleton).

One ordered rule system, ``Policy -> Completion -> Arrival -> Combat -> QueenTerminal ->
Harvest -> Upkeep -> Lay -> Regeneration -> AutomaticDepart -> Terminal/Log``
(FOR-1 v0.2.1 errata A). Implemented here: Policy, Arrival (with the post-arrival intel
sensor), AutomaticDepart and the window accounting of Terminal/Log. **Completion, Combat,
QueenTerminal, Harvest, Upkeep, Lay and Regeneration are not implemented and do nothing**;
a run is therefore not a match, and ``step_tick`` refuses to run unless the caller passes
``movement_only=True``.

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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .actions import ActionV2, ValidatedActionV2, get_action_mask, validate_action
from .events import Event, Phase
from .intel import IntelPhase, update_local_intel
from .model import SimState
from .movement import arrive_units, dispatch_available
from .observation import ObservationV2, build_observation
from .orders import apply_joint_orders
from .policy import Policy, decide
from .rules import Q_TOTAL, SIDES
from .runstate import RunState
from .snapshot import WindowStats, begin_policy_phase


class IncompleteRulesError(RuntimeError):
    """The tick engine implements movement only; pass ``movement_only=True`` to acknowledge it."""


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
    intel_backup = [c.intel.clone() for c in state.colonies]
    run_backup = run.clone()
    try:
        snapshot = begin_policy_phase(state, (run.windows[0], run.windows[1]))
        observations = [build_observation(snapshot, side) for side in SIDES]  # before any decision
        actions = tuple(ctx.controllers[side].act(ctx.tick, observations[side], run) for side in SIDES)
        if not all(isinstance(a, ValidatedActionV2) for a in actions):
            raise TypeError("controllers must return ValidatedActionV2")
    except BaseException:
        for colony, backup in zip(state.colonies, intel_backup):
            colony.intel = backup
        run.streams, run.windows = run_backup.streams, run_backup.windows
        raise
    ctx.events.extend(apply_joint_orders(state, actions, Phase.POLICY))
    ctx.events.extend(dispatch_available(state, Phase.POLICY))
    run.windows = [WindowStats(), WindowStats()]  # a new policy window starts


def _phase_arrival(ctx: TickContext) -> None:
    ctx.events.extend(arrive_units(ctx.state, Phase.ARRIVAL))
    for side in SIDES:  # sensor sub-phase: after Arrival, before Combat
        update_local_intel(ctx.state, side, IntelPhase.POST_ARRIVAL_PRE_COMBAT)


def _phase_automatic_depart(ctx: TickContext) -> None:
    ctx.events.extend(dispatch_available(ctx.state, Phase.AUTOMATIC_DEPART))


def _phase_terminal_log(ctx: TickContext) -> None:
    # Window accounting only; the terminal check itself needs combat/economy (not implemented).
    ctx.run.windows = [
        WindowStats(w.harvest, w.upkeep_paid, w.egg_spend, w.starvation_debt_delta, w.window_ticks + 1)
        for w in ctx.run.windows
    ]


PhaseFunc = Callable[[TickContext], None]

DEFAULT_PHASES: dict[Phase, PhaseFunc | None] = {
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
IMPLEMENTED_PHASES = tuple(p for p, f in DEFAULT_PHASES.items() if f is not None)
NOT_IMPLEMENTED_PHASES = tuple(p for p, f in DEFAULT_PHASES.items() if f is None)


def step_tick(
    state: SimState,
    run: RunState,
    controllers: Sequence[Controller],
    *,
    movement_only: bool = False,
    phases: Mapping[Phase, PhaseFunc | None] | None = None,
) -> TickResult:
    if not movement_only:
        raise IncompleteRulesError(
            "only movement phases are implemented (not implemented: "
            + ", ".join(p.name for p in NOT_IMPLEMENTED_PHASES) + "); pass movement_only=True"
        )
    if len(controllers) != len(SIDES):
        raise ValueError("one controller per side")
    if state.tick >= state.config.t_max:
        raise ValueError("the episode horizon T_max has been reached")
    table = DEFAULT_PHASES if phases is None else phases
    ctx = TickContext(state=state, run=run, controllers=controllers, tick=state.tick)
    reached: list[PhaseRecord] = []
    for phase in Phase:
        func = table[phase]
        reached.append(PhaseRecord(phase, func is not None))
        if func is not None:
            func(ctx)
            if ctx.terminal is not None:
                return TickResult(ctx.tick, tuple(ctx.events), tuple(reached), terminal=ctx.terminal)
    state.tick = ctx.tick + 1
    return TickResult(ctx.tick, tuple(ctx.events), tuple(reached))
