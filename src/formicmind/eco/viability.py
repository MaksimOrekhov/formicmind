"""Viability of a colony per world tick, the reward of an interval, and time-aware GAE (contracts 9.1, 9.2, 9.2a). Stdlib only.

``ViabilityTracker`` is **not part of the world rules**: it watches a running world from outside (after every ``step_world_tick``), reads only
the colony's *own* state (its cohorts, stored food, queen) and its own per-tick accounting from the tick record (food delivered into its nest,
upkeep due), and keeps integer fixed-point EMAs of them. It never reads another colony, the map or any kill / combat event, so a reward built from
it can pay for nothing but the colony's own viability (contracts 9.3).

Per tick ``tau`` (after the whole tick, Terminal/Log included), with ``Q = 65536``::

    E_pop   += (x_pop  - E_pop)  / 300          x_pop = workers + soldiers/2          (tau = 300 ticks)
    income  += (delivered - income) / 120       food actually delivered into the nest this tick
    burn    += (due - burn) / 120               upkeep due this tick (eggs are an investment, not burn)
    pop      = min(1, E_pop / 60)
    runway   = food / (burn - income)           if burn > income else +inf
    security = min(1, runway / 40) * gate       gate = min(1, E_pop / 10)   [reward v2; v1: gate = 1]
    v        = queen_alive * (0.6 pop + 0.4 security)        v_q = floor(Q v)

Reward versions (contracts 9.1): v1 is the first FOR-49 pilot's reward. It paid ``security = 1`` to a colony with no adults at all (no upkeep, so an
infinite runway), which PPO found and exploited. v2 multiplies ``security`` by the population gate ``min(1, E_pop / 10)``, so a hollow colony earns
nothing for its stored food. ``REWARD_VERSION`` is the version used for training; v1 stays available only to re-score old runs.

[CHOICE] The contract fixes tau = 300 only for the population EMA; income and burn use tau = 120 ticks (four planned decision windows), the
same memory as the windowed EMAs of the observation (``observation.next_rolling``, weight 1/4 per 30 ticks), because deliveries are lumpy.
All arithmetic is integer; a rounded-to-nearest EMA step keeps it free of the floor bias.

Reward v3 (contracts 9.1 / 9.5a) keeps the v2 formula and adds one **agent-episode** terminal, ``functional_extinction``: the colony is physically
``alive`` (queen alive, status ``alive``) but has no adult worker, no adult soldier and no worker brood, whatever its store holds. The tracker then ends
the agent's episode exactly like a death (``v_q(tau) = 0``, the penalty is paid once, nothing is recorded afterwards) and says why
(``terminal_reason``). This is a definition of the *reward and the episode*, not a world rule: the tracker never writes to the world, the colony keeps
its queen, its store and its status, and the physical collapse rule of contracts 9.5 is unchanged. A physical terminal (queen death, collapse) wins
over the functional one when both hold in the same tick, so a colony is terminated once and pays one penalty.

Interval bookkeeping: a colony that dies in tick ``tau_d`` (queen killed, or collapse, or - v3 - functional extinction) has ``v_q(tau_d) = 0`` and its terminal interval ends at
``death_tick = tau_d + 1`` (so a planned decision that was taken at the start of ``tau_d`` still has a one-tick interval and the penalty is paid
once, in it). Rewards are float64 and computed outside the rules from the integer trace.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

Q = 65536
POP_CAP = 60            # workers-equivalents at which ``pop`` saturates
TAU_POP = 300
TAU_ECON = 120
RUNWAY_CAP = 40
W_POP_Q, W_SECURITY_Q = 6, 4   # 0.6 / 0.4 in tenths
K_NOMINAL = 30
GAMMA_30 = 0.997
LAMBDA_30 = 0.95
DEATH_PENALTY = 1.0
REWARD_VERSION = 3          # reward revision: 1 = first FOR-49 pilot, 2 = + population gate on security, 3 = v2 + functional-extinction episode terminal
REWARD_VERSIONS = (1, 2, 3)
REASON_QUEEN_DEATH, REASON_COLLAPSE, REASON_FUNCTIONAL = "queen_death", "collapse", "functional_extinction"
GATE_POP_WE = 10            # workers-equivalents of E_pop at which the population gate of ``security`` opens fully (v2)


def _ema(previous: int, sample: int, tau: int) -> int:
    """Integer EMA step ``previous + (sample - previous) / tau`` rounded half up (works for negative differences)."""
    return previous + ((sample - previous) * 2 + tau) // (2 * tau)


def viability_q(pop_ema_q: int, income_ema_q: int, burn_ema_q: int, food: int, queen_alive: bool, version: int = REWARD_VERSION) -> int:
    """``v_q = floor(Q * v)`` of contracts 9.1 from the three fixed-point EMAs, the stored food and the queen (``version``: reward revision)."""
    if version not in REWARD_VERSIONS:
        raise ValueError(f"unknown reward version {version!r}")
    if not queen_alive:
        return 0
    pop_q = min(Q, pop_ema_q // POP_CAP)
    net = burn_ema_q - income_ema_q
    security_q = Q if net <= 0 else min(Q, max(0, food) * Q * Q // (RUNWAY_CAP * net))
    if version >= 2:
        security_q = security_q * min(Q, pop_ema_q // GATE_POP_WE) // Q
    return (W_POP_Q * pop_q + W_SECURITY_Q * security_q) // 10


def functional_extinction(adult_workers: int, adult_soldiers: int, worker_brood: int) -> bool:
    """The agent-episode terminal of reward v3 for a physically alive colony: nobody to work or fight and no worker brood that could become workers.
    Soldier brood alone does not count (it cannot forage), and the store is deliberately not part of the condition."""
    return adult_workers == 0 and adult_soldiers == 0 and worker_brood == 0


@dataclass(slots=True)
class _ColonyTrace:
    pop_ema_q: int
    income_ema_q: int
    burn_ema_q: int
    v_q: list[int] = field(default_factory=list)   # v_q[tau], one entry per world tick since the start (zero in the tick of death)
    death_tick: int | None = None                  # tau_d + 1; None while the colony lives
    reason: str | None = None                      # queen_death | collapse | functional_extinction (v3 only); None while the colony lives


class ViabilityTracker:
    """Per-tick viability of every colony of one world. Call ``update(state, record)`` once after each ``step_world_tick``."""

    def __init__(self, state, version: int = REWARD_VERSION) -> None:
        if version not in REWARD_VERSIONS:
            raise ValueError(f"unknown reward version {version!r}")
        self.version = version
        self.traces = [_ColonyTrace(c.rolling.pop_ema_q, 0, c.rolling.burn_ema_q) for c in state.colonies]
        self.tick = state.tick   # next tick to be recorded

    def update(self, state, record) -> None:
        if record.tick != self.tick or state.tick != self.tick + 1:
            raise ValueError(f"tracker expected tick {self.tick}, got record {record.tick} / state {state.tick}")
        adults = [[0, 0] for _ in state.colonies]
        for cohort in state.cohorts:
            adults[cohort.colony_id][0 if cohort.caste == "worker" else 1] += cohort.count
        for colony in state.colonies:
            trace = self.traces[colony.colony_id]
            if trace.death_tick is not None:
                continue
            workers, soldiers = adults[colony.colony_id]
            alive = colony.status == "alive" and colony.queen_hp > 0
            reason = None if alive else (REASON_QUEEN_DEATH if colony.queen_hp <= 0 else REASON_COLLAPSE)
            if alive and self.version >= 3 and functional_extinction(workers, soldiers, sum(1 for b in colony.brood if b.caste == "worker")):
                alive, reason = False, REASON_FUNCTIONAL
            trace.pop_ema_q = _ema(trace.pop_ema_q, (2 * workers + soldiers) * Q // 2, TAU_POP)
            trace.income_ema_q = _ema(trace.income_ema_q, record.delivered_by_colony.get(colony.colony_id, 0) * Q, TAU_ECON)
            trace.burn_ema_q = _ema(trace.burn_ema_q, record.upkeep_due_by_colony.get(colony.colony_id, 0) * Q, TAU_ECON)
            trace.v_q.append(viability_q(trace.pop_ema_q, trace.income_ema_q, trace.burn_ema_q, colony.food, alive, self.version))
            if not alive:
                trace.death_tick, trace.reason = self.tick + 1, reason
        self.tick += 1

    def terminated(self) -> set[int]:
        """Colonies whose agent episode has ended (any reason). The rollout stops asking them for decisions."""
        return {i for i, trace in enumerate(self.traces) if trace.death_tick is not None}

    def terminal_reason(self, colony: int) -> str | None:
        return self.traces[colony].reason

    def death_tick(self, colony: int) -> int | None:
        return self.traces[colony].death_tick

    def window(self, colony: int, start: int, stop: int) -> list[int]:
        """``v_q`` of the ticks ``[start, stop)``; ``stop`` may not exceed what was recorded."""
        trace = self.traces[colony].v_q
        if not 0 <= start <= stop <= len(trace):
            raise ValueError(f"colony {colony}: window [{start}, {stop}) is outside the {len(trace)} recorded ticks")
        return trace[start:stop]

    def current(self, colony: int) -> int:
        trace = self.traces[colony].v_q
        return trace[-1] if trace else 0


# ------------------------------------------------------------------------------------------------ reward of an interval (9.2)


def gamma_tick(gamma30: float = GAMMA_30) -> float:
    return gamma30 ** (1.0 / K_NOMINAL)


def interval_discount(delta: int, gamma30: float = GAMMA_30) -> float:
    """``gamma_i = gamma_30 ** (delta_i / 30)``: the discount over an interval of ``delta`` world ticks."""
    return gamma30 ** (delta / K_NOMINAL)


def interval_lambda(delta: int, lambda30: float = LAMBDA_30) -> float:
    return lambda30 ** (delta / K_NOMINAL)


def interval_rewards(trace: Sequence[int], horizon: int, *, terminated: bool, gamma30: float = GAMMA_30,
                     penalty: float = DEATH_PENALTY) -> tuple[float, float]:
    """``(r_i, r_i^gamma)`` of one interval from its integer viability trace ``v_q`` (one entry per elapsed tick, ``delta = len(trace) >= 1``).

    ``r_i`` is undiscounted (reports); ``r_i^gamma`` is discounted inside the interval with the per-tick factor (the PPO target). Death pays
    ``-penalty`` once (``-penalty * gamma_tick ** delta`` in the discounted one); nothing else is paid for fighting or killing.
    """
    delta = len(trace)
    if delta < 1:
        raise ValueError("an interval lasts at least one tick")
    g = gamma_tick(gamma30)
    plain = sum(trace) / (Q * horizon)
    discounted = sum(v * g ** k for k, v in enumerate(trace)) / (Q * horizon)
    if terminated:
        plain -= penalty
        discounted -= penalty * g ** delta
    return plain, discounted


@dataclass(frozen=True, slots=True)
class IntervalStep:
    """What time-aware GAE needs from one decision interval (contracts 9.2a)."""

    delta: int            # elapsed world ticks
    reward_disc: float    # r_i^gamma
    terminated: bool      # the colony died at the end of this interval: no bootstrap, no continuation of the advantage chain
    value: float          # Value_critic(o_i), the learned critic at the start of the interval


def gae_intervals(steps: Sequence[IntervalStep], last_value: float, *, gamma30: float = GAMMA_30,
                  lambda30: float = LAMBDA_30) -> tuple[list[float], list[float]]:
    """Time-aware GAE of one colony trajectory. ``last_value`` is the learned critic of the observation at the end of a *truncated* trajectory
    (ignored when the last step is terminated). Returns ``(advantages, return_targets)``."""
    advantages = [0.0] * len(steps)
    carry = 0.0
    for i in range(len(steps) - 1, -1, -1):
        step = steps[i]
        gamma = interval_discount(step.delta, gamma30)
        if step.terminated:
            next_value, carry_in = 0.0, 0.0
        else:
            next_value = steps[i + 1].value if i + 1 < len(steps) else last_value
            carry_in = carry
        delta_td = step.reward_disc + gamma * next_value - step.value
        carry = delta_td + gamma * interval_lambda(step.delta, lambda30) * carry_in
        advantages[i] = carry
    if not all(math.isfinite(a) for a in advantages):
        raise FloatingPointError("non-finite advantage")
    return advantages, [a + s.value for a, s in zip(advantages, steps)]
