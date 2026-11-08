"""FOR-43 design aid: reference arithmetic for the Ecosystem v1 reward contract (docs/ecosystem_v1_contracts.md section 9).

NOT the Ecosystem engine and NOT a trainer. It pins down, with executable arithmetic, how a colony's reward, discount and
GAE are computed when the interval between policy decisions varies (K = 30 ticks normally, 10 ticks while in combat):

* reward is paid for *elapsed world ticks*: ``r_i = sum(v[tau] for tau in interval) / (Q * H)``; a decision (alert or not)
  adds nothing, so the undiscounted total over a horizon does not depend on how it is split into decisions;
* discounting is *time-aware*: ``gamma_i = gamma_30 ** (delta_i / 30)``; the training reward of an interval is discounted
  inside the interval with the per-tick factor, so the discounted return is also independent of the split;
* GAE uses the same ``gamma_i`` and ``lambda_i = lambda_30 ** (delta_i / 30)``;
* ``v`` is the *viability* of the colony (fixed-point, 0..Q), computed by the engine each tick; it is never a value
  estimate. Bootstrapping at a horizon truncation uses a learned critic ``Value_critic(o_H)``, passed in explicitly;
  a terminal transition (queen death / collapse) bootstraps with 0 and receives the death penalty exactly once.

Stdlib only; imports nothing from ``formicmind``. Usage: ``python scripts/ecosystem_reward_ref.py`` prints a summary.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

Q = 65536  # fixed point of the viability score (1.0 == Q)
K = 30  # nominal decision interval, ticks
GAMMA_30 = 0.997
LAMBDA_30 = 0.95
DEATH_PENALTY = 1.0


def viability_exact(e_pop, income, burn, food, queen_alive: bool, version: int = 2) -> Fraction:
    """Viability ``v`` of contracts 9.1 in exact rational arithmetic (no fixed point, no floor), from natural units:
    ``e_pop`` workers-equivalents (the EMA), ``income`` / ``burn`` food per tick, ``food`` stored.

    * reward v1 (first FOR-49 pilot): ``v = alive * (0.6 pop + 0.4 security)``;
    * reward v2: ``security`` is multiplied by the population gate ``min(1, e_pop / 10)``, so a colony with no adults is paid nothing
      for the food in its store.
    Written without reading ``formicmind``; the engine's integer ``viability_q`` must stay within a few units of ``floor(Q * v)``."""
    if version not in (1, 2):
        raise ValueError(f"unknown reward version {version}")
    if not queen_alive:
        return Fraction(0)
    e_pop, income, burn, food = Fraction(e_pop), Fraction(income), Fraction(burn), Fraction(food)
    pop = min(Fraction(1), e_pop / 60)
    security = Fraction(1) if burn <= income else min(Fraction(1), food / (burn - income) / 40)
    if version == 2:
        security *= min(Fraction(1), e_pop / 10)
    return Fraction(6, 10) * pop + Fraction(4, 10) * security


def viability_q_exact(e_pop, income, burn, food, queen_alive: bool, version: int = 2) -> int:
    return math.floor(Q * viability_exact(e_pop, income, burn, food, queen_alive, version))


@dataclass(frozen=True, slots=True)
class Transition:
    start: int  # first world tick of the interval
    delta: int  # elapsed world ticks
    reward: float  # undiscounted reward (metrics, reports); includes the death penalty on the terminal transition
    reward_disc: float  # reward discounted inside the interval with the per-tick factor (used by PPO targets)
    discount: float  # gamma_i = gamma_30 ** (delta / K), applied to the value of the next state
    terminated: bool  # colony died at the end of this interval (no bootstrap)
    truncated: bool  # horizon reached (bootstrap with the learned critic)


def gamma_tick(gamma30: float = GAMMA_30) -> float:
    return gamma30 ** (1.0 / K)


def interval_discount(delta: int, gamma30: float = GAMMA_30) -> float:
    return gamma30 ** (delta / K)


def build_transitions(v_q: Sequence[int], decision_ticks: Sequence[int], horizon: int, *, death_tick: int | None = None,
                      gamma30: float = GAMMA_30, penalty: float = DEATH_PENALTY) -> list[Transition]:
    """Transitions of one colony.

    ``v_q[tau]`` is the viability (0..Q) after world tick ``tau``; only ticks ``< end`` are used, where ``end`` is
    ``death_tick`` (the colony died at that tick: viability 0, no further decisions) or ``horizon``.
    ``decision_ticks`` are the ticks at which the colony decided (must start with 0, strictly increasing, all < end)."""
    end = horizon if death_tick is None else death_tick
    if end <= 0 or len(v_q) < end:
        raise ValueError("viability trace shorter than the episode")
    ticks = list(decision_ticks)
    if not ticks or ticks[0] != 0 or any(b <= a for a, b in zip(ticks, ticks[1:])) or ticks[-1] >= end:
        raise ValueError("decision ticks must start at 0, increase strictly and stay below the episode end")
    g = gamma_tick(gamma30)
    out: list[Transition] = []
    for index, start in enumerate(ticks):
        stop = ticks[index + 1] if index + 1 < len(ticks) else end
        delta = stop - start
        undiscounted = sum(v_q[start:stop]) / (Q * horizon)
        discounted = sum(v_q[tau] * g ** (tau - start) for tau in range(start, stop)) / (Q * horizon)
        last = index + 1 == len(ticks)
        terminated = last and death_tick is not None
        if terminated:  # the penalty is paid once, at the moment of death (weight g ** delta inside the interval)
            undiscounted -= penalty
            discounted -= penalty * g ** delta
        out.append(Transition(start, delta, undiscounted, discounted, interval_discount(delta, gamma30), terminated,
                              last and death_tick is None))
    return out


def discounted_return_ticks(v_q: Sequence[int], horizon: int, *, death_tick: int | None = None,
                            gamma30: float = GAMMA_30, penalty: float = DEATH_PENALTY) -> float:
    """Tick-level reference: sum over world ticks of g**tau * v[tau] / (Q * H), minus the discounted death penalty."""
    end = horizon if death_tick is None else death_tick
    g = gamma_tick(gamma30)
    total = sum(v_q[tau] * g ** tau for tau in range(end)) / (Q * horizon)
    if death_tick is not None:
        total -= penalty * g ** end
    return total


def gae(transitions: Sequence[Transition], values: Sequence[float], critic_last_value: float = 0.0, *,
        lambda30: float = LAMBDA_30) -> tuple[list[float], list[float]]:
    """Time-aware GAE over one colony trajectory.

    ``values[i]`` is the learned critic ``Value_critic(o_i)`` at the start of interval ``i``. ``critic_last_value`` is the
    critic at the observation reached at the end of a *truncated* trajectory; it is ignored for a terminated one (0).
    ``delta_i = r_disc_i + gamma_i * V_next * (not terminated) - V_i``; ``A_i = delta_i + gamma_i * lambda_i * A_{i+1}``
    with ``lambda_i = lambda30 ** (delta_i / K)``. Returns ``(advantages, return_targets)``."""
    n = len(transitions)
    if len(values) != n:
        raise ValueError("one critic value per transition is required")
    advantages = [0.0] * n
    carry = 0.0
    for i in range(n - 1, -1, -1):
        t = transitions[i]
        if t.terminated:
            next_value, carry_in = 0.0, 0.0
        else:
            next_value = values[i + 1] if i + 1 < n else critic_last_value
            carry_in = carry
        lam = lambda30 ** (t.delta / K)
        delta = t.reward_disc + t.discount * next_value - values[i]
        carry = delta + t.discount * lam * carry_in
        advantages[i] = carry
    return advantages, [a + v for a, v in zip(advantages, values)]


def partition(horizon: int, base: int = K, alert: int = 10, alert_windows: Sequence[tuple[int, int]] = ()) -> list[int]:
    """Decision ticks: every ``base`` ticks, every ``alert`` ticks inside the given ``(from, to)`` combat windows."""
    ticks, t = [0], 0
    while True:
        step = alert if any(a <= t < b for a, b in alert_windows) else base
        t += step
        if t >= horizon:
            return ticks
        ticks.append(t)


def main() -> int:
    horizon = 6000
    v = [int(0.6 * Q)] * horizon
    print("constant viability 0.6 over", horizon, "ticks")
    for name, ticks in (("200 x 30 ticks", partition(horizon)), ("600 x 10 ticks", partition(horizon, base=10)),
                        ("30 with a combat window 900..1200 at 10", partition(horizon, alert_windows=[(900, 1200)]))):
        tr = build_transitions(v, ticks, horizon)
        g = gamma_tick()
        disc, scale = 0.0, 1.0
        for t in tr:
            disc += scale * t.reward_disc
            scale *= t.discount
        print(f"{name:42} decisions={len(tr):4}  undiscounted={sum(t.reward for t in tr):.12f}  discounted={disc:.12f}")
    print(f"tick-level discounted reference            {discounted_return_ticks(v, horizon):.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
