"""Reward schedule (FOR-42, phase 1): terminal outcome only, plus optional potential-based shaping.

* Terminal: win +1, draw 0, loss -1 from the learner's seat, ``2 * score - 1`` with the FOR-1 section 9 score 1/0.5/0.
  The two seats of one episode always receive opposite terminal rewards, so a mirror self-play episode sums to 0.
* A *truncated* episode (harness cutoff, ``outcome is None``) gets **no** terminal reward: the last step bootstraps from
  the value of the observation reached at the cutoff. It is never turned into a loss or a draw.
* Optional shaping (default coefficient 0): ``r'_t = r_t + gamma * Phi(o_{t+1}) - Phi(o_t)`` with ``Phi(terminal) = 0``
  (Ng et al. 1999: preserves the optimal policy), ``Phi`` a bounded function of the agent's **own public** colony
  health (queen HP + worker/soldier HP sums from ``own_global``, each already in [0, 1)), scaled by ``coef`` so that
  ``0 <= Phi < coef``. The whole-episode shaping sum is therefore bounded by ``2 * coef`` and cannot dominate +-1.
  Nothing here rewards idling, farming, a specific caste or a specific move.
* Rewards are attached to policy *decisions* (every ``policy_interval_k`` ticks), not to simulation ticks;
  ``gamma`` is expressed in decisions.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..observation import OBS_V2_OFFSETS, OwnGlobalIndex

_OWN_GLOBAL_START = OBS_V2_OFFSETS["own_global"][0]


def terminal_reward(score: float) -> float:
    if score not in (0.0, 0.5, 1.0):
        raise ValueError(f"score must be 0, 0.5 or 1, got {score!r}")
    return 2.0 * score - 1.0


def shaping_potential(observation_values: Sequence[float], coef: float) -> float:
    if coef == 0.0:
        return 0.0
    g = _OWN_GLOBAL_START
    health = (
        observation_values[g + OwnGlobalIndex.QUEEN_HP]
        + observation_values[g + OwnGlobalIndex.WORKER_HP_SUM]
        + observation_values[g + OwnGlobalIndex.SOLDIER_HP_SUM]
    ) / 3.0
    return coef * health


def step_rewards(
    potentials: Sequence[float],
    *,
    terminal: float | None,
    gamma: float,
    final_potential: float = 0.0,
) -> list[float]:
    """Per-decision rewards of one trajectory.

    ``terminal`` is the +-1/0 outcome reward, or ``None`` for a truncated trajectory. ``final_potential`` is
    ``Phi`` of the observation after the last decision: ``0`` for a terminal episode (absorbing state), the real
    potential for a truncated one.
    """
    n = len(potentials)
    rewards: list[float] = []
    for t in range(n):
        following = potentials[t + 1] if t + 1 < n else final_potential
        rewards.append(gamma * following - potentials[t])
    if terminal is not None and n:
        rewards[-1] += terminal
    return rewards


__all__ = ["shaping_potential", "step_rewards", "terminal_reward"]
