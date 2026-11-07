"""One full engine match with learner/opponent controllers that record PPO trajectories (FOR-42).

``run_match`` is the only place that touches the engine: ``reset`` + ``step_tick`` FULL (the same loop as
``evaluation.execute_episode``), no partial mode, no second physics. Controllers see exactly
``(ObservationV2, ActionMaskV2, own PCG32 stream)``.
"""

from __future__ import annotations

import time
from array import array
from dataclasses import dataclass, field
from typing import Any

import torch

from ..actions import get_action_mask, validate_action
from ..baselines import baseline_spec
from ..decoder import PolicyLogitsV2, decode_action_logits
from ..match import _register_streams
from ..observation import build_observation
from ..reset import reset
from ..rng import policy_stream_id
from ..rules import SOLDIER, WORKER
from ..snapshot import begin_policy_phase
from ..tick import PolicyController, step_tick
from .distribution import HEAD_DIM, INT_FIELDS, MASK_LEN, Z_SLOTS, evaluate_latent, latent_to_logits, sample_latent
from .maps import map_config
from .network import ActorCritic
from .reward import shaping_potential, step_rewards, terminal_reward


@dataclass(slots=True)
class ActionStats:
    decisions: int = 0
    lay: list[int] = field(default_factory=lambda: [0, 0, 0])
    army: list[int] = field(default_factory=lambda: [0] * 7)
    avoid: int = 0
    k_hist: list[int] = field(default_factory=lambda: [0, 0, 0])
    forage_support: list[int] = field(default_factory=lambda: [0, 0, 0, 0])  # executed support size 1..3 (index = size)
    home_share_sum: float = 0.0
    entropy_discrete_sum: float = 0.0
    entropy_share_sum: float = 0.0
    inference_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__slots__}


class RLController:
    """Stochastic (or greedy) actor controller. Records the PPO sample when ``record`` is set."""

    def __init__(
        self, actor: ActorCritic, stream_id: str, *, greedy: bool = False, record: bool = False, shaping_coef: float = 0.0,
    ) -> None:
        self.actor = actor
        self.stream_id = stream_id
        self.greedy = greedy
        self.record = record
        self.shaping_coef = shaping_coef
        self.log_std_tensor = actor.log_std.detach()
        self.log_std = self.log_std_tensor.tolist()
        self.stats = ActionStats()
        self.obs = array("f")
        self.mask = array("B")
        self.ints = array("h")
        self.z = array("f")
        self.logp = array("f")
        self.phi = array("f")
        self.ticks = array("i")
        self.executed = array("H")  # forage_q (6) per decision, for audit
        self.executed_choices = array("b")  # lay, army, avoid per decision, for audit

    def act(self, tick, observation, run):  # Controller protocol
        started = time.perf_counter()
        mask = get_action_mask(observation)
        flat = mask.flatten()
        obs_t = torch.tensor(observation.values, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            heads = self.actor.policy(obs_t)
        ints, z = sample_latent(heads[0].tolist(), self.log_std, flat, run.stream(self.stream_id), greedy=self.greedy)
        action = decode_action_logits(PolicyLogitsV2(latent_to_logits(ints, z)), mask)
        validated = validate_action(observation, mask, action)

        stats = self.stats
        stats.decisions += 1
        stats.lay[action.lay_order] += 1
        stats.army[action.army_choice] += 1
        stats.avoid += int(action.avoid_center)
        stats.k_hist[ints[3]] += 1
        stats.forage_support[sum(1 for q in action.forage_share_q if q > 0)] += 1
        stats.home_share_sum += action.forage_share_q[0] / 65535.0
        if self.record:
            with torch.inference_mode():
                evaluation = evaluate_latent(
                    heads, self.log_std_tensor, torch.tensor([flat], dtype=torch.bool),
                    torch.tensor([ints], dtype=torch.int64), torch.tensor([z], dtype=torch.float32),
                )
            stats.entropy_discrete_sum += float(evaluation.entropy_discrete[0])
            stats.entropy_share_sum += float(evaluation.entropy_share[0])
            self.obs.extend(observation.values)
            self.mask.extend(flat)
            self.ints.extend(ints)
            self.z.extend(z)
            self.logp.append(float(evaluation.log_prob[0]))
            self.phi.append(shaping_potential(observation.values, self.shaping_coef))
            self.ticks.append(tick)
            self.executed.extend(action.forage_share_q)
            self.executed_choices.extend((action.lay_order, action.army_choice, int(action.avoid_center)))
        stats.inference_seconds += time.perf_counter() - started
        return validated

    def trajectory(
        self, *, score: float | None, gamma: float, final_observation: tuple[float, ...] | None,
    ) -> dict[str, Any]:
        """Serialise to plain bytes (no NumPy needed; ``torch.frombuffer`` on the receiving side)."""
        steps = len(self.logp)
        truncated = score is None
        final_phi = 0.0
        if truncated and final_observation is not None:
            final_phi = shaping_potential(final_observation, self.shaping_coef)
        rewards = step_rewards(
            list(self.phi), terminal=None if truncated else terminal_reward(score), gamma=gamma,
            final_potential=final_phi,
        )
        return {
            "steps": steps,
            "obs": self.obs.tobytes(),
            "mask": self.mask.tobytes(),
            "ints": self.ints.tobytes(),
            "z": self.z.tobytes(),
            "logp": self.logp.tobytes(),
            "reward": array("f", rewards).tobytes(),
            "phi": self.phi.tobytes(),
            "ticks": self.ticks.tobytes(),
            "executed": self.executed.tobytes(),
            "executed_choices": self.executed_choices.tobytes(),
            "truncated": truncated,
            "bootstrap_obs": array("f", final_observation).tobytes() if truncated and final_observation else b"",
            "terminal_reward": None if truncated else terminal_reward(score),
        }


def decode_trajectory(traj: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Bytes -> CPU tensors with the layout PPO needs."""
    n = traj["steps"]

    def tensor(key: str, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
        raw = traj[key]
        if n == 0 or not raw:
            return torch.empty(shape, dtype=dtype)
        return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape).clone()

    return {
        "obs": tensor("obs", torch.float32, (n, 223)),
        "mask": tensor("mask", torch.uint8, (n, MASK_LEN)).bool(),
        "ints": tensor("ints", torch.int16, (n, INT_FIELDS)).long(),
        "z": tensor("z", torch.float32, (n, Z_SLOTS)),
        "logp": tensor("logp", torch.float32, (n,)),
        "reward": tensor("reward", torch.float32, (n,)),
        "phi": tensor("phi", torch.float32, (n,)),
        "ticks": tensor("ticks", torch.int32, (n,)),
        "executed": tensor("executed", torch.int16, (n, 6)).long() & 0xFFFF,
        "executed_choices": tensor("executed_choices", torch.int8, (n, 3)),
    }


def _final_counts(state) -> dict[str, tuple[int, int]]:
    colonies = state.colonies

    def count(caste: int) -> tuple[int, int]:
        return tuple(sum(1 for u in c.units if u.caste == caste) for c in colonies)  # type: ignore[return-value]

    return {
        "queen_hp": (colonies[0].queen_hp, colonies[1].queen_hp),
        "food": (colonies[0].food, colonies[1].food),
        "workers": count(WORKER),
        "soldiers": count(SOLDIER),
        "brood": (len(colonies[0].brood), len(colonies[1].brood)),
    }


def _bootstrap_observation(state, run, side: int) -> tuple[float, ...]:
    """Observation of the post-cutoff state, built on a clone so the finished episode is not touched."""
    clone, run_clone = state.clone(), run.clone()
    return build_observation(begin_policy_phase(clone, (run_clone.windows[0], run_clone.windows[1])), side).values


def _make_controller(seat: dict[str, Any], actors: dict[str, ActorCritic], shaping_coef: float):
    kind = seat["kind"]
    if kind == "rl":
        policy_id, replicate = seat["stream"]
        return RLController(
            actors[seat["ref"]], policy_stream_id(policy_id, replicate), greedy=bool(seat.get("greedy", False)),
            record=bool(seat.get("record", False)), shaping_coef=shaping_coef,
        )
    if kind == "baseline":
        spec = baseline_spec(seat["ref"])
        return PolicyController(spec.build(), policy_stream_id(spec.policy_id, int(seat.get("replicate", 0))))
    raise ValueError(f"unknown seat kind {kind!r}")


def run_match(spec: dict[str, Any], actors: dict[str, ActorCritic]) -> dict[str, Any]:
    """Play one full match. ``spec['seats']`` describes seat 0 (A) and seat 1 (B).

    A controller error or an invalid action propagates (a technical failure is never an outcome).
    """
    started = time.perf_counter()
    config = map_config(spec["map_id"])
    shaping_coef = float(spec.get("shaping_coef", 0.0))
    gamma = float(spec.get("gamma", 0.997))
    max_ticks = spec.get("max_ticks")
    limit = config.t_max if max_ticks is None else int(max_ticks)
    if not 0 <= limit <= config.t_max:
        raise ValueError("max_ticks must be within 0..T_max")
    if limit % config.policy_interval_k:
        raise ValueError("max_ticks must be a multiple of the policy interval K so a truncated run ends on a policy tick")
    controllers = tuple(_make_controller(seat, actors, shaping_coef) for seat in spec["seats"])
    state, run = reset(config, int(spec["seed"]))
    _register_streams(run, controllers, ())
    executed = 0
    sim_started = time.perf_counter()
    while run.outcome is None and executed < limit and state.tick < config.t_max:
        step_tick(state, run, controllers)
        executed += 1
    sim_seconds = time.perf_counter() - sim_started
    outcome = run.outcome
    truncated = outcome is None
    seats_out: list[dict[str, Any]] = []
    for side, (seat, controller) in enumerate(zip(spec["seats"], controllers)):
        entry: dict[str, Any] = {"kind": seat["kind"], "ref": seat["ref"], "side": side}
        score = None if outcome is None else outcome.score[side]
        entry["score"] = score
        if isinstance(controller, RLController):
            entry["stats"] = controller.stats.to_dict()
            if controller.record:
                final_obs = _bootstrap_observation(state, run, side) if truncated else None
                entry["trajectory"] = controller.trajectory(score=score, gamma=gamma, final_observation=final_obs)
        seats_out.append(entry)
    return {
        "match_id": spec.get("match_id"),
        "seed": spec["seed"],
        "map_id": spec["map_id"],
        "tag": spec.get("tag"),
        "truncated": truncated,
        "winner": None if outcome is None else outcome.winner.value,
        "reason": None if outcome is None else outcome.reason.value,
        "terminal_tick": None if outcome is None else outcome.terminal_tick,
        "ticks_executed": executed,
        "final": _final_counts(state),
        "seats": seats_out,
        "wall_seconds": time.perf_counter() - started,
        "sim_seconds": sim_seconds,
    }


__all__ = ["ActionStats", "RLController", "decode_trajectory", "run_match"]
