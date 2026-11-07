"""Run configuration of one RL training run (FOR-42). Everything is JSON-serialisable and hashed for resume checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from typing import Any

from .network import HIDDEN_SIZES
from .ppo import PPOConfig

# Fields that may change when a run is resumed (budget, parallelism, cadence); everything else defines the learning run.
RESUMABLE_FIELDS = frozenset({
    "iterations", "max_matches", "max_wall_seconds", "workers", "chunk_size", "device", "snapshot_every",
    "checkpoint_every", "eval_every", "eval_suite", "name",
})


@dataclass(frozen=True, slots=True)
class RunConfig:
    name: str = "rl-run"
    run_seed: int = 42
    hidden_sizes: tuple[int, ...] = HIDDEN_SIZES
    # budget (the run ends at the first limit reached; nothing runs unbounded)
    iterations: int = 80
    matches_per_iteration: int = 128
    max_matches: int = 0  # 0 = no match cap beyond iterations
    max_wall_seconds: float = 10800.0  # 3 h; 0 = no wall cap beyond iterations
    max_ticks: int | None = None  # harness cutoff; None = full T_max matches
    # PPO
    gamma: float = 0.997
    lam: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    ent_coef_share: float = 0.001
    lr: float = 3.0e-4
    epochs: int = 4
    minibatch: int = 512
    max_grad_norm: float = 0.5
    target_kl: float = 0.05
    # reward
    shaping_coef: float = 0.0
    # self-play
    p_current: float = 0.5
    baseline_sparring: float = 0.0
    gen_map_fraction: float = 0.5
    snapshot_every: int = 5
    max_active_snapshots: int = 12
    # cadence
    checkpoint_every: int = 5
    eval_every: int = 10
    eval_suite: str = "quick"
    # compute
    workers: int = 10
    chunk_size: int = 4
    device: str = "auto"

    def ppo(self) -> PPOConfig:
        return PPOConfig(
            gamma=self.gamma, lam=self.lam, clip=self.clip, vf_coef=self.vf_coef, ent_coef=self.ent_coef,
            ent_coef_share=self.ent_coef_share, lr=self.lr, epochs=self.epochs, minibatch=self.minibatch,
            max_grad_norm=self.max_grad_norm, target_kl=self.target_kl,
        )

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["hidden_sizes"] = list(self.hidden_sizes)
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "RunConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(data) - names
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown)}")
        values = dict(data)
        if "hidden_sizes" in values:
            values["hidden_sizes"] = tuple(values["hidden_sizes"])
        return cls(**values)

    def identity_sha256(self) -> str:
        data = {k: v for k, v in self.to_json().items() if k not in RESUMABLE_FIELDS}
        return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def validate(self) -> None:
        if not 0.0 <= self.p_current <= 1.0 or not 0.0 <= self.baseline_sparring <= 1.0:
            raise ValueError("p_current and baseline_sparring must be probabilities")
        if self.p_current + self.baseline_sparring > 1.0 + 1e-9:
            raise ValueError("p_current + baseline_sparring must not exceed 1")
        if not 0.0 < self.gamma <= 1.0 or not 0.0 <= self.lam <= 1.0:
            raise ValueError("gamma in (0, 1], lam in [0, 1]")
        if self.shaping_coef < 0.0 or self.shaping_coef > 0.5:
            raise ValueError("shaping_coef must be within [0, 0.5] so shaping cannot dominate the +-1 outcome")
        if self.matches_per_iteration < 1 or self.iterations < 0 or self.minibatch < 1 or self.epochs < 1:
            raise ValueError("matches_per_iteration, minibatch, epochs must be positive")
        if self.max_ticks is not None and self.max_ticks < 1:
            raise ValueError("max_ticks must be positive or None")


__all__ = ["RESUMABLE_FIELDS", "RunConfig"]
