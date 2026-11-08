"""Frozen snapshot pool and the seeded match planner (FOR-42, phase 2).

* A snapshot is the **actor** weights of one learner iteration, written once and never modified (``add`` refuses to
  overwrite; the manifest carries the SHA-256 of the weights file and every later load re-verifies it). Opponents are
  therefore fixed functions during a PPO batch: no moving-policy targets.
* ``plan_matches`` is a pure function of ``(run_seed, iteration, pool, config)``: the same inputs give the same
  specs. Every match gets a unique engineering seed (``TRAIN_SEED_BASE + global match index``); eval seeds live in a
  disjoint range (see ``evaluate.py``). Policy RNG streams are keyed by policy id + replicate, never by seat.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .maps import training_map_id
from .network import actor_state_from_bytes

TRAIN_SEED_BASE = 4_200_000_000
EVAL_SEED_BASE = 9_000_000_000
SNAPSHOT_SCHEMA = "formicmind.rl-snapshot/v1"
BASELINE_SPARRING = ("random:v2", "turtle:v2", "rush:v2", "balanced:v2", "raider:v2")


class PoolError(ValueError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    with open(temp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))


@dataclass(frozen=True, slots=True)
class SnapshotInfo:
    snapshot_id: int
    iteration: int
    path: str  # weights file
    sha256: str

    @property
    def ref(self) -> str:
        return f"snap:{self.snapshot_id:04d}"

    @property
    def policy_id(self) -> str:
        return f"rl-snap-{self.snapshot_id:04d}"

    def to_json(self) -> dict[str, Any]:
        return {"snapshot_id": self.snapshot_id, "iteration": self.iteration, "path": self.path, "sha256": self.sha256}


class SnapshotPool:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, snapshot_id: int) -> Path:
        return self.root / f"snap_{snapshot_id:04d}"

    def add(self, snapshot_id: int, iteration: int, actor_state: dict[str, torch.Tensor]) -> SnapshotInfo:
        directory = self._dir(snapshot_id)
        weights = directory / "weights.pt"
        if weights.exists():
            raise PoolError(f"snapshot {snapshot_id} already exists and is immutable")
        directory.mkdir(parents=True, exist_ok=True)
        import io
        buffer = io.BytesIO()
        torch.save(actor_state, buffer)
        atomic_write_bytes(weights, buffer.getvalue())
        info = SnapshotInfo(snapshot_id, iteration, str(weights), file_sha256(weights))
        atomic_write_json(directory / "manifest.json", {"schema": SNAPSHOT_SCHEMA, **info.to_json()})
        return info

    def load_info(self, snapshot_id: int) -> SnapshotInfo:
        manifest = json.loads((self._dir(snapshot_id) / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema") != SNAPSHOT_SCHEMA:
            raise PoolError("unsupported snapshot manifest")
        info = SnapshotInfo(manifest["snapshot_id"], manifest["iteration"], str(self._dir(snapshot_id) / "weights.pt"),
                            manifest["sha256"])
        if file_sha256(Path(info.path)) != info.sha256:
            raise PoolError(f"snapshot {snapshot_id} weights do not match their manifest SHA-256")
        return info

    def load_state(self, info: SnapshotInfo) -> dict[str, torch.Tensor]:
        if file_sha256(Path(info.path)) != info.sha256:
            raise PoolError(f"snapshot {info.snapshot_id} weights do not match their manifest SHA-256")
        return actor_state_from_bytes(Path(info.path).read_bytes())


def plan_matches(
    run_seed: int, iteration: int, count: int, active: list[SnapshotInfo], *, p_current: float, baseline_sparring: float,
    gen_map_fraction: float, shaping_coef: float, gamma: float, max_ticks: int | None,
) -> list[dict[str, Any]]:
    """Seeded planner for one iteration. Opponent = current weights, a frozen snapshot, or (optionally) a baseline."""
    rng = random.Random(f"formicmind-rl-plan-{run_seed}-{iteration}")
    specs: list[dict[str, Any]] = []
    for index in range(count):
        match_id = iteration * count + index
        map_id = training_map_id(rng, gen_map_fraction)
        roll = rng.random()
        if not active or roll < p_current:
            opponent = {"kind": "current"}
        elif roll < p_current + baseline_sparring:
            opponent = {"kind": "baseline", "ref": rng.choice(BASELINE_SPARRING)}
        else:
            opponent = {"kind": "snapshot", "ref": rng.choice(active).ref}
        learner_seat = rng.randrange(2)
        learner = {"kind": "rl", "ref": "current", "stream": ("rl-current", 0), "record": True}
        if opponent["kind"] == "current":
            seats = [learner, {"kind": "rl", "ref": "current", "stream": ("rl-current", 1), "record": True}]
        elif opponent["kind"] == "snapshot":
            snap = next(s for s in active if s.ref == opponent["ref"])
            other = {"kind": "rl", "ref": snap.ref, "stream": (snap.policy_id, 0), "record": False}
            seats = [learner, other] if learner_seat == 0 else [other, learner]
        else:
            other = {"kind": "baseline", "ref": opponent["ref"], "replicate": 0}
            seats = [learner, other] if learner_seat == 0 else [other, learner]
        specs.append({
            "match_id": match_id, "seed": TRAIN_SEED_BASE + match_id, "map_id": map_id, "seats": seats,
            "tag": {"opponent": opponent["kind"] if opponent["kind"] != "snapshot" else opponent["ref"],
                    "opponent_kind": opponent["kind"], "learner_seat": learner_seat},
            "shaping_coef": shaping_coef, "gamma": gamma, "max_ticks": max_ticks,
        })
    return specs


def thin_active(infos: list[SnapshotInfo], limit: int) -> list[SnapshotInfo]:
    """Keep the first and newest snapshots and spread the rest evenly (files stay on disk, only sampling is limited)."""
    if limit <= 0 or len(infos) <= limit:
        return list(infos)
    if limit == 1:
        return [infos[-1]]
    picks = {0, len(infos) - 1}
    step = (len(infos) - 1) / (limit - 1)
    picks.update(round(i * step) for i in range(limit))
    ordered = sorted(picks)
    while len(ordered) > limit:
        ordered.pop(len(ordered) // 2)
    return [infos[i] for i in ordered]


__all__ = [
    "BASELINE_SPARRING", "EVAL_SEED_BASE", "TRAIN_SEED_BASE", "PoolError", "SnapshotInfo", "SnapshotPool",
    "atomic_write_bytes", "atomic_write_json", "file_sha256", "plan_matches", "thin_active",
]
