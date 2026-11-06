"""Manifest-first, checksum-verified checkpoints for resumable PPO runs (FOR-42).

``checkpoints/ckpt_<iteration>/`` holds ``weights.pt`` (full actor+critic state dict), ``train_state.pt`` (Adam state) and
``manifest.json`` (SHA-256 of both, counters, config identity, active snapshot list, contract versions). Both tensor
files are loaded with ``weights_only=True``. The run directory's ``latest.json`` is written last, atomically, so a crash
mid-save never leaves a pointer to a half-written checkpoint.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import torch

from ..versions import ACTION_VERSION, DECODER_VERSION, OBSERVATION_VERSION
from .distribution import HEAD_DIM
from .network import MODEL_SCHEMA, ActorCritic
from .pool import atomic_write_bytes, atomic_write_json, file_sha256

CHECKPOINT_SCHEMA = "formicmind.rl-checkpoint/v1"


class RLCheckpointError(ValueError):
    pass


def contracts() -> dict[str, Any]:
    return {"observation": OBSERVATION_VERSION, "action": ACTION_VERSION, "decoder": DECODER_VERSION,
            "head_dim": HEAD_DIM, "model": MODEL_SCHEMA}


def _tensor_bytes(obj: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.getvalue()


def save_checkpoint(
    run_dir: Path, iteration: int, model: ActorCritic, optimizer: torch.optim.Optimizer, meta: dict[str, Any],
) -> Path:
    directory = Path(run_dir) / "checkpoints" / f"ckpt_{iteration:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    weights = directory / "weights.pt"
    train_state = directory / "train_state.pt"
    atomic_write_bytes(weights, _tensor_bytes({k: v.detach().to("cpu") for k, v in model.state_dict().items()}))
    atomic_write_bytes(train_state, _tensor_bytes(optimizer.state_dict()))
    manifest = {
        "schema": CHECKPOINT_SCHEMA, "iteration": iteration, "contracts": contracts(),
        "hidden_sizes": list(model.hidden_sizes),
        "weights": {"file": weights.name, "sha256": file_sha256(weights)},
        "train_state": {"file": train_state.name, "sha256": file_sha256(train_state)},
        **meta,
    }
    atomic_write_json(directory / "manifest.json", manifest)
    atomic_write_json(Path(run_dir) / "latest.json", {"iteration": iteration, "path": f"checkpoints/ckpt_{iteration:06d}"})
    return directory


def load_checkpoint(directory: Path, *, device: torch.device | str = "cpu") -> tuple[ActorCritic, dict[str, Any], dict[str, Any]]:
    """Return ``(model, manifest, optimizer_state)``; every file is checked against the manifest first."""
    directory = Path(directory)
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RLCheckpointError(f"cannot read checkpoint manifest: {exc}") from exc
    if manifest.get("schema") != CHECKPOINT_SCHEMA:
        raise RLCheckpointError("unsupported RL checkpoint schema")
    if manifest.get("contracts") != contracts():
        raise RLCheckpointError("checkpoint contract versions do not match this simulator")
    for key in ("weights", "train_state"):
        entry = manifest[key]
        path = directory / entry["file"]
        if path.is_symlink() or not path.is_file():
            raise RLCheckpointError(f"missing or linked checkpoint file {path.name}")
        if file_sha256(path) != entry["sha256"]:
            raise RLCheckpointError(f"{path.name} does not match the manifest SHA-256")
    state = torch.load(directory / "weights.pt", map_location="cpu", weights_only=True)
    model = ActorCritic(tuple(manifest["hidden_sizes"]))
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as exc:
        raise RLCheckpointError(f"checkpoint tensors do not match the network: {exc}") from exc
    for name, tensor in model.state_dict().items():
        if not torch.isfinite(tensor).all().item():
            raise RLCheckpointError(f"non-finite parameter {name}")
    optimizer_state = torch.load(directory / "train_state.pt", map_location="cpu", weights_only=True)
    model.to(device)
    return model, manifest, optimizer_state


def latest_checkpoint(run_dir: Path) -> Path | None:
    pointer = Path(run_dir) / "latest.json"
    if not pointer.is_file():
        return None
    data = json.loads(pointer.read_text(encoding="utf-8"))
    return Path(run_dir) / data["path"]


__all__ = ["CHECKPOINT_SCHEMA", "RLCheckpointError", "contracts", "latest_checkpoint", "load_checkpoint", "save_checkpoint"]
