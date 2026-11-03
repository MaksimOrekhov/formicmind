"""Manifest-first, checksum-verified and weights-only checkpoint loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .dataset import _write_atomic
from .model import HIDDEN_SIZES, INPUT_SIZE, MODEL_SCHEMA, OUTPUT_SIZE, PolicyMLP, resolve_device, torch
from ..versions import ACTION_VERSION, DECODER_VERSION, OBSERVATION_VERSION

CHECKPOINT_SCHEMA = "formicmind.ml-checkpoint/v1"
WEIGHTS_NAME = "weights.pt"


class CheckpointError(ValueError):
    """Checkpoint does not match the supported model/contract or failed integrity checks."""


def checkpoint_contract() -> dict[str, str | int]:
    return {
        "observation": OBSERVATION_VERSION,
        "observation_size": INPUT_SIZE,
        "action": ACTION_VERSION,
        "decoder": DECODER_VERSION,
    }


def save_checkpoint(model: PolicyMLP, directory: str | Path, manifest: dict[str, Any]) -> dict[str, Any]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    weights_path = directory / WEIGHTS_NAME
    temp_weights = directory / f".{WEIGHTS_NAME}.tmp"
    state = {name: tensor.detach().to("cpu") for name, tensor in model.state_dict().items()}
    try:
        torch.save(state, temp_weights)
        temp_weights.replace(weights_path)
    finally:
        temp_weights.unlink(missing_ok=True)
    weights_sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    result = {
        **manifest,
        "schema": CHECKPOINT_SCHEMA,
        "model": {
            "schema": MODEL_SCHEMA,
            "input_size": INPUT_SIZE,
            "hidden_sizes": list(HIDDEN_SIZES),
            "output_size": OUTPUT_SIZE,
            "activation": "relu",
            "loss_target": "baseline-v2 PolicyLogitsV2 regression",
        },
        "contracts": checkpoint_contract(),
        "weights": {"file": WEIGHTS_NAME, "sha256": weights_sha, "serialization": "torch.state_dict/weights_only"},
    }
    manifest_bytes = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    _write_atomic(directory / "manifest.json", manifest_bytes)
    return result


def load_checkpoint(directory: str | Path, requested_device: str = "auto") -> tuple[PolicyMLP, dict[str, Any], torch.device]:
    directory = Path(directory)
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"cannot read checkpoint manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != CHECKPOINT_SCHEMA:
        raise CheckpointError("unsupported checkpoint manifest schema")
    if manifest.get("contracts") != checkpoint_contract():
        raise CheckpointError("checkpoint contract versions do not match this simulator")
    expected_model = {
        "schema": MODEL_SCHEMA,
        "input_size": INPUT_SIZE,
        "hidden_sizes": list(HIDDEN_SIZES),
        "output_size": OUTPUT_SIZE,
        "activation": "relu",
        "loss_target": "baseline-v2 PolicyLogitsV2 regression",
    }
    if manifest.get("model") != expected_model:
        raise CheckpointError("unsupported model architecture in checkpoint manifest")
    weights = manifest.get("weights")
    if (not isinstance(weights, dict) or weights.get("file") != WEIGHTS_NAME
            or weights.get("serialization") != "torch.state_dict/weights_only"):
        raise CheckpointError("manifest does not name the supported weights-only file")
    weights_path = directory / WEIGHTS_NAME
    if weights_path.is_symlink():
        raise CheckpointError("checkpoint weights must not be a symbolic link")
    try:
        raw = weights_path.read_bytes()
    except OSError as exc:
        raise CheckpointError(f"cannot read checkpoint weights: {exc}") from exc
    if hashlib.sha256(raw).hexdigest() != weights.get("sha256"):
        raise CheckpointError("checkpoint weight SHA-256 does not match the manifest")

    device = resolve_device(requested_device)
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except Exception as exc:
        raise CheckpointError(f"safe weights-only load failed: {exc}") from exc
    if not isinstance(state, dict) or not state or any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise CheckpointError("weights file must contain only a state_dict of tensors")
    model = PolicyMLP().to(device)
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as exc:
        raise CheckpointError(f"checkpoint tensors do not match the fixed MLP: {exc}") from exc
    for tensor in model.state_dict().values():
        if not torch.isfinite(tensor).all().item():
            raise CheckpointError("checkpoint contains non-finite model parameters")
    model.eval()
    return model, manifest, device


__all__ = ["CHECKPOINT_SCHEMA", "CheckpointError", "checkpoint_contract", "load_checkpoint", "save_checkpoint"]
