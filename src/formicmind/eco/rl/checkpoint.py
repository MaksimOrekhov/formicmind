"""Saving and loading a trained ``ActorCriticE1`` (FOR-49): ``weights.pt`` (a plain ``state_dict``) next to ``manifest.json`` that binds it to the
observation / action versions, the network configuration, the SHA-256 of the weights file and how it was trained. A model that does not match the
running code is refused, never adapted silently.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from ..versions import ACTION_VERSION, ECO_RULES_VERSION, OBSERVATION_VERSION
from ..viability import REWARD_VERSION
from .network import ActorCriticE1, NetworkConfig

MODEL_SCHEMA = "formicmind.eco.rl.model/1"
WEIGHTS = "weights.pt"
MANIFEST = "manifest.json"


class ModelError(ValueError):
    """The model directory is missing, corrupt, or built for other versions."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_model(directory: str | Path, model: ActorCriticE1, *, training: dict | None = None) -> dict:
    """Write ``weights.pt`` and ``manifest.json`` into ``directory`` and return the manifest."""
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    torch.save(state, folder / WEIGHTS)
    manifest = {
        "schema": MODEL_SCHEMA, "eco_rules_version": ECO_RULES_VERSION, "observation_version": OBSERVATION_VERSION,
        "action_version": ACTION_VERSION, "reward_version": REWARD_VERSION, "network": model.config.to_dict(), "weights_file": WEIGHTS,
        "weights_sha256": sha256_file(folder / WEIGHTS), "parameters": sum(p.numel() for p in model.parameters()),
        "torch": torch.__version__, "training": training or {},
    }
    (folder / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
    return manifest


def load_model(directory: str | Path) -> tuple[ActorCriticE1, dict]:
    """Load and verify a model directory: schema, versions, weights hash, then the weights themselves."""
    folder = Path(directory)
    try:
        manifest = json.loads((folder / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelError(f"{folder}: no readable {MANIFEST}: {exc}") from exc
    if manifest.get("schema") != MODEL_SCHEMA:
        raise ModelError(f"{folder}: not a {MODEL_SCHEMA} model")
    for key, current in (("eco_rules_version", ECO_RULES_VERSION), ("observation_version", OBSERVATION_VERSION), ("action_version", ACTION_VERSION)):
        if manifest.get(key) != current:
            raise ModelError(f"{folder}: trained for {key}={manifest.get(key)!r}, this code is {current!r}")
    weights = folder / manifest.get("weights_file", WEIGHTS)
    if not weights.is_file() or sha256_file(weights) != manifest.get("weights_sha256"):
        raise ModelError(f"{folder}: the weights file is missing or does not match the SHA-256 in the manifest")
    try:
        model = ActorCriticE1(NetworkConfig(**manifest["network"]))
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    except (RuntimeError, TypeError, KeyError) as exc:
        raise ModelError(f"{folder}: the weights do not fit the network of the manifest: {exc}") from exc
    model.eval()
    manifest.setdefault("reward_version", 1)     # models trained before the reward revision (first FOR-49 pilot) carry no field: they were trained on v1
    return model, manifest


def train_config_dict(config) -> dict:
    return asdict(config)
