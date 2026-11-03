"""Validate imitation and exercise a checkpoint in real full-engine matches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..baselines import baseline_spec
from ..match import run_episode
from ..rng import policy_stream_id
from ..rules import default_config
from ..serialization import config_hash
from ..tick import PolicyController
from .checkpoint import load_checkpoint
from .dataset import DatasetError, _git_provenance, _write_atomic, load_dataset
from .model import MLPPolicy
from .training import _score_examples, benchmark_inference


def evaluate_model(
    model_dir: str | Path,
    dataset_path: str | Path,
    *,
    seeds: tuple[int, ...] | list[int],
    opponent: str = "balanced:v2",
    max_ticks: int = 120,
    device: str = "auto",
    output: str | Path | None = None,
) -> dict[str, Any]:
    dataset_manifest, examples = load_dataset(dataset_path)
    model, checkpoint_manifest, resolved_device = load_checkpoint(model_dir, device)
    validation_examples = [example for example in examples if example.split == "validation"]
    if not validation_examples:
        raise DatasetError("dataset has no validation examples")
    requested_seeds = tuple(seeds)
    if not requested_seeds or len(set(requested_seeds)) != len(requested_seeds):
        raise ValueError("evaluation needs one or more unique seeds")
    if any(type(seed) is not int or not 0 <= seed <= (1 << 64) - 1 for seed in requested_seeds):
        raise ValueError("evaluation seeds must be unsigned 64-bit integers")
    training_seeds = {int(seed) for seed in dataset_manifest.get("seeds", [])}
    overlap = training_seeds.intersection(requested_seeds)
    if overlap:
        raise ValueError(f"evaluation seeds overlap collected train/validation seeds: {sorted(overlap)}")
    config = default_config()
    if dataset_manifest.get("simulator", {}).get("config_sha256") != config_hash(config):
        raise DatasetError("evaluation config differs from the dataset simulator configuration")
    if type(max_ticks) is not int or not 1 <= max_ticks <= config.t_max:
        raise ValueError(f"max_ticks must be within 1..T_max ({config.t_max})")
    opponent_spec = baseline_spec(opponent)
    if opponent_spec.version != "v2":
        raise ValueError("match evaluation requires an explicit Baselines v2 opponent")

    policy = MLPPolicy(model, resolved_device)
    imitation = _score_examples(model, validation_examples, resolved_device)
    inference = benchmark_inference(policy, validation_examples)
    model_hash = checkpoint_manifest["weights"]["sha256"]
    model_policy_id = f"mlp-{model_hash[:16]}"
    model_stream = policy_stream_id(model_policy_id, 0)
    opponent_stream = policy_stream_id(opponent_spec.policy_id, 1)
    matches: list[dict[str, Any]] = []
    for seed in requested_seeds:
        for model_side in (0, 1):
            neural = PolicyController(policy, model_stream)
            teacher = PolicyController(opponent_spec.build(), opponent_stream)
            controllers = (neural, teacher) if model_side == 0 else (teacher, neural)
            result = run_episode(config, seed, controllers[0], controllers[1], max_ticks=max_ticks)
            score = None if result.score is None else result.score[model_side]
            matches.append({
                "seed": seed,
                "model_side": model_side,
                "opponent": opponent_spec.full_id,
                "ticks_executed": result.ticks_executed,
                "final_tick": result.final_tick,
                "truncated": result.truncated,
                "winner": result.winner,
                "reason": result.reason,
                "model_score": score,
                "queen_hp": {"model": result.queen_hp[model_side], "opponent": result.queen_hp[1 - model_side]},
                "food": {"model": result.food[model_side], "opponent": result.food[1 - model_side]},
                "workers": {"model": result.workers[model_side], "opponent": result.workers[1 - model_side]},
                "soldiers": {"model": result.soldiers[model_side], "opponent": result.soldiers[1 - model_side]},
                "config_sha256": result.config_hash,
                "events_sha256": result.events_hash,
                "state_sha256": result.state_hash,
            })
    completed = [row["model_score"] for row in matches if row["model_score"] is not None]
    git_sha, git_dirty = _git_provenance()
    report: dict[str, Any] = {
        "schema": "formicmind.ml-evaluation/v1",
        "classification": "technical imitation check; not a test of strategic superiority",
        "model": {
            "weights_sha256": model_hash,
            "checkpoint_manifest": "manifest.json",
            "device": str(resolved_device),
        },
        "dataset": {
            "file": Path(dataset_path).name,
            "sha256": dataset_manifest["dataset_sha256"],
            "validation_examples": len(validation_examples),
            "teacher": dataset_manifest["teacher"],
        },
        "contracts": checkpoint_manifest["contracts"],
        "simulator": {"config_sha256": config_hash(config)},
        "imitation": imitation,
        "performance": {"inference": inference},
        "match_summary": {
            "opponent": opponent_spec.full_id,
            "paired_seat": True,
            "matches": len(matches),
            "completed_matches": len(completed),
            "truncated_matches": len(matches) - len(completed),
            "invalid_actions": 0,
            "mean_score_when_completed": sum(completed) / len(completed) if completed else None,
        },
        "matches": matches,
        "source": {"git_sha": git_sha, "git_dirty": git_dirty},
        "seeds": list(requested_seeds),
        "max_ticks": max_ticks,
    }
    if output is not None:
        path = Path(output)
        _write_atomic(path, (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    return report


__all__ = ["evaluate_model"]
