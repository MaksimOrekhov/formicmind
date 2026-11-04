"""FOR-37 training and evaluation; kept separate from the backward-compatible v1 CLI path."""

from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

from ..baselines import baseline_spec
from ..match import run_episode
from ..rng import policy_stream_id
from ..tick import PolicyController
from .checkpoint import load_checkpoint, save_checkpoint
from .dataset import DatasetError, _write_atomic
from .model import (
    MLPPolicy,
    PolicyMLP,
    configure_determinism,
    resolve_device,
    synchronize,
    torch,
)
from .quality_dataset import (
    QUALITY_DATASET_SCHEMA,
    QUALITY_MATCH_OPPONENTS,
    QUALITY_MATCH_SEEDS,
    QUALITY_TRAIN_SEED,
    QUALITY_UNTRAINED_SEED,
    QualityExample,
    load_quality_dataset,
    quality_config,
)
from .training import _score_examples, benchmark_inference


def train_quality_model(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = QUALITY_TRAIN_SEED,
    epochs: int = 8,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    requested_device: str = "auto",
) -> dict[str, Any]:
    """Fit only on train examples; held-out examples are not scored before checkpoint save."""
    if type(seed) is not int or not 0 <= seed < (1 << 63):
        raise ValueError("training seed must be an integer in 0..2^63-1")
    if type(epochs) is not int or epochs <= 0 or type(batch_size) is not int or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive integers")
    if learning_rate <= 0 or not learning_rate < float("inf"):
        raise ValueError("learning_rate must be finite and positive")
    dataset_manifest, examples = load_quality_dataset(dataset_path)
    train_examples = [example for example in examples if example.split == "train"]
    validation_examples = [example for example in examples if example.split == "validation"]
    if not train_examples or not validation_examples:
        raise DatasetError("quality training needs non-empty train and validation splits")

    device = resolve_device(requested_device)
    configure_determinism(seed, device)
    model = PolicyMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    x_train = torch.tensor([example.observation for example in train_examples], dtype=torch.float32, device=device)
    y_train = torch.tensor([example.teacher_logits for example in train_examples], dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    epoch_losses: list[float] = []
    step_seconds: list[float] = []
    model.train()
    training_start = time.perf_counter()
    for _epoch in range(epochs):
        permutation = torch.randperm(len(train_examples), generator=generator)
        weighted_loss = 0.0
        observed = 0
        for offset in range(0, len(train_examples), batch_size):
            indices = permutation[offset:offset + batch_size].to(device)
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_train.index_select(0, indices))
            loss = torch.mean((prediction - y_train.index_select(0, indices)) ** 2)
            loss.backward()
            optimizer.step()
            synchronize(device)
            step_seconds.append(time.perf_counter() - started)
            batch_count = int(indices.numel())
            weighted_loss += loss.detach().item() * batch_count
            observed += batch_count
        epoch_losses.append(weighted_loss / observed)
    synchronize(device)
    total_training_seconds = time.perf_counter() - training_start
    model.eval()
    train_metrics = _score_examples(model, train_examples, device)
    validation_metrics = _score_examples(model, validation_examples, device)
    policy = MLPPolicy(model, device)
    inference = benchmark_inference(policy, validation_examples)
    try:
        cuda_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    except Exception:
        cuda_name = None
    all_seeds = sorted(int(seed_value) for seed_value in dataset_manifest["split_by_seed"])
    source_sha, source_dirty = dataset_manifest.get("source", {}).get("git_sha"), dataset_manifest.get("source", {}).get("git_dirty")
    checkpoint_manifest: dict[str, Any] = {
        "objective": "FOR-37 technical behavior cloning; not evidence of strategic superiority",
        "loss": {"name": "mean_squared_error", "target": "balanced:v2 PolicyLogitsV2 (17 float32)"},
        "training": {
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "shuffle": "torch.randperm with CPU generator seeded by training seed, per epoch",
            "epoch_teacher_logit_mse": epoch_losses,
            "train": train_metrics,
            "validation": validation_metrics,
        },
        "performance": {
            "training_device": str(device),
            "cuda_device_name": cuda_name,
            "optimizer_steps": len(step_seconds),
            "training_seconds": total_training_seconds,
            "mean_step_milliseconds": statistics.fmean(step_seconds) * 1000 if step_seconds else 0.0,
            "training_samples_per_second": epochs * len(train_examples) / total_training_seconds
            if total_training_seconds else float("inf"),
            "inference": inference,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "torch_threads": torch.get_num_threads(),
        },
        "data": {
            "schema": QUALITY_DATASET_SCHEMA,
            "file": Path(dataset_path).name,
            "sha256": dataset_manifest["dataset_sha256"],
            "example_count": dataset_manifest["example_count"],
            "match_count": dataset_manifest["match_count"],
            "seeds": all_seeds,
            "split_by_seed": dataset_manifest["split_by_seed"],
            "seed_sets": dataset_manifest["seed_sets"],
            "teacher": dataset_manifest["teacher"],
            "contexts": dataset_manifest["context_matrix"],
            "deduplication": dataset_manifest["deduplication"],
            "collector_source": dataset_manifest["source"],
        },
        "source": {"git_sha": source_sha, "git_dirty": source_dirty},
        "determinism": {
            "algorithms": "torch.use_deterministic_algorithms(True)",
            "seeded_cpu_rng": True,
            "cuda_reproducible_across_hardware": False,
        },
    }
    return save_checkpoint(model, output_dir, checkpoint_manifest)


def _checkpoint_quality_data(checkpoint_manifest: dict[str, Any], dataset_manifest: dict[str, Any]) -> None:
    data = checkpoint_manifest.get("data")
    if not isinstance(data, dict) or data.get("schema") != QUALITY_DATASET_SCHEMA:
        raise DatasetError("checkpoint is not bound to a FOR-37 quality dataset")
    if data.get("sha256") != dataset_manifest.get("dataset_sha256"):
        raise DatasetError("checkpoint dataset SHA-256 does not match quality dataset")
    for key in ("split_by_seed", "seed_sets", "teacher", "contexts", "deduplication"):
        expected_key = "context_matrix" if key == "contexts" else key
        if data.get(key) != dataset_manifest.get(expected_key):
            raise DatasetError(f"checkpoint quality {key} provenance differs from dataset")


def _score_by_context(
    model: PolicyMLP,
    examples: list[QualityExample],
    device: torch.device,
    expected_context_ids: set[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for context_id in sorted(expected_context_ids):
        subset = [example for example in examples if example.context_id == context_id]
        base = {
            "context_id": context_id,
            "map_id": context_id.split("__", 1)[0],
            "opponent_id": context_id.split("__", 1)[1],
            "examples": len(subset),
            "unique_decisions": len({example.decision_sha256 for example in subset}),
            "unique_trajectories": len({example.trajectory_sha256 for example in subset}),
        }
        if subset:
            base["metrics"] = _score_examples(model, subset, device)
            base["status"] = "scored"
        else:
            base["status"] = "no independent examples after global deduplication"
        result[context_id] = base
    return result


def _oracle_metrics(examples: list[QualityExample]) -> dict[str, Any]:
    return {
        "examples": len(examples),
        "teacher_logit_mse": 0.0,
        "legal_action_rate": 1.0,
        "invalid_action_count": 0,
        "exact_action_accuracy": 1.0,
        "lay_accuracy": 1.0,
        "army_accuracy": 1.0,
        "forage_share_exact_accuracy": 1.0,
        "avoid_center_accuracy": 1.0,
    }


def _match_record(result: Any, seed: int, model_side: int, opponent_id: str) -> dict[str, Any]:
    terminal_score = None if result.score is None else result.score[model_side]
    return {
        "seed": seed,
        "model_side": model_side,
        "opponent": opponent_id,
        "ticks_executed": result.ticks_executed,
        "final_tick": result.final_tick,
        "truncated": result.truncated,
        "winner": result.winner,
        "reason": result.reason,
        "model_score": terminal_score,
        "queen_hp": {"model": result.queen_hp[model_side], "opponent": result.queen_hp[1 - model_side]},
        "food": {"model": result.food[model_side], "opponent": result.food[1 - model_side]},
        "workers": {"model": result.workers[model_side], "opponent": result.workers[1 - model_side]},
        "soldiers": {"model": result.soldiers[model_side], "opponent": result.soldiers[1 - model_side]},
        "config_sha256": result.config_hash,
        "events_sha256": result.events_hash,
        "state_sha256": result.state_hash,
    }


def _match_summaries(matches: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for opponent in QUALITY_MATCH_OPPONENTS:
        subset = [match for match in matches if match["opponent"] == opponent]
        completed = [match for match in subset if not match["truncated"] and match["model_score"] is not None]
        scores = [match["model_score"] for match in completed]
        result[opponent] = {
            "matches": len(subset),
            "completed_matches": len(completed),
            "truncated_matches": sum(match["truncated"] for match in subset),
            "model_wins": sum(score == 1.0 for score in scores),
            "draws": sum(score == 0.5 for score in scores),
            "model_losses": sum(score == 0.0 for score in scores),
            "mean_model_score_completed_only": statistics.fmean(scores) if scores else None,
        }
    return result


def evaluate_quality_model(
    model_dir: str | Path,
    dataset_path: str | Path,
    *,
    device: str = "auto",
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate immutable held-out examples, random-init control, oracle and paired real matches."""
    dataset_manifest, examples = load_quality_dataset(dataset_path)
    model, checkpoint_manifest, resolved_device = load_checkpoint(model_dir, device)
    _checkpoint_quality_data(checkpoint_manifest, dataset_manifest)
    heldout = [example for example in examples if example.split == "heldout"]
    if not heldout:
        raise DatasetError("quality dataset has no held-out examples")
    dataset_seeds = {int(seed_value) for seed_value in dataset_manifest["split_by_seed"]}
    match_seed_overlap = dataset_seeds.intersection(QUALITY_MATCH_SEEDS)
    if match_seed_overlap:
        raise DatasetError(f"paired match-evaluation seeds overlap quality dataset seeds: {sorted(match_seed_overlap)}")
    split_examples = {split: [example for example in examples if example.split == split]
                      for split in ("train", "validation", "heldout")}
    expected_context_ids = {
        split: {match["context_id"] for match in dataset_manifest["matches"] if match["split"] == split}
        for split in split_examples
    }

    trained_metrics = {split: _score_examples(model, rows, resolved_device)
                       for split, rows in split_examples.items()}
    trained_contexts = {split: _score_by_context(model, rows, resolved_device, expected_context_ids[split])
                        for split, rows in split_examples.items()}
    policy = MLPPolicy(model, resolved_device)
    latency = benchmark_inference(policy, heldout, measured_calls=512)

    configure_determinism(QUALITY_UNTRAINED_SEED, resolved_device)
    untrained_model = PolicyMLP().to(resolved_device)
    untrained_model.eval()
    untrained_metrics = _score_examples(untrained_model, heldout, resolved_device)
    untrained_contexts = _score_by_context(
        untrained_model, heldout, resolved_device, expected_context_ids["heldout"],
    )
    oracle = _oracle_metrics(heldout)

    weights_hash = checkpoint_manifest["weights"]["sha256"]
    model_policy_id = f"mlp-{weights_hash[:16]}"
    model_stream = policy_stream_id(model_policy_id, 0)
    config = quality_config("classic")
    matches: list[dict[str, Any]] = []
    for opponent_id in QUALITY_MATCH_OPPONENTS:
        opponent_spec = baseline_spec(opponent_id)
        opponent_stream = policy_stream_id(opponent_spec.policy_id, 1)
        for seed in QUALITY_MATCH_SEEDS:
            for model_side in (0, 1):
                neural = PolicyController(policy, model_stream)
                opponent = PolicyController(opponent_spec.build(), opponent_stream)
                controllers = (neural, opponent) if model_side == 0 else (opponent, neural)
                match_result = run_episode(config, seed, controllers[0], controllers[1], max_ticks=config.t_max)
                matches.append(_match_record(match_result, seed, model_side, opponent_spec.full_id))
    truncated = [match for match in matches if match["truncated"]]
    completed = [match for match in matches if not match["truncated"] and match["model_score"] is not None]
    heldout_metrics = trained_metrics["heldout"]
    validation_metrics = trained_metrics["validation"]
    low_sample_contexts = [context_id for context_id, details in trained_contexts["heldout"].items()
                           if details["examples"] < 100]
    sample_size_ok = (
        heldout_metrics["examples"] >= 200
        and dataset_manifest["deduplication"]["split_unique_trajectory_counts"]["heldout"] >= 4
        and not low_sample_contexts
    )
    accuracy_margin = (heldout_metrics["exact_action_accuracy"]
                       - untrained_metrics["exact_action_accuracy"])
    validation_test_gap = abs(validation_metrics["exact_action_accuracy"]
                              - heldout_metrics["exact_action_accuracy"])
    checks = {
        "sample_size": sample_size_ok,
        "heldout_accuracy_at_least_0_80": heldout_metrics["exact_action_accuracy"] >= 0.80,
        "heldout_margin_over_untrained_at_least_0_20": accuracy_margin >= 0.20,
        "heldout_logit_mse_below_untrained": (
            heldout_metrics["teacher_logit_mse"] < untrained_metrics["teacher_logit_mse"]
        ),
        "heldout_legal_action_rate_1_0": heldout_metrics["legal_action_rate"] == 1.0,
        "validation_heldout_accuracy_gap_at_most_0_15": validation_test_gap <= 0.15,
        "checkpoint_dataset_provenance": True,
    }
    git_sha, git_dirty = None, None
    try:
        import subprocess

        repository = Path(__file__).resolve().parents[3]
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository, check=True,
                                 capture_output=True, text=True).stdout.strip()
        git_dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=repository, check=True,
                                        capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    report: dict[str, Any] = {
        "schema": "formicmind.ml-quality-evaluation/v1",
        "protocol": "docs/ml_quality_protocol.md",
        "classification": "technical imitation generalization check; not strategic superiority",
        "model": {
            "weights_sha256": weights_hash,
            "device": str(resolved_device),
            "checkpoint_manifest": "manifest.json",
        },
        "dataset": {
            "file": Path(dataset_path).name,
            "sha256": dataset_manifest["dataset_sha256"],
            "example_count": dataset_manifest["example_count"],
            "deduplication": dataset_manifest["deduplication"],
        },
        "contracts": checkpoint_manifest["contracts"],
        "imitation": {
            "trained": trained_metrics,
            "trained_by_context": trained_contexts,
            "untrained_heldout": untrained_metrics,
            "untrained_heldout_by_context": untrained_contexts,
            "teacher_oracle_heldout": oracle,
        },
        "protocol_assessment": {
            "decision": "GO to separate self-learning design" if all(checks.values()) else "NO-GO / insufficient evidence",
            "checks": checks,
            "heldout_accuracy_margin_over_untrained": accuracy_margin,
            "validation_heldout_accuracy_gap": validation_test_gap,
            "low_sample_heldout_contexts": low_sample_contexts,
            "explanation": (
                "A GO authorizes only a separately designed next experiment; it is not a strategic-strength claim."
                if all(checks.values()) else
                "At least one pre-registered technical or sample-size threshold failed; no self-learning transition is justified."
            ),
        },
        "performance": {"inference": latency},
        "matches": {
            "paired_seat": True,
            "max_ticks": config.t_max,
            "summary_by_opponent": _match_summaries(matches),
            "completed": completed,
            "truncated": truncated,
            "all": matches,
        },
        "source": {"git_sha": git_sha, "git_dirty": git_dirty},
        "match_seeds": list(QUALITY_MATCH_SEEDS),
    }
    if output is not None:
        path = Path(output)
        _write_atomic(path, (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    return report


__all__ = ["evaluate_quality_model", "train_quality_model"]
