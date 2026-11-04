"""Reproducible CPU/CUDA behavior cloning and inference diagnostics."""

from __future__ import annotations

import hashlib
import platform
import statistics
import time
from pathlib import Path
from typing import Any

from ..actions import ActionV2, get_action_mask, validate_action
from ..decoder import PolicyLogitsV2, decode_action_logits
from ..observation import ObservationV2
from ..rng import new_stream
from ..versions import ACTION_VERSION, DECODER_VERSION, OBSERVATION_VERSION
from .checkpoint import save_checkpoint
from .dataset import DatasetError, DatasetExample, _git_provenance, load_dataset
from .model import (
    HIDDEN_SIZES,
    INPUT_SIZE,
    OUTPUT_SIZE,
    MLPPolicy,
    PolicyMLP,
    configure_determinism,
    resolve_device,
    synchronize,
    torch,
)


def _score_examples(model: PolicyMLP, examples: list[DatasetExample], device: torch.device) -> dict[str, Any]:
    if not examples:
        raise DatasetError("cannot score an empty dataset split")
    model.eval()
    x = torch.tensor([example.observation for example in examples], dtype=torch.float32, device=device)
    y = torch.tensor([example.teacher_logits for example in examples], dtype=torch.float32, device=device)
    with torch.inference_mode():
        outputs = model(x)
        mse = torch.mean((outputs - y) ** 2).item()
        predictions = outputs.detach().to("cpu").tolist()
    exact = lay = army = forage = avoid = legal = 0
    invalid = 0
    for example, prediction in zip(examples, predictions, strict=True):
        try:
            logits = PolicyLogitsV2(prediction)
            action = validate_action(
                ObservationV2(example.observation),
                example.mask,
                decode_action_logits(logits, example.mask),
            ).action
        except (ValueError, OverflowError, TypeError):
            invalid += 1
            continue
        legal += 1
        exact += action == example.teacher_action
        lay += action.lay_order == example.teacher_action.lay_order
        army += action.army_choice == example.teacher_action.army_choice
        forage += action.forage_share_q == example.teacher_action.forage_share_q
        avoid += action.avoid_center == example.teacher_action.avoid_center
    total = len(examples)
    return {
        "examples": total,
        "teacher_logit_mse": mse,
        "legal_action_rate": legal / total,
        "invalid_action_count": invalid,
        "exact_action_accuracy": exact / total,
        "lay_accuracy": lay / total,
        "army_accuracy": army / total,
        "forage_share_exact_accuracy": forage / total,
        "avoid_center_accuracy": avoid / total,
    }


def benchmark_inference(policy: MLPPolicy, examples: list[DatasetExample], *, measured_calls: int = 128) -> dict[str, float | int]:
    """Time end-to-end ObservationV2 → PolicyLogitsV2 on the selected device."""
    if not examples or measured_calls <= 0:
        raise ValueError("inference benchmark needs examples and a positive call count")
    device = policy.device
    benchmark_rng = new_stream(0, "policy:mlp-benchmark:0")
    probe_count = min(8, len(examples))
    for index in range(probe_count):
        example = examples[index]
        policy(
            ObservationV2(example.observation),
            example.mask,
            benchmark_rng,
        )
    synchronize(device)
    latencies: list[float] = []
    for index in range(measured_calls):
        example = examples[index % len(examples)]
        started = time.perf_counter()
        policy(
            ObservationV2(example.observation),
            example.mask,
            benchmark_rng,
        )
        synchronize(device)
        latencies.append(time.perf_counter() - started)
    elapsed = sum(latencies)

    def percentile(values: list[float], fraction: float) -> float:
        ordered = sorted(values)
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "calls": measured_calls,
        "seconds": elapsed,
        "milliseconds_per_decision": elapsed * 1000 / measured_calls,
        "milliseconds_p50": percentile(latencies, 0.50) * 1000,
        "milliseconds_p95": percentile(latencies, 0.95) * 1000,
        "decisions_per_second": measured_calls / elapsed if elapsed else float("inf"),
    }


def train_model(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 20261136,
    epochs: int = 8,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    requested_device: str = "auto",
) -> dict[str, Any]:
    """Fit the fixed MLP to teacher PolicyLogitsV2 using deterministic MSE regression."""
    if type(seed) is not int or not 0 <= seed < (1 << 63):
        raise ValueError("training seed must be an integer in 0..2^63-1")
    if type(epochs) is not int or epochs <= 0 or type(batch_size) is not int or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive integers")
    if learning_rate <= 0 or not (learning_rate < float("inf")):
        raise ValueError("learning_rate must be finite and positive")
    dataset_manifest, examples = load_dataset(dataset_path)
    train_examples = [example for example in examples if example.split == "train"]
    validation_examples = [example for example in examples if example.split == "validation"]
    if not train_examples or not validation_examples:
        raise DatasetError("dataset must contain both whole-match train and validation splits")

    device = resolve_device(requested_device)
    configure_determinism(seed, device)
    model = PolicyMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    x_train = torch.tensor([example.observation for example in train_examples], dtype=torch.float32, device=device)
    y_train = torch.tensor([example.teacher_logits for example in train_examples], dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    step_seconds: list[float] = []
    epoch_losses: list[float] = []
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
    git_sha, git_dirty = _git_provenance()
    try:
        cuda_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    except Exception:
        cuda_name = None
    training_steps = len(step_seconds)
    samples_per_second = (epochs * len(train_examples) / total_training_seconds
                          if total_training_seconds else float("inf"))
    manifest: dict[str, Any] = {
        "objective": "technical behavior cloning; not evidence of strategic superiority",
        "loss": {"name": "mean_squared_error", "target": "teacher PolicyLogitsV2 (17 float32)"},
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
            "optimizer_steps": training_steps,
            "training_seconds": total_training_seconds,
            "mean_step_milliseconds": statistics.fmean(step_seconds) * 1000 if step_seconds else 0.0,
            "training_samples_per_second": samples_per_second,
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
            "file": Path(dataset_path).name,
            "sha256": dataset_manifest["dataset_sha256"],
            "example_count": len(examples),
            "match_count": dataset_manifest["match_count"],
            "seeds": dataset_manifest["seeds"],
            "split_by_seed": dataset_manifest["split_by_seed"],
            "matches": dataset_manifest["matches"],
            "teacher": dataset_manifest["teacher"],
            "opponent": dataset_manifest["opponent"],
            "simulator": dataset_manifest["simulator"],
            "collector_source": dataset_manifest["source"],
        },
        "source": {"git_sha": git_sha, "git_dirty": git_dirty},
        "versions": {
            "observation": OBSERVATION_VERSION,
            "action": ACTION_VERSION,
            "decoder": DECODER_VERSION,
        },
        "determinism": {
            "algorithms": "torch.use_deterministic_algorithms(True)",
            "seeded_cpu_rng": True,
            "cuda_reproducible_across_hardware": False,
        },
    }
    result = save_checkpoint(model, output_dir, manifest)
    result["model_sha256"] = result["weights"]["sha256"]
    return result


__all__ = ["benchmark_inference", "train_model"]
