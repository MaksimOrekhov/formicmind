"""FOR-40 training: factorised MSE vs action-aware losses, validation-only early stopping, selection record."""

from __future__ import annotations

import copy
import json
import platform
import time
from pathlib import Path
from typing import Any

from ..versions import ACTION_VERSION, DECODER_VERSION, OBSERVATION_VERSION
from .checkpoint import save_checkpoint
from .dataset import DatasetError, _write_atomic
from .for40 import FOR40_PROTOCOL, FOR40_PROTOCOL_COMMIT, TRAIN_SEEDS, load_for40_dataset, require_data_go
from .for40_metrics import compact, records_from_logits, summarize
from .model import PolicyMLP, configure_determinism, resolve_device, synchronize, torch

CONDITIONS = ("A", "B", "B1", "B2")
LOSS_SPEC = {
    "A": {"discrete": "mse", "forage": "mse"},
    "B": {"discrete": "cross_entropy", "forage": "share_surrogate"},
    "B1": {"discrete": "cross_entropy", "forage": "mse"},
    "B2": {"discrete": "mse", "forage": "share_surrogate"},
}
WEIGHTS = {"bce_avoid": 0.5, "bce_support": 0.5}
Q_TOTAL = 65535
MASK_FILL = -1e9
HYPERPARAMETERS = {
    "optimizer": "Adam", "learning_rate": 1e-3, "weight_decay": 0.0, "batch_size": 128,
    "max_epochs": 400, "min_epochs": 80, "patience_epochs": 50, "relative_improvement": 1e-3,
}


def build_tensors(examples: list[Any], device: torch.device) -> dict[str, torch.Tensor]:
    """Observation, teacher logits, masks and class targets of a split as device tensors."""
    def t(values: Any, dtype: Any) -> torch.Tensor:
        return torch.tensor(values, dtype=dtype, device=device)

    shares = [[q / Q_TOTAL for q in e.teacher_action.forage_share_q] for e in examples]
    support = [[1.0 if q > 0 else 0.0 for q in e.teacher_action.forage_share_q] for e in examples]
    conditional = [1.0 if tuple(e.mask.army_avoid_center[e.teacher_action.army_choice]) == (True, True) else 0.0
                   for e in examples]
    return {
        "x": t([e.observation for e in examples], torch.float32),
        "y": t([e.teacher_logits for e in examples], torch.float32),
        "lay": t([e.teacher_action.lay_order for e in examples], torch.long),
        "army": t([e.teacher_action.army_choice for e in examples], torch.long),
        "avoid": t([1.0 if e.teacher_action.avoid_center else 0.0 for e in examples], torch.float32),
        "conditional": t(conditional, torch.float32),
        "lay_mask": t([list(e.mask.lay) for e in examples], torch.bool),
        "army_mask": t([list(e.mask.army_choice) for e in examples], torch.bool),
        "share": t(shares, torch.float32),
        "support": t(support, torch.float32),
    }


def loss_components(output: torch.Tensor, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Every loss term of the protocol on one batch (means over the batch)."""
    F = torch.nn.functional
    y = batch["y"]
    lay_logits = output[:, 0:3].masked_fill(~batch["lay_mask"], MASK_FILL)
    army_logits = output[:, 9:16].masked_fill(~batch["army_mask"], MASK_FILL)
    forage = output[:, 3:9]
    # Discrete part = lay (3) + army (7) + avoid (1) = 11 logits.
    mse_disc = torch.cat([(output[:, 0:3] - y[:, 0:3]) ** 2, (output[:, 9:17] - y[:, 9:17]) ** 2], dim=1).mean()
    return {
        "mse_all": ((output - y) ** 2).mean(),
        "mse_disc": mse_disc,
        "mse_forage": ((forage - y[:, 3:9]) ** 2).mean(),
        "ce_lay": F.cross_entropy(lay_logits, batch["lay"]),
        "ce_army": F.cross_entropy(army_logits, batch["army"]),
        "bce_avoid": (F.binary_cross_entropy_with_logits(output[:, 16], batch["avoid"], reduction="none")
                      * batch["conditional"]).mean(),
        "ce_share": -(batch["share"] * F.log_softmax(forage, dim=1)).sum(dim=1).mean(),
        "bce_support": F.binary_cross_entropy_with_logits(forage, batch["support"]),
    }


def total_loss(condition: str, parts: dict[str, torch.Tensor]) -> torch.Tensor:
    if condition == "A":
        return parts["mse_all"]
    discrete = (parts["ce_lay"] + parts["ce_army"] + WEIGHTS["bce_avoid"] * parts["bce_avoid"]
                if LOSS_SPEC[condition]["discrete"] == "cross_entropy" else parts["mse_disc"])
    forage = (parts["ce_share"] + WEIGHTS["bce_support"] * parts["bce_support"]
              if LOSS_SPEC[condition]["forage"] == "share_surrogate" else parts["mse_forage"])
    return discrete + forage


def predict(model: PolicyMLP, tensors: dict[str, torch.Tensor]) -> list[list[float]]:
    model.eval()
    with torch.inference_mode():
        return model(tensors["x"]).detach().to("cpu").tolist()


def train_run(
    condition: str, seed: int, train_examples: list[Any], validation_examples: list[Any],
    device: torch.device, out_dir: Path, *, dataset_manifest: dict[str, Any], hyper: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one (condition, seed) run; selection uses only the condition's own validation loss."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown FOR-40 condition {condition!r}")
    hyper = {**HYPERPARAMETERS, **(hyper or {})}
    configure_determinism(seed, device)
    model = PolicyMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper["learning_rate"], weight_decay=hyper["weight_decay"])
    train_t = build_tensors(train_examples, device)
    val_t = build_tensors(validation_examples, device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    n_train = len(train_examples)
    curves: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    best_val, reference_val = float("inf"), float("inf")
    best_state, best_epoch, since_improvement, steps = None, 0, 0, 0
    stopped_by = "cap"
    started = time.perf_counter()
    for epoch in range(1, hyper["max_epochs"] + 1):
        model.train()
        permutation = torch.randperm(n_train, generator=generator)
        weighted, observed = 0.0, 0
        for offset in range(0, n_train, hyper["batch_size"]):
            index = permutation[offset:offset + hyper["batch_size"]].to(device)
            batch = {key: value.index_select(0, index) for key, value in train_t.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = total_loss(condition, loss_components(model(batch["x"]), batch))
            loss.backward()
            optimizer.step()
            steps += 1
            weighted += loss.detach().item() * int(index.numel())
            observed += int(index.numel())
        model.eval()
        with torch.inference_mode():
            val_loss = total_loss(condition, loss_components(model(val_t["x"]), val_t)).item()
        curves.append({"epoch": epoch, "train_loss": weighted / observed, "validation_loss": val_loss})
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        if val_loss < reference_val * (1.0 - hyper["relative_improvement"]):
            reference_val, since_improvement = val_loss, 0
        else:
            since_improvement += 1
        if epoch % 10 == 0:
            summary = summarize(records_from_logits(validation_examples, predict(model, val_t)))
            checks.append({"epoch": epoch, **{k: compact(summary)[k] for k in
                           ("composite", "distinct_decoded", "collapse")}})
        if epoch >= hyper["min_epochs"] and since_improvement >= hyper["patience_epochs"]:
            stopped_by = "patience"
            break
    synchronize(device)
    seconds = time.perf_counter() - started
    assert best_state is not None
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    val_summary = summarize(records_from_logits(validation_examples, predict(model, val_t)))
    train_summary = summarize(records_from_logits(train_examples, predict(model, train_t)))
    epochs_run = len(curves)
    tail_start = max(0, int(epochs_run * 0.75) - 1)
    tail_improvement = ((curves[tail_start]["validation_loss"] - min(c["validation_loss"] for c in curves[tail_start:]))
                        / abs(curves[tail_start]["validation_loss"])) if curves else 0.0
    try:
        cuda_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    except Exception:  # noqa: BLE001
        cuda_name = None
    manifest = {
        "objective": "FOR-40 data-first imitation; technical behavior cloning, not evidence of strategic superiority",
        "protocol": {"path": FOR40_PROTOCOL, "commit": FOR40_PROTOCOL_COMMIT},
        "loss": {"condition": condition, **LOSS_SPEC[condition], "weights": WEIGHTS,
                 "target": "balanced:v2 PolicyLogitsV2 / ActionV2 labels"},
        "training": {
            "seed": seed, **hyper, "epochs_run": epochs_run, "optimizer_steps": steps, "best_epoch": best_epoch,
            "best_validation_loss": best_val, "stopped_by": stopped_by, "converged": stopped_by == "patience",
            "tail_quarter_relative_validation_improvement": tail_improvement,
            "curves": curves, "periodic_validation_checks": checks,
            "train": compact(train_summary), "validation": compact(val_summary),
        },
        "performance": {"training_device": str(device), "cuda_device_name": cuda_name, "training_seconds": seconds},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                    "cuda_available": torch.cuda.is_available(), "device": str(device),
                    "torch_threads": torch.get_num_threads()},
        "data": {"schema": dataset_manifest["schema"], "sha256": dataset_manifest["dataset_sha256"],
                 "map_plan_sha256": dataset_manifest["map_plan"]["sha256"],
                 "train_examples": n_train, "validation_examples": len(validation_examples)},
        "versions": {"observation": OBSERVATION_VERSION, "action": ACTION_VERSION, "decoder": DECODER_VERSION},
        "determinism": {"algorithms": "torch.use_deterministic_algorithms(True)", "seeded_cpu_rng": True,
                        "cuda_reproducible_across_hardware": False},
    }
    result = save_checkpoint(model, out_dir, manifest)
    return {
        "condition": condition, "seed": seed, "checkpoint": out_dir.name, "weights_sha256": result["weights"]["sha256"],
        "device": str(device), "epochs_run": epochs_run, "optimizer_steps": steps, "best_epoch": best_epoch,
        "best_validation_loss": best_val, "stopped_by": stopped_by, "converged": stopped_by == "patience",
        "validation_composite": val_summary["composite"], "validation": compact(val_summary),
        "train": compact(train_summary), "seconds": seconds,
        "tail_quarter_relative_validation_improvement": tail_improvement,
    }


def select_models(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Protocol selection: best validation composite per condition, then A vs B for the GO candidate."""
    per_condition = {}
    for condition in CONDITIONS:
        group = [r for r in runs if r["condition"] == condition and r["validation_composite"] is not None]
        if group:
            per_condition[condition] = max(group, key=lambda r: (r["validation_composite"], -r["seed"]))
    candidate = None
    if "A" in per_condition and "B" in per_condition:
        candidate = "B" if per_condition["B"]["validation_composite"] >= per_condition["A"]["validation_composite"] else "A"
    return {"selected": {c: {"seed": r["seed"], "checkpoint": r["checkpoint"], "weights_sha256": r["weights_sha256"],
                             "validation_composite": r["validation_composite"]} for c, r in per_condition.items()},
            "candidate_condition": candidate}


def train_all_for40(dataset_path: str | Path, output_dir: str | Path, *, device_name: str = "cuda",
                    cpu_reproduction: bool = True, seeds: tuple[int, ...] = TRAIN_SEEDS,
                    hyper: dict[str, Any] | None = None, expect_full_protocol: bool = True) -> dict[str, Any]:
    """Run the 4 × len(seeds) preregistered trainings; ``hyper``/``seeds`` are overridden only by smoke tests."""
    manifest, examples = load_for40_dataset(dataset_path, expect_full_protocol=expect_full_protocol)
    if expect_full_protocol:  # smoke tests (False) keep working on tiny plans; the real dataset needs DATA GO
        require_data_go(manifest, examples)
    train = [e for e in examples if e.split == "train"]
    validation = [e for e in examples if e.split == "validation"]
    if not train or not validation:
        raise DatasetError("FOR-40 training needs non-empty train and validation splits")
    output_dir = Path(output_dir)
    device = resolve_device(device_name)
    runs: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for seed in seeds:
            name = f"{condition}_{seed}"
            runs.append(train_run(condition, seed, train, validation, device,
                                  output_dir / "checkpoints" / name, dataset_manifest=manifest, hyper=hyper))
    reproduction = []
    if cpu_reproduction and device.type != "cpu":
        cpu = resolve_device("cpu")
        for condition in ("A", "B"):
            original = next(r for r in runs if r["condition"] == condition and r["seed"] == seeds[0])
            repro = train_run(condition, seeds[0], train, validation, cpu,
                              output_dir / "checkpoints_cpu_reproduction" / f"{condition}_{seeds[0]}",
                              dataset_manifest=manifest, hyper=hyper)
            reproduction.append({
                "condition": condition, "seed": seeds[0],
                "cuda_best_validation_loss": original["best_validation_loss"], "cpu_best_validation_loss": repro["best_validation_loss"],
                "cuda_validation_composite": original["validation_composite"], "cpu_validation_composite": repro["validation_composite"],
                "cuda_best_epoch": original["best_epoch"], "cpu_best_epoch": repro["best_epoch"],
                "cuda_stopped_by": original["stopped_by"], "cpu_stopped_by": repro["stopped_by"],
            })
    selection = {
        "schema": "formicmind.ml-for40-selection/v1",
        "protocol": {"path": FOR40_PROTOCOL, "commit": FOR40_PROTOCOL_COMMIT},
        "dataset_sha256": manifest["dataset_sha256"],
        "rule": "max validation composite per condition (tie: smaller seed); candidate = higher composite of selected A and B (tie: B)",
        **select_models(runs),
        "runs": runs,
        "cpu_reproduction": reproduction,
        "hyperparameters": {**HYPERPARAMETERS, **(hyper or {})},
        "frozen_before_heldout": True,
    }
    _write_atomic(output_dir / "selection.json",
                  (json.dumps(selection, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    return selection


__all__ = ["CONDITIONS", "HYPERPARAMETERS", "LOSS_SPEC", "build_tensors", "loss_components", "predict",
           "select_models", "total_loss", "train_all_for40", "train_run"]
