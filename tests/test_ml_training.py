from __future__ import annotations

import pytest

from formicmind.ml.dataset import collect_dataset


def test_cpu_training_safe_load_and_actual_match_smoke(tmp_path):
    torch = pytest.importorskip("torch", reason="CPU ML smoke is installed and run by Windows CI")
    from formicmind.ml.checkpoint import CheckpointError, load_checkpoint
    from formicmind.ml.evaluation import evaluate_model
    from formicmind.ml.training import train_model

    dataset_path = tmp_path / "pilot.jsonl"
    collect_dataset(dataset_path, seeds=[301, 302], max_ticks=1, validation_fraction=0.5)

    first_dir = tmp_path / "model-first"
    first = train_model(dataset_path, first_dir, seed=1234, epochs=2, batch_size=2, requested_device="cpu")
    second_dir = tmp_path / "model-second"
    second = train_model(dataset_path, second_dir, seed=1234, epochs=2, batch_size=2, requested_device="cpu")
    assert first["performance"]["training_device"] == "cpu"
    assert first["training"]["validation"]["examples"] > 0
    assert first["training"]["validation"]["legal_action_rate"] == 1.0
    assert first["training"]["validation"]["invalid_action_count"] == 0
    assert first["training"]["validation"]["exact_action_accuracy"] >= 0.0
    assert first["performance"]["optimizer_steps"] > 0
    assert first["performance"]["inference"]["decisions_per_second"] > 0
    assert first["weights"]["sha256"] == second["weights"]["sha256"]

    loaded, manifest, device = load_checkpoint(first_dir, "cpu")
    assert isinstance(loaded, torch.nn.Module)
    assert str(device) == "cpu"
    assert manifest["contracts"]["observation_size"] == 223
    assert manifest["model"]["output_size"] == 17

    report_path = tmp_path / "match-report.json"
    report = evaluate_model(
        first_dir, dataset_path, seeds=[901], opponent="balanced:v2", max_ticks=1,
        device="cpu", output=report_path,
    )
    assert report["match_summary"]["matches"] == 2
    assert report["imitation"]["legal_action_rate"] == 1.0
    assert {match["model_side"] for match in report["matches"]} == {0, 1}
    assert all(match["ticks_executed"] == 1 for match in report["matches"])
    assert report_path.exists()
    with pytest.raises(ValueError, match="overlap"):
        evaluate_model(
            first_dir, dataset_path, seeds=[301], opponent="balanced:v2", max_ticks=1, device="cpu",
        )

    weights = first_dir / "weights.pt"
    weights.write_bytes(weights.read_bytes() + b"bad")
    with pytest.raises(CheckpointError, match="SHA-256"):
        load_checkpoint(first_dir, "cpu")


def test_explicit_cuda_fails_loudly_without_cuda():
    torch = pytest.importorskip("torch", reason="CPU ML smoke is installed and run by Windows CI")
    from formicmind.ml.model import resolve_device

    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="device=cuda was requested"):
            resolve_device("cuda")
