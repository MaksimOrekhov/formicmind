"""FOR-38 diagnostics: synthetic decoder strictness, metric controls, frozen-artifact guard, provenance failures."""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from pathlib import Path

import pytest

from formicmind.decoder import PolicyLogitsV2, decode_forage_logits, quantize_shares
from formicmind.ml import diagnostics as dg
from formicmind.ml.dataset import DatasetError
from formicmind.ml.quality_dataset import collect_quality_dataset, load_quality_dataset

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "results" / "ml_quality_for37"
ALL = (True,) * 6


# ----------------------------------------------------------------------------- frozen inputs are untouched

def test_frozen_for37_artifacts_have_pinned_hashes():
    result = dg.verify_frozen_artifacts(FROZEN)
    assert set(result) == set(dg.FROZEN_FOR37_SHA256)
    for name, info in result.items():
        assert info["present"], name
        assert info["match"], f"{name} changed: {info['sha256']}"
    # binary / byte-exact artifacts are hashed without any line-ending normalisation
    assert result["quality.jsonl"]["hashed_with_lf_normalization"] is False
    assert result["checkpoint/weights.pt"]["hashed_with_lf_normalization"] is False


def test_file_sha256_can_ignore_checkout_line_endings(tmp_path):
    lf, crlf = tmp_path / "a.json", tmp_path / "b.json"
    lf.write_bytes(b'{"a": 1}\n')
    crlf.write_bytes(b'{"a": 1}\r\n')
    assert dg.file_sha256(lf) != dg.file_sha256(crlf)
    assert dg.file_sha256(lf, normalize_lf=True) == dg.file_sha256(crlf, normalize_lf=True)


# ----------------------------------------------------------------------------- pair metrics / categories

def _q(*values: int) -> tuple[int, ...]:
    assert sum(values) == 65535
    return tuple(values)


def test_total_variation_matches_definition():
    a, b = _q(65535, 0, 0, 0, 0, 0), _q(0, 65535, 0, 0, 0, 0)
    assert dg.total_variation(a, a) == 0.0
    assert dg.total_variation(a, b) == 1.0
    assert dg.total_variation(_q(32768, 32767, 0, 0, 0, 0), _q(32767, 32768, 0, 0, 0, 0)) == pytest.approx(1 / 65535)


def test_forage_categories_separate_selection_errors_from_share_errors():
    teacher = _q(0, 20000, 20000, 25535, 0, 0)
    cases = {
        "exact": teacher,
        "same_support_small_share_error": _q(0, 20001, 20000, 25534, 0, 0),
        "same_support_large_redistribution": _q(0, 40000, 20000, 5535, 0, 0),
        "support_pred_superset": _q(5000, 20000, 20000, 20535, 0, 0),
        "support_pred_subset": _q(0, 32768, 0, 32767, 0, 0),
        "support_partial_overlap": _q(0, 20000, 0, 25535, 20000, 0),
        "support_disjoint": _q(30000, 0, 0, 0, 20000, 15535),
    }
    for expected, predicted in cases.items():
        assert dg.forage_pair_metrics(predicted, teacher)["category"] == expected, expected
    small = dg.forage_pair_metrics(cases["same_support_small_share_error"], teacher)
    assert small["support_equal"] and not small["exact"] and small["tv"] == pytest.approx(1 / 65535)
    assert dg.forage_pair_metrics(cases["support_disjoint"], teacher)["jaccard"] == 0.0
    # top-1 target can match while the support does not (share error, not target error)
    assert dg.forage_pair_metrics(cases["support_pred_superset"], teacher)["top1_match"] is True
    assert dg.forage_pair_metrics(cases["support_disjoint"], teacher)["top1_match"] is False


def test_forage_pair_metrics_reject_malformed_vectors():
    with pytest.raises(ValueError):
        dg.forage_pair_metrics((1, 2, 3), _q(65535, 0, 0, 0, 0, 0))
    with pytest.raises(ValueError):
        dg.forage_pair_metrics((1, 0, 0, 0, 0, 0), _q(65535, 0, 0, 0, 0, 0))


# ----------------------------------------------------------------------------- synthetic decoder behaviour

def test_equal_logits_split_evenly_and_remainder_goes_to_lower_index():
    three = decode_forage_logits([0.0, 1.0, 1.0, 1.0, -1.0, -1.0], ALL)
    assert three == (0, 21845, 21845, 21845, 0, 0)
    two = decode_forage_logits([-1.0, 1.0, 1.0, -1.0, -1.0, -1.0], ALL)
    assert two == (0, 32768, 32767, 0, 0, 0)  # 65535 is odd: the extra unit goes to the lower category


def test_softmax_is_shift_invariant_but_not_quantization_free():
    base = [0.0, 1.5, 1.0, 0.5, -1.0, -1.0]
    shifted = [v + 0.25 if i in (1, 2, 3) else v for i, v in enumerate(base)]
    assert decode_forage_logits(base, ALL) == decode_forage_logits(shifted, ALL)
    q = decode_forage_logits(base, ALL)
    probs = dg.forage_continuous_probs(base, ALL)
    assert sum(q) == 65535
    for share, p in zip(q, probs):
        assert abs(share / 65535 - p) <= 1 / 65535  # largest remainder moves each share by less than one unit


def test_zero_is_not_positive_and_all_nonpositive_logits_keep_one_target():
    all_zero = decode_forage_logits([0.0] * 6, ALL)
    assert all_zero == (65535, 0, 0, 0, 0, 0)  # k = 1 -> lowest index wins the tie
    tiny_positive = decode_forage_logits([0.0, 1e-6, 0.0, 0.0, 0.0, 0.0], ALL)
    assert tiny_positive[1] == 65535  # one positive logit -> support 1
    two = decode_forage_logits([0.0, 1e-6, 2e-6, 0.0, 0.0, 0.0], ALL)
    assert sum(1 for v in two if v) == 2  # crossing zero changes the *support*, not just the share


def test_one_quantum_requires_logit_precision_near_1e_minus_4():
    base = [-1.0, 1.0, 0.8, 0.6, -1.0, -1.0]
    q0 = decode_forage_logits(base, ALL)
    nudged = list(base)
    nudged[1] += 2e-4
    assert decode_forage_logits(nudged, ALL) != q0  # a 2e-4 logit error already moves a share
    below_float32 = list(base)
    below_float32[1] += 1e-9  # lost when the logit is stored as float32
    assert PolicyLogitsV2([*below_float32, *([0.0] * 11)]).values[1] == PolicyLogitsV2([*base, *([0.0] * 11)]).values[1]
    assert decode_forage_logits(below_float32, ALL) == q0


def test_masked_forage_targets_never_receive_mass():
    mask = (True, False, True, True, False, True)
    q = decode_forage_logits([0.3, 5.0, 0.2, 0.1, 9.0, 0.4], mask)
    assert q[1] == 0 and q[4] == 0 and sum(q) == 65535


def test_independent_reference_decoder_agrees_with_shared_decoder():
    rng = random.Random(3838)
    for _ in range(300):
        logits = [rng.choice([-1.0, -0.4, 0.0, 0.3, 1.0, 1.7]) + rng.uniform(-0.05, 0.05) for _ in range(6)]
        mask = tuple(rng.random() > 0.2 for _ in range(6))
        if not any(mask):
            continue
        float32 = PolicyLogitsV2([*logits, *([0.0] * 11)]).values[:6]
        assert dg._independent_forage_reference(float32, mask) == decode_forage_logits(logits, mask)


def test_quantize_shares_total_and_tie_breaking():
    assert sum(quantize_shares([1, 1, 1, 0, 0, 0])) == 65535
    assert quantize_shares([1, 1, 0, 0, 0, 0]) == (32768, 32767, 0, 0, 0, 0)


# ----------------------------------------------------------------------------- analysis on a real tiny dataset

@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("diag")
    path = root / "quality.jsonl"
    collect_quality_dataset(path, train_seeds=[6101, 6102], validation_seeds=[6201, 6202],
                            heldout_seeds=[6301, 6302], max_ticks=120)
    manifest, examples = load_quality_dataset(path)
    return {"root": root, "path": path, "manifest": manifest, "examples": examples}


def test_oracle_predictions_score_perfectly_through_the_same_metric_path(tiny_dataset):
    examples = tiny_dataset["examples"]
    summary = dg.analyze_predictions(examples, [e.teacher_logits for e in examples])
    assert summary["action_accuracy"]["full_action"] == 1.0
    assert summary["action_accuracy"]["forage_share_exact"] == 1.0
    assert summary["forage"]["categories"]["exact"] == len(examples)
    assert summary["logit_mse"]["all"] == 0.0
    assert summary["forage"]["tv_mean"] == 0.0
    assert dg.teacher_oracle_roundtrip(examples) == {"examples": len(examples), "full_action_exact_rate": 1.0,
                                                    "forage_exact_rate": 1.0}


def test_constant_predictions_are_visible_in_variability_and_majority_baselines(tiny_dataset):
    examples = tiny_dataset["examples"]
    constant = [0.0] * 17
    summary = dg.analyze_predictions(examples, [constant] * len(examples))
    assert summary["variability"]["distinct_decoded_lay_army_avoid"] == 1
    assert summary["variability"]["by_group"]["lay"]["pred_std_mean"] == 0.0
    assert 0.0 < summary["majority_class_accuracy"]["lay"] <= 1.0
    # decoding of an all-zero forage block keeps one target, so the support is always size one
    assert set(summary["forage"]["support_k_distribution"]["pred"]) == {1}


def test_mean_logit_baseline_ignores_observation(tiny_dataset):
    examples = tiny_dataset["examples"]
    train = [e for e in examples if e.split == "train"]
    baseline = dg.mean_logit_baseline(train, examples)
    assert len(baseline["mean_logits"]) == 17
    assert baseline["logit_mse"] >= 0.0
    assert baseline["action_accuracy"]["full_action"] <= 1.0


def test_masked_logit_errors_are_reported_separately(tiny_dataset):
    examples = tiny_dataset["examples"]
    wrong = [list(e.teacher_logits) for e in examples]
    for row, example in zip(wrong, examples):
        for index, active in enumerate(dg._active_logit_flags(example.mask, example.teacher_action)):
            if not active:
                row[index] += 5.0
    summary = dg.analyze_predictions(examples, wrong)
    masked = summary["logit_mse"]["masked_or_inactive_entries"]
    assert masked["count"] > 0
    assert masked["mse"] > 0.0
    # perturbing only masked entries cannot change the decoded action
    assert summary["action_accuracy"]["full_action"] == 1.0


def test_noise_sensitivity_zero_noise_is_exact_and_large_noise_is_not(tiny_dataset):
    examples = tiny_dataset["examples"]
    rows = dg.noise_sensitivity(examples, sigmas=(0.0, 1e-6, 1e-1), repeats=2)
    assert rows[0]["forage_exact_rate"] == 1.0
    assert rows[-1]["forage_exact_rate"] < rows[0]["forage_exact_rate"]
    assert rows[-1]["forage_exact_rate"] <= rows[1]["forage_exact_rate"]
    assert rows[-1]["samples"] == 2 * len(examples)
    again = dg.noise_sensitivity(examples, sigmas=(0.0, 1e-6, 1e-1), repeats=2)
    assert again == rows  # fixed seed -> reproducible


def test_share_sensitivity_is_thousands_of_counts_per_logit_unit(tiny_dataset):
    sensitivity = dg.share_sensitivity(tiny_dataset["examples"])
    assert sensitivity["counts_per_logit_unit_median"] > 1000
    assert 1e-6 < sensitivity["logit_error_for_one_count_median"] < 1e-3


def test_analyze_predictions_rejects_misaligned_inputs(tiny_dataset):
    with pytest.raises(ValueError):
        dg.analyze_predictions(tiny_dataset["examples"], [])
    assert dg.analyze_predictions([], []) == {"examples": 0}


def test_rollout_summary_reports_first_divergence():
    log = [
        {"tick": 0, "full_agree": True, "lay_agree": True, "army_agree": True, "avoid_agree": True,
         "forage_exact": True, "forage_support_equal": True, "forage_tv": 0.0},
        {"tick": 30, "full_agree": False, "lay_agree": False, "army_agree": True, "avoid_agree": True,
         "forage_exact": False, "forage_support_equal": True, "forage_tv": 0.1},
    ]
    summary = dg.summarize_rollout_log(log)
    assert summary["decisions"] == 2
    assert summary["first_divergence_tick_full_action"] == 30
    assert summary["first_divergence_tick_lay_army_avoid"] == 30
    assert summary["lay_agree_rate"] == 0.5
    assert dg.summarize_rollout_log([]) == {"decisions": 0}


# ----------------------------------------------------------------------------- checkpoint-facing (needs torch)

@pytest.fixture(scope="module")
def trained_tiny(tiny_dataset):
    pytest.importorskip("torch", reason="diagnostics smoke runs in Windows CI")
    from formicmind.ml.quality import train_quality_model

    checkpoint = tiny_dataset["root"] / "checkpoint"
    train_quality_model(tiny_dataset["path"], checkpoint, seed=3838, epochs=1, batch_size=32, requested_device="cpu")
    return checkpoint


def _tree_hashes(*roots: Path) -> dict[str, str]:
    result = {}
    for root in roots:
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        for path in files:
            result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def test_diagnose_quality_is_read_only_and_reports_pipeline_controls(tiny_dataset, trained_tiny, tmp_path):
    before = _tree_hashes(tiny_dataset["path"], trained_tiny)
    output = tmp_path / "diag.json"
    report = dg.diagnose_quality(trained_tiny, tiny_dataset["path"], device="cpu", output=output)
    assert _tree_hashes(tiny_dataset["path"], trained_tiny) == before
    assert json.loads(output.read_text(encoding="utf-8")) == json.loads(json.dumps(report))
    checks = report["pipeline_checks"]
    assert checks["outputs_have_17_values"] and checks["outputs_finite"] and checks["outputs_are_exact_float32_values"]
    assert checks["stored_masks_equal_rederived_masks"]
    assert checks["teacher_oracle_roundtrip"]["full_action_exact_rate"] == 1.0
    agree = checks["independent_forage_reference_agrees_with_teacher_labels"]
    assert agree["agree"] == agree["of"] == len(tiny_dataset["examples"])
    assert set(report["splits"]) == {"train", "validation", "heldout"}
    assert report["provenance"]["checkpoint_dataset_bound"] is True
    assert report["provenance"]["dataset_sha256"] == tiny_dataset["manifest"]["dataset_sha256"]
    assert set(report["mean_logit_baseline"]) == {"train", "validation", "heldout"}
    assert report["training_facts"]["optimizer_steps_total"] >= 1
    assert report["reproduces_reported_metrics"] is None  # no report.json next to the tiny dataset


def test_diagnose_quality_rejects_checkpoint_bound_to_another_dataset(tiny_dataset, trained_tiny, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    shutil.copytree(trained_tiny, checkpoint)
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["data"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="checkpoint dataset SHA-256 does not match quality dataset"):
        dg.diagnose_quality(checkpoint, tiny_dataset["path"], device="cpu")


def test_diagnose_quality_rejects_malformed_dataset_provenance(tiny_dataset, trained_tiny, tmp_path):
    dataset = tmp_path / "quality.jsonl"
    shutil.copy(tiny_dataset["path"], dataset)
    shutil.copy(tiny_dataset["path"].with_name("quality.manifest.json"), tmp_path / "quality.manifest.json")
    lines = dataset.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["teacher_side"] = 1 - row["teacher_side"]
    lines[0] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    raw = ("\n".join(lines) + "\n").encode()
    dataset.write_bytes(raw)
    manifest_path = tmp_path / "quality.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_sha256"] = hashlib.sha256(raw).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="example provenance differs from its retained match"):
        dg.diagnose_quality(trained_tiny, dataset, device="cpu")


def test_diagnose_quality_rejects_tampered_dataset_bytes(tiny_dataset, trained_tiny, tmp_path):
    dataset = tmp_path / "quality.jsonl"
    shutil.copy(tiny_dataset["path"], dataset)
    shutil.copy(tiny_dataset["path"].with_name("quality.manifest.json"), tmp_path / "quality.manifest.json")
    dataset.write_bytes(dataset.read_bytes() + b"\n")
    with pytest.raises(DatasetError, match="SHA-256 does not match"):
        dg.diagnose_quality(trained_tiny, dataset, device="cpu")


def test_prediction_shape_dtype_and_finiteness_are_enforced(tiny_dataset):
    torch = pytest.importorskip("torch", reason="diagnostics smoke runs in Windows CI")
    examples = tiny_dataset["examples"][:3]

    class Wrong(torch.nn.Module):
        def forward(self, x):
            return torch.zeros((x.shape[0], 16))

    class Bad(torch.nn.Module):
        def forward(self, x):
            return torch.full((x.shape[0], 17), float("nan"))

    with pytest.raises(DatasetError, match="expected float32"):
        dg.predict_logits(Wrong(), examples, torch.device("cpu"))
    with pytest.raises(DatasetError, match="non-finite"):
        dg.predict_logits(Bad(), examples, torch.device("cpu"))


def test_rollout_divergence_smoke_with_short_cutoff(tiny_dataset, trained_tiny):
    pytest.importorskip("torch", reason="diagnostics smoke runs in Windows CI")
    from formicmind.ml.checkpoint import load_checkpoint

    model, manifest, device = load_checkpoint(trained_tiny, "cpu")
    result = dg.rollout_divergence(model, manifest, device, None, opponents=("turtle:v2",), sides=(0,), max_ticks=61)
    assert len(result["games"]) == 1
    game = result["games"][0]
    assert game["summary"]["decisions"] == 3  # ticks 0, 30, 60
    assert game["reproduces_reported_match"] is None
    assert all(set(d) >= {"tick", "full_agree", "forage_tv", "model_action", "teacher_action"} for d in game["decisions"])
