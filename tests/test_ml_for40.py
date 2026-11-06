"""FOR-40 CPU smoke and integrity tests: map plan, plan, loader/leakage, losses, training, metrics, evaluation.

The real held-out maps are never played here: a module fixture relabels three *train* maps as train / validation /
held-out so the full pipeline runs on a tiny plan.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from formicmind.ml import for40, for40_eval, for40_train  # noqa: E402
from formicmind.ml.checkpoint import load_checkpoint  # noqa: E402
from formicmind.ml.dataset import DatasetError  # noqa: E402
from formicmind.ml.for40_metrics import (  # noqa: E402
    majority_summary,
    records_from_logits,
    summarize,
    train_reference,
)
from formicmind.ml.for40_train import build_tensors, loss_components, total_loss  # noqa: E402


# ------------------------------------------------------------------------------------------------ plan / maps

def test_map_plan_reproduces_preregistered_checksum_and_properties() -> None:
    plan = for40.map_plan()
    assert for40.map_plan_sha256() == for40.MAP_PLAN_SHA256
    assert [len(for40.maps_of_split(s)) for s in for40.SPLITS] == [24, 8, 8]
    assert for40.edge_orbits() == ((0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 5))
    references = list(for40.reference_vectors().values())
    assert len(references) == 9
    for i, spec in enumerate(plan):
        assert all(for40.vector_distance(spec.vector, other) >= 2 for other in references)
        assert all(for40.vector_distance(spec.vector, other.vector) >= 2 for other in plan[i + 1:])
        config = for40.for40_config(spec.map_id)  # validate_config + ObservationV2 limit + routing
        assert max(edge.length for edge in config.topology.edges) <= 4
        assert 7 <= len(config.topology.edges) <= 13
    assert len({m.map_id for m in plan}) == 40


def test_plan_sizes_seed_ranges_and_deferred_opponents() -> None:
    plan = for40.plan_for40()
    assert len(plan) == 1216 and len({p.match_id for p in plan}) == 1216
    assert {s: sum(p.split == s for p in plan) for s in for40.SPLITS} == for40.EXPECTED_MATCHES
    assert all(not p.disturbed for p in plan if p.split == "heldout")
    for item in plan:
        assert 400_000_000 < item.seed < 406_000_000  # not FOR-4/25/35/37/38 ranges
        if item.split != "heldout":
            assert item.opponent not in ("rush:v2", "raider:v2") and "rush" not in item.opponent and "raider" not in item.opponent
    assert {p.opponent for p in plan if p.split == "heldout"} == {
        "turtle:v2", "balanced:v2", "rush:v2", "raider:v2", "random:v2"}
    # whole maps never cross splits
    assert all(len({p.split for p in plan if p.map_id == m.map_id}) == 1 for m in for40.map_plan())


# ------------------------------------------------------------------------------------------------ mini pipeline

@pytest.fixture(scope="module")
def mini(tmp_path_factory: pytest.TempPathFactory):
    base = [m for m in for40.map_plan() if m.split == "train"][:3]
    relabeled = tuple(for40.MapSpec(m.seed, m.vector, split)
                      for m, split in zip(base, ("train", "validation", "heldout"), strict=True))
    patch = pytest.MonkeyPatch()
    patch.setattr(for40, "map_plan", lambda: relabeled)
    for40.map_by_id.cache_clear()
    patch.setattr(for40, "MAP_PLAN_SHA256", for40.map_plan_sha256())
    patch.setattr(for40_eval, "maps_of_split", for40.maps_of_split)
    full = for40.plan_for40(maps=relabeled)

    def pick(split: str, source: str, count: int) -> list:
        return [p for p in full if p.split == split and p.source == source][:count]

    plan = (pick("train", "P_det", 2) + pick("train", "P_random", 3) + pick("train", "S_eps", 3)
            + pick("train", "S_switch", 2) + pick("validation", "P_det", 2) + pick("validation", "P_random", 2)
            + pick("validation", "S_eps", 1) + pick("heldout", "H_det", 4) + pick("heldout", "H_random", 2))
    root = tmp_path_factory.mktemp("for40")
    dataset = root / "ds.jsonl"
    manifest = for40.collect_for40_dataset(dataset, plan=plan)
    yield {"root": root, "dataset": dataset, "manifest": manifest, "relabeled": relabeled, "plan": plan}
    patch.undo()
    for40.map_by_id.cache_clear()


def test_collection_manifest_and_loader_round_trip(mini) -> None:
    manifest, examples = for40.load_for40_dataset(mini["dataset"], expect_full_protocol=False)
    assert manifest["schema"] == "formicmind.behavior-cloning/v3"
    assert manifest["protocol"]["commit"] == for40.FOR40_PROTOCOL_COMMIT
    assert manifest["leakage"]["clean"] and len(examples) == manifest["example_count"]
    assert {e.split for e in examples} == {"train", "validation", "heldout"}
    assert manifest["disturbed_cap"]["share_after"]["train"] <= for40.DISTURBED_CAP
    assert all(len(e.observation) == 223 for e in examples)
    # funnel: matches -> trajectories -> decisions
    train = manifest["funnel"]["train"]
    assert train["planned_matches"] >= train["source_trajectories"] >= train["retained_trajectories"] > 0
    assert train["source_decisions"] >= train["retained_decisions"] == train["unique_decisions"]
    gate = for40.evaluate_gate(manifest, examples, full_protocol=False)
    assert gate["decision"] == "DATA NO-GO" and "D4" in gate["failed"]  # tiny plan cannot meet the volume gate
    assert gate["criteria"]["D3"]["pass"] and gate["criteria"]["D3"]["clean"]


def _rewrite(mini, tmp_path: Path, mutate) -> Path:
    target = tmp_path / "ds.jsonl"
    shutil.copy(mini["dataset"], target)
    shutil.copy(mini["dataset"].with_name("ds.manifest.json"), tmp_path / "ds.manifest.json")
    mutate(target, tmp_path / "ds.manifest.json")
    return target


def test_loader_rejects_tampering(mini, tmp_path: Path) -> None:
    import hashlib

    def refresh(jsonl: Path, manifest_path: Path, lines: list[str]) -> None:
        data = ("\n".join(lines) + "\n").encode()
        jsonl.write_bytes(data)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["dataset_sha256"] = hashlib.sha256(data).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def hash_only(jsonl: Path, manifest_path: Path) -> None:
        jsonl.write_bytes(jsonl.read_bytes() + b"\n")

    (tmp_path / "a").mkdir()
    with pytest.raises(DatasetError, match="SHA-256"):
        for40.load_for40_dataset(_rewrite(mini, tmp_path / "a", hash_only), expect_full_protocol=False)

    def flip_label(jsonl: Path, manifest_path: Path) -> None:
        lines = jsonl.read_text(encoding="utf-8").splitlines()
        row = json.loads(lines[0])
        row["teacher_action"]["lay_order"] = (row["teacher_action"]["lay_order"] + 1) % 3
        lines[0] = json.dumps(row, sort_keys=True, separators=(",", ":"))
        refresh(jsonl, manifest_path, lines)

    (tmp_path / "b").mkdir()
    with pytest.raises(DatasetError):
        for40.load_for40_dataset(_rewrite(mini, tmp_path / "b", flip_label), expect_full_protocol=False)

    def move_split(jsonl: Path, manifest_path: Path) -> None:
        lines = jsonl.read_text(encoding="utf-8").splitlines()
        row = json.loads(lines[0])
        row["split"] = "heldout" if row["split"] != "heldout" else "train"
        lines[0] = json.dumps(row, sort_keys=True, separators=(",", ":"))
        refresh(jsonl, manifest_path, lines)

    (tmp_path / "c").mkdir()
    with pytest.raises(DatasetError, match="split"):
        for40.load_for40_dataset(_rewrite(mini, tmp_path / "c", move_split), expect_full_protocol=False)

    def wrong_maps(jsonl: Path, manifest_path: Path) -> None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["map_plan"]["maps"][0]["split"] = "heldout"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    (tmp_path / "d").mkdir()
    with pytest.raises(DatasetError, match="map"):
        for40.load_for40_dataset(_rewrite(mini, tmp_path / "d", wrong_maps), expect_full_protocol=False)


def test_gzip_dataset_round_trip_is_byte_stable(mini, tmp_path: Path) -> None:
    import gzip
    import hashlib

    target = tmp_path / "ds.jsonl.gz"
    manifest = for40.collect_for40_dataset(target, plan=mini["plan"])
    assert manifest["dataset_sha256"] == mini["manifest"]["dataset_sha256"]  # deterministic re-collection
    assert hashlib.sha256(gzip.decompress(target.read_bytes())).hexdigest() == manifest["dataset_sha256"]
    loaded, examples = for40.load_for40_dataset(target, expect_full_protocol=False)
    assert loaded["dataset_file"] == "ds.jsonl.gz" and len(examples) == manifest["example_count"]
    assert (tmp_path / "ds.manifest.json").exists()


def test_full_protocol_check_rejects_a_partial_plan(mini) -> None:
    with pytest.raises(DatasetError, match="plan"):
        for40.load_for40_dataset(mini["dataset"], expect_full_protocol=True)


# ------------------------------------------------------------------------------------------------ losses

def _batch(examples, device="cpu"):
    return build_tensors(examples, torch.device(device))


def test_loss_components_and_conditions(mini) -> None:
    _, examples = for40.load_for40_dataset(mini["dataset"], expect_full_protocol=False)
    train = [e for e in examples if e.split == "train"][:64]
    batch = _batch(train)
    output = torch.randn(len(train), 17)
    parts = loss_components(output, batch)
    assert torch.isclose(parts["mse_all"], ((output - batch["y"]) ** 2).mean())
    # 11 discrete + 6 forage logits make up the 17 outputs
    expected_all = (parts["mse_disc"] * 11 + parts["mse_forage"] * 6) / 17
    assert torch.isclose(parts["mse_all"], expected_all, atol=1e-5)
    assert torch.isclose(for40_train.total_loss("A", parts), parts["mse_all"])
    discrete = parts["ce_lay"] + parts["ce_army"] + 0.5 * parts["bce_avoid"]
    forage = parts["ce_share"] + 0.5 * parts["bce_support"]
    assert torch.isclose(total_loss("B", parts), discrete + forage)
    assert torch.isclose(total_loss("B1", parts), discrete + parts["mse_forage"])
    assert torch.isclose(total_loss("B2", parts), parts["mse_disc"] + forage)
    # perfect classifier logits drive the cross-entropies to ~0 and the surrogate to the entropy of the shares
    perfect = torch.zeros(len(train), 17)
    perfect[torch.arange(len(train)), batch["lay"]] = 50.0
    perfect[torch.arange(len(train)), 9 + batch["army"]] = 50.0
    low = loss_components(perfect, batch)
    assert low["ce_lay"] < 1e-3 and low["ce_army"] < 1e-3
    assert torch.all(torch.isfinite(torch.stack(list(parts.values()))))


def test_cross_entropy_ignores_masked_options_and_avoid_is_conditional() -> None:
    from formicmind.actions import ActionV2

    class _Mask:
        lay = (True, False, False)
        army_choice = (True, False, False, False, False, False, False)
        army_avoid_center = ((True, False),) + ((False, False),) * 6

    class _Example:
        observation = tuple([0.0] * 223)
        teacher_logits = tuple([0.0] * 17)
        mask = _Mask()
        teacher_action = ActionV2(0, (65535, 0, 0, 0, 0, 0), 0, False)

    batch = build_tensors([_Example()], torch.device("cpu"))
    output = torch.zeros(1, 17)
    output[0, 1] = 1000.0  # masked-out lay option must not matter
    output[0, 16] = 1000.0  # avoid is not a conditional decision here
    parts = loss_components(output, batch)
    assert parts["ce_lay"] < 1e-4 and parts["bce_avoid"] == 0.0
    assert batch["support"].tolist() == [[1.0, 0, 0, 0, 0, 0]]


# ------------------------------------------------------------------------------------------------ metrics

def test_metrics_oracle_constant_and_majority(mini) -> None:
    _, examples = for40.load_for40_dataset(mini["dataset"], expect_full_protocol=False)
    train = [e for e in examples if e.split == "train"]
    held = [e for e in examples if e.split == "heldout"]
    oracle = summarize(records_from_logits(held, [list(e.teacher_logits) for e in held]))
    assert oracle["legal_rate"] == 1.0 and oracle["forage"]["exact_rate"] == 1.0
    assert oracle["lay"]["macro_recall"] == 1.0 and oracle["composite"] == pytest.approx(1.0)
    assert oracle["mse"]["all"] == 0.0 and not oracle["collapse"]
    reference = train_reference(train)
    constant = summarize(records_from_logits(held, [reference["mean_logits"]] * len(held)))
    assert constant["distinct_decoded"]["lay"] == 1 and constant["collapse"]
    assert constant["lay"]["macro_recall"] <= 0.5 + 1e-9  # a constant cannot recall more than one class
    majority = majority_summary(held, reference)
    assert majority["lay"]["accuracy"] == pytest.approx(
        sum(e.teacher_action.lay_order == reference["lay_majority"] for e in held) / len(held))


# ------------------------------------------------------------------------------------------------ training + evaluation

@pytest.fixture(scope="module")
def trained(mini):
    out = mini["root"] / "out"
    selection = for40_train.train_all_for40(
        mini["dataset"], out, device_name="cpu", cpu_reproduction=False, seeds=(for40.TRAIN_SEEDS[0],),
        hyper={"max_epochs": 6, "min_epochs": 2, "patience_epochs": 2}, expect_full_protocol=False)
    return {"out": out, "selection": selection}


def test_training_budget_checkpoint_and_selection(mini, trained) -> None:
    selection = trained["selection"]
    assert selection["frozen_before_heldout"] and selection["dataset_sha256"] == mini["manifest"]["dataset_sha256"]
    assert {r["condition"] for r in selection["runs"]} == {"A", "B", "B1", "B2"}
    assert set(selection["selected"]) == {"A", "B", "B1", "B2"} and selection["candidate_condition"] in ("A", "B")
    for run in selection["runs"]:
        assert 2 <= run["epochs_run"] <= 6 and run["optimizer_steps"] >= run["epochs_run"]
        model, manifest, _ = load_checkpoint(trained["out"] / "checkpoints" / run["checkpoint"], "cpu")
        assert manifest["weights"]["sha256"] == run["weights_sha256"]
        assert manifest["training"]["best_epoch"] == run["best_epoch"]
        assert len(manifest["training"]["curves"]) == run["epochs_run"]
        assert manifest["loss"]["condition"] == run["condition"]
        assert model is not None


def test_evaluation_pipeline_and_selection_binding(mini, trained, tmp_path: Path) -> None:
    out = trained["out"]
    report = for40_eval.evaluate_for40(
        mini["dataset"], out / "selection.json", out, device_name="cpu", match_max_ticks=90, expect_full_protocol=False)
    assert report["q1_gate"]["decision"] in ("NO-GO", "GO to separate design of the next stage")
    assert {"G0_legal_rate_1", "G1_no_collapse", "G10_stability_2_of_3_seeds"} <= set(report["q1_gate"]["checks"])
    assert set(report["controls"]) >= {"constant_mean_logits", "majority_actions", "untrained_mlp", "teacher_oracle"}
    assert report["controls"]["teacher_oracle"]["composite"] == pytest.approx(1.0)
    matches = report["real_matches_records"]
    assert {m["policy"] for m in matches} == {"candidate", "other", "constant_mean_logits", "teacher"}
    assert {m["side"] for m in matches} == {0, 1} and all(m["decisions_logged"] > 0 for m in matches)
    assert report["real_matches"]["match_count"] == len(matches)
    assert (out / "evaluation.json").exists()
    # a selection bound to a different dataset must be refused
    tampered = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    tampered["dataset_sha256"] = "0" * 64
    bad = tmp_path / "selection.json"
    bad.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(DatasetError, match="selection"):
        for40_eval.evaluate_for40(mini["dataset"], bad, out, device_name="cpu", real_matches=False,
                                  expect_full_protocol=False)


def test_q1_criteria_flag_a_collapsed_model(mini) -> None:
    _, examples = for40.load_for40_dataset(mini["dataset"], expect_full_protocol=False)
    train = [e for e in examples if e.split == "train"]
    held = [e for e in examples if e.split == "heldout"]
    reference = train_reference(train)
    records = records_from_logits(held, [reference["mean_logits"]] * len(held))
    summary = summarize(records)
    majority = majority_summary(held, reference)
    by_map = {m: summarize([r for r in records if r["map_id"] == m]) for m in {e.map_id for e in held}}
    verdict = for40_eval.evaluate_q1_criteria(summary, summary, majority, by_map, by_map, summary, summary, 0.5)
    assert verdict["checks"]["G1_no_collapse"] is False  # a constant predictor is a collapse whatever its MSE
    assert verdict["checks"]["G5_forage"] is False and verdict["checks"]["G6_logit_mse_below_constant"] is False
