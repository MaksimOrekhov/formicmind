from __future__ import annotations

import hashlib
import json
import shutil
from types import SimpleNamespace

import pytest

from formicmind.invariants import validate_config
from formicmind.ml.dataset import DatasetError
from formicmind.routing import shortest_path
from formicmind.ml.quality_dataset import (
    QUALITY_CONTEXTS,
    _deduplicate_rows,
    _decision_payload,
    _sha256,
    collect_quality_dataset,
    load_quality_dataset,
    quality_config,
)


def test_quality_maps_are_mirror_valid_and_change_only_topology():
    maps = {map_id: quality_config(map_id) for map_id in ("classic", "cross", "dense")}
    for config in maps.values():
        validate_config(config)
        assert config.rules_version == maps["classic"].rules_version
        assert config.balance_version == maps["classic"].balance_version
        assert config.balance == maps["classic"].balance
        assert config.t_max == maps["classic"].t_max
    assert {map_id: len(config.topology.edges) for map_id, config in maps.items()} == {
        "classic": 9, "cross": 10, "dense": 11,
    }
    for config in maps.values():
        for source in range(7):
            for destination in range(7):
                shortest_path(config.topology, source, destination, side=0)
                if destination != 3:
                    shortest_path(config.topology, source, destination, side=0, avoid_center=True)


def test_quality_collection_uses_three_splits_and_deduplicates_globally(tmp_path):
    dataset = tmp_path / "quality.jsonl"
    manifest = collect_quality_dataset(
        dataset,
        train_seeds=[1101, 1102, 1103],
        validation_seeds=[1201, 1202],
        heldout_seeds=[1301, 1302],
        max_ticks=120,
    )
    loaded_manifest, examples = load_quality_dataset(dataset)

    assert loaded_manifest == manifest
    assert manifest["schema"] == "formicmind.behavior-cloning/v2"
    assert manifest["match_count"] == 2 * (3 * 4 + 2 * 4 + 2 * 5)
    assert set(example.split for example in examples) == {"train", "validation", "heldout"}
    assert {match["teacher_side"] for match in manifest["matches"]} == {0, 1}
    assert {example.map_id for example in examples} == {"classic", "cross", "dense"}
    assert {example.opponent_id for example in examples} == {"turtle:v2", "random:v2", "rush:v2"}
    assert all(len(example.observation) == 223 for example in examples)
    assert all(len(example.teacher_logits) == 17 for example in examples)
    assert set(QUALITY_CONTEXTS) == {"train", "validation", "heldout"}
    assert manifest["deduplication"]["split_example_counts"] == {
        split: sum(example.split == split for example in examples)
        for split in ("train", "validation", "heldout")
    }
    for left, right in (("train", "validation"), ("train", "heldout"), ("validation", "heldout")):
        left_rows = [example for example in examples if example.split == left]
        right_rows = [example for example in examples if example.split == right]
        assert {example.seed for example in left_rows}.isdisjoint(example.seed for example in right_rows)
        assert {example.trajectory_sha256 for example in left_rows}.isdisjoint(
            example.trajectory_sha256 for example in right_rows
        )
        assert {example.decision_sha256 for example in left_rows}.isdisjoint(
            example.decision_sha256 for example in right_rows
        )
    first = json.loads(dataset.read_text(encoding="utf-8").splitlines()[0])
    assert "trajectory_group_id" in first and "decision_sha256" in first
    assert "observation" in first and "state" not in first


def test_quality_collection_rejects_overlap_before_running_matches(tmp_path):
    with pytest.raises(DatasetError, match="seed sets overlap"):
        collect_quality_dataset(
            tmp_path / "bad.jsonl",
            train_seeds=[1], validation_seeds=[1], heldout_seeds=[3], max_ticks=1,
        )


def _row(match_id: str, split: str, tick: int, marker: int) -> dict:
    row = {
        "match_id": match_id,
        "split": split,
        "tick": tick,
        "seed": 10 if split == "train" else 20,
        "teacher_side": 0,
        "context_id": "classic__turtle:v2",
        "observation": [float(marker)],
        "mask": [True],
        "teacher_logits": [float(marker)],
        "teacher_action": {"lay_order": marker},
    }
    row["decision_sha256"] = _sha256(_decision_payload(row))
    return row


def _match(match_id: str, split: str, seed: int) -> dict:
    return {
        "match_id": match_id,
        "split": split,
        "seed": seed,
        "teacher_side": 0,
        "context_id": "classic__turtle:v2",
    }


def test_deduplication_keeps_decisions_in_one_split_and_drops_full_trajectory_aliases():
    train_a = [_row("train-a", "train", 0, 1), _row("train-a", "train", 1, 2)]
    validation_overlap = [_row("validation", "validation", 0, 1), _row("validation", "validation", 1, 3)]
    train_duplicate = [_row("train-b", "train", 0, 1), _row("train-b", "train", 1, 2)]
    matches = [_match("train-a", "train", 1), _match("validation", "validation", 2),
               _match("train-b", "train", 3)]
    rows, summary = _deduplicate_rows(
        {"train-a": train_a, "validation": validation_overlap, "train-b": train_duplicate}, matches,
    )
    assert summary["duplicate_trajectories_removed"] == 1
    assert summary["duplicate_decisions_removed"] == 1
    assert not next(match for match in matches if match["match_id"] == "train-b")["trajectory_retained"]
    train_decisions = {row["decision_sha256"] for row in rows if row["split"] == "train"}
    validation_decisions = {row["decision_sha256"] for row in rows if row["split"] == "validation"}
    assert train_decisions.isdisjoint(validation_decisions)
    assert len(validation_decisions) == 1


def test_quality_cpu_training_saves_v2_provenance_and_weights_only_checkpoint(tmp_path):
    pytest.importorskip("torch", reason="quality CPU training smoke runs in Windows CI")
    from formicmind.ml.checkpoint import load_checkpoint
    from formicmind.ml.quality import train_quality_model

    dataset = tmp_path / "quality.jsonl"
    collect_quality_dataset(
        dataset,
        train_seeds=[2101, 2102],
        validation_seeds=[2201, 2202],
        heldout_seeds=[2301, 2302],
        max_ticks=120,
    )
    checkpoint_dir = tmp_path / "checkpoint"
    result = train_quality_model(
        dataset, checkpoint_dir, seed=12345, epochs=1, batch_size=32,
        requested_device="cpu",
    )
    model, manifest, device = load_checkpoint(checkpoint_dir, "cpu")
    assert str(device) == "cpu"
    assert model is not None
    assert result["data"]["schema"] == "formicmind.behavior-cloning/v2"
    assert manifest["data"]["seed_sets"]["heldout"] == [2301, 2302]
    assert manifest["weights"]["serialization"] == "torch.state_dict/weights_only"


@pytest.fixture(scope="module")
def quality_eval_artifacts(tmp_path_factory):
    pytest.importorskip("torch", reason="quality eval smoke runs in Windows CI")
    from formicmind.ml.quality import train_quality_model

    root = tmp_path_factory.mktemp("quality-eval")
    dataset = root / "quality.jsonl"
    collect_quality_dataset(
        dataset,
        train_seeds=[4101, 4102],
        validation_seeds=[4201, 4202],
        heldout_seeds=[4301, 4302],
        max_ticks=120,
    )
    checkpoint = root / "checkpoint"
    train_quality_model(
        dataset, checkpoint, seed=410037, epochs=1, batch_size=32, requested_device="cpu",
    )
    return {"root": root, "dataset": dataset, "checkpoint": checkpoint}


def test_quality_loader_rejects_teacher_side_that_differs_from_match_metadata(
    quality_eval_artifacts, tmp_path,
):
    dataset = quality_eval_artifacts["dataset"]
    source_manifest = json.loads(dataset.with_name("quality.manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
    row = rows[0]
    source_match = next(match for match in source_manifest["matches"] if match["match_id"] == row["match_id"])
    assert row["teacher_side"] == source_match["teacher_side"]
    row["teacher_side"] = 1 - source_match["teacher_side"]

    tampered_dataset = tmp_path / "teacher-side-tampered.jsonl"
    raw = ("\n".join(json.dumps(item, sort_keys=True, separators=(",", ":")) for item in rows) + "\n").encode()
    tampered_dataset.write_bytes(raw)
    source_manifest["dataset_file"] = tampered_dataset.name
    source_manifest["dataset_sha256"] = hashlib.sha256(raw).hexdigest()
    tampered_dataset.with_name("teacher-side-tampered.manifest.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )

    with pytest.raises(DatasetError, match="example provenance differs from its retained match"):
        load_quality_dataset(tampered_dataset)


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("sha", "checkpoint dataset SHA-256 does not match quality dataset"),
        ("teacher", "checkpoint quality teacher provenance differs from dataset"),
        ("splits", "checkpoint quality split_by_seed provenance differs from dataset"),
    ],
)
def test_eval_quality_rejects_checkpoint_dataset_provenance_mismatch(
    quality_eval_artifacts, tmp_path, mismatch, message,
):
    from formicmind.ml.quality import evaluate_quality_model

    checkpoint = tmp_path / "checkpoint"
    shutil.copytree(quality_eval_artifacts["checkpoint"], checkpoint)
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = manifest["data"]
    if mismatch == "sha":
        data["sha256"] = "0" * 64
    elif mismatch == "teacher":
        data["teacher"]["policy_id"] = "tampered-teacher"
    else:
        data["split_by_seed"]["4101"] = "heldout"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(DatasetError, match=message):
        evaluate_quality_model(checkpoint, quality_eval_artifacts["dataset"], device="cpu")


def test_eval_quality_smoke_reports_provenance_metrics_and_terminal_status(
    quality_eval_artifacts, monkeypatch,
):
    from formicmind.ml import quality
    from formicmind.ml.quality_dataset import load_quality_dataset

    dataset_manifest, examples = load_quality_dataset(quality_eval_artifacts["dataset"])
    match_seed = 433700037
    monkeypatch.setattr(quality, "QUALITY_MATCH_SEEDS", (match_seed,))
    monkeypatch.setattr(quality, "QUALITY_MATCH_OPPONENTS", ("turtle:v2",))
    calls = []

    def fake_run_episode(config, seed, controller0, controller1, *, max_ticks):
        calls.append((seed, max_ticks))
        terminal = len(calls) == 1
        return SimpleNamespace(
            score=(1.0, 0.0) if terminal else None,
            ticks_executed=20 if terminal else max_ticks,
            final_tick=20 if terminal else max_ticks,
            truncated=not terminal,
            winner=0 if terminal else None,
            reason="queen_death" if terminal else "max_ticks",
            queen_hp=(240, 0) if terminal else (240, 240),
            food=(100, 0) if terminal else (100, 100),
            workers=(10, 0) if terminal else (10, 10),
            soldiers=(2, 0) if terminal else (2, 2),
            config_hash="config-sha",
            events_hash="events-sha",
            state_hash="state-sha",
        )

    monkeypatch.setattr(quality, "run_episode", fake_run_episode)
    report = quality.evaluate_quality_model(
        quality_eval_artifacts["checkpoint"], quality_eval_artifacts["dataset"], device="cpu",
    )

    assert calls == [(match_seed, 3000), (match_seed, 3000)]
    assert report["dataset"]["sha256"] == dataset_manifest["dataset_sha256"]
    assert report["model"]["weights_sha256"] == json.loads(
        (quality_eval_artifacts["checkpoint"] / "manifest.json").read_text(encoding="utf-8")
    )["weights"]["sha256"]
    assert report["match_seeds"] == [match_seed]
    assert set(report["imitation"]["trained"]) == {"train", "validation", "heldout"}
    for split in ("train", "validation", "heldout"):
        assert report["imitation"]["trained"][split]["examples"] == sum(
            example.split == split for example in examples
        )
    assert report["imitation"]["untrained_heldout"]["examples"] == sum(
        example.split == "heldout" for example in examples
    )
    oracle = report["imitation"]["teacher_oracle_heldout"]
    assert oracle["examples"] == report["imitation"]["untrained_heldout"]["examples"]
    assert oracle["teacher_logit_mse"] == 0.0
    assert oracle["legal_action_rate"] == oracle["exact_action_accuracy"] == 1.0
    matches = report["matches"]
    assert len(matches["completed"]) == 1
    assert matches["completed"][0]["truncated"] is False
    assert matches["completed"][0]["model_score"] == 1.0
    assert len(matches["truncated"]) == 1
    assert matches["truncated"][0]["truncated"] is True
    assert matches["truncated"][0]["model_score"] is None
    summary = matches["summary_by_opponent"]["turtle:v2"]
    assert summary["matches"] == 2
    assert summary["completed_matches"] == 1
    assert summary["truncated_matches"] == 1
    assert summary["model_wins"] == 1
    assert summary["mean_model_score_completed_only"] == 1.0


def test_eval_quality_rejects_match_seeds_overlapping_dataset(quality_eval_artifacts, monkeypatch):
    from formicmind.ml import quality

    monkeypatch.setattr(quality, "QUALITY_MATCH_SEEDS", (4101,))
    with pytest.raises(DatasetError, match="match-evaluation seeds overlap quality dataset seeds"):
        quality.evaluate_quality_model(
            quality_eval_artifacts["checkpoint"], quality_eval_artifacts["dataset"], device="cpu",
        )
