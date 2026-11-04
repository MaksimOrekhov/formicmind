from __future__ import annotations

import json

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
