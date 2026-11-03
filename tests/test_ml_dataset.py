from __future__ import annotations

import json
import subprocess
import sys

import pytest

from formicmind.ml.dataset import DatasetError, collect_dataset, load_dataset
from formicmind.actions import get_action_mask
from formicmind.observation import ObservationV2


def test_collection_uses_real_v2_matches_and_splits_whole_seeds(tmp_path):
    dataset_path = tmp_path / "pilot.jsonl"
    manifest = collect_dataset(
        dataset_path,
        seeds=[711, 712],
        teacher="balanced:v2",
        opponent="turtle:v2",
        max_ticks=1,
        validation_fraction=0.5,
    )
    loaded_manifest, examples = load_dataset(dataset_path)

    assert loaded_manifest == manifest
    assert manifest["match_count"] == 4
    assert manifest["teacher"]["id"] == "balanced:v2"
    assert manifest["opponent"]["id"] == "turtle:v2"
    assert manifest["contracts"]["observation_length"] == 223
    assert set(manifest["split_by_seed"].values()) == {"train", "validation"}
    assert {example.teacher_side for example in examples} == {0, 1}
    assert {len(example.observation) for example in examples} == {223}
    assert {len(example.teacher_logits) for example in examples} == {17}
    assert {example.seed for example in examples if example.split == "train"}
    assert {example.seed for example in examples if example.split == "validation"}
    assert all(example.split == manifest["split_by_seed"][str(example.seed)] for example in examples)
    assert all(example.mask == get_action_mask(ObservationV2(example.observation)) for example in examples)
    row = json.loads(dataset_path.read_text(encoding="utf-8").splitlines()[0])
    assert set(row) == {
        "schema", "match_id", "seed", "teacher_side", "tick", "split",
        "observation", "mask", "teacher_logits", "teacher_action",
    }


def test_dataset_rejects_baseline_v1_for_collection(tmp_path):
    with pytest.raises(DatasetError, match="Baselines v2"):
        collect_dataset(tmp_path / "bad.jsonl", seeds=[1, 2], teacher="balanced", max_ticks=1)
    with pytest.raises(DatasetError, match="technical teacher"):
        collect_dataset(tmp_path / "random.jsonl", seeds=[1, 2], teacher="random:v2", max_ticks=1)


def test_dataset_rejects_modified_jsonl_before_training(tmp_path):
    path = tmp_path / "pilot.jsonl"
    collect_dataset(path, seeds=[81, 82], max_ticks=1, validation_fraction=0.5)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(DatasetError, match="SHA-256"):
        load_dataset(path)


def test_importing_core_and_ml_package_does_not_load_torch():
    script = "import sys; sys.path.insert(0, 'src'); import formicmind; import formicmind.ml; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", script], check=True)
