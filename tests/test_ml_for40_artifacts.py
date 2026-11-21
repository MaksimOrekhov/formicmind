"""FOR-40 committed artifacts: integrity, leakage, decoder agreement and the recorded data-gate result.

Kept in its own module because ``test_ml_for40`` temporarily relabels maps for its tiny pipeline fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from formicmind.ml import for40


_DOCS = Path(__file__).resolve().parents[1] / "docs"
_DATASET = _DOCS / "ml_for40" / "dataset.jsonl.gz"


@pytest.mark.skipif(not _DATASET.exists(), reason="FOR-40 dataset artifact is not present")
def test_committed_for40_artifacts_match_the_recorded_data_gate() -> None:
    """Integrity + leakage + decoder checks of the full committed dataset; the gate result is reproduced exactly."""
    import gzip
    import hashlib

    manifest, examples = for40.load_for40_dataset(_DATASET)  # hashes, masks, decoder labels, split isolation
    assert hashlib.sha256(gzip.decompress(_DATASET.read_bytes())).hexdigest() == manifest["dataset_sha256"]
    assert manifest["leakage"]["clean"] and len(examples) == manifest["example_count"] == 10400
    committed = json.loads((_DOCS / "ml_for40" / "gate.json").read_text(encoding="utf-8"))
    recomputed = for40.evaluate_gate(manifest, examples)
    assert committed == json.loads(json.dumps(recomputed))
    assert committed["decision"] == "DATA NO-GO" and committed["failed"] == ["D7"]
    protocol = (_DOCS / "ml_for40_protocol.md").read_text(encoding="utf-8")
    assert for40.MAP_PLAN_SHA256 in protocol and for40.map_plan_sha256() == for40.MAP_PLAN_SHA256
    # a DATA NO-GO means no model was trained or evaluated on held-out data
    assert not (_DOCS / "ml_for40" / "checkpoints").exists() and not (_DOCS / "ml_for40" / "selection.json").exists()


# ------------------------------------------------------------------------------------------------ DATA NO-GO lock

def _forbid(monkeypatch: pytest.MonkeyPatch, module, *names: str) -> None:
    def boom(*args, **kwargs):
        raise AssertionError("must not be reached after DATA NO-GO")

    for name in names:
        monkeypatch.setattr(module, name, boom)


@pytest.mark.skipif(not _DATASET.exists(), reason="FOR-40 dataset artifact is not present")
def test_train_for40_refuses_the_committed_data_no_go_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                              capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("torch")
    from formicmind.ml import cli, for40_train
    from formicmind.ml.dataset import DatasetError

    _forbid(monkeypatch, for40_train, "resolve_device", "train_run", "save_checkpoint")
    out = tmp_path / "out"
    with pytest.raises(DatasetError, match=r"DATA NO-GO.*D7"):
        for40_train.train_all_for40(_DATASET, out, device_name="cuda")
    assert not out.exists()
    assert cli.main(["train-for40", "--dataset", str(_DATASET), "--output-dir", str(out)]) == 2
    assert "DATA NO-GO" in capsys.readouterr().err and not out.exists()


@pytest.mark.skipif(not _DATASET.exists(), reason="FOR-40 dataset artifact is not present")
def test_eval_for40_refuses_the_committed_data_no_go_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                             capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("torch")
    from formicmind.ml import cli, for40_eval
    from formicmind.ml.dataset import DatasetError

    _forbid(monkeypatch, for40_eval, "resolve_device", "load_checkpoint", "run_real_matches")
    out = tmp_path / "out"
    selection = tmp_path / "selection.json"  # deliberately absent: the gate must fire before it is read
    with pytest.raises(DatasetError, match=r"DATA NO-GO.*D7"):
        for40_eval.evaluate_for40(_DATASET, selection, out, device_name="cuda")
    assert not out.exists()
    assert cli.main(["eval-for40", "--dataset", str(_DATASET), "--selection", str(selection),
                     "--output-dir", str(out)]) == 2
    assert "DATA NO-GO" in capsys.readouterr().err and not out.exists()


def test_require_data_go_passes_only_a_data_go_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    from formicmind.ml.dataset import DatasetError

    monkeypatch.setattr(for40, "evaluate_gate", lambda m, e: {"decision": "DATA GO", "failed": []})
    assert for40.require_data_go({}, [])["decision"] == "DATA GO"
    monkeypatch.setattr(for40, "evaluate_gate", lambda m, e: {"decision": "DATA NO-GO", "failed": ["D4", "D7"]})
    with pytest.raises(DatasetError, match="D4, D7"):
        for40.require_data_go({}, [])
