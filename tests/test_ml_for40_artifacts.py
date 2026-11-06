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
