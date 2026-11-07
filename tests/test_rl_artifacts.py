"""FOR-42: the committed pilot artefacts are intact and consistent (integrity only; nothing is regenerated)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from formicmind.rl.checkpoint import load_checkpoint  # noqa: E402
from formicmind.rl.pool import SnapshotPool  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "results" / "rl_for42"
RUNS = ("p1_seed42", "p1b_seed43", "p1c_seed44", "p2_shaping_seed42")


@pytest.mark.parametrize("run", RUNS)
def test_final_checkpoint_is_intact_and_contract_matched(run):
    latest = json.loads((ROOT / run / "latest.json").read_text(encoding="utf-8"))
    model, manifest, _ = load_checkpoint(ROOT / run / latest["path"]) if (ROOT / run / latest["path"] / "train_state.pt").exists() \
        else (None, None, None)
    if model is None:  # replicate runs keep weights only; verify the weights hash from the manifest directly
        from formicmind.rl.pool import file_sha256
        manifest = json.loads((ROOT / run / latest["path"] / "manifest.json").read_text(encoding="utf-8"))
        assert file_sha256(ROOT / run / latest["path"] / "weights.pt") == manifest["weights"]["sha256"]
    assert manifest["iteration"] == 80 and manifest["counters"]["matches"] == 10240
    config = json.loads((ROOT / run / "config.json").read_text(encoding="utf-8"))
    assert config["baseline_sparring"] == 0.0  # no scripted opponent in training: nothing to imitate


def test_main_pilot_snapshots_are_immutable_files_with_matching_hashes():
    pool = SnapshotPool(ROOT / "p1_seed42" / "snapshots")
    infos = [pool.load_info(i) for i in range(17)]  # raises if any SHA-256 differs
    assert [i.iteration for i in infos] == [5 * k for k in range(17)]
    state = pool.load_state(infos[0])
    assert all(torch.isfinite(t).all() for t in state.values())


@pytest.mark.parametrize("run", RUNS)
def test_final_evaluation_and_gates_are_consistent(run):
    report = json.loads((ROOT / run / "eval" / "final_full.json").read_text(encoding="utf-8"))
    assert report["matches"] == 480 and report["truncated_matches"] == 0
    for opponent, modes in report["results"].items():
        for mode, splits in modes.items():
            entry = splits["all"]
            assert entry["wins"] + entry["draws"] + entry["losses"] + entry["truncated"] == entry["n"], (opponent, mode)
    gates = json.loads((ROOT / run / "analysis.json").read_text(encoding="utf-8"))["gates"]
    assert gates["technical_pass"] and gates["run_go"] and gates["matches"] == 10240
    assert gates["T2_max_identity_ratio_dev"] < 1e-3
