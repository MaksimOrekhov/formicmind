"""FOR-44: the reproducible CLI - ``python -m formicmind.eco map|verify|bench|sweep``."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from formicmind.canonical import canonical_dumps
from formicmind.eco.cli import main

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "docs" / "ecosystem_examples"
MAP_ARGS = ["map", "--seed", "42", "--colonies", "8", "--scenario", "moderate", "--report", "--state"]


def run(capsys, *argv):
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_map_output_is_stable_json(capsys):
    code, out1, _ = run(capsys, *MAP_ARGS)
    _, out2, _ = run(capsys, *MAP_ARGS)
    assert code == 0 and out1 == out2 and out1.endswith("}\n")
    doc = json.loads(out1)
    assert doc["schema"] == "formicmind.eco.map-cli/1" and doc["map_seed"] == "42"
    assert doc["config"]["n_colonies"] == 8 and doc["config"]["scenario"] == "moderate"
    assert len(doc["map"]["nodes"]) == 96 and sum(1 for n in doc["map"]["nodes"] if n["patch"]) == 32
    assert len(doc["map"]["nests"]) == 8 and len(set(doc["map"]["nests"])) == 8
    assert doc["report"]["all_checks_passed"] is True and doc["initial_state"]["tick"] == 0
    claimed = doc.pop("document_sha256")
    assert claimed == hashlib.sha256(canonical_dumps(doc)).hexdigest()
    assert list(json.loads(out1)) == sorted(json.loads(out1)), "top-level keys are sorted"


def test_compact_and_pretty_outputs_carry_the_same_document(capsys):
    _, pretty, _ = run(capsys, *MAP_ARGS)
    _, compact, _ = run(capsys, *MAP_ARGS, "--compact")
    assert json.loads(pretty) == json.loads(compact) and len(compact) < len(pretty)
    assert " " not in compact.strip()[:200]


def test_the_json_contains_no_policy_observation_metadata(capsys):
    _, out, _ = run(capsys, *MAP_ARGS)
    doc = json.loads(out)
    keys = set()

    def walk(value):
        if isinstance(value, dict):
            keys.update(value)
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    walk(doc)
    assert not {"observation", "mask", "action", "logits", "policy", "critic", "reward", "target", "owner", "ownership"} & keys
    assert doc["versions"]["observation_version"] == "ObservationE1/v1"  # a contract name only; no observation is built here


@pytest.mark.parametrize("colonies", [6, 7, 8])
@pytest.mark.parametrize("scenario", ["abundance", "moderate", "severe"])
def test_every_size_and_scenario_generates_and_verifies(capsys, tmp_path, colonies, scenario):
    out = tmp_path / "map.json"
    code, _, err = run(capsys, "map", "--seed", "7", "--colonies", str(colonies), "--scenario", scenario, "--report", "--out", str(out),
                       "--summary")
    assert code == 0 and "ALL PASSED" in err and f"colonies={colonies}" in err
    assert out.read_bytes().count(b"\r") == 0, "LF newlines on every platform"
    code, text, _ = run(capsys, "verify", str(out))
    assert code == 0 and text.startswith("OK  map_hash=")


def test_verify_detects_tampering_with_the_map_the_hashes_or_the_report(capsys, tmp_path):
    _, out, _ = run(capsys, *MAP_ARGS)
    doc = json.loads(out)
    cases = {}
    d = json.loads(out); d["map"]["edges"][0]["length"] = 2 if d["map"]["edges"][0]["length"] != 2 else 3; cases["edge length"] = d
    d = json.loads(out); d["map"]["nodes"][5]["x"] += 1; cases["coordinate"] = d
    d = json.loads(out); d["map_seed"] = "43"; cases["seed"] = d
    d = json.loads(out); d["report"]["counts"]["edges"] += 1; cases["report"] = d
    d = json.loads(out); d["config"]["scenario"] = "severe"; cases["scenario"] = d
    d = json.loads(out); d["map_hash"] = "0" * 64; cases["map hash"] = d
    d = json.loads(out); d["config_hash"] = "0" * 64; cases["config hash"] = d
    d = json.loads(out); d["schema"] = "other"; cases["schema"] = d
    assert doc["document_sha256"]
    for name, bad in cases.items():
        path = tmp_path / f"{name.replace(' ', '_')}.json"
        path.write_text(json.dumps(bad), encoding="utf-8")
        code, _, err = run(capsys, "verify", str(path))
        assert code == 3 and err.startswith("error:"), name
    # a checksum-valid but non-generated map is also rejected: swap two nests of the same map and refresh every hash
    from dataclasses import replace

    from formicmind.canonical import canonical_dumps as cd
    from formicmind.eco.codec import to_canonical
    from formicmind.eco.config import config_from_canonical
    from formicmind.eco.mapgen import generate_map
    from formicmind.eco.serialization import map_from_canonical, seal_map

    config = config_from_canonical(doc["config"])
    forged = seal_map(replace(map_from_canonical(doc["map"], config), nests=tuple(reversed(generate_map(config, 42).nests))))
    d = json.loads(out)
    d["map"], d["map_hash"] = to_canonical(forged), forged.map_hash
    d.pop("report"), d.pop("report_hash"), d.pop("document_sha256")
    d["document_sha256"] = hashlib.sha256(cd(d)).hexdigest()
    path = tmp_path / "forged.json"
    path.write_text(json.dumps(d), encoding="utf-8")
    code, _, err = run(capsys, "verify", str(path))
    assert code == 3 and "error:" in err


def test_verify_rejects_garbage_files(capsys, tmp_path):
    for name, content in (("a.json", "not json"), ("b.json", "[]"), ("c.json", "{}"), ("d.json", '{"schema": "formicmind.eco.map-cli/1"}')):
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        code, _, err = run(capsys, "verify", str(path))
        assert code == 3 and "error" in err, name
    code, _, err = run(capsys, "verify", str(tmp_path / "missing.json"))
    assert code == 3


def test_invalid_arguments_are_rejected(capsys):
    for argv in (["map", "--colonies", "5"], ["map", "--colonies", "9"], ["map", "--scenario", "mild"], ["map", "--seed", "x"], ["nope"], []):
        with pytest.raises(SystemExit) as exc:
            main(argv)
        assert exc.value.code == 2
    assert run(capsys, "map", "--seed", "-3")[0] == 3  # a seed outside u64 is a clean error, not a traceback
    assert run(capsys, "map", "--nodes", "60")[0] == 3


def test_document_hash_and_map_hash_do_not_depend_on_the_output_format(capsys):
    _, a, _ = run(capsys, "map", "--seed", "5")
    _, b, _ = run(capsys, "map", "--seed", "5", "--compact")
    assert json.loads(a)["map_hash"] == json.loads(b)["map_hash"] and json.loads(a)["document_sha256"] == json.loads(b)["document_sha256"]
    _, c, _ = run(capsys, "map", "--seed", "6")
    assert json.loads(c)["map_hash"] != json.loads(a)["map_hash"]


def test_bench_reports_generation_and_report_latency_only(capsys):
    code, out, _ = run(capsys, "bench", "--maps", "5", "--colonies", "6", "--scenario", "severe")
    doc = json.loads(out)
    assert code == 0 and doc["maps"] == 5 and doc["unit"] == "microseconds"
    assert "not worlds per second" in doc["benchmark"]
    for key in ("generation_incl_acceptance_checks_us", "fairness_report_only_us"):
        assert set(doc[key]) == {"mean", "median", "p95", "max", "min"} and 0 < doc[key]["min"] <= doc[key]["median"] <= doc[key]["max"]
    assert doc["attempts"]["max"] >= 1


def test_sweep_summarises_fairness_per_size_and_scenario(capsys):
    code, out, _ = run(capsys, "sweep", "--maps", "4", "--sizes", "6", "8", "--scenarios", "moderate")
    doc = json.loads(out)
    assert code == 0 and [(r["colonies"], r["scenario"], r["failed"]) for r in doc["sweep"]] == [(6, "moderate", 0), (8, "moderate", 0)]
    assert all(r["near_access_cv_permille"]["max"] <= 100 and r["shared_supply_cv_permille"]["max"] <= 200 for r in doc["sweep"])


def test_module_entry_point_runs_as_a_script():
    result = subprocess.run([sys.executable, "-m", "formicmind.eco", "map", "--seed", "1", "--compact"], capture_output=True, text=True,
                            cwd=ROOT, check=True)
    assert json.loads(result.stdout)["map_seed"] == "1"
    helped = subprocess.run([sys.executable, "-m", "formicmind.eco", "--help"], capture_output=True, text=True, cwd=ROOT, check=True)
    assert "map" in helped.stdout and "verify" in helped.stdout


def test_the_committed_example_is_what_the_cli_produces_today(capsys):
    """docs/ecosystem_examples/map_seed42_moderate_8.json is a real, reproducible output (regenerate it with the command in the doc)."""
    path = EXAMPLES / "map_seed42_moderate_8.json"
    assert path.exists(), "commit the example next to the doc"
    _, out, _ = run(capsys, *MAP_ARGS)
    assert path.read_bytes().replace(b"\r\n", b"\n") == out.encode("ascii")
    code, text, _ = run(capsys, "verify", str(path))
    assert code == 0 and "seed=42" in text
