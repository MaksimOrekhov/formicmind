"""FOR-15: the minimal CLI (``python -m formicmind.replay --verify``, ``sim replay --verify``) and the smoke recorder."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from formicmind.replay import ReplayReader, verify
from formicmind.replay.cli import EXIT_FORMAT, EXIT_IO, EXIT_MISMATCH, EXIT_OK, EXIT_USAGE, main as replay_main
from formicmind.replay_smoke import ALL_SCENARIOS, main as smoke_main
from formicmind.cli import main as sim_main

from replay_helpers import find_event, fix_event_totals, forge

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def run_module(module, *args, cwd):
    """A fresh interpreter that only sees ``src`` - it has none of the test or policy code."""
    env = {**os.environ, "PYTHONPATH": str(SRC), "PYTHONUTF8": "0"}
    return subprocess.run([sys.executable, "-m", module, *args], capture_output=True, text=True, cwd=cwd, env=env,
                          check=False)


def test_verify_cli_succeeds_with_hashes_and_exit_code_zero(replays, tmp_path):
    proc = run_module("formicmind.replay", "--verify", str(replays["raid_win"].path), cwd=tmp_path)
    assert proc.returncode == EXIT_OK == 0 and proc.stderr == ""
    result = replays["raid_win"].result
    for needle in (result.events_hash, result.state_hash, result.checkpoint_hash, "winner=A", "reason=queen_death",
                   "terminal_tick=175", "terminal_phase=4", "cut mid-tick at QueenTerminal", "ticks=176"):
        assert needle in proc.stdout, needle
    assert proc.stdout.isascii() and len(proc.stdout.splitlines()) == 6  # a short, fixed-shape report


def test_cli_verifies_an_rng_policy_match_in_a_process_that_has_no_policy_code(replays, tmp_path):
    proc = run_module("formicmind.replay", "--verify", str(replays["rng_policy"].path), cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert replays["rng_policy"].result.checkpoint_hash in proc.stdout


def test_cli_reports_a_truncated_run_without_inventing_an_outcome(replays, tmp_path):
    proc = run_module("formicmind.replay", "--verify", str(replays["cutoff"].path), cwd=tmp_path)
    assert proc.returncode == 0 and "outcome=None (truncated harness run)" in proc.stdout and "winner" not in proc.stdout


def test_sim_entry_point_is_declared_and_works(replays, tmp_path):
    assert 'sim = "formicmind.cli:main"' in (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    ok = run_module("formicmind.cli", "replay", "--verify", str(replays["collapse"].path), cwd=tmp_path)
    assert ok.returncode == 0 and "economic_collapse" in ok.stdout and "tick completed" in ok.stdout
    for args in ([], ["run"], ["replay"], ["replay", "--verify"], ["replay", "--bogus", "x"]):
        proc = run_module("formicmind.cli", *args, cwd=tmp_path)
        assert proc.returncode == EXIT_USAGE == 64 and proc.stdout == "" and "usage" in proc.stderr.lower()


def test_exit_codes_distinguish_mismatch_corruption_and_missing_files(replays, tmp_path, capsys):
    good = replays["mutual_raid_draw"].data
    corrupt = tmp_path / "corrupt.jsonl"
    corrupt.write_bytes(good[:-100])
    mismatch = tmp_path / "mismatch.jsonl"

    def forged(r):
        find_event(r, "harvest", 3)["e"]["harvested"] += 1
        fix_event_totals(r)

    mismatch.write_bytes(forge(good, forged))
    assert replay_main(["--verify", str(corrupt)]) == EXIT_FORMAT == 2
    assert "FAIL format" in capsys.readouterr().err
    assert replay_main(["--verify", str(mismatch)]) == EXIT_MISMATCH == 1
    err = capsys.readouterr().err
    assert "FAIL verify" in err and "harvest" in err
    assert replay_main(["--verify", str(tmp_path / "nope.jsonl")]) == EXIT_IO == 3
    assert "FAIL io" in capsys.readouterr().err
    assert replay_main(["--verify", str(replays["mutual_raid_draw"].path)]) == 0
    assert sim_main(["replay", "--verify", str(corrupt)]) == 2


def test_exit_code_and_output_are_deterministic(replays, tmp_path):
    first = run_module("formicmind.replay", "--verify", str(replays["mutual_raid_draw"].path), cwd=tmp_path)
    second = run_module("formicmind.replay", "--verify", str(replays["mutual_raid_draw"].path), cwd=tmp_path)
    assert (first.returncode, first.stdout) == (second.returncode, second.stdout)


def test_non_ascii_paths_work_on_windows_and_the_output_stays_ascii(replays, tmp_path):
    folder = tmp_path / "реплеи"
    folder.mkdir()
    target = folder / "матч.jsonl"
    target.write_bytes(replays["mutual_raid_draw"].data)
    proc = run_module("formicmind.replay", "--verify", str(target), cwd=tmp_path)
    assert proc.returncode == 0 and proc.stdout.isascii() and "\\u0440" in proc.stdout


# --- replay_smoke ------------------------------------------------------------------------------------------------------------

def test_smoke_recorder_writes_verifies_and_reports_a_subset(tmp_path):
    lines = []
    code = smoke_main(["--out", str(tmp_path / "out"), "--scenario", "mutual_raid_draw", "--scenario", "cutoff"],
                      write=lines.append)
    text = "\n".join(lines)
    assert code == 0 and "FUNCTIONAL CHECK" in text
    for name in ("mutual_raid_draw", "cutoff"):
        assert verify((tmp_path / "out" / f"{name}.jsonl")).finished is (name != "cutoff")
    assert "93b401faab189be2c6ab6d851614c8ce19c8ef7abb830e156e44449d87c1805b" in text  # pinned events_hash of the draw


def test_smoke_recorder_single_file_mode_and_usage(tmp_path):
    target = tmp_path / "one.jsonl"
    assert smoke_main(["--out", str(target), "--scenario", "collapse"], write=lambda _: None) == 0
    assert ReplayReader.open(target).read_header().source == "replay_smoke:collapse"
    with pytest.raises(SystemExit):
        smoke_main(["--out", str(tmp_path / "x.jsonl")], write=lambda _: None)  # all scenarios into one file
    with pytest.raises(SystemExit):
        smoke_main(["--out", str(tmp_path), "--scenario", "nope"], write=lambda _: None)
    assert set(ALL_SCENARIOS) == {"raid_win", "mutual_raid_draw", "collapse", "timeout_draw", "cutoff", "rng_policy"}


def test_smoke_recorder_runs_as_a_module(tmp_path):
    proc = run_module("formicmind.replay_smoke", "--out", str(tmp_path / "r"), "--scenario", "rng_policy", cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "rng_policy" in proc.stdout and (tmp_path / "r" / "rng_policy.jsonl").exists()
