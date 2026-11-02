"""The movement-only smoke run: pinned output, determinism across seeds and PYTHONHASHSEED."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from formicmind.smoke import SMOKE_TICKS, run_movement_smoke

SRC = Path(__file__).resolve().parents[1] / "src"

EXPECTED_LINES = (
    "t=0   A sends 8 workers A->N1 (arrive t=2)",
    "t=0   A sends 4 soldiers A->N1 (arrive t=2)",
    "t=2   A 8 workers arrive at N1",
    "t=2   A 4 soldiers arrive at N1",
    "t=2   A sends 4 soldiers N1->C (arrive t=4)",
    "t=4   A 4 soldiers arrive at C",
    "t=30  A sends 4 soldiers C->N4 (arrive t=32)",
    "t=32  A 4 soldiers arrive at N4",
    "t=32  A sends 4 soldiers N4->B (arrive t=34)",
    "t=34  A 4 soldiers arrive at B",
)
EXPECTED_POSITIONS = (
    "A: 4 workers at A, 4 soldiers at B, 8 workers at N1",
    "B: 12 workers at B, 4 soldiers at B",
)
# Regression hashes of this exact scenario (produced by this implementation).
EXPECTED_EVENTS_HASH = "ad06d68fb3b116727cfc4340ac30782061fcddb4e75b3dae128c77354813477a"
EXPECTED_STATE_HASH = "17e7e19e339a6a74c567ee6bcb935add397b9f4c3fb299da6c37b25a416bdc3a"


def test_smoke_prints_ten_readable_movement_events_and_final_positions():
    report = run_movement_smoke()
    assert 60 <= report.ticks == SMOKE_TICKS <= 100
    assert report.lines == EXPECTED_LINES and 5 <= len(report.lines) <= 10
    assert report.positions == EXPECTED_POSITIONS
    assert report.event_count == 70
    assert "A's observation at t=60: enemy home known=1, current=1" in report.observation_note
    assert "window_ticks/K=1" in report.observation_note
    text = report.render()
    assert "not a match" in text and "no combat/economy/hatching" in text


def test_smoke_hashes_are_pinned_and_independent_of_the_root_seed():
    base = run_movement_smoke()
    assert base.events_hash == EXPECTED_EVENTS_HASH and base.state_hash == EXPECTED_STATE_HASH
    other = run_movement_smoke(seed=77)
    assert (other.events_hash, other.state_hash, other.lines) == (base.events_hash, base.state_hash, base.lines)
    assert other.checkpoint_hash != base.checkpoint_hash          # the RNG registry still differs


@pytest.mark.parametrize("ticks", [60, 61, 100])
def test_smoke_runs_for_other_lengths(ticks):
    report = run_movement_smoke(ticks)
    assert report.ticks == ticks and report.lines[:3] == EXPECTED_LINES[:3]


def _run_in_subprocess(hashseed: str) -> str:
    script = (
        f"import sys; sys.path.insert(0, {str(SRC)!r})\n"
        "from formicmind.smoke import run_movement_smoke\n"
        "r = run_movement_smoke()\n"
        "print(r.events_hash, r.state_hash, r.checkpoint_hash, sep=' ')\n"
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          env={**os.environ, "PYTHONHASHSEED": hashseed}, check=False)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


@pytest.mark.parametrize("other", ["1", "4242", "random"])
def test_smoke_is_independent_of_pythonhashseed(other):
    assert _run_in_subprocess("0") == _run_in_subprocess(other)


def test_smoke_matches_in_process_result():
    r = run_movement_smoke()
    assert _run_in_subprocess("5") == f"{r.events_hash} {r.state_hash} {r.checkpoint_hash}"
