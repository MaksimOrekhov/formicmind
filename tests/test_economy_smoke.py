"""The economy smoke run: partial-mode label, hand-checkable rows, pinned regression hashes, determinism."""

import os
import subprocess
import sys
from pathlib import Path

from formicmind.economy_smoke import ECONOMY_SMOKE_TICKS, LABEL, run_economy_smoke

SRC = Path(__file__).resolve().parents[1] / "src"

# Regression hashes of this exact scenario (produced by this implementation, default candidate balance).
EXPECTED_EVENTS_HASH = "907e1fbef9e815d138438b0ac7ef8062306764828a5cd14117c72151b6f122bf"
EXPECTED_STATE_HASH = "a4df02a5da0144a90f77f311986295e8dbc12f6bc714459f395a63bc3a0e229f"
EXPECTED_CHECKPOINT_HASH = "4b35c370efc86e183adf1a72407a13e98731499a3c7cec51430a720c3bead842"


def test_smoke_is_labelled_partial_and_runs_100_to_150_ticks_with_a_closed_food_ledger():
    report = run_economy_smoke()
    assert 100 <= report.ticks == ECONOMY_SMOKE_TICKS <= 150
    text = report.render()
    assert LABEL == "PARTIAL (NO COMBAT/TERMINAL)" and LABEL in text and "not a match" in text
    assert report.ledger.endswith(": OK")
    assert "start 400 + regen" in report.ledger


def test_first_rows_agree_with_the_hand_calculation():
    rows = run_economy_smoke().rows
    # ticks 0..9: harvest 8 + 12 * 8 = 92 (workers arrive at t=2 and t=3), upkeep 8 * 10 = 80,
    # A paid one worker egg (10), B one soldier egg (20); warehouses 60 + 92 - 80 - egg
    a10 = next(r for r in rows if r.startswith("  10 | A"))
    b10 = next(r for r in rows if r.startswith("  10 | B"))
    assert " 62 |" in a10 and "      92     80   10" in a10
    assert " 52 |" in b10 and "      92     80   20" in b10
    assert a10.endswith("| 76/60/120/76/60")


def test_totals_and_hashes_are_pinned_and_unit_counts_are_consistent():
    report = run_economy_smoke()
    a, b = report.totals[1], report.totals[2]
    assert "A: harvest=1556 upkeep_paid=1188" in a and "egg_spend=120 eggs_laid=9" in a
    assert "hatched W/S=6/3" in a and "adults W/S=18/7, brood W/S=0/0" in a
    assert "B: harvest=1439 upkeep_paid=1312" in b and "egg_spend=180 eggs_laid=12" in b
    assert "hatched W/S=4/6" in b and "adults W/S=16/10, brood W/S=2/0" in b
    assert (report.events_hash, report.state_hash, report.checkpoint_hash) == (
        EXPECTED_EVENTS_HASH, EXPECTED_STATE_HASH, EXPECTED_CHECKPOINT_HASH)


def test_the_candidate_egg_batch_sizes_all_run_and_keep_the_ledger():
    for eggs in (1, 2, 4):
        report = run_economy_smoke(ticks=120, eggs_per_lay=eggs)
        assert report.ledger.endswith(": OK") and report.eggs_per_lay == eggs


def test_smoke_output_is_independent_of_pythonhashseed():
    code = (f"import sys; sys.path.insert(0, {str(SRC)!r}); "
            "from formicmind.economy_smoke import run_economy_smoke; print(run_economy_smoke().render())")

    def run(seed):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False)
        assert proc.returncode == 0, proc.stderr
        return proc.stdout

    first = run("0")
    assert first == run("1") == run("987654") == run("random")
    assert first.strip() == run_economy_smoke().render().strip()
