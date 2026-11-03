from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

from formicmind.canonical import canonical_dumps
from formicmind.gate1_context import (
    _attach_paired_block_digests,
    _build_configs,
    _build_summary,
    _canonical_missing,
    _context_label,
    _degeneracy_summary,
    _execute_match,
    _expected_matchups,
    _paired_seat_score,
    _rank_reversal_label,
    _row_score,
    _schedule,
)
from formicmind.rules import default_config
from formicmind.serialization import config_hash, config_to_canonical


def _block_rows(
    *, first: str, second: str, seed: int, trajectory_seed: str,
) -> list[dict]:
    rows = []
    for family in ("R0", "R1"):
        for first_on_a in (True, False):
            rows.append({
                "status": "complete",
                "map_id": f"{family}-T0",
                "resource_family": family,
                "terrain_profile": "T0",
                "matchup": f"{first}-{second}",
                "policies": [first, second],
                "seat_assignment": {
                    "A": first if first_on_a else second,
                    "B": second if first_on_a else first,
                },
                "seed": seed,
                "trajectory_sha256": f"{family}-{first_on_a}-{trajectory_seed}",
                "config_sha256": f"config-{family}",
                "paired_trajectory_block_sha256": None,
            })
    return rows


def test_fixed_map_matrix_preserves_default_and_invariants():
    before = canonical_dumps(config_to_canonical(default_config()))
    configs = _build_configs()

    assert list(configs) == [
        "R0-T0", "R0-T1", "R0-T2", "R1-T0", "R1-T1", "R1-T2",
    ]
    assert configs["R0-T0"] == default_config()
    assert replace(
        configs["R1-T1"].balance, resources=configs["R0-T1"].balance.resources,
    ) == configs["R0-T1"].balance
    assert configs["R0-T0"].topology == configs["R1-T0"].topology
    assert configs["R0-T1"].topology == configs["R1-T1"].topology
    assert configs["R0-T2"].topology == configs["R1-T2"].topology
    assert configs["R0-T0"].topology != configs["R0-T1"].topology
    assert configs["R0-T0"].topology != configs["R0-T2"].topology
    assert len({config_hash(config) for config in configs.values()}) == 6
    assert canonical_dumps(config_to_canonical(default_config())) == before


def test_tournament_schedule_has_exact_fixed_size_and_no_self_play():
    schedule = _schedule()
    assert len(schedule) == 720
    assert len(_expected_matchups()) == 10
    assert {row[0] for row in schedule} == {
        "R0-T0", "R0-T1", "R0-T2", "R1-T0", "R1-T1", "R1-T2",
    }
    assert {row[3] for row in schedule} == set(range(20261221, 20261227))
    assert {row[4] for row in schedule} == {False, True}
    assert all(left != right for _, left, right, _, _ in schedule)
    assert len({(map_id, left, right, seed, seat)
                for map_id, left, right, seed, seat in schedule}) == 720


def test_independent_blocks_collapse_deterministic_seed_duplicates_but_keep_random_paths():
    deterministic = (
        _block_rows(first="turtle", second="rush", seed=20261221, trajectory_seed="same")
        + _block_rows(first="turtle", second="rush", seed=20261222, trajectory_seed="same")
    )
    _attach_paired_block_digests(deterministic)
    assert {
        row["paired_trajectory_block_sha256"] for row in deterministic
    } == {deterministic[0]["paired_trajectory_block_sha256"]}

    random_rows = (
        _block_rows(first="random", second="rush", seed=20261221, trajectory_seed="draw-a")
        + _block_rows(first="random", second="rush", seed=20261222, trajectory_seed="draw-b")
    )
    _attach_paired_block_digests(random_rows)
    first_block = random_rows[0]["paired_trajectory_block_sha256"]
    second_block = random_rows[4]["paired_trajectory_block_sha256"]
    assert first_block != second_block


def test_scores_collapse_seats_exactly_and_missing_is_not_a_draw():
    assert _row_score({"score_x2": {"random": 2}}, "random") == 1
    assert _row_score({"score_x2": {"random": 1}}, "random") == Fraction(1, 2)
    assert _row_score({"score_x2": None}, "random") is None
    assert _paired_seat_score([
        {"score_x2": {"random": 2}},
        {"score_x2": {"random": 1}},
    ], "random") == Fraction(3, 4)
    assert _paired_seat_score([
        {"score_x2": {"random": 2}},
        {"score_x2": None},
    ], "random") is None


def test_preregistered_threshold_boundaries_and_exact_zero_direction():
    assert _context_label(
        Fraction(15, 100),
        [Fraction(15, 100), Fraction(1, 5), Fraction(0)],
        adequate=True, complete=True,
    ) == "CONTEXT_SENSITIVITY_CANDIDATE"
    assert _context_label(
        Fraction(15, 100),
        [Fraction(15, 100), Fraction(0), Fraction(0)],
        adequate=True, complete=True,
    ) == "NO_CONTEXT_SIGNAL_IN_SCHEDULE"
    assert _context_label(
        Fraction(149, 1000),
        [Fraction(1, 5), Fraction(1, 5), Fraction(0)],
        adequate=True, complete=True,
    ) == "NO_CONTEXT_SIGNAL_IN_SCHEDULE"
    assert _context_label(
        Fraction(1, 2),
        [Fraction(1, 2)] * 3,
        adequate=False, complete=True,
    ) == "INCONCLUSIVE"

    assert _rank_reversal_label(
        {"R0": Fraction(1, 10), "R1": Fraction(-1, 10)},
        adequate=True, complete=True,
    ) == "PAIRWISE_RANK_REVERSAL_CANDIDATE"
    assert _rank_reversal_label(
        {"R0": Fraction(99, 1000), "R1": Fraction(-1, 10)},
        adequate=True, complete=True,
    ) == "NO_CONTEXT_SIGNAL_IN_SCHEDULE"
    assert _rank_reversal_label(
        {"R0": Fraction(1, 2), "R1": Fraction(-1, 2)},
        adequate=True, complete=False,
    ) == "INCONCLUSIVE"


def test_degeneracy_summary_keeps_censored_rows_out_of_timeout_counts():
    result = _degeneracy_summary([{
        "status": "censored_incomplete",
        "map_id": "R0-T0",
        "matchup": "random-turtle",
        "terminal": None,
        "ticks_executed": 3000,
        "event_counts": {},
        "terminal_metrics": {},
        "policy_event_metrics": {},
    }])
    group = next(
        row for row in result
        if row["map_id"] == "R0-T0" and row["matchup"] == "random-turtle"
    )
    assert group["rows_present"] == 1
    assert group["timeout_count"] == 0
    assert group["timeout_fraction"] == {"numerator": 0, "denominator": 1}


def test_summary_uses_paired_seed_blocks_and_emits_full_matrices():
    rows = []
    for map_id, first, second, seed, swapped in _schedule():
        family, terrain = map_id.split("-")
        first_score = (
            2 if (first, second) == ("random", "turtle") and family == "R1"
            else 0 if (first, second) == ("random", "turtle") and family == "R0"
            else 1
        )
        rows.append({
            "schema": "formicmind.gate1-context-match/v1",
            "status": "complete",
            "map_id": map_id,
            "resource_family": family,
            "terrain_profile": terrain,
            "config_sha256": f"config-{map_id}",
            "matchup": f"{first}-{second}",
            "policies": [first, second],
            "seed": seed,
            "seat_assignment": {
                "A": second if swapped else first,
                "B": first if swapped else second,
            },
            "score_x2": {first: first_score, second: 2 - first_score},
            "terminal": {
                "winner": "A" if first_score == 2 else "B" if first_score == 0 else "draw",
                "reason": "queen_death",
            },
            "ticks_executed": 1,
            "event_counts": {},
            "terminal_metrics": {
                first: {"queen_hp": 200, "soldiers": 2, "food": 0},
                second: {"queen_hp": 200, "soldiers": 2, "food": 0},
            },
            "policy_event_metrics": {
                first: {"repetitive_orders_flag": False},
                second: {"repetitive_orders_flag": False},
            },
            "trajectory_sha256": f"{map_id}-{first}-{second}-{seed}",
            "random_stream_trajectory_sha256": (
                f"random-{seed}" if "random" in (first, second) else None
            ),
            "paired_trajectory_block_sha256": None,
        })

    summary = _build_summary(rows, calls=720)
    assert summary["status"] == "complete"
    assert summary["scheduled_matches"] == summary["completed_matches"] == 720
    assert summary["scheduled_paired_seed_blocks"] == 180
    assert len(summary["map_matchup_score_matrix"]) == 60
    assert len(summary["policy_map_score_matrix"]) == 30
    assert len(summary["policy_family_rankings"]) == 2
    pair = next(
        item for item in summary["pairwise_context_analysis"]
        if item["matchup"] == "random-turtle"
    )
    assert pair["context_sensitivity_label"] == "CONTEXT_SENSITIVITY_CANDIDATE"
    assert pair["rank_reversal_label"] == "PAIRWISE_RANK_REVERSAL_CANDIDATE"
    assert all(profile["distinct_paired_trajectory_blocks"] == 6
               for profile in pair["profiles"])


def test_degeneracy_flags_use_the_preregistered_three_of_twelve_cutoff():
    rows = []
    for index in range(12):
        timeout = index < 3
        rows.append({
            "status": "complete",
            "map_id": "R0-T0",
            "matchup": "random-turtle",
            "terminal": {
                "reason": "timeout" if timeout else "queen_death",
                "state_tick": 1000 if timeout else 500,
            },
            "ticks_executed": 1000 if timeout else 500,
            "event_counts": {},
            "terminal_metrics": {
                "random": {"queen_hp": 100, "soldiers": 1, "food": 300},
                "turtle": {"queen_hp": 100, "soldiers": 1, "food": 300},
            },
            "policy_event_metrics": {
                "random": {"repetitive_orders_flag": timeout},
                "turtle": {"repetitive_orders_flag": False},
            },
        })
    group = next(
        row for row in _degeneracy_summary(rows)
        if row["map_id"] == "R0-T0" and row["matchup"] == "random-turtle"
    )
    assert group["timeout_count"] == 3
    assert group["timeout_concentration_flag"] is True
    assert group["extended_zero_combat_flag"] is True
    assert group["food_stockpile_live_army_timeout_flag"] is True
    assert group["repetitive_policy_order_flag"] is True


def test_real_matches_replay_deterministically_and_keep_policy_identity_when_swapped():
    configs = _build_configs()
    first_run = _execute_match(
        "R0-T0", configs["R0-T0"], "random", "turtle", 20261221, False, [0],
    )
    repeated = _execute_match(
        "R0-T0", configs["R0-T0"], "random", "turtle", 20261221, False, [0],
    )
    swapped = _execute_match(
        "R0-T0", configs["R0-T0"], "random", "turtle", 20261221, True, [0],
    )
    local_rich = _execute_match(
        "R1-T0", configs["R1-T0"], "random", "turtle", 20261221, False, [0],
    )

    assert first_run["status"] == repeated["status"] == swapped["status"] == "complete"
    assert local_rich["status"] == "complete"
    assert first_run["trajectory_sha256"] == repeated["trajectory_sha256"]
    assert first_run["final_state_sha256"] == repeated["final_state_sha256"]
    assert first_run["final_checkpoint_sha256"] == repeated["final_checkpoint_sha256"]
    assert first_run["score_x2"] == repeated["score_x2"]
    assert first_run["trajectory_sha256"] == swapped["trajectory_sha256"]
    assert first_run["score_x2"] == swapped["score_x2"]
    assert first_run["horizons"]["100"]["status"] in {"reached", "not_measured"}
    assert canonical_dumps(_canonical_missing(first_run))
