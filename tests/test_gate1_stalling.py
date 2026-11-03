from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from formicmind.actions import ActionMaskV2, ActionV2, InvalidAction, validate_action
from formicmind.baselines import baseline_spec
from formicmind.canonical import canonical_dumps
from formicmind.gate1_stalling import (
    BASELINE_BUDGET,
    EXPECTED_BASELINES,
    EXPECTED_PROBES,
    INTERVENTIONS,
    MAPS,
    PAIRS,
    PROBE_TICKS,
    SEEDS,
    _RunBudget,
    _action_diff,
    _action_object,
    _attach_block_digests,
    _baseline_id,
    _build_configs,
    _candidate_action,
    _finish_metrics,
    _manifest,
    _new_policy_metrics,
    _probe_schedule,
    _review_flag,
    _run_baseline,
    _run_probe,
    _schedule,
    _source_fingerprint,
    _unique_probe_context_sha256,
    _verify_control_suffix,
    verify,
    Gate1StallingError,
)
from formicmind.observation import build_observation
from formicmind.reset import reset
from formicmind.rng import policy_stream_id
from formicmind.rules import Q_TOTAL
from formicmind.snapshot import begin_policy_phase
from formicmind.serialization import config_hash


def test_fixed_schedule_and_probes_match_protocol() -> None:
    assert len(_schedule()) == EXPECTED_BASELINES == 288
    assert len(_probe_schedule()) == EXPECTED_PROBES == 384
    assert tuple(_build_configs()) == MAPS
    assert set(SEEDS) == set(range(20270201, 20270207))
    assert set(PAIRS) == {
        ("turtle", "balanced"), ("turtle", "raider"),
        ("rush", "balanced"), ("random", "turtle"),
    }
    assert set(PROBE_TICKS) == {300, 900}
    assert set(INTERVENTIONS) == {"raid_once", "home_forage_once"}
    assert len({row for row in _schedule()}) == 288
    assert hashlib.sha256(canonical_dumps(list(SEEDS))).hexdigest() == "ed35d5d2f19af9a36ae72dd6afc33709fd5d43240e626679144b6ee8f3c5b18a"
    seats = [{"A": "left", "B": "right"}, {"A": "right", "B": "left"}]
    assert hashlib.sha256(canonical_dumps(seats)).hexdigest() == "edbd1306cfd9beea32c7703bb5e77cbb6a89277ef33d6bf999dddc46d6c93252"


def test_map_and_policy_hashes_are_the_published_for22_values() -> None:
    expected = {
        "R0-T0": "3f6aaa36a0d3d15888fa42f142df1151376794e62b553ab8e5f7659c9794145d",
        "R0-T1": "78e9c6979a2fb3973ae9a275f79ad5b62cb4b6ed3893a8f8469e4cc599e5fb34",
        "R0-T2": "b9e43b2aec34f303f371a9f9e5a2bcd8b5c5d894e608a3ca9e71386339f632f1",
        "R1-T0": "407bb3907099533adfe10b7c300a480b45b5d604dc019693b1ac99f47502303f",
        "R1-T1": "3bf8dabf6011ed88275c3fcfec3875c4da46ef849e7da7b0ac17bb2cd0e9e605",
        "R1-T2": "1b6f1f999246711919fb4dee9238e5fed31c5baac0dca779ea76c47afe167ceb",
    }
    configs = _build_configs()
    assert {key: config_hash(value) for key, value in configs.items()} == expected
    assert baseline_spec("turtle").digest() == "6c9fa1c676a553294c26ffb4f3372953f93ecec73b5d8669f99a451a05e0ee4e"
    assert baseline_spec("balanced").digest() == "03c6760da2b2fc42108133c8b81a91b70e3c086327fbabedf9d3f30eb7dcdd0f"
    assert baseline_spec("raider").digest() == "6d4a23b0071c6ee308b8149850debe488e83d79c7fda50f19ee5053310c39b19"
    assert baseline_spec("rush").digest() == "cfea106a71257d8ebe79cfa6ac090b1927fd3010ca905516e2e41512460ea496"
    assert baseline_spec("random").digest() == "254037fbbe5f089fc1f909405ebcc3567573682553d3715d9bab8cf292b07f2c"


def test_review_flag_boundary_is_exactly_three_of_twelve() -> None:
    assert _review_flag(2) == {
        "numerator": 2, "denominator": 12, "flag": False,
        "unit": "descriptive_raw_scheduled_episodes",
    }
    assert _review_flag(3)["flag"] is True
    assert _review_flag(3)["numerator"] == 3


def test_repetitive_orders_boundaries_at_ten_ninety_and_seventy_five_percent() -> None:
    exact_army_row = _new_policy_metrics()
    exact_army_row.update({
        "order_count": 10, "lay_none_orders": 0,
        "army_choice_counts": {"1": 9, "2": 1},
        "forage_vector_counts": {"a": 5, "b": 5},
    })
    exact_army = {"turtle": exact_army_row}
    assert _finish_metrics(exact_army)["turtle"]["repetitive_orders_flag"] is True
    exact_forage_row = _new_policy_metrics()
    exact_forage_row.update({
        "order_count": 10, "lay_none_orders": 0,
        "army_choice_counts": {"1": 8, "2": 2},
        "forage_vector_counts": {"a": 9, "b": 1},
    })
    assert _finish_metrics({"turtle": exact_forage_row})["turtle"]["repetitive_orders_flag"] is True
    exact_lay_row = _new_policy_metrics()
    exact_lay_row.update({
        "order_count": 12, "lay_none_orders": 9,
        "army_choice_counts": {"1": 8, "2": 4},
        "forage_vector_counts": {"a": 8, "b": 4},
    })
    exact_lay_none = {"turtle": exact_lay_row}
    assert _finish_metrics(exact_lay_none)["turtle"]["repetitive_orders_flag"] is True
    below_minimum_row = _new_policy_metrics()
    below_minimum_row.update({
        "order_count": 9, "lay_none_orders": 9,
        "army_choice_counts": {"1": 9}, "forage_vector_counts": {"a": 9},
    })
    below_minimum = {"turtle": below_minimum_row}
    assert _finish_metrics(below_minimum)["turtle"]["repetitive_orders_flag"] is False


def test_predeclared_candidates_are_single_axis_and_masked_raid_is_ineligible() -> None:
    action = ActionV2(1, (0, Q_TOTAL, 0, 0, 0, 0), 0, False)
    mask = ActionMaskV2(
        lay=(True, True, True), forage_target=(True,) * 6,
        army_choice=(True,) * 7, army_avoid_center=((True, True),) * 7,
    )
    raid, status, _reason = _candidate_action("raid_once", action, mask)
    assert status == "eligible" and raid is not None
    assert _action_diff(action, raid) == ["army_choice"]

    shares, status, _reason = _candidate_action("home_forage_once", action, mask)
    assert status == "eligible" and shares is not None
    assert shares.forage_share_q == (Q_TOTAL, 0, 0, 0, 0, 0)
    assert _action_diff(action, shares) == ["forage_share_q"]

    masked = ActionMaskV2(
        lay=mask.lay, forage_target=mask.forage_target,
        army_choice=mask.army_choice[:6] + (False,), army_avoid_center=mask.army_avoid_center,
    )
    assert _candidate_action("raid_once", action, masked) == (None, "ineligible", "army_choice_6_masked")
    assert _candidate_action("raid_once", ActionV2(1, action.forage_share_q, 6, False), mask)[1] == "no_effect"
    assert _candidate_action("home_forage_once", ActionV2(1, (Q_TOTAL, 0, 0, 0, 0, 0), 0, False), mask)[1] == "no_effect"


def test_mirror_and_deterministic_trajectory_collapse_preserves_random_paths() -> None:
    rows = []
    for map_id, left, right, seed, swapped in _schedule():
        random_pair = left == "random" or right == "random"
        trace = f"{map_id}:{left}:{right}:{seed if random_pair else 'same'}"
        a, b = (right, left) if swapped else (left, right)
        rows.append({
            "map_id": map_id, "matchup": f"{left}-{right}", "seed": seed,
            "policies": [left, right], "seat_assignment": {"A": a, "B": b},
            "policy_relative_trajectory_sha256": trace,
        })
    _attach_block_digests(rows)
    deterministic = {
        row["paired_trajectory_block_sha256"] for row in rows
        if row["map_id"] == "R0-T0" and row["matchup"] == "turtle-balanced"
    }
    stochastic = {
        row["paired_trajectory_block_sha256"] for row in rows
        if row["map_id"] == "R0-T0" and row["matchup"] == "random-turtle"
    }
    assert len(deterministic) == 1
    assert len(stochastic) == 6


def test_unique_probe_context_collapses_same_normalized_checkpoint_across_seed_and_seat() -> None:
    first = {"map_id": "R0-T0", "matchup": "turtle-balanced", "baseline_id": "seed1-first_A"}
    duplicate = {"map_id": "R0-T0", "matchup": "turtle-balanced", "baseline_id": "seed6-first_B"}
    checkpoint = "a" * 64
    assert _unique_probe_context_sha256(first, 300, "turtle", "raid_once", checkpoint) == \
        _unique_probe_context_sha256(duplicate, 300, "turtle", "raid_once", checkpoint)
    assert _unique_probe_context_sha256(first, 300, "turtle", "raid_once", checkpoint) != \
        _unique_probe_context_sha256(duplicate, 300, "turtle", "home_forage_once", checkpoint)


def test_engine_timeout_and_one_shot_continuation_use_real_step_tick() -> None:
    config = _build_configs()["R0-T0"]
    baseline, checkpoints, physical = _run_baseline(
        "R0-T0", config, "turtle", "balanced", 20270201, False, _RunBudget(3000),
    )
    assert baseline["status"] == "complete"
    assert baseline["terminal"]["reason"] == "timeout"
    assert baseline["terminal"]["terminal_tick"] == 2999
    assert baseline["ticks_executed"] == 3000
    checkpoint = checkpoints[(300, "turtle", "home_forage_once")]
    assert checkpoint.eligibility_status == "eligible"
    descriptor, tick, target, intervention = next(
        item for item in _probe_schedule()
        if item[0]["baseline_id"] == _baseline_id("R0-T0", "turtle", "balanced", 20270201, False)
        and item[1:] == (300, "turtle", "home_forage_once")
    )
    probe = _run_probe(descriptor, baseline, checkpoint, tick, target, intervention,
                       _RunBudget(1_152_000), physical)
    assert probe["eligibility_status"] == "eligible"
    assert probe["altered_decision_count"] == 1
    assert probe["first_changed_order_tick"] == 300
    assert probe["first_physical_divergence_tick"] != "not_measured"

    control_calls = _verify_control_suffix(
        checkpoints[(300, "turtle", "raid_once")], "turtle", "balanced", False,
        baseline, _RunBudget(BASELINE_BUDGET),
    )
    assert control_calls == 2700
    assert probe["terminal"]["reason"] in {"timeout", "queen_death", "economic_collapse"}


def test_strict_verifier_rejects_missing_or_tampered_rows_before_replay(tmp_path: Path) -> None:
    study = tmp_path / "study"
    study.mkdir()
    baselines = canonical_dumps({"baseline_id": "not-a-scheduled-row"}) + b"\n"
    probes = b""
    summary = canonical_dumps({"schema": "tampered"}) + b"\n"
    artifacts = {
        "baselines.jsonl": hashlib.sha256(baselines).hexdigest(),
        "probes.jsonl": hashlib.sha256(probes).hexdigest(),
        "summary.json": hashlib.sha256(summary).hexdigest(),
    }
    from formicmind.gate1_stalling import _git, _write_json

    manifest = _manifest(_git("rev-parse", "HEAD"), 0, 0, artifacts)
    (study / "baselines.jsonl").write_bytes(baselines)
    (study / "probes.jsonl").write_bytes(probes)
    (study / "summary.json").write_bytes(summary)
    _write_json(study / "manifest.json", manifest)
    with pytest.raises(Gate1StallingError, match="baseline rows are missing"):
        verify(study)
