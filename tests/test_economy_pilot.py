"""FOR-20 preregistered economic pilot matrix and harness tests."""

import pytest

from formicmind.economy_pilot import (
    DEFAULT_TOPOLOGY,
    _canonical_json,
    _classify_lay_opportunities,
    _plateau_tick,
    _run_one,
    _update_event_totals,
    aggregate_runs,
    build_config,
    forage_shares_for,
    lay_order_for,
    matrix_design,
)
from formicmind.events import LayEvent, Phase, StarvationEvent, UpkeepEvent
from formicmind.rules import LAY_NONE, LAY_SOLDIER, LAY_WORKER, default_config
from formicmind.serialization import config_hash


def test_locked_matrix_has_144_unique_rows_and_three_open_seeds():
    design = matrix_design()
    keys = [row["run_key"] for row in design]
    assert len(design) == len(set(keys)) == 144
    assert {row["seed"] for row in design} == {20261111, 20261112, 20261113}
    assert {row["eggs_per_lay"] for row in design} == {1, 2, 4}
    assert {row["lay_regime"] for row in design} == {
        "no-lay", "worker-first", "soldier-focused", "alternating",
    }
    assert {row["forage_regime"] for row in design} == {"single-local", "split-local"}
    assert {row["environment_profile"] for row in design} == {"default", "scarcity"}


def test_experimental_configs_do_not_mutate_default_balance_or_versions():
    default = default_config()
    baseline_hash = config_hash(default)
    candidate = build_config(4, "scarcity")
    assert config_hash(default) == baseline_hash
    assert default.balance.eggs_per_lay == 1
    assert candidate.balance.eggs_per_lay == 4
    assert candidate.rules_version == default.rules_version
    assert candidate.balance_version == default.balance_version
    assert candidate.balance.start_food == default.balance.start_food
    for original, stressed in zip(default.balance.resources, candidate.balance.resources):
        assert stressed.start_food == original.start_food // 2
        assert stressed.regen == original.regen // 2
        assert stressed.capacity == original.capacity
        assert stressed.max_harvest == original.max_harvest
    barrier = build_config(1, "default", "local-barrier")
    assert {(edge.a, edge.b) for edge in barrier.topology.edges} == {(0, 1), (0, 2), (4, 6), (5, 6)}


def test_scripted_regimes_obey_policy_cadence_and_forage_contract():
    assert [lay_order_for("alternating", tick) for tick in (0, 30, 90, 120, 150, 240)] == [
        LAY_WORKER, LAY_WORKER, LAY_WORKER, LAY_SOLDIER, LAY_SOLDIER, LAY_WORKER,
    ]
    assert lay_order_for("no-lay", 0) == LAY_NONE
    assert lay_order_for("soldier-focused", 90) == LAY_SOLDIER
    assert forage_shares_for("single-local") == (0, 65535, 0, 0, 0, 0)
    split = forage_shares_for("split-local")
    assert split == (0, 32768, 32767, 0, 0, 0)
    assert sum(split) == 65535 and sum(value > 0 for value in split) == 2


def test_lay_event_is_required_only_for_due_requests_not_none_or_cooldown():
    _classify_lay_opportunities(
        3, (LAY_NONE, LAY_WORKER), (10, 10), (240, 240), (),
    )
    _classify_lay_opportunities(
        5, (LAY_WORKER, LAY_SOLDIER), (10, 10), (240, 240), (),
    )
    laid = LayEvent(
        tick=10, phase=int(Phase.LAY), side=0, laid=True, caste=0, eggs=1,
        unit_cost=10, total_cost=10, food_before=100, food_after=90,
        unit_ids=(1,), hatch_tick=30, next_lay_tick=20,
    )
    rejected = LayEvent(
        tick=10, phase=int(Phase.LAY), side=1, laid=False, caste=0, eggs=1,
        unit_cost=10, total_cost=10, food_before=0, food_after=0,
        unit_ids=(), hatch_tick=-1, next_lay_tick=0,
    )
    _classify_lay_opportunities(
        10, (LAY_WORKER, LAY_WORKER), (10, 10), (240, 240), (laid, rejected),
    )
    with pytest.raises(ValueError, match="emitted 0 LayEvents"):
        _classify_lay_opportunities(10, (LAY_WORKER, LAY_NONE), (10, 0), (240, 240), ())


def test_plateau_definition_requires_three_complete_stable_windows():
    assert _plateau_tick([{"tick": 0, "adult_workers": 12}, {"tick": 100, "adult_workers": 12},
                          {"tick": 200, "adult_workers": 13}]) is None
    points = [{"tick": tick, "adult_workers": 12 + (tick // 100) % 2} for tick in range(0, 401, 100)]
    assert _plateau_tick(points) == 300
    assert _plateau_tick([{"tick": 0, "adult_workers": 1}, {"tick": 100, "adult_workers": 0},
                          {"tick": 200, "adult_workers": 0}, {"tick": 300, "adult_workers": 0}]) is None


def test_upkeep_starvation_event_accounting_and_strict_json_outputs():
    totals = [{name: 0 for name in (
        "harvested_food", "upkeep_paid", "upkeep_missing", "lay_none_ticks", "cooldown_ticks",
        "eligible_lay_attempts", "laid_batches", "food_rejected_batches", "eggs_laid", "egg_spend",
        "hatched_workers", "hatched_soldiers", "starved_workers", "starved_soldiers", "starvation_deaths",
    )} for _ in (0, 1)]
    upkeep = UpkeepEvent(
        tick=5, phase=int(Phase.UPKEEP), side=0, adult_workers=12, adult_soldiers=4,
        remainder_before=0, remainder_after=0, due=8, paid=6, missing=2,
        food_before=6, food_after=0, debt_before=0, debt_after=2, deaths=0, debt_reset=False,
    )
    starvation = StarvationEvent(
        tick=5, phase=int(Phase.UPKEEP), side=0, seq=0, unit_id=8, caste=1, hp=30,
        node=0, to_node=-1, debt_before=5, debt_after=0,
    )
    counts = {}
    from collections import Counter
    event_counts = Counter(counts)
    _update_event_totals(upkeep, totals, event_counts)
    _update_event_totals(starvation, totals, event_counts)
    assert totals[0]["upkeep_paid"] == 6 and totals[0]["upkeep_missing"] == 2
    assert totals[0]["starved_soldiers"] == totals[0]["starvation_deaths"] == 1
    assert event_counts == {"upkeep": 1, "starvation": 1}
    assert _canonical_json({"value": 1}).endswith(b"\n")
    with pytest.raises(ValueError):
        _canonical_json({"value": float("nan")})


def test_aggregation_collapses_duplicate_deterministic_seed_rows():
    sample_checkpoint = {"tick": 100, "side": 0, "status": "missing", "missing_reason": "pilot_horizon"}
    runs = []
    for design in matrix_design():
        runs.append({
            **design,
            "trajectory_signature": "same-open-trajectory",
            "side_summaries": [{"side": 0}, {"side": 1}],
            "viability": [], "red_flags": [],
            "checkpoints": [{**sample_checkpoint, "side": side} for side in (0, 1)],
            "mirror_state_equal_every_tick": True,
        })
    summary = aggregate_runs(runs)
    assert summary["run_count"] == 144
    assert summary["condition_count"] == 48
    assert summary["unique_trajectory_count"] == 48
    assert summary["scheduled_duplicate_rows"] == 96
    assert all(row["unique_trajectory_count"] == 1 for row in summary["conditions"])
    assert all(row["unique_trajectories"][0]["duplicate_seed_rows"] == 2 for row in summary["conditions"])


def test_full_step_tick_pilot_is_mirrored_isolated_and_deterministic():
    config = build_config(2, "default")
    row = {
        "run_key": "integration-eggs-2-worker-single",
        "eggs_per_lay": 2,
        "lay_regime": "worker-first",
        "forage_regime": "single-local",
        "environment_profile": "default",
        "seed": 20261111,
        "topology_mode": DEFAULT_TOPOLOGY,
        "config_hash": config_hash(config),
    }
    first = _run_one(row, topology_mode=DEFAULT_TOPOLOGY, horizon=40)
    second = _run_one(row, topology_mode=DEFAULT_TOPOLOGY, horizon=40)
    assert first == second
    assert first["ticks_executed"] == 40
    assert first["pilot_cutoff"] and first["terminal_reason"] is None
    assert first["mirror_state_equal_every_tick"] and first["isolation_passed"]
    assert first["data_quality_status"] == "valid" and first["isolation_check_count"] == 40
    assert first["event_counts"].get("combat", 0) == 0
    assert first["event_counts"].get("queen_damage", 0) == 0
    assert first["side_totals"][0]["laid_batches"] >= 1
    assert first["side_totals"][0]["hatched_workers"] >= 2
    assert first["final_state"]["state_hash"] == second["final_state"]["state_hash"]
    assert first["final_state"]["checkpoint_hash"] == second["final_state"]["checkpoint_hash"]
