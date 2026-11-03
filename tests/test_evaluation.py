"""FOR-18 deterministic evaluation and full-path benchmark regression tests."""

import json

import pytest

from formicmind.cli import EXIT_IO, EXIT_OK, EXIT_USAGE, main as sim_main
from formicmind.evaluation import (
    EvaluationError,
    SingleCoreAffinity,
    controller_pair,
    evaluate_episodes,
    execute_episode,
    load_manifest,
    nearest_rank,
    run_benchmark,
)
from formicmind.match import run_episode
from formicmind.match_smoke import SCENARIOS, MATCH_SMOKE_SEED
from formicmind.rules import default_config
from formicmind.tick import ScriptedController


def test_streaming_episode_matches_the_existing_episode_runner():
    scenario = SCENARIOS["raid_win"]
    config = default_config()
    expected = run_episode(config, MATCH_SMOKE_SEED, ScriptedController(scenario.script_a),
                           ScriptedController(scenario.script_b))
    actual = execute_episode(MATCH_SMOKE_SEED, controller_pair("raid_win"), config=config)

    assert actual.winner == expected.winner
    assert actual.reason == expected.reason
    assert actual.terminal_tick == expected.terminal_tick
    assert actual.state_tick == expected.final_tick
    assert actual.tick_completed == expected.outcome.tick_completed
    assert actual.ticks_executed == expected.ticks_executed
    assert actual.ticks_executed == actual.terminal_tick + 1  # queen terminal is an incomplete executed tick
    assert actual.score == expected.score
    assert actual.queen_hp == expected.queen_hp
    assert actual.food == expected.food
    assert actual.workers == expected.workers
    assert actual.soldiers == expected.soldiers
    assert actual.brood == expected.brood
    assert actual.config_hash == expected.config_hash
    assert actual.state_hash == expected.state_hash
    assert actual.checkpoint_hash == expected.checkpoint_hash
    assert not hasattr(actual, "events")


def test_phase_timing_observer_does_not_change_the_simulation():
    pair = controller_pair("raid_win")
    expected = execute_episode(MATCH_SMOKE_SEED, pair)
    observations = []
    actual = execute_episode(MATCH_SMOKE_SEED, pair, on_phase_timing=lambda phase, ns: observations.append((phase, ns)))

    assert actual.state_hash == expected.state_hash
    assert actual.checkpoint_hash == expected.checkpoint_hash
    assert observations
    assert all(type(ns) is int and ns >= 0 for _, ns in observations)


def test_eval_seed_schedule_paired_seats_and_truncation_accounting():
    first, aggregate = evaluate_episodes(
        episodes=2, seed_start=900, scenario="mutual_raid_draw", paired_seat=True,
    )
    again, same_aggregate = evaluate_episodes(
        episodes=2, seed_start=900, scenario="mutual_raid_draw", paired_seat=True,
    )
    assert first == again
    assert aggregate == same_aggregate
    assert [row["seed"] for row in first] == [900, 900, 901, 901]
    assert [row["seat_assignment"]["A"].rsplit(":", 1)[-1] for row in first] == [
        "script_a", "script_b", "script_a", "script_b",
    ]
    assert aggregate["attempted"] == aggregate["finished"] == 4
    assert aggregate["draws"] == 4

    cutoffs, cutoff_aggregate = evaluate_episodes(
        episodes=2, seed_start=900, scenario="mutual_raid_draw", max_ticks=0,
    )
    assert all(row["truncated"] and row["winner"] is None and row["score"] is None for row in cutoffs)
    assert cutoff_aggregate["finished"] == 0
    assert cutoff_aggregate["truncated"] == 2
    assert cutoff_aggregate["draws"] == cutoff_aggregate["wins_A"] == cutoff_aggregate["wins_B"] == 0
    assert cutoff_aggregate["mean_score_finished_only"] is None


def test_nearest_rank_p10_and_median_contract():
    values = [3, 1, 5, 2, 4]
    assert nearest_rank(values, 0.10) == 1
    assert nearest_rank(values, 0.50) == 3
    with pytest.raises(EvaluationError):
        nearest_rank([], 0.10)


def test_benchmark_uses_exact_step_tick_budget_and_reports_boundary_cutoff():
    result = run_benchmark(
        runs=2, ticks_per_run=50, warmup_ticks=10, seed_start=900,
        scenario="mutual_raid_draw", pin_core=None,
    )
    measurement = result["measurement"]
    assert measurement["warmup"]["ticks"] == 10
    assert len(measurement["samples"]) == 2
    for sample in measurement["samples"]:
        assert sample["ticks"] == 50
        assert sample["seed_schedule_repeated"] is True
        assert "events" not in sample
        assert sample["full_terminal_episodes"] == 1
        assert sample["benchmark_boundary_cutoffs"] == 1
        assert sample["full_terminal_episodes"] + sample["benchmark_boundary_cutoffs"] == sample["episodes"]
    throughput = measurement["throughput"]
    assert throughput["ticks_per_second_p10"] == min(s["ticks_per_second"] for s in measurement["samples"])
    assert throughput["ticks_per_second_p50"] == pytest.approx(
        sorted(s["ticks_per_second"] for s in measurement["samples"])[0] / 2
        + sorted(s["ticks_per_second"] for s in measurement["samples"])[1] / 2,
    )


def test_phase_diagnostic_is_separate_from_sample_throughput():
    result = run_benchmark(
        runs=1, ticks_per_run=20, warmup_ticks=0, seed_start=901,
        scenario="technical", phase_timing_ticks=10,
    )
    measurement = result["measurement"]
    assert measurement["samples"][0]["ticks"] == 20
    diagnostic = measurement["phase_diagnostics"]
    assert diagnostic["ticks"] == 10
    assert diagnostic["excluded_from_throughput_samples"] is True
    assert sum(row["calls"] for row in diagnostic["phase_timings"].values()) > 0


def test_affinity_is_best_effort_and_restored():
    affinity = SingleCoreAffinity()
    with affinity:
        pass
    assert "affinity_applied" in affinity.report
    if affinity.report["affinity_applied"]:
        assert affinity.report["restored"] is True
    else:
        assert affinity.report["status"] == "not_applied"


def test_cli_run_eval_and_manifest_replay_write_reproducible_rows(tmp_path, capsys):
    run_path = tmp_path / "run.json"
    assert sim_main([
        "run", "--seed", "903", "--a", "scripted", "--b", "scripted", "--scenario", "raid_win",
        "--max-ticks", "10", "--json-out", str(run_path),
    ]) == EXIT_OK
    row = json.loads(run_path.read_text(encoding="utf-8"))["results"][0]
    assert row["ticks_executed"] == 10 and row["truncated"]
    assert "brood_A/B=" in capsys.readouterr().out

    manifest_path = tmp_path / "eval.manifest.json"
    first_path = tmp_path / "eval.json"
    assert sim_main([
        "eval", "--episodes", "2", "--seed-start", "904", "--a", "scripted", "--b", "scripted",
        "--scenario", "mutual_raid_draw", "--paired", "--manifest-out", str(manifest_path),
        "--json-out", str(first_path),
    ]) == EXIT_OK
    first = json.loads(first_path.read_text(encoding="utf-8"))
    second_path = tmp_path / "eval-repeat.json"
    assert sim_main([
        "eval", "--manifest", str(manifest_path), "--json-out", str(second_path),
    ]) == EXIT_OK
    second = json.loads(second_path.read_text(encoding="utf-8"))
    assert first["results"] == second["results"]
    assert first["aggregate"] == second["aggregate"]
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert capsys.readouterr().out.count("manifest_sha256=") == 2


def test_cli_benchmark_writes_raw_samples_and_invalid_inputs_have_nonzero_codes(tmp_path):
    result_path = tmp_path / "benchmark.json"
    manifest_path = tmp_path / "benchmark.manifest.json"
    assert sim_main([
        "benchmark", "--runs", "2", "--ticks-per-run", "15", "--warmup-ticks", "4", "--seed-start", "905",
        "--a", "scripted", "--b", "scripted", "--scenario", "technical", "--json-out", str(result_path),
        "--manifest-out", str(manifest_path),
    ]) == EXIT_OK
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert [sample["ticks"] for sample in result["measurement"]["samples"]] == [15, 15]
    assert result["measurement"]["throughput"]["gate0_status"] == "NOT MEASURED"
    repeat_path = tmp_path / "benchmark-repeat.json"
    assert sim_main([
        "benchmark", "--manifest", str(manifest_path), "--json-out", str(repeat_path),
    ]) == EXIT_OK
    repeated = json.loads(repeat_path.read_text(encoding="utf-8"))
    assert [sample["ticks"] for sample in repeated["measurement"]["samples"]] == [15, 15]
    assert repeated["manifest_sha256"] == result["manifest_sha256"]

    assert sim_main([
        "run", "--seed", "18446744073709551616", "--a", "scripted", "--b", "scripted",
    ]) == EXIT_USAGE
    assert sim_main([
        "run", "--seed", "1", "--a", "scripted", "--b", "scripted", "--json-out", str(tmp_path),
    ]) == EXIT_IO


def test_manifest_rejects_changes_to_contract_parameters(tmp_path):
    config = default_config()
    path = tmp_path / "manifest.json"
    params = {
        "scenario": "technical", "seed": 906, "max_ticks": None,
    }
    from formicmind.evaluation import build_manifest, write_json

    manifest = build_manifest("run", config, params)
    manifest["parameters"]["max_ticks"] = config.t_max + 1
    write_json(path, manifest)
    with pytest.raises(EvaluationError, match="max_ticks"):
        load_manifest(path, "run", config)
