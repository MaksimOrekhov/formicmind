"""FOR-43: the design calculator behind docs/ecosystem_v1_economy.md is deterministic and self-consistent.

It is a design aid, not the Ecosystem engine: nothing here imports ``formicmind`` and nothing touches L0.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ecosystem_budget.py"


@pytest.fixture(scope="module")
def budget():
    spec = importlib.util.spec_from_file_location("ecosystem_budget", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report(budget):
    return budget.report()


def test_the_calculator_does_not_import_the_engine():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "import formicmind" not in text and "from formicmind" not in text and "from ." not in text


def test_calculator_is_deterministic(budget):
    e = budget.Econ()
    sc = budget.scenario("moderate", 1.0)
    first = json.dumps(budget.simulate(sc, e, policy="greedy", ticks=1500), sort_keys=True, default=str)
    second = json.dumps(budget.simulate(sc, e, policy="greedy", ticks=1500), sort_keys=True, default=str)
    assert first == second
    assert json.dumps(budget.duel_grid(e), sort_keys=True) == json.dumps(budget.duel_grid(e), sort_keys=True)


def test_forager_rate_formula_and_break_even(budget):
    e = budget.Econ()
    assert budget.forager_rate(3, e) == pytest.approx(16 / 14)
    assert budget.forager_rate(10, e) == pytest.approx(16 / 28)
    assert budget.break_even_distance(e) == pytest.approx(16.0)
    assert budget.net_per_forager(16, e) == pytest.approx(0.0)
    assert budget.net_per_forager(17, e) < 0


def test_food_is_conserved_in_every_fluid_run(report):
    for run in report["dynamic"]:
        assert abs(run["ledger_residual"]) < 1e-5, run["scenario"]


def test_scarcity_ordering_of_carrying_capacity(budget):
    e = budget.Econ()
    capacity = {}
    for sc in budget.SCENARIOS:
        rows = budget.static_budget(sc, e)["classes"]
        capacity[sc.name] = rows[-1]["cumulative_WE_fed"]
    assert capacity["severe"] < capacity["moderate"] < capacity["abundance"]
    assert capacity["moderate"] == 60.0 and capacity["severe"] == 37.5 and capacity["abundance"] == 92.5


def test_initial_surplus_exists_in_moderate_and_abundant_worlds(budget):
    e = budget.Econ()
    moderate = budget.static_budget(budget.scenario("moderate", 1.0), e)
    severe = budget.static_budget(budget.scenario("severe", 0.6), e)
    assert moderate["start_surplus"] > 0 and moderate["start_utilisation"] < 0.6
    assert severe["start_surplus"] > 0 and severe["start_utilisation"] > 0.9  # almost no headroom


def test_peaceful_fair_share_has_no_famine_and_greedy_expansion_has(report):
    for run in report["dynamic"]:
        if run["policy"] == "fair":
            assert run["events"]["first_famine"] == -1 and run["starved_colonies"] == 0
        elif run["soldier_share"] == 0.0:
            assert run["events"]["first_famine"] > 0, "unchecked growth must starve without any war script"


def test_combat_rule_matches_l0(budget):
    even = budget.engagement(4, 4, 0)
    assert even["attacker_survivors"] == 0 and even["defender_soldier_survivors"] == 0  # mutual annihilation
    # home bonus: floor(2 * n * 5 / 4); one defender deals 2, two deal 5
    assert budget.engagement(1, 2, 0, defender_home=True)["attacker_survivors"] == 0
    # square law: doubling the force wins with far fewer losses
    big = budget.engagement(8, 4, 12)
    assert big["attacker_lost"] <= 2 and big["defender_soldiers_lost"] == 4


@pytest.fixture(scope="module")
def grid(report):
    return report["duel"]


def test_peace_by_labour_beats_a_fair_share_and_war_pays_only_in_a_narrow_window(grid):
    rows = grid
    attack_best = [r for r in rows if r["best"] == "attack"]
    assert attack_best, "war must be the best option somewhere, otherwise the economy is a peace script"
    assert len(attack_best) < len(rows) / 2, "and not in most cases"
    for r in rows:
        if r["third"] == 1:
            assert r["best"] != "attack", "a third colony at the patch makes attacking one neighbour a bad trade"
        if r["best"] == "attack":
            assert r["b_behavior"] in ("fair", "greedy") and r["third"] == 0


def test_a_built_army_rarely_pays_for_itself(grid):
    rows = grid
    built_attack = [r for r in rows if r["best_if_army_built"] == "attack"]
    existing_attack = [r for r in rows if r["best"] == "attack"]
    assert len(built_attack) < len(existing_attack)
    assert all(r["b_behavior"] == "greedy" and r["third"] == 0 and r["t_remaining"] >= 3000 for r in built_attack)


def test_retreat_beats_a_hopeless_attack_and_guarding_costs_without_threat(grid):
    for r in grid:
        weak = r["values"]["attack_weak"]["vs_peace"]
        assert weak < 0, "equal force is mutual annihilation: not attacking is strictly better"
        assert r["values"]["guarded_fair"]["vs_peace"] < 0, "a standing garrison is a pure cost without raids"


def test_longer_horizon_makes_a_profitable_attack_more_profitable(budget):
    e = budget.Econ()
    sc = budget.scenario("moderate", 1.0)
    short = budget.duel(sc, e, b_behavior="greedy", third=0, t_remaining=600)
    long = budget.duel(sc, e, b_behavior="greedy", third=0, t_remaining=3000)
    assert short["values"]["attack"]["vs_peace"] > 0
    assert long["values"]["attack"]["vs_peace"] > short["values"]["attack"]["vs_peace"]
    # a very short remainder cannot repay the losses
    tiny = budget.duel(sc, e, b_behavior="fair", third=0, t_remaining=60)
    assert tiny["best"] != "attack"


def test_guard_pays_only_if_raids_are_frequent(budget):
    g = budget.guard_break_even_raids(budget.Econ())
    assert 100 < g["break_even_one_raid_per_ticks"] < 400
    assert g["garrison_cost_per_tick"] == pytest.approx(1.6)


def test_finite_lifespan_costs_little_and_keeps_the_population(report):
    rows = [r for r in report["lifespan"] if r["policy"] == "fair"]
    inf = {r["scenario"]: r for r in rows if r["lifespan"] == "inf"}
    for r in rows:
        if r["lifespan"] in (2000, 4000):
            base = inf[r["scenario"]]
            assert abs(r["WE_6000"] - base["WE_6000"]) <= 0.12 * base["WE_6000"], r
            assert r["eggs_per_harvest"] <= 0.02 and r["alive"] == 8
        if r["lifespan"] == 2000:
            assert r["old_age_deaths"] > 0


def test_orphans_die_out_naturally_in_about_one_lifespan(report):
    rows = report["orphan"]
    for r in rows:
        assert r["others_alive"] == 7, "a queen death must not end the world or hurt the other colonies"
        assert 0.5 * r["lifespan"] <= r["ticks_after_queen_death"] <= 1.1 * r["lifespan"], r
    assert [r["extinct_tick"] for r in rows] == sorted(r["extinct_tick"] for r in rows)


def test_age_ledger_helpers_conserve_units(budget):
    e = budget.Econ()
    ledger = budget._seed_ages(10, 2000, e)
    assert sum(ledger.values()) == 10 and max(ledger) <= 0
    budget._book_deaths(ledger, 4)
    assert sum(ledger.values()) == 6
    budget._book_deaths(ledger, 99)
    assert ledger == {}


def test_share_gain_is_zero_when_the_patch_is_not_scarce(budget):
    assert budget.share_gain(100.0, 5.0, [5.0, 5.0], 0) == 0.0
    assert budget.share_gain(6.0, 6.0, [6.0], 0) == pytest.approx(3.0)


def test_main_prints_and_dumps_json(budget, capsys):
    assert budget.main(["--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert math.isclose(data["break_even_distance"], 16.0)


def test_age_bucket_semantics_nominal_lifespan_is_a_maximum(budget):
    e = budget.Econ()
    L, B = e.lifespan_worker, e.age_bucket
    # the first unit of a bucket lives exactly L, the last one B - 1 ticks less; the whole bucket dies together
    assert budget.lived_ticks(1200, L, B) == L
    assert budget.lived_ticks(1299, L, B) == L - (B - 1)
    assert budget.expiry_tick(1200, L, B) == budget.expiry_tick(1299, L, B) == 1200 + L
    lived = [budget.lived_ticks(h, L, B) for h in range(-500, 3000)]
    assert max(lived) == L and min(lived) == L - B + 1
    assert sum(lived) / len(lived) == pytest.approx(L - (B - 1) / 2, abs=0.6)  # mean ~ L - 49.5
    # negative start ages use floor division too: age 150 -> bucket -200, expiry L - 200
    assert budget.bucket_start(-150, B) == -200 and budget.expiry_tick(-150, L, B) == L - 200


def test_the_simulation_kills_a_bucket_exactly_at_its_expiry_tick(budget):
    e = budget.Econ(start_age_spread=0.5)
    sc = budget.scenario("moderate", 1.0)
    # colony 0 loses its queen at tick 0 (no new adults): only the seeded ages decide when its start adults die
    r = budget.simulate(sc, e, policy="fair", ticks=2100, queen_death={0: 0}, trace=True)
    adults = {t: v[0][0] + v[0][1] for t, v in r["series"]}
    seeds = list(budget._seed_ages(e.start_workers, e.lifespan_worker, e)) + list(budget._seed_ages(e.start_soldiers, e.lifespan_soldier, e))
    first = min(budget.expiry_tick(b, e.lifespan_worker, e.age_bucket) for b in seeds)
    last = max(budget.expiry_tick(b, e.lifespan_worker, e.age_bucket) for b in seeds)
    assert adults[950] == e.start_workers + e.start_soldiers and first >= 1000
    assert adults[last - (last % 50)] <= adults[950] and adults[2050] == 0 and last <= 2000
