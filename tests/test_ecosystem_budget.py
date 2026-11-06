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


def test_report_is_deterministic(budget, report):
    first = json.dumps(report, sort_keys=True, default=str)
    second = json.dumps(budget.report(), sort_keys=True, default=str)
    assert first == second


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


def test_war_economics_depend_on_garrison_and_third_parties(budget):
    e = budget.Econ()
    kwargs = dict(supply=6, attackers=8, rival_soldiers=4, rival_foragers=8, my_foragers=8, distance=7,
                  remaining_ticks=3000)
    free = budget.war_roi(e, garrison=0, **kwargs)
    held = budget.war_roi(e, garrison=4, **kwargs)
    crowded = budget.war_roi(e, garrison=2, other_rivals=1, **kwargs)
    assert free["profitable_over_remaining"] and not held["profitable_over_remaining"]
    assert free["net_per_tick"] > held["net_per_tick"]
    assert crowded["gain_per_tick"] < budget.war_roi(e, garrison=2, **kwargs)["gain_per_tick"]


def test_share_gain_is_zero_when_the_patch_is_not_scarce(budget):
    assert budget.share_gain(100.0, 5.0, [5.0, 5.0], 0) == 0.0
    assert budget.share_gain(6.0, 6.0, [6.0], 0) == pytest.approx(3.0)


def test_main_prints_and_dumps_json(budget, capsys):
    assert budget.main(["--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert math.isclose(data["break_even_distance"], 16.0)
