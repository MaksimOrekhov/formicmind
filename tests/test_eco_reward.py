"""FOR-49: the per-tick viability trace, the time-aware interval reward and the variable-interval GAE of contracts 9.1 / 9.2 / 9.2a (T-21, T-22)
on the real engine, checked against the independent reference arithmetic ``scripts/ecosystem_reward_ref.py``. No torch here."""

from __future__ import annotations

import copy
import importlib.util
import math
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from formicmind.eco import default_config
from formicmind.eco.engine import RouteTable, step_world_tick
from formicmind.eco.fixtures import fixture_controller
from formicmind.eco.initial import build_world
from formicmind.eco.rl.play import colony_metrics, play_world
from formicmind.eco.viability import (
    DEATH_PENALTY, GAMMA_30, REWARD_VERSION, REWARD_VERSIONS, Q, IntervalStep, ViabilityTracker, gae_intervals, gamma_tick, interval_discount,
    interval_rewards, viability_q,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ecosystem_reward_ref.py"
H = 6000


@pytest.fixture(scope="module")
def ref():
    spec = importlib.util.spec_from_file_location("ecosystem_reward_ref_for49", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _wave(n=H):
    return [min(Q, int(Q * (0.5 + 0.4 * math.sin(t / 211.0)) * (0.5 if 2500 <= t < 3100 else 1.0))) for t in range(n)]


def _partitions(ref):
    return {"200x30": ref.partition(H), "600x10": ref.partition(H, base=10),
            "mixed": ref.partition(H, alert_windows=[(900, 1200), (4000, 4090), (5800, 5990)]),
            "irregular": [0, 7, 30, 31, 100, 1500, 2999, 3000, 5500]}


def _intervals(ticks, end):
    return [(start, (ticks[i + 1] if i + 1 < len(ticks) else end)) for i, start in enumerate(ticks)]


# ------------------------------------------------------------------------------------------------ T-21 / T-22 on the production functions


def test_interval_rewards_match_the_reference_for_every_split_and_pay_the_penalty_once(ref):
    trace = _wave()
    for name, ticks in _partitions(ref).items():
        for death in (None, 4321):
            end = H if death is None else death
            kept = [t for t in ticks if t < end]
            reference = ref.build_transitions(trace, kept, H, death_tick=death)
            mine = []
            for i, (start, stop) in enumerate(_intervals(kept, end)):
                terminal = death is not None and i == len(kept) - 1
                mine.append(interval_rewards(trace[start:stop], H, terminated=terminal))
            for (r, rd), t in zip(mine, reference):
                assert r == pytest.approx(t.reward, abs=1e-12) and rd == pytest.approx(t.reward_disc, abs=1e-12), (name, death)
            total = sum(r for r, _ in mine)
            assert total == pytest.approx(sum(trace[:end]) / (Q * H) - (DEATH_PENALTY if death is not None else 0.0), abs=1e-12), (name, death)


def test_the_discounted_return_is_the_tick_level_one_for_every_split(ref):
    trace = _wave()
    for death in (None, 3000):
        end = H if death is None else death
        expected = ref.discounted_return_ticks(trace, H, death_tick=death)
        for name, ticks in _partitions(ref).items():
            kept = [t for t in ticks if t < end]
            total, scale = 0.0, 1.0
            for i, (start, stop) in enumerate(_intervals(kept, end)):
                r, rd = interval_rewards(trace[start:stop], H, terminated=death is not None and i == len(kept) - 1)
                total += scale * rd
                scale *= interval_discount(stop - start)
            assert total == pytest.approx(expected, abs=1e-12), (name, death)


def test_gamma_is_time_aware_and_alerts_add_nothing():
    assert interval_discount(30) == pytest.approx(GAMMA_30) and interval_discount(10) ** 3 == pytest.approx(GAMMA_30, rel=1e-12)
    assert gamma_tick() ** 30 == pytest.approx(GAMMA_30, rel=1e-12)
    constant = [Q // 2] * H
    calm = sum(interval_rewards(constant[s:s + 30], H, terminated=False)[0] for s in range(0, H, 30))
    fighting = sum(interval_rewards(constant[s:s + 10], H, terminated=False)[0] for s in range(0, H, 10))
    assert calm == pytest.approx(fighting, abs=1e-12) == pytest.approx(0.5, abs=1e-12)


def test_gae_matches_the_reference_for_variable_intervals_terminal_and_truncated(ref):
    trace = _wave()
    for name, ticks in _partitions(ref).items():
        for death in (None, 4321):
            end = H if death is None else death
            kept = [t for t in ticks if t < end]
            reference = ref.build_transitions(trace, kept, H, death_tick=death)
            values = [0.3 + 0.2 * math.sin(i / 7.0) for i in range(len(kept))]
            steps = []
            for i, (start, stop) in enumerate(_intervals(kept, end)):
                terminal = death is not None and i == len(kept) - 1
                steps.append(IntervalStep(stop - start, interval_rewards(trace[start:stop], H, terminated=terminal)[1], terminal, values[i]))
            adv, ret = gae_intervals(steps, 0.7)
            ref_adv, ref_ret = ref.gae(reference, values, 0.7)
            assert adv == pytest.approx(ref_adv, abs=1e-12) and ret == pytest.approx(ref_ret, abs=1e-12), (name, death)


def test_bootstrap_is_the_critic_and_only_after_a_truncation():
    steps = [IntervalStep(30, 0.01, False, 0.5), IntervalStep(30, 0.01, False, 0.4)]
    low, _ = gae_intervals(steps, 0.0)
    high, _ = gae_intervals(steps, 5.0)
    assert high[1] - low[1] == pytest.approx(interval_discount(30) * 5.0)          # a truncated trajectory leans on the critic's value
    dead = [IntervalStep(30, 0.01, False, 0.5), IntervalStep(12, -1.0, True, 0.4)]
    assert gae_intervals(dead, 0.0)[0] == gae_intervals(dead, 123.0)[0]            # a terminal one ignores whatever value is offered
    assert gae_intervals(dead, 0.0)[0][1] == pytest.approx(-1.0 - 0.4)             # terminal: delta = r - V(o), no continuation
    # an exact critic (V_i = r_i + gamma_i V_{i+1}) gives zero advantage whatever the split
    deltas, rewards = (30, 10, 20, 7), (0.02, 0.011, 0.015, 0.003)
    values, value = [], 0.0
    for delta, reward in reversed(list(zip(deltas, rewards))):
        value = reward + interval_discount(delta) * value
        values.append(value)
    values.reverse()
    exact = [IntervalStep(d, r, False, v) for d, r, v in zip(deltas, rewards, values)]
    assert all(abs(a) < 1e-15 for a in gae_intervals(exact, 0.0)[0])


# ------------------------------------------------------------------------------------------------ the viability trace on the real engine


def test_viability_formula_and_bounds():
    full = 60 * Q
    assert viability_q(full, 0, 0, 0, True) == Q                       # no burn: infinite runway; saturated population
    assert viability_q(full, 0, 0, 0, False) == 0                      # no queen
    assert viability_q(0, 0, 5 * Q, 0, True) == 0                      # nobody, no food, burning
    assert viability_q(full, 0, 5 * Q, 200, True) == (6 * Q + 4 * Q) // 10     # runway 40 ticks saturates security
    assert viability_q(full, 0, 5 * Q, 100, True) == (6 * Q + 4 * (Q // 2)) // 10   # runway 20 -> security 0.5
    assert viability_q(full, 3 * Q, 5 * Q, 100, True) == (6 * Q + 4 * Q) // 10      # a steady income of 3 stretches 100 food to 50 ticks
    assert viability_q(full // 2, 0, 0, 0, True) == (6 * (Q // 2) + 4 * Q) // 10    # half the population cap
    assert all(0 <= viability_q(p, i, b, f, True) <= Q for p in (0, Q, 40 * Q, 99 * Q) for i in (0, Q, 9 * Q) for b in (0, 3 * Q, 9 * Q)
               for f in (0, 1, 80, 600))


# ------------------------------------------------------------------------------------------------ reward v2: no pay for a hollow colony (contracts 9.1)


def test_reward_versions_are_separate_and_the_current_one_is_two():
    assert REWARD_VERSION == 2 and REWARD_VERSIONS == (1, 2)
    assert viability_q(0, 0, 0, 600, True, 1) == 4 * Q // 10           # v1 (first pilot): nobody alive, food in the store -> 0.4 forever
    assert viability_q(0, 0, 0, 600, True, 2) == 0                      # v2: the same colony earns nothing
    assert viability_q(0, 0, 0, 600, True) == viability_q(0, 0, 0, 600, True, REWARD_VERSION)
    with pytest.raises(ValueError):
        viability_q(0, 0, 0, 600, True, 3)
    with pytest.raises(ValueError):
        ViabilityTracker(build_world(default_config(8, "moderate"), 50012).state, 0)


def test_security_is_gated_by_min_one_epop_over_ten():
    full = 60 * Q
    for e_pop, gate in ((0, 0), (1, 0.1), (5, 0.5), (9, 0.9), (10, 1), (30, 1)):
        v = viability_q(int(e_pop * Q), 0, 0, 600, True)               # unlimited runway: security would be 1 without the gate
        expected = 0.6 * min(1, e_pop / 60) + 0.4 * gate
        assert abs(v / Q - expected) < 1e-3, (e_pop, v / Q, expected)
    # a colony that is really there keeps its old score: the gate is open from 10 workers-equivalents up
    for version in (1, 2):
        assert viability_q(full, 0, 5 * Q, 100, True, version) == (6 * Q + 4 * (Q // 2)) // 10
    assert viability_q(10 * Q, 0, 5 * Q, 100, True, 1) == viability_q(10 * Q, 0, 5 * Q, 100, True, 2)
    # hoarding is no strategy under v2: a hollow colony is worth at most what its tiny population is worth
    assert viability_q(Q // 2, 0, 0, 600, True, 2) < Q // 20


def test_viability_matches_the_independent_exact_reference_for_both_versions(ref):
    checked = 0
    for version in REWARD_VERSIONS:
        for pop_q in (0, Q // 4, Q, 5 * Q, 9 * Q + 12345, 10 * Q, 24 * Q, 60 * Q, 90 * Q):
            for income_q, burn_q in ((0, 0), (0, 3 * Q), (2 * Q, 3 * Q), (4 * Q, 3 * Q), (Q // 3, 5 * Q + 7)):
                for food in (0, 1, 80, 233, 600):
                    for alive in (True, False):
                        got = viability_q(pop_q, income_q, burn_q, food, alive, version)
                        want = ref.viability_q_exact(Fraction(pop_q, Q), Fraction(income_q, Q), Fraction(burn_q, Q), food, alive, version)
                        assert abs(got - want) <= 3, (version, pop_q, income_q, burn_q, food, alive, got, want)
                        checked += 1
    assert checked == 2 * 9 * 5 * 5 * 2
    with pytest.raises(ValueError):
        ref.viability_exact(1, 0, 0, 1, True, version=3)
    # the reference alone already says: hollow + full store = 0.4 under v1, 0 under v2
    assert ref.viability_exact(0, 0, 0, 600, True, 1) == Fraction(2, 5) and ref.viability_exact(0, 0, 0, 600, True, 2) == 0


def _hollow_trace(version: int, ticks: int) -> list[int]:
    """Colony 0 of a real map loses every adult and all brood, keeps a full store and its queen; the world around it is not simulated."""
    world = build_world(default_config(8, "moderate"), 50012)
    state = world.state
    state.cohorts = [c for c in state.cohorts if c.colony_id != 0]
    state.colonies[0].brood = []
    state.colonies[0].food = 600
    tracker = ViabilityTracker(state, version)
    for _ in range(ticks):
        record = SimpleNamespace(tick=state.tick, delivered_by_colony={}, upkeep_due_by_colony={})
        state.tick += 1
        tracker.update(state, record)
    return tracker.traces[0].v_q


def test_a_hollow_colony_on_a_full_store_is_paid_by_the_old_reward_and_not_by_the_new_one():
    old, new = _hollow_trace(1, 2400), _hollow_trace(2, 2400)
    assert old[-1] >= 0.39 * Q                                          # the exploit of the first pilot, reproduced exactly
    assert new[-1] < 0.01 * Q                                           # the population EMA decays, nothing replaces it
    assert all(n <= o for n, o in zip(new, old))
    assert sum(new[1200:]) / (Q * 1200) < 0.02 and sum(old[1200:]) / (Q * 1200) > 0.39   # the tail of the run: nothing against 0.4 forever
    assert sum(new) < 0.25 * sum(old)


def _run(config, seed, ticks, roles=None):
    return play_world(config, seed, fixture_controller("peaceful_fair", roles), ticks)


@pytest.fixture(scope="module")
def raid():
    """Seed 50004 with colony 3 raiding: colony 0 loses its queen at tick 2621 (docs/ecosystem_v1_interaction.md) and dies out at tick 4400."""
    return _run(default_config(8, "moderate"), 50004, 2700, {3: "raider"})


def test_the_trace_ends_in_the_tick_of_death_and_pays_the_penalty_once(raid):
    tracker = raid.tracker
    assert tracker.death_tick(0) == 2622 and len(tracker.traces[0].v_q) == 2622 and tracker.traces[0].v_q[-1] == 0
    assert all(tracker.death_tick(c) is None and len(tracker.traces[c].v_q) == 2700 for c in range(1, 8))
    metrics = colony_metrics(raid)
    assert metrics[0]["survived"] is False and metrics[0]["death_tick"] == 2621
    assert metrics[0]["reward"] == pytest.approx(sum(tracker.traces[0].v_q) / (Q * H) - 1.0, abs=1e-12)
    assert all(m["reward"] == pytest.approx(sum(tracker.traces[m["colony"]].v_q) / (Q * H), abs=1e-12) for m in metrics[1:])
    assert all(0 <= v <= Q for trace in tracker.traces for v in trace.v_q)


def test_the_report_carries_both_rewards_and_the_behaviour_counts(raid):
    """Evaluation looks at what the colonies did, not only at the reward: workers over time, brood, eggs, combat, and the first pilot's reward."""
    metrics = colony_metrics(raid)
    for row in metrics:
        assert row["reward_v1"] is not None and row["reward_v1"] >= row["reward"] - 1e-9        # v2 never pays more than v1 for the same run
        assert row["workers_peak"] >= row["workers_mean"] > 0 and row["eggs_laid"] >= 0 and row["brood_end"] >= 0
    assert metrics[0]["combat"] > 0 and metrics[3]["combat"] > 0 and metrics[0]["queen_damage"] > 0       # the raid: colony 3 against colony 0
    assert sum(1 for row in metrics if row["combat"] == 0) >= 3


def test_a_trace_matches_an_independent_recomputation_from_the_raw_per_tick_data():
    config = default_config(8, "moderate")
    world = build_world(config, 50012)
    routes, controller = RouteTable(world.world_map), fixture_controller("peaceful_fair")
    tracker = ViabilityTracker(world.state)
    pop = [c.rolling.pop_ema_q for c in world.state.colonies]
    income = [0] * 8
    burn = [c.rolling.burn_ema_q for c in world.state.colonies]
    expected = [[] for _ in range(8)]

    def ema(prev, x, tau):                     # the documented step, written independently: round half up of prev + (x - prev) / tau
        return prev + (2 * (x - prev) + tau) // (2 * tau)

    for _ in range(900):
        record = step_world_tick(world.state, world.run, config, world.world_map, controller, routes=routes, validate_input=world.state.tick == 0)
        tracker.update(world.state, record)
        for c in world.state.colonies:
            i = c.colony_id
            w = sum(x.count for x in world.state.cohorts if x.colony_id == i and x.caste == "worker")
            s = sum(x.count for x in world.state.cohorts if x.colony_id == i and x.caste == "soldier")
            pop[i] = ema(pop[i], (2 * w + s) * Q // 2, 300)
            income[i] = ema(income[i], record.delivered_by_colony.get(i, 0) * Q, 120)
            burn[i] = ema(burn[i], record.upkeep_due_by_colony.get(i, 0) * Q, 120)
            security = Q if burn[i] <= income[i] else min(Q, c.food * Q * Q // (40 * (burn[i] - income[i])))
            security = security * min(Q, pop[i] // 10) // Q          # reward v2: the population gate of security
            expected[i].append((6 * min(Q, pop[i] // 60) + 4 * security) // 10 if c.queen_hp > 0 else 0)
    assert [t.v_q for t in tracker.traces] == expected
    assert any(max(t.v_q) > 0 for t in tracker.traces)


def test_the_reward_reads_only_the_colonys_own_state():
    """Poison everything that is not colony 0's: other colonies' food, queens and cohorts, and their entries in the tick record."""
    config = default_config(8, "moderate")
    worlds = []
    for _ in range(2):
        world = build_world(config, 50012)
        worlds.append((world, RouteTable(world.world_map), fixture_controller("peaceful_fair"), ViabilityTracker(world.state)))
    for _ in range(200):
        records = []
        for world, routes, controller, _tracker in worlds:
            records.append(step_world_tick(world.state, world.run, config, world.world_map, controller, routes=routes,
                                           validate_input=world.state.tick == 0))
        (a, _, _, ta), (b, _, _, tb) = worlds
        ra, rb = records
        poisoned = copy.deepcopy(b.state)
        for colony in poisoned.colonies[1:]:
            colony.food, colony.queen_hp, colony.status = 999, 0, "orphan"
        poisoned.cohorts = [c for c in poisoned.cohorts if c.colony_id == 0]
        rb2 = copy.deepcopy(rb)
        rb2.delivered_by_colony = {k: (v if k == 0 else 777) for k, v in rb.delivered_by_colony.items()}
        rb2.upkeep_due_by_colony = {k: (v if k == 0 else 555) for k, v in rb.upkeep_due_by_colony.items()}
        ta.update(a.state, ra)
        tb.update(poisoned, rb2)
        assert ta.traces[0].v_q == tb.traces[0].v_q
        # undo the poison for the next tick: tb must keep following the real world for colony 0 only
        tb.traces[1:] = copy.deepcopy(ta.traces[1:])
        for t in (tb.traces[0],):
            t.death_tick = ta.traces[0].death_tick


def test_a_world_cut_by_the_horizon_is_truncated_not_terminal():
    result = _run(default_config(8, "moderate"), 50012, 400)
    assert all(t.death_tick is None and len(t.v_q) == 400 for t in result.tracker.traces)
    assert result.state.world_outcome.finished is False
