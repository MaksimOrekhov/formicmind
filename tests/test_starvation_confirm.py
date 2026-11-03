"""FOR-35 harness tests (isolated study module; no engine rule is changed). Real matches here use seeds that
are NOT in the preregistered schedule, so no result of the study is looked at while testing."""

from __future__ import annotations

import copy
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from formicmind import economy as economy_module
from formicmind import starvation_confirm as sc
from formicmind import tick as tick_module

ROOT = Path(__file__).resolve().parents[1]
OFF_SCHEDULE_SEED = 20270301  # used by FOR-25, never by FOR-35's schedule


# --- schedule, frozen constants ---------------------------------------------------------------------

def test_schedule_is_exact_and_frozen():
    assert sc.EXPECTED_ROWS == 4800 == len(sc.schedule_row_ids()) == len(set(sc.schedule_row_ids()))
    assert sc.EXPECTED_ANCHORS == 960 == len(sc.schedule_groups())
    assert len(sc.VARIANTS) == 5 and sc.VARIANTS[0] == "A0"
    assert len(sc.MAPS) == 6 and len(sc.PAIRS) == 10 and len(sc.SEEDS) == 8 and len(sc.SEATS) == 2
    assert sc.SEEDS == tuple(range(20270501, 20270509))
    assert sc._canon_sha(list(sc.SEEDS)) == sc.SEEDS_SHA256
    assert sc._canon_sha(list(sc.SEATS)) == sc.SEATS_SHA256
    assert sc.MAIN_BUDGET == 14_400_000 and sc.ANCHOR_BUDGET == 2_880_000
    assert sc.PAIRS[0] == ("random", "turtle") and sc.PAIRS[-1] == ("balanced", "raider")


def test_configs_and_policies_match_the_protocol():
    for name, expected in sc.CONFIG_SHA256.items():
        assert sc.config_hash(sc._configs()[name]) == expected
    for name, descriptor in sc._policy_descriptors().items():
        assert descriptor["parameter_sha256"] == sc.POLICY_PARAMETER_SHA256[name]
        assert descriptor["policy_id"] == f"{name}-v2"


def test_frozen_documents_and_engine_fingerprint():
    assert sc._file_sha(ROOT / sc.PROTOCOL_PATH) == sc.PROTOCOL_SHA256
    for path, digest in sc.ERRATA.values():
        assert sc._file_sha(ROOT / path) == digest
    assert sc.source_fingerprint(exclude_harness=True) == sc.BASE_ENGINE_FINGERPRINT


def test_new_seeds_do_not_overlap_any_other_file():
    assert sc._scan_seed_collisions() == []
    used = set(range(20261102, 20261108)) | set(range(20270301, 20270308)) | set(range(20270201, 20270208))
    assert not set(sc.SEEDS) & used


def test_package_is_imported_from_this_checkout():
    import formicmind
    assert Path(formicmind.__file__).resolve() == (ROOT / "src" / "formicmind" / "__init__.py").resolve()


# --- debt rules and stage M -------------------------------------------------------------------------

@pytest.mark.parametrize("variant,due,food,debt,streak,expected", [
    ("A0", 5, 0, 3, 0, (0, 0, True, 0)),
    ("A0", 0, 9, 3, 0, (3, 9, False, 0)),
    ("A0", 5, 7, 0, 0, (0, 7, False, 0)),
    ("A1a", 5, 0, 3, 0, (3, 0, False, 0)),
    ("A1a", 5, 1, 3, 0, (0, 1, True, 0)),
    ("A1a", 0, 4, 3, 0, (3, 4, False, 0)),
    ("A1b", 5, 4, 3, 0, (3, 4, False, 0)),
    ("A1b", 5, 5, 3, 0, (0, 5, True, 0)),
    ("A2", 5, 2, 3, 0, (1, 0, False, 0)),
    ("A2", 5, 40, 3, 0, (0, 37, False, 0)),
    ("A2", 0, 4, 3, 0, (0, 1, False, 0)),
    ("A2", 5, 0, 3, 0, (3, 0, False, 0)),
    ("A3b", 5, 0, 3, 8, (3, 0, False, 9)),
    ("A3b", 5, 0, 3, 9, (0, 0, True, 0)),
    ("A3b", 0, 0, 3, 9, (3, 0, False, 9)),
    ("A3b", 5, 0, 0, 9, (0, 0, False, 10)),
])
def test_debt_rule_branches(variant, due, food, debt, streak, expected):
    assert sc.debt_rule(variant, due=due, food=food, debt=debt, streak=streak) == expected


def test_unknown_variant_is_rejected():
    with pytest.raises(sc.StarvationConfirmError):
        sc.debt_rule("A9", due=1, food=0, debt=1, streak=0)


def test_a0_equals_the_unchanged_upkeep_on_random_states():
    rng = random.Random(35)
    for _ in range(300):
        setup = dict(workers=rng.randint(0, 40), soldiers=rng.randint(0, 6), food=rng.randint(0, 30),
                     debt=rng.randint(0, 12), rem=rng.randint(0, 9))
        assert sc.single_tick("A0", **setup) == sc.single_tick(None, **setup)


@pytest.mark.parametrize("name", sorted(sc.EDGE_CASES))
def test_edge_cases_are_consistent(name):
    setup = sc.EDGE_CASES[name]
    assert sc.single_tick("A0", **setup) == sc.single_tick(None, **setup)
    for variant in sc.VARIANTS:
        r = sc.single_tick(variant, **setup)
        assert r["paid"] + r["missing"] == r["due"] and r["debt_after"] >= 0 and r["deaths"] >= 0


def test_threshold_edge_cases():
    assert sc.single_tick("A0", workers=10, food=0, debt=0, rem=0)["deaths"] == 0       # debt 4
    assert sc.single_tick("A0", workers=12, food=0, debt=0, rem=2)["deaths"] == 1        # debt 5
    nine = sc.single_tick("A0", workers=12, food=0, debt=4, rem=2)
    assert (nine["deaths"], nine["debt_after"]) == (1, 4)
    fourteen = sc.single_tick("A0", workers=25, food=0, debt=4, rem=0)
    assert (fourteen["deaths"], fourteen["debt_after"]) == (2, 4)
    last = sc.single_tick("A0", workers=1, food=0, debt=100, rem=9)
    assert last["adults_left"] == 0 and last["debt_after"] > 0


def test_tenths_do_not_get_lost():
    state, run = sc._toy_state(1, 0, food=100, debt=0, rem=0)
    dues = []
    for t in range(10):
        state.tick = t
        up, _ = sc._side0(economy_module.upkeep(state, run))
        dues.append(up.due)
    assert sum(dues) == 4 and state.colonies[0].upkeep_remainder10 == 0


def test_stage_m_reproduces_the_mechanism():
    demo = sc.build_minimal_demo()
    assert demo["k1_mechanism_reproduced"] is True
    m1 = demo["scenarios"]["M1"]
    a0 = m1["A0"]["totals"]
    assert a0["deaths"] == 0 and a0["missing"] > 0 and a0["forgiven"] == a0["missing"]
    assert demo["scenarios"]["M4"]["A0"]["totals"]["deaths"] > 0
    assert all(v["totals"]["deaths"] == 0 and v["totals"]["missing"] == 0 for v in [demo["scenarios"]["M3"][x]
                                                                                  for x in sc.VARIANTS])
    assert m1["A1a"]["totals"]["forgiven"] > 0 and demo["k3_mechanism_removed"]["A1a"] is False
    for variant in ("A1b", "A2", "A3b"):
        assert demo["k3_mechanism_removed"][variant] is True
    for scenario in demo["scenarios"].values():
        assert all(v["debt_identity_holds"] for v in scenario.values())


def test_variants_are_restored_even_after_an_exception():
    original = tick_module.upkeep
    with pytest.raises(RuntimeError):
        with sc.use_variant("A1b"):
            assert tick_module.upkeep is not original
            raise RuntimeError("boom")
    assert tick_module.upkeep is original is economy_module.upkeep
    with sc.use_variant("A0"):
        with pytest.raises(sc.StarvationConfirmError):
            with sc.use_variant("A2"):
                pass
    assert tick_module.upkeep is economy_module.upkeep
    with pytest.raises(sc.StarvationConfirmError):
        with sc.use_variant("A9"):
            pass


# --- windows and the primary stagnation measure ------------------------------------------------------

def _series(completed=3000, adults=20, brood=0, missing=0, forgiven=0, hatches=0, dst=0, dcb=0):
    s = sc.SideSeries()
    s.adults = [adults] * (completed + 1)
    s.brood = [brood] * (completed + 1)
    s.queen = [1] * (completed + 1)
    for name in ("due", "paid", "missing", "forgiven", "repaid", "starvation_deaths", "combat_deaths", "hatches"):
        setattr(s, name, [0] * completed)
    s.missing = [missing] * completed
    s.forgiven = [forgiven] * completed
    s.hatches = [hatches] * completed
    s.starvation_deaths = [dst] * completed
    s.combat_deaths = [dcb] * completed
    return s


def _field(window, name):
    return window[sc.SIDE_WINDOW_FIELDS.index(name)]


def test_incomplete_windows_are_not_measured_rather_than_zero():
    windows = sc.analyze_side_windows(_series(completed=700), 700)
    assert windows[0] is not None and windows[1] is not None and windows[2] is None
    assert sc.analyze_side_windows(_series(completed=299), 299) == [None] * sc.N_WINDOWS


@pytest.mark.parametrize("delta,flat", [(2, True), (3, False), (-2, True), (-3, False)])
def test_pop_flat_boundary(delta, flat):
    s = _series(completed=600)
    s.adults = [20] * 300 + [20 + delta] * 301
    w = sc.analyze_side_windows(s, 600)[0]
    assert _field(w, "pop_flat") == int(flat)


@pytest.mark.parametrize("hatches,deaths,churn", [(3, 3, 1), (2, 3, 0), (3, 2, 0)])
def test_churn_boundary(hatches, deaths, churn):
    s = _series(completed=300)
    s.hatches[0], s.starvation_deaths[1] = hatches, deaths
    assert _field(sc.analyze_side_windows(s, 300)[0], "churn") == churn


def test_churn_counts_combat_deaths_too():
    s = _series(completed=300)
    s.hatches[0], s.starvation_deaths[1], s.combat_deaths[2] = 3, 1, 2
    assert _field(sc.analyze_side_windows(s, 300)[0], "churn") == 1


@pytest.mark.parametrize("peak_gain,new_peak", [(2, 0), (3, 1)])
def test_new_peak_margin(peak_gain, new_peak):
    s = _series(completed=300)
    s.brood = [0] * 150 + [peak_gain] * 151  # N rises to 20 + gain mid-window and stays
    assert _field(sc.analyze_side_windows(s, 300)[0], "new_peak") == new_peak


def test_a_cycling_colony_is_stalled_but_a_growing_one_is_not():
    cycle = _series(completed=600)
    cycle.adults = [32 if (m // 100) % 2 == 0 else 24 for m in range(601)]
    cycle.hatches, cycle.starvation_deaths = [1] * 600, [1] * 600
    w = sc.analyze_side_windows(cycle, 600)
    assert _field(w[1], "new_peak") == 0 and _field(w[1], "churn") == 1 and _field(w[1], "stalled") == 1
    growing = _series(completed=600)
    growing.adults = [20 + m // 20 for m in range(601)]
    assert _field(sc.analyze_side_windows(growing, 600)[1], "stalled") == 0


@pytest.mark.parametrize("ticks,deficit", [(29, 0), (30, 1)])
def test_deficit_window_boundary(ticks, deficit):
    s = _series(completed=300)
    s.missing = [1] * ticks + [0] * (300 - ticks)
    assert _field(sc.analyze_side_windows(s, 300)[0], "deficit") == deficit


def _match(first, second, conflict=None, completed=300):
    a = sc.analyze_side_windows(first, completed)
    b = sc.analyze_side_windows(second, completed)
    return a, b, sc.analyze_match_windows(a, b, conflict or [0] * completed, completed)


def test_stagnation_needs_both_sides_stalled_and_no_conflict():
    a, b, w = _match(_series(), _series(), completed=300)
    assert w[0] == [0, 1, 0, 1]
    quiet = [0] * 300
    quiet[17] = 1
    assert _match(_series(), _series(), quiet)[2][0][1] == 0
    growing = _series(completed=300)
    growing.adults = [20 + m // 10 for m in range(301)]
    assert _match(growing, _series())[2][0][1] == 0


def test_stagnation_with_deficit_vs_saturated():
    deficit = _series(completed=300)
    deficit.missing = [1] * 30 + [0] * 270
    assert _match(deficit, _series())[2][0] == [0, 1, 1, 0]
    assert _match(_series(), _series())[2][0] == [0, 1, 0, 1]


def test_primary_stagnation_ignores_forgiveness_and_debt():
    base = _match(_series(), _series())[2]
    forgiving = _series(completed=300, forgiven=7)
    assert _match(forgiving, _series(completed=300, forgiven=3))[2] == base
    a, b, w = _match(_series(missing=1, forgiven=1), _series(missing=1, forgiven=0))
    c, d, w2 = _match(_series(missing=1, forgiven=0), _series(missing=1, forgiven=1))
    assert w == w2


def test_derived_counts_and_attribution():
    first = [[20] * len(sc.SIDE_WINDOW_FIELDS) for _ in range(sc.N_WINDOWS)]
    si = sc.SIDE_WINDOW_FIELDS.index
    for k, row in enumerate(first):
        row[si("missing")], row[si("forgiven")] = 10, 6
    second = copy.deepcopy(first)
    windows = [[0, 1, 1, 0] for _ in range(5)] + [[0, 1, 0, 1]] * 3 + [[1, 0, 0, 0]] + [None]
    d = sc.derive_stagnation(windows, first, second)
    assert (d["stagnant"], d["stagnant_deficit"], d["stagnant_sat"]) == (8, 5, 3)
    assert d["sd_missing"] == 5 * 20 and d["sd_forgiven"] == 5 * 12
    assert d["persistent"] and d["persistent_deficit"] and d["forgiveness_supported"]
    few = sc.derive_stagnation([[0, 1, 1, 0]] * 4 + [None] * 6, first, second)
    assert not few["persistent_deficit"] and not few["forgiveness_supported"]
    barely = copy.deepcopy(first)
    for row in barely:
        row[si("forgiven")] = 4  # 2*8 < 20 -> not supported
    assert not sc.derive_stagnation(windows, barely, barely)["forgiveness_supported"]


def test_episodes_and_censoring():
    s = _series(completed=1000, adults=10)
    s.starvation_deaths[100] = s.starvation_deaths[120] = 1  # one episode (gap 20)
    s.starvation_deaths[500] = 1                              # second episode
    e = sc.analyze_episodes(s, 1000)
    assert e["episodes"] == 2 and e["observable"] == 2 and e["recovered_economy"] == 2
    assert sc.analyze_episodes(s, 700)["observable"] == 1      # the second episode ends at 500; 500+300 > 700
    for t in range(130, 700, 90):
        s.missing[t] = 1                                                # a shortfall every 90 ticks until 700
    assert sc.analyze_episodes(s, 1000)["recovered_economy"] == 1      # only the second episode finds a clean run
    s.adults = [10] * 101 + [7] * 900
    assert sc.analyze_episodes(s, 1000)["recovered_population"] == 1  # only episode 2 (7 -> 7) recovers


# --- clusters and erratum E1 ------------------------------------------------------------------------

def _fake_row(variant, map_id, pair, seed, seat, trajectory, reason="timeout", winner="draw", sides=(),
              stagnant_deficit=0, stagnant=0, supported=False, deficit=0):
    stag = {"stagnant": stagnant, "stagnant_deficit": stagnant_deficit, "stagnant_sat": 0, "persistent": stagnant >= 5,
            "persistent_deficit": stagnant_deficit >= 5, "forgiveness_supported": supported}
    side = lambda policy, world: {"policy": policy, "world_side": world, "totals": {"deficit_ticks": deficit}}
    return {"variant": variant, "map_id": map_id, "first": pair[0], "second": pair[1], "seed": seed, "seat": seat,
            "row_id": f"{variant}|{map_id}|{pair[0]}-{pair[1]}|{seed}|{seat}", "trajectory_sha256": trajectory,
            "terminal": {"reason": reason, "winner": winner, "sides": list(sides)}, "stagnation": stag,
            "sides": {"first": side(pair[0], "A" if seat == "first_A" else "B"),
                      "second": side(pair[1], "B" if seat == "first_A" else "A")}}


def _toy_study(variant_hashes, *, reason_v="timeout", seeds=(1, 2, 3, 4), map_id="R0-T0", pair=("turtle", "raider")):
    """A0 rows share one trajectory hash per seat; variant rows use ``variant_hashes(seed, seat)``."""
    a0 = [_fake_row("A0", map_id, pair, s, seat, f"a0-{seat}") for s in seeds for seat in sc.SEAT_NAMES]
    v = [_fake_row("A1b", map_id, pair, s, seat, variant_hashes(s, seat), reason=reason_v)
         for s in seeds for seat in sc.SEAT_NAMES]
    return a0, v


def test_clusters_come_from_a0_only_and_group_equal_blocks():
    a0, v = _toy_study(lambda s, seat: f"v-{s}-{seat}")
    clusters = sc.build_clusters(a0)
    assert len(clusters) == 1 and clusters[0].seeds == (1, 2, 3, 4)
    with pytest.raises(sc.StarvationConfirmError):
        sc.build_clusters(v)
    distinct = sc.build_clusters([r for r in a0 if r["seed"] != 4] + [
        _fake_row("A0", "R0-T0", ("turtle", "raider"), 4, seat, "other") for seat in sc.SEAT_NAMES])
    assert [c.seeds for c in distinct] == [(1, 2, 3), (4,)]


def test_splitting_one_trajectory_into_many_hashes_creates_no_fictitious_growth():
    """Erratum E1.6 (1): same outcomes, any number of distinct variant hashes -> zero differences."""
    a0, v_same = _toy_study(lambda s, seat: f"a0-{seat}")
    _, v_split = _toy_study(lambda s, seat: f"v-{s}-{seat}")
    clusters = sc.build_clusters(a0)
    a0_by = {sc.row_key(r): r for r in a0}
    t0 = sc.indicator_table(a0, a0_by)
    results = []
    for rows in (v_same, v_split):
        counts = sc.e1_counts(clusters, sc.indicator_table(rows, a0_by), t0, [])
        results.append(counts)
        assert counts["t_loss"] == counts["t_loss_a0"] and counts["collapses"] == counts["collapses_a0"]
        assert counts["x_changed"] == 0 and sc.k5_ok(counts) and sc.k6_ok(counts)
    assert results[0] == results[1]  # also independent of the number of variant hashes (E1.6 (2))
    # the old per-variant grouping would have seen 1 block for A0 and 4 for the variant:
    assert len({r["trajectory_sha256"] for r in v_split}) > len({r["trajectory_sha256"] for r in a0})


def test_real_worsening_counts_once_per_cluster_not_per_row_or_hash():
    a0, _ = _toy_study(lambda s, seat: "x")
    # in one seed of the cluster, on one seat, Turtle (world side A) loses to a collapse
    bad = []
    for s in (1, 2, 3, 4):
        for seat in sc.SEAT_NAMES:
            if s in (2, 3) and seat == "first_A":
                bad.append(_fake_row("A1b", "R0-T0", ("turtle", "raider"), s, seat, f"v-{s}-{seat}",
                                     reason="economic_collapse", winner="B", sides=(0,)))
            else:
                bad.append(_fake_row("A1b", "R0-T0", ("turtle", "raider"), s, seat, f"v-{s}-{seat}"))
    clusters = sc.build_clusters(a0)
    a0_by = {sc.row_key(r): r for r in a0}
    counts = sc.e1_counts(clusters, sc.indicator_table(bad, a0_by), sc.indicator_table(a0, a0_by), [])
    assert counts["t_loss"] - counts["t_loss_a0"] == 1 and counts["collapses"] - counts["collapses_a0"] == 1
    assert counts["t_loss_worse"] == 1 and counts["t_loss_better"] == 0
    assert counts["t_loss_heterogeneous"] == [clusters[0].cluster_id]   # published, counted once
    assert counts["x_changed"] == 1


def test_common_denominators_do_not_depend_on_the_variant():
    a0, v1 = _toy_study(lambda s, seat: f"v-{s}-{seat}")
    _, v2 = _toy_study(lambda s, seat: "same", reason_v="economic_collapse")
    clusters = sc.build_clusters(a0)
    a0_by = {sc.row_key(r): r for r in a0}
    t0 = sc.indicator_table(a0, a0_by)
    c1 = sc.e1_counts(clusters, sc.indicator_table(v1, a0_by), t0, [])
    c2 = sc.e1_counts(clusters, sc.indicator_table(v2, a0_by), t0, [])
    for key in ("n_clusters", "n_turtle_clusters", "n_exposed", "n_non_exposed"):
        assert c1[key] == c2[key]
    assert c1["n_clusters"] == 1 and c1["n_turtle_clusters"] == 1


def test_a0_against_itself_has_no_differences():
    a0, _ = _toy_study(lambda s, seat: "x")
    clusters = sc.build_clusters(a0)
    a0_by = {sc.row_key(r): r for r in a0}
    t0 = sc.indicator_table(a0, a0_by)
    counts = sc.e1_counts(clusters, t0, t0, [])
    assert counts["t_loss"] == counts["t_loss_a0"] and counts["collapses"] == counts["collapses_a0"]
    assert counts["x_changed"] == 0 and counts["t_loss_worse"] == counts["t_loss_better"] == 0


def test_turtle_clusters_are_only_pairs_with_turtle():
    a0, _ = _toy_study(lambda s, seat: "x", pair=("rush", "raider"))
    clusters = sc.build_clusters(a0)
    a0_by = {sc.row_key(r): r for r in a0}
    counts = sc.e1_counts(clusters, sc.indicator_table(a0, a0_by), sc.indicator_table(a0, a0_by), [])
    assert counts["n_turtle_clusters"] == 0 and counts["t_loss"] == 0


def test_k5_k6_thresholds():
    ok = {"t_loss": 7, "t_loss_a0": 4, "collapses": 10, "collapses_a0": 8, "x_changed": 3, "n_non_exposed": 30}
    assert sc.k5_ok(ok) and sc.k6_ok(ok)
    assert not sc.k5_ok({**ok, "t_loss": 8})                       # +4 > max(3, ceil(1))
    assert sc.k5_ok({**ok, "t_loss": 24, "t_loss_a0": 20})          # +4 <= ceil(5)
    assert not sc.k5_ok({**ok, "t_loss": 26, "t_loss_a0": 20})
    assert not sc.k6_ok({**ok, "collapses": 12})
    assert not sc.k6_ok({**ok, "x_changed": 4})                     # > max(3, ceil(3.0))
    assert sc.k6_ok({**ok, "x_changed": 4, "n_non_exposed": 31})    # ceil(3.1) = 4


def test_resolution_classes_and_worst_case_order():
    r0 = _fake_row("A0", "R0-T0", ("turtle", "raider"), 1, "first_A", "h", stagnant_deficit=6, stagnant=6, supported=True)
    def v(**kw):
        return _fake_row("A1b", "R0-T0", ("turtle", "raider"), 1, "first_A", "h2", **kw)
    assert sc.resolution_class(r0, v(stagnant_deficit=3, stagnant=3)) == "PERSISTS"            # ceil(6/2)
    assert sc.resolution_class(r0, v(stagnant_deficit=2, stagnant=2)) == "ADJUSTED_ACTIVE"
    assert sc.resolution_class(r0, v(stagnant_deficit=2, stagnant=5)) == "ADJUSTED_STATIC"
    assert sc.resolution_class(r0, v(reason="economic_collapse", stagnant_deficit=0)) == "COLLAPSED"
    assert sc.resolution_class(r0, v(reason="queen_death", stagnant_deficit=0)) == "QUEEN_DEATH"
    assert sc.CLASS_ORDER.index("PERSISTS") > sc.CLASS_ORDER.index("COLLAPSED") > sc.CLASS_ORDER.index("QUEEN_DEATH")
    assert sc.resolution_class(_fake_row("A0", "R0-T0", ("turtle", "raider"), 1, "first_A", "h", stagnant_deficit=5,
                                         supported=True), v(stagnant_deficit=3)) == "PERSISTS"


def test_cluster_resolution_takes_the_worst_member():
    a0 = [_fake_row("A0", "R0-T0", ("turtle", "raider"), s, seat, f"a0-{seat}", stagnant_deficit=6, supported=True)
          for s in (1, 2) for seat in sc.SEAT_NAMES]
    cluster = sc.build_clusters(a0)[0]
    a0_by = {sc.row_key(r): r for r in a0}
    rows = {}
    for s in (1, 2):
        for seat in sc.SEAT_NAMES:
            reason = "queen_death" if (s, seat) == (2, "first_B") else "timeout"
            rows[("R0-T0", "turtle", "raider", s, seat)] = _fake_row(
                "A1b", "R0-T0", ("turtle", "raider"), s, seat, "z", reason=reason)
    assert sc.cluster_resolution(cluster, a0_by, rows) == "QUEEN_DEATH"
    assert sc.exposed_clusters([cluster], a0_by) == [cluster]


# --- verdict ----------------------------------------------------------------------------------------

def _decide(**over):
    base = dict(integrity_ok=True, k1=True, exposure=8, spread_ok=True, n_clusters=200,
                passing={"A1a": False, "A1b": True, "A2": True, "A3b": False},
                collateral={"A1a": (0, 0), "A1b": (4, 1), "A2": (2, 1), "A3b": (0, 0)})
    base.update(over)
    return sc.decide(**base)


def test_verdict_order_and_candidate_choice():
    assert _decide()["verdict"] == "NEW_RULES_VERSION" and _decide()["candidate"] == "A2"   # fewer side effects
    tie = _decide(collateral={"A1a": (0, 0), "A1b": (2, 1), "A2": (2, 1), "A3b": (0, 0)})
    assert tie["candidate"] == "A1b"                                                             # simpler rule
    assert _decide(integrity_ok=False)["verdict"] == "INSUFFICIENT_DATA"
    assert _decide(k1=False)["verdict"] == "INSUFFICIENT_DATA"
    assert _decide(n_clusters=119)["verdict"] == "INSUFFICIENT_DATA"
    assert _decide(exposure=3)["verdict"] == "INSUFFICIENT_DATA"
    assert _decide(spread_ok=False)["verdict"] == "INSUFFICIENT_DATA"
    assert _decide(exposure=0)["verdict"] == "KEEP_A0"
    assert _decide(passing={"A1a": True, "A1b": False, "A2": False, "A3b": False}
                   | {"A1a": False})["verdict"] == "KEEP_A0"
    assert _decide(exposure=6)["weak"] is True and _decide(exposure=8)["weak"] is False
    assert _decide(exposure=0, n_clusters=10)["verdict"] == "INSUFFICIENT_DATA"                  # coverage first


# --- real matches (off-schedule seed) ---------------------------------------------------------------

@pytest.fixture(scope="module")
def group_result():
    return sc.execute_group(("R0-T0", "turtle", "raider", OFF_SCHEDULE_SEED, False))


def test_group_runs_all_variants_and_the_a0_anchor_matches(group_result):
    rows, anchor = group_result
    assert [r["variant"] for r in rows] == list(sc.VARIANTS)
    assert anchor["matches"] is True and sc.anchor_matches(rows[0], anchor)
    assert tick_module.upkeep is economy_module.upkeep
    for r in rows:
        assert r["status"] == "complete" and r["config_sha256"] == sc.CONFIG_SHA256["R0-T0"]
        assert r["terminal"]["reason"] in ("timeout", "queen_death", "economic_collapse")
        assert r["calls"] <= sc.T_MAX and r["sides"]["first"]["policy"] == "turtle"
        for rel in ("first", "second"):
            t = r["sides"][rel]["totals"]
            assert t["debt_end"] == t["missing"] - 5 * t["starvation_deaths"] - t["forgiven"] - t["repaid"]
    assert rows[0]["first_divergence_tick"] == -1
    assert all(r["first_divergence_tick"] >= -1 for r in rows)


def test_group_is_deterministic_and_a0_is_a_pure_copy_of_the_engine(group_result):
    again, anchor2 = sc.execute_group(("R0-T0", "turtle", "raider", OFF_SCHEDULE_SEED, False))
    assert [sc.canonical_dumps(r) for r in again] == [sc.canonical_dumps(r) for r in group_result[0]]
    assert anchor2 == group_result[1]


def test_variants_only_act_through_the_debt_rule(group_result):
    rows, _ = group_result
    a0 = rows[0]
    for r in rows[1:]:
        if r["sides"]["first"]["totals"]["forgiven"] + r["sides"]["second"]["totals"]["forgiven"] == 0 and \
                r["sides"]["first"]["totals"]["repaid"] + r["sides"]["second"]["totals"]["repaid"] == 0 and \
                a0["sides"]["first"]["totals"]["deficit_ticks"] + a0["sides"]["second"]["totals"]["deficit_ticks"] == 0:
            assert r["trajectory_sha256"] == a0["trajectory_sha256"]
    assert rows[3]["sides"]["first"]["totals"]["forgiven"] == 0 == rows[3]["sides"]["second"]["totals"]["forgiven"]


def test_hashseed_does_not_change_a_row():
    code = ("from formicmind import starvation_confirm as sc, json as _j\n") if False else (
        "import json\nfrom formicmind import starvation_confirm as sc\n"
        "rows,_=sc.execute_group(('R0-T1','rush','balanced',%d,True))\n"
        "print(json.dumps([r['trajectory_sha256']+r['final_state_sha256'] for r in rows]))\n" % OFF_SCHEDULE_SEED)
    outs = []
    for hashseed in ("1", "97"):
        env = {**os.environ, "PYTHONHASHSEED": hashseed, "PYTHONPATH": str(ROOT / "src")}
        outs.append(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                                   env=env, cwd=ROOT).stdout)
    assert outs[0] == outs[1] and json.loads(outs[0])


def test_a0_row_is_accepted_by_the_replay_free_checkpoint_identity(group_result):
    rows, anchor = group_result
    assert rows[0]["final_checkpoint_sha256"] == anchor["final_checkpoint_sha256"]
    assert rows[0]["events_trace_sha256"] == anchor["events_trace_sha256"]


# --- artifacts: tamper detection --------------------------------------------------------------------

@pytest.fixture(scope="module")
def mini_gen():
    groups = [("R0-T0", "turtle", "raider", OFF_SCHEDULE_SEED, False), ("R0-T0", "turtle", "raider", OFF_SCHEDULE_SEED, True)]
    return sc.generate(2, groups=groups)


def _write(directory, gen):
    artifacts = sc._artifact_bytes(gen)
    for name, data in artifacts.items():
        (directory / name).write_bytes(data)
    return {"artifacts": {n: sc._sha(sc._lf(d)) for n, d in artifacts.items()}, "step_tick_calls": gen["calls"]}


def test_compare_artifacts_accepts_the_original_and_rejects_every_tamper(tmp_path, mini_gen):
    manifest = _write(tmp_path, mini_gen)
    assert sc.compare_artifacts(tmp_path, manifest, mini_gen) == []
    rows_path = tmp_path / "matches.jsonl"
    original = rows_path.read_bytes()
    lines = original.split(b"\n")
    rows_path.write_bytes(b"\n".join(lines[1:]))                                    # missing row
    assert sc.compare_artifacts(tmp_path, manifest, mini_gen)
    forged = json.loads(lines[0])
    forged["terminal"]["reason"] = "queen_death"
    rows_path.write_bytes(b"\n".join([sc.canonical_dumps(forged), *lines[1:]]))      # false terminal
    assert sc.compare_artifacts(tmp_path, manifest, mini_gen)
    rows_path.write_bytes(original + original.split(b"\n")[0] + b"\n")               # extra row
    assert sc.compare_artifacts(tmp_path, manifest, mini_gen)
    rows_path.write_bytes(original)
    summary = json.loads((tmp_path / "summary.json").read_text())
    summary["decision"]["verdict"] = "NEW_RULES_VERSION"
    (tmp_path / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=1) + "\n")
    assert sc.compare_artifacts(tmp_path, manifest, mini_gen)
    (tmp_path / "summary.json").write_bytes(sc._artifact_bytes(mini_gen)["summary.json"])
    bad = {**manifest, "artifacts": {**manifest["artifacts"], "anchors.json": "0" * 64}}
    assert sc.compare_artifacts(tmp_path, bad, mini_gen)
    assert sc.compare_artifacts(tmp_path, {**manifest, "step_tick_calls": {"main": 1, "anchor": 1}}, mini_gen)


def test_mini_summary_is_consistent(mini_gen):
    summary = mini_gen["summary"]
    assert summary["preregistered"] is True and summary["integrity"]["schedule_complete"] is False
    assert summary["decision"]["verdict"] == "INSUFFICIENT_DATA"        # a two-row sample can never decide
    assert summary["design"]["a0_clusters"] == 1


def test_summary_is_derived_from_stored_rows_only(mini_gen):
    again = sc.build_summary(mini_gen["rows"], mini_gen["demo"], mini_gen["anchors"])
    assert sc._dump_json(again) == sc._dump_json(mini_gen["summary"])
    forged = copy.deepcopy(mini_gen["rows"])
    flipped = not forged[0]["stagnation"]["forgiveness_supported"]
    forged[0]["stagnation"]["forgiveness_supported"] = flipped
    forged[0]["terminal"]["reason"] = "queen_death" if forged[0]["terminal"]["reason"] == "timeout" else "timeout"
    assert sc._dump_json(sc.build_summary(forged, mini_gen["demo"], mini_gen["anchors"])) != sc._dump_json(mini_gen["summary"])


def test_protected_files_exclude_only_study_files():
    protected = sc.protected_files()
    assert "src/formicmind/economy.py" in protected and "docs/starvation_confirm_protocol.md" not in protected
    assert "src/formicmind/starvation_confirm.py" not in protected and "docs/reliability.md" not in protected
    assert "src/formicmind/starvation_debt.py" in protected


def test_default_config_and_economy_are_untouched():
    from formicmind.rules import default_config
    from formicmind.serialization import config_hash
    before = config_hash(default_config())
    sc.build_minimal_demo()
    assert config_hash(default_config()) == before
    assert tick_module.upkeep is economy_module.upkeep


def test_rows_with_unmeasured_windows_serialise_and_stay_null():
    rows, _ = sc.execute_group(("R0-T0", "rush", "raider", OFF_SCHEDULE_SEED, False))
    assert any(None in r["windows"] for r in rows)                  # an early terminal leaves windows unmeasured
    data = sc._dump_lines(rows)
    back = [json.loads(line) for line in data.splitlines()]
    assert back == json.loads(json.dumps(rows))


# --- committed results (cheap consistency; the full regeneration is `verify`) ---------------------------

RESULTS = ROOT / "results" / "starvation_confirm"


@pytest.mark.skipif(not RESULTS.is_dir(), reason="results are produced after the preregistered run")
def test_committed_results_match_the_manifest_and_the_summary_is_derivable():
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["artifacts"].items():
        assert sc._sha(sc._lf((RESULTS / name).read_bytes())) == digest, name
    assert manifest["protocol_sha256"] == sc.PROTOCOL_SHA256
    assert {n: e["sha256"] for n, e in manifest["errata"].items()} == {str(n): h for n, (_, h) in sc.ERRATA.items()}
    assert manifest["engine_fingerprint"] == sc.BASE_ENGINE_FINGERPRINT
    assert manifest["seeds"] == list(sc.SEEDS) and manifest["schedule"]["rows"] == sc.EXPECTED_ROWS
    rows = [json.loads(line) for line in (RESULTS / "matches.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["row_id"] for r in rows] == sc.schedule_row_ids()
    assert sum(r["calls"] for r in rows) == manifest["step_tick_calls"]["main"] <= sc.MAIN_BUDGET
    anchors = json.loads((RESULTS / "anchors.json").read_text(encoding="utf-8"))["anchors"]
    demo = json.loads((RESULTS / "minimal_demo.json").read_text(encoding="utf-8"))
    stored = (RESULTS / "summary.json").read_bytes()
    assert sc._lf(sc._dump_json(sc.build_summary(rows, demo, anchors))) == sc._lf(stored)
    assert len(anchors) == sc.EXPECTED_ANCHORS and all(a["matches"] for a in anchors)
    assert sum(a["calls"] for a in anchors) == manifest["step_tick_calls"]["anchor"] <= sc.ANCHOR_BUDGET
