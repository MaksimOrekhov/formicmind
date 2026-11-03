"""FOR-25: starvation-debt study harness (variants, scenarios, windows, blocks, verdict, strict verification)."""

import json
import random
from dataclasses import replace

import pytest

from formicmind import economy as economy_module
from formicmind import starvation_debt as sd
from formicmind import tick as tick_module
from formicmind.canonical import canonical_dumps
from formicmind.rules import default_config
from formicmind.serialization import config_hash
from formicmind.tick import ScriptedController, step_tick

ALL = sd.VARIANTS


def single(variant, **kw):
    return sd.run_single_tick(variant, **kw)


# --- frozen inputs and the schedule ---------------------------------------------------------------------

def test_the_protocol_erratum_and_engine_sources_are_frozen():
    assert sd.protocol_sha256() == sd.PROTOCOL_SHA256
    assert sd._file_sha(sd._ROOT / sd.ERRATUM_PATH) == sd.ERRATUM_SHA256
    # every engine file except the experimental harness is byte-identical to the frozen base
    assert sd.source_fingerprint(exclude_harness=True) == sd.BASE_ENGINE_FINGERPRINT
    assert tick_module.upkeep is economy_module.upkeep


def test_the_schedule_has_exactly_4320_rows_and_the_frozen_axes():
    rows = sd.schedule()
    assert len(rows) == sd.EXPECTED_ROWS == 4320 == 6 * 6 * 10 * 6 * 2
    assert len(set(rows)) == 4320
    assert len(sd.PAIRS) == 10 and sd.SEEDS == tuple(range(20270301, 20270307))
    assert sd.VARIANTS == ("A0", "A1a", "A1b", "A2", "A3a", "A3b")
    assert sd.MAIN_BUDGET == 12_960_000 and sd.ANCHOR_BUDGET == 2_160_000
    assert sd.hashlib.sha256(canonical_dumps(list(sd.SEEDS))).hexdigest() == sd.SEEDS_SHA256
    assert {row[0] for row in rows} == set(ALL) and {row[5] for row in rows} == {False, True}


def test_the_default_config_and_map_hashes_are_untouched():
    for name, config in sd._configs().items():
        assert config_hash(config) == sd.CONFIG_SHA256[name]
    assert default_config().balance.starvation_unit_cost == 5


# --- rule arithmetic: A0 is the current rule, the others differ only where defined -----------------------

def test_a0_is_the_unchanged_economy_upkeep_on_random_states():
    rng = random.Random(25)
    for _ in range(400):
        setup = dict(workers=rng.randint(0, 40), soldiers=rng.randint(0, 8), food=rng.randint(0, 25),
                     debt=rng.randint(0, 14), rem=rng.randint(0, 9), cost=rng.choice([None, 3, 7]))
        assert single("A0", **setup) == single(None, **setup)


def test_a0_events_windows_and_state_equal_the_production_function_exactly():
    from formicmind.starvation_debt import VariantContext, variant_upkeep
    for setup in (dict(workers=12, soldiers=4, food=3, debt=2, rem=5), dict(workers=40, soldiers=0, food=0, debt=19, rem=9)):
        s1, r1 = sd._toy_state(setup["workers"], setup["soldiers"], food=setup["food"], debt=setup["debt"], rem=setup["rem"])
        s2, r2 = sd._toy_state(setup["workers"], setup["soldiers"], food=setup["food"], debt=setup["debt"], rem=setup["rem"])
        e1 = economy_module.upkeep(s1, r1)
        e2 = variant_upkeep(VariantContext("A0"), s2, r2)
        assert [e.to_canonical() for e in e1] == [e.to_canonical() for e in e2]
        assert r1.windows == r2.windows
        assert [(c.food, c.starvation_debt, c.upkeep_remainder10, len(c.units)) for c in s1.colonies] == \
            [(c.food, c.starvation_debt, c.upkeep_remainder10, len(c.units)) for c in s2.colonies]


def test_edge_cases_agree_for_every_variant_when_no_tick_is_fully_paid():
    cases = {
        "due_zero_keeps_debt": (3, 0, False), "zero_food_charges_debt": (1, 0, False),
        "debt_4_no_death": (4, 0, False), "debt_5_one_death": (0, 1, False),
        "debt_9_one_death_leaves_4": (4, 1, False), "debt_14_two_deaths_leaves_4": (4, 2, False),
        "debt_68_many_deaths": (3, 13, False), "last_adult_keeps_residual_debt": (96, 1, False),
    }
    for name, (debt_after, deaths, reset) in cases.items():
        for variant in ALL:
            got = single(variant, **sd.EDGE_CASES[name])
            assert (got["debt_after"], got["deaths"], got["debt_reset"]) == (debt_after, deaths, reset), (name, variant)


def test_a_tick_with_nothing_due_never_changes_debt_in_any_variant():
    for variant in ("A0", "A1a", "A1b", "A3a", "A3b"):
        got = single(variant, workers=1, food=40, debt=3, rem=0)
        assert got["due"] == 0 and got["debt_after"] == 3 and not got["debt_reset"] and got["food_after"] == 40
    a2 = single("A2", workers=1, food=40, debt=3, rem=0)  # A2 repays from surplus even when nothing is due
    assert a2["debt_after"] == 0 and a2["food_after"] == 37 and not a2["debt_reset"]


def test_a1a_resets_only_with_a_positive_surplus_and_a1b_only_when_the_stock_covers_another_tick():
    exact = sd.EDGE_CASES["exact_payment_with_debt"]      # food == due: surplus 0
    one = sd.EDGE_CASES["one_unit_surplus_with_debt"]     # surplus 1, due 5
    big = sd.EDGE_CASES["large_surplus_with_debt"]        # surplus 35 >= due
    assert single("A0", **exact)["debt_after"] == 0
    assert single("A1a", **exact)["debt_after"] == 3 and single("A1b", **exact)["debt_after"] == 3
    assert single("A1a", **one)["debt_reset"] and single("A1a", **one)["debt_after"] == 0
    assert single("A1b", **one)["debt_after"] == 3 and not single("A1b", **one)["debt_reset"]
    for variant in ("A0", "A1a", "A1b"):
        assert single(variant, **big)["debt_after"] == 0 and single(variant, **big)["food_after"] == 35


def test_a2_repays_debt_with_surplus_food_and_destroys_that_food():
    exact = single("A2", **sd.EDGE_CASES["exact_payment_with_debt"])
    assert exact["debt_after"] == 3 and exact["food_after"] == 0
    one = single("A2", **sd.EDGE_CASES["one_unit_surplus_with_debt"])
    assert (one["debt_after"], one["food_after"]) == (2, 0)
    big = single("A2", **sd.EDGE_CASES["large_surplus_with_debt"])
    assert (big["debt_after"], big["food_after"], big["debt_reset"]) == (0, 32, False)
    more = single("A2", **sd.EDGE_CASES["surplus_exceeds_debt"])
    assert (more["debt_after"], more["food_after"]) == (0, 5)


def test_a3_resets_after_n_consecutive_fully_paid_ticks_and_underpayment_restarts_the_count():
    setup = sd.EDGE_CASES["large_surplus_with_debt"]
    assert single("A3a", **setup, streak=1)["debt_after"] == 3       # becomes 2 < 3
    assert single("A3a", **setup, streak=2)["debt_after"] == 0       # becomes 3
    assert single("A3b", **setup, streak=8)["debt_after"] == 3       # becomes 9 < 10
    assert single("A3b", **setup, streak=9)["debt_after"] == 0       # becomes 10
    ctx = sd.VariantContext("A3a", [2, 0])
    state, run = sd._toy_state(12, 0, food=0, debt=0, rem=2)
    sd.variant_upkeep(ctx, state, run)                               # underpaid: streak back to 0
    assert ctx.streak[0] == 0
    state2, run2 = sd._toy_state(1, 0, food=0, debt=0, rem=0)        # due == 0 leaves the streak alone
    ctx2 = sd.VariantContext("A3a", [2, 0])
    sd.variant_upkeep(ctx2, state2, run2)
    assert ctx2.streak[0] == 2


def test_tenths_of_upkeep_are_carried_and_sum_exactly():
    got = sd._tenths_rounding()
    assert got["dues"] == [0, 0, 1, 0, 1, 0, 0, 1, 0, 1] and got["sum"] == 4 and got["remainder_end"] == 0


# --- minimal reproducible scenarios ----------------------------------------------------------------------

def test_m1_is_the_chronic_deficit_the_rule_forgives_and_candidates_stop_it():
    demo = sd.build_minimal_demo()
    m1 = demo["scenarios"]["M1"]
    a0 = m1["A0"]["totals"]
    assert a0["deaths"] == 0 and a0["missing"] == 481 and a0["forgiven"] == 481 and a0["repaid"] == 0
    assert m1["A0"]["adults_end"] == 32 and m1["A0"]["max_debt_seen"] < 5
    assert m1["A1b"]["totals"]["deaths"] > 0 and m1["A3a"]["totals"]["deaths"] > 0 and m1["A2"]["totals"]["deaths"] > 0
    assert demo["r1_mechanism_reproduced"] is True
    for scenario in demo["scenarios"].values():
        for result in scenario.values():
            assert result["debt_identity_holds"]


def test_erratum_e1_m2_is_not_a_control_but_m4_is():
    demo = sd.build_minimal_demo()
    m2, m4 = demo["scenarios"]["M2"]["A0"]["totals"], demo["scenarios"]["M4"]["A0"]["totals"]
    assert m2["deaths"] == 0 and m2["deficit_ticks"] > 400 and m2["forgiven"] > 0   # constant shortfall still forgiven
    assert m4["deaths"] > 0                                                         # no tick can be fully paid initially
    m3 = demo["scenarios"]["M3"]
    assert all(m3[v]["totals"]["missing"] == 0 and m3[v]["adults_end"] == 32 for v in ALL)


def test_a0_demo_matches_the_production_function_tick_by_tick():
    assert sd.run_minimal("A0", "M1")["a0_equals_production_upkeep"] is True


# --- swapping the rule never leaks -----------------------------------------------------------------------

def test_use_variant_swaps_and_always_restores_the_phase_function():
    original = economy_module.upkeep
    with sd.use_variant("A2") as ctx:
        assert tick_module.upkeep is not original and ctx.variant == "A2"
        with pytest.raises(sd.StarvationDebtError):
            with sd.use_variant("A1a"):
                pass
    assert tick_module.upkeep is original and economy_module.upkeep is original
    with pytest.raises(RuntimeError):
        with sd.use_variant("A1b"):
            raise RuntimeError("boom")
    assert tick_module.upkeep is original
    with pytest.raises(sd.StarvationDebtError):
        with sd.use_variant("A9"):
            pass
    assert tick_module.upkeep is original


def test_the_swapped_rule_is_really_used_by_step_tick_and_the_default_is_not_changed():
    state, run = sd._toy_state(12, 0, food=0, debt=0, rem=2)
    ctrl = (ScriptedController({}), ScriptedController({}))
    with sd.use_variant("A0") as ctx:
        step_tick(state, run, ctrl)
        assert ctx.calls == 1
    state2, run2 = sd._toy_state(12, 0, food=0, debt=0, rem=2)
    before = tick_module.upkeep
    step_tick(state2, run2, ctrl)
    assert tick_module.upkeep is before


# --- real engine: collapse, anchor, determinism -----------------------------------------------------------

def test_economic_collapse_through_the_real_engine_under_every_variant():
    for variant in ALL:
        state, run = sd._toy_state(1, 0, food=0, debt=4, rem=6)
        state.colonies[0].brood.clear()
        state.colonies[0].queen_hp = 240
        ctrl = (ScriptedController({}), ScriptedController({}))
        with sd.use_variant(variant):
            for _ in range(5):
                if run.outcome is not None:
                    break
                step_tick(state, run, ctrl)
        assert run.outcome is not None and run.outcome.reason.value == "economic_collapse", variant
        assert run.outcome.sides == (0,) and state.colonies[0].units == []


def test_a_short_schedule_runs_the_real_engine_and_a0_equals_the_unchanged_engine():
    rows = sd.run_schedule(sd.schedule(("A0", "A2"), ("R0-T0",), (("turtle", "balanced"),), (7,)), 1)
    assert len(rows) == 4 and all(r["status"] == "complete" for r in rows)
    a0 = [r for r in rows if r["variant"] == "A0"]
    assert all(r["anchor_matches_unchanged_engine"] and r["anchor_calls"] == r["ticks_executed"] for r in a0)
    assert all(r["anchor_calls"] == 0 for r in rows if r["variant"] == "A2")
    assert all(side["debt_identity_holds"] for r in rows for side in (r["sides"]["first"], r["sides"]["second"]))
    again = sd.run_schedule(sd.schedule(("A0", "A2"), ("R0-T0",), (("turtle", "balanced"),), (7,)), 1)
    assert canonical_dumps(rows) == canonical_dumps(again)             # deterministic
    assert tick_module.upkeep is economy_module.upkeep


def test_parallel_workers_produce_the_same_rows_as_serial_execution():
    sched = sd.schedule(("A0", "A1b"), ("R1-T0",), (("rush", "balanced"),), (7,))
    assert canonical_dumps(sd.run_schedule(sched, 1)) == canonical_dumps(sd.run_schedule(sched, 2))


# --- windows, episodes, blocks, verdict -------------------------------------------------------------------

def make_trace(missing, forgiven, adults, deaths=None):
    tr = sd._SideTrace()
    n = len(missing)
    tr.missing, tr.forgiven = list(missing), list(forgiven)
    tr.repaid = [0] * n
    tr.deaths = list(deaths) if deaths is not None else [0] * n
    tr.due = [10] * n
    tr.adults = list(adults)
    return tr


def window_flags(deficit_ticks, forgiven, missing, a_start, a_end, alive=True, completed=300):
    miss = [0] * 300
    forg = [0] * 300
    for i in range(deficit_ticks):
        miss[i] = 1
    miss[0] += missing - deficit_ticks          # total missing as requested
    forg[1] = forgiven
    adults = [a_start] + [a_start] * 299 + [a_end]
    tr = make_trace(miss, forg, adults)
    return sd._flags_for_windows(tr, completed, alive)[1]


def test_window_flag_boundaries_are_inclusive_and_exact():
    assert window_flags(30, 10, 20 + 10, 20, 20)["chronic"] == 1
    assert window_flags(29, 10, 29, 20, 20)["chronic"] == 0
    assert window_flags(30, 15, 30, 20, 20)["forgiving"] == 1               # 2*forgiven == missing
    assert window_flags(30, 14, 30, 20, 20)["forgiving"] == 0               # 2*14 < 30
    assert window_flags(30, 0, 30, 20, 20)["forgiving"] == 0                # nothing forgiven
    assert window_flags(30, 15, 30, 20, 22)["zombie"] == 1                  # |delta| == 2
    assert window_flags(30, 15, 30, 20, 17)["zombie"] == 0                  # |delta| == 3
    assert window_flags(30, 15, 30, 20, 18)["zombie"] == 1


def test_a_window_is_not_measured_unless_it_is_fully_inside_the_match():
    tr = make_trace([1] * 450, [0] * 450, [10] * 451)
    windows, flags = sd._flags_for_windows(tr, 450, True)
    assert [w[0] for w in windows] == [0]
    assert sd._flags_for_windows(make_trace([1] * 299, [0] * 299, [10] * 300), 299, True)[0] == []


def test_a_queen_death_in_the_last_window_prevents_the_zombie_label():
    assert window_flags(30, 15, 30, 20, 20, alive=False, completed=300)["zombie"] == 0


def test_death_episodes_split_at_a_gap_over_30_and_report_recovery_and_censoring():
    n = 1000
    deaths = [0] * n
    for t in (100, 130):                  # gap 30: one episode
        deaths[t] = 1
    deaths[161] = 1                       # gap 31 from 130: second episode
    deaths[900] = 1                       # too late to observe 300 ticks
    adults = [20] * (n + 1)
    for t in range(101, n + 1):
        adults[t] = 18
    for t in range(500, n + 1):
        adults[t] = 20                    # population back to the pre-death count
    tr = make_trace([0] * n, [0] * n, adults, deaths)
    episodes = sd._episodes(tr, n)
    assert [e[0:2] for e in episodes] == [[100, 130], [161, 161], [900, 900]]
    assert [e[4] for e in episodes] == [1, 1, 0]
    assert episodes[2][5:7] == [-1, -1]
    assert episodes[0][5] == 1                                   # 100 undeficient ticks follow
    assert episodes[1][6] == 1                                   # adults >= 18 immediately after 161?
    deficit = [0] * n
    for t in range(131, 431):
        deficit[t] = 1
    tr2 = make_trace(deficit, [0] * n, adults, [0] * 130 + [1] + [0] * (n - 131))
    assert sd._episodes(tr2, n)[0][5] == 0


def fake_row(map_id, matchup, seed, seat, digest, *, zombie=0, missing=0, reason="timeout", variant="A0", trace="t"):
    side = {"missing": missing, "flags": {"zombie": zombie, "chronic": 0, "forgiving": 0}, "debt_identity_holds": True}
    return {"map_id": map_id, "matchup": matchup, "seed": seed, "seat": seat, "trajectory_sha256": digest,
            "sides": {"first": dict(side), "second": dict(side, flags={"zombie": 0, "chronic": 0, "forgiving": 0})},
            "terminal": {"reason": reason, "terminal_tick": 2999, "winner": "draw"}, "variant": variant,
            "events_trace_sha256": f"{trace}-{seat}", "status": "complete", "last600_combat_events": 0, "adults_end": 1}


def test_duplicate_deterministic_trajectories_collapse_but_distinct_random_paths_do_not():
    rows = []
    for seed in (1, 2, 3):
        for seat in ("first_A", "first_B"):
            rows.append(fake_row("R0-T0", "turtle-balanced", seed, seat, f"same-{seat}"))
            rows.append(fake_row("R0-T0", "random-turtle", seed, seat, f"rnd-{seed}-{seat}"))
    blocks = sd._group_blocks(rows)
    det = [b for b in blocks if b["matchup"] == "turtle-balanced"]
    rnd = [b for b in blocks if b["matchup"] == "random-turtle"]
    assert len(det) == 1 and det[0]["seeds"] == [1, 2, 3] and len(det[0]["rows"]) == 6
    assert len(rnd) == 3
    assert len(sd._representative(det[0])) == 2
    with pytest.raises(sd.StarvationDebtError):
        sd._group_blocks([r for r in rows if not (r["seed"] == 1 and r["seat"] == "first_B" and r["matchup"] == "turtle-balanced")])


def a0_block(map_id, matchup, zombie, deficit=True):
    return {"map_id": map_id, "matchup": matchup, "zombie": zombie, "deficit": deficit,
            "zombie_windows": 2 if zombie else 0, "seeds": [1], "rows": []}


def comparison(support=6, remaining=3, z_v=2, z_0=4, c_v=1, c_0=1, false=0):
    return {"support_S": support, "r3_efficacy": bool(support >= 6 and remaining <= support // 2 and z_v <= z_0 // 2),
            "r4_safety": bool(not false and c_v - c_0 <= max(3, -(-c_0 // 4)))}


def verdict(blocks, comps, integrity=True):
    return sd.decide_verdict(blocks, comps, r1=True, integrity_ok=integrity)


def test_verdict_thresholds_are_exact():
    maps, pairs = ("R0-T0", "R1-T1"), ("turtle-balanced", "turtle-raider")
    base = [a0_block(maps[i % 2], pairs[i % 2], True) for i in range(6)] + [a0_block("R0-T0", "rush-balanced", False) for _ in range(10)]
    good = {"A2": comparison()}
    assert verdict(base, good)["verdict"] == "ИЗМЕНИТЬ"
    assert verdict(base, good)["qualifying_variants"] == ["A2"]
    assert verdict(base[1:], good)["verdict"] == "ОСТАВИТЬ"                             # only 5 zombie blocks
    one_pair = [a0_block(maps[i % 2], "turtle-balanced", True) for i in range(6)] + base[6:]
    assert verdict(one_pair, good)["verdict"] == "ОСТАВИТЬ"                             # one pair only
    one_map = [a0_block("R0-T0", pairs[i % 2], True) for i in range(6)] + base[6:]
    assert verdict(one_map, good)["verdict"] == "ОСТАВИТЬ"                             # one map only
    assert verdict(base, {"A2": comparison(remaining=4)})["verdict"] == "ОСТАВИТЬ"      # R(V) > floor(S/2)
    assert verdict(base, {"A2": comparison(z_v=3, z_0=4)})["verdict"] == "ОСТАВИТЬ"     # Z(V) > floor(Z0/2)
    assert verdict(base, {"A2": comparison(c_v=5, c_0=1)})["verdict"] == "ОСТАВИТЬ"     # collapse guard (+4 > max(3, 1))
    assert verdict(base, {"A2": comparison(c_v=4, c_0=1)})["verdict"] == "ИЗМЕНИТЬ"     # +3 allowed
    assert verdict(base, {"A2": comparison(false=1)})["verdict"] == "ОСТАВИТЬ"
    assert verdict(base, {"A2": comparison(support=5)})["verdict"] == "НЕДОСТАТОЧНО ДАННЫХ"   # S3
    assert verdict(base, good, integrity=False)["verdict"] == "НЕДОСТАТОЧНО ДАННЫХ"           # S1
    few = [a0_block(maps[i % 2], pairs[i % 2], True) for i in range(6)] + [a0_block("R0-T0", "rush-balanced", False, deficit=False)] * 6
    assert verdict(few, good)["exposure_E"] == 6 and verdict(few, good)["verdict"] == "НЕДОСТАТОЧНО ДАННЫХ"   # S2
    exposed = [a0_block("R0-T0", "rush-balanced", False) for _ in range(12)]
    assert verdict(exposed, good)["verdict"] == "ОСТАВИТЬ"                              # E == 12, no zombies


# --- strict verification of a short run ----------------------------------------------------------------------

SHORT = sd.schedule(("A0", "A3a"), ("R0-T0",), (("turtle", "balanced"),), (7,))


@pytest.fixture(scope="module")
def short_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("sd") / "out"
    manifest = sd.run(out, 1, _schedule=SHORT)
    return out, manifest


def test_a_short_run_verifies_strictly(short_run):
    out, manifest = short_run
    assert manifest["expected_rows"] == 4 and manifest["protocol_sha256"] == sd.PROTOCOL_SHA256
    assert manifest["engine_fingerprint_excluding_harness"] == sd.BASE_ENGINE_FINGERPRINT
    assert sd.verify(out, 1, _schedule=SHORT)["ok"] is True


def test_run_refuses_a_non_empty_directory(short_run):
    out, _ = short_run
    with pytest.raises(sd.StarvationDebtError):
        sd.run(out, 1, _schedule=SHORT)


def tampered_copy(out, tmp_path, name, transform):
    target = tmp_path / "copy"
    target.mkdir()
    for f in out.iterdir():
        (target / f.name).write_bytes(f.read_bytes())
    path = target / name
    path.write_bytes(transform(path.read_bytes()))
    return target


def test_verification_rejects_altered_missing_or_extra_rows_and_summaries(short_run, tmp_path):
    out, _ = short_run

    def refresh(target):
        m = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        for n in m["artifacts"]:
            m["artifacts"][n] = sd._sha((target / n).read_bytes())
        (target / "manifest.json").write_bytes(canonical_dumps(m) + b"\n")

    for i, (name, transform) in enumerate([
        ("matches.jsonl", lambda b: b.replace(b'"timeout"', b'"queen_death"', 1)),          # false terminal
        ("matches.jsonl", lambda b: b.split(b"\n", 1)[1]),                                  # missing row
        ("matches.jsonl", lambda b: b + b.split(b"\n", 1)[0] + b"\n"),                      # extra row
        ("summary.json", lambda b: b.replace(b'"verdict"', b'"verdict_"', 1)),              # modified summary
        ("reproduction.json", lambda b: b.replace(b'"example_confirmed"', b'"example_x"', 1)),
    ]):
        sub = tmp_path / f"t{i}"
        sub.mkdir()
        target = tampered_copy(out, sub, name, transform)
        with pytest.raises(sd.StarvationDebtError):
            sd.verify(target, 1, _schedule=SHORT)                                          # manifest hash mismatch
        refresh(target)                                                                    # even with refreshed hashes
        with pytest.raises(sd.StarvationDebtError):
            sd.verify(target, 1, _schedule=SHORT)


def test_verification_rejects_a_forged_manifest_binding(short_run, tmp_path):
    out, _ = short_run
    for i, key in enumerate(("protocol_sha256", "package_fingerprint", "engine_fingerprint_excluding_harness")):
        sub = tmp_path / f"m{i}"
        sub.mkdir()
        target = tampered_copy(out, sub, "manifest.json",
                               lambda b, key=key: json.dumps({**json.loads(b), key: "0" * 64}).encode())
        with pytest.raises(sd.StarvationDebtError):
            sd.verify(target, 1, _schedule=SHORT)


def test_reproduction_report_is_attached_to_the_unchanged_engine(short_run):
    out, _ = short_run
    repro = json.loads((out / "reproduction.json").read_text(encoding="utf-8"))
    assert repro["unchanged_engine"] is True and repro["case"]["seed"] == 20270201
    assert repro["match"]["terminal"]["reason"] == "timeout"
    assert repro["archive_comparison"]["status"] in {"found", "not_measured"}
    if repro["archive_comparison"]["status"] == "found":
        assert repro["archive_comparison"]["reproduced_terminal"] is True
        assert repro["archive_comparison"]["reproduced_final_state"] is True
        assert repro["archive_comparison"]["reproduced_events_trace"] is True
