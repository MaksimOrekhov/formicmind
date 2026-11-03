"""FOR-25 (current tree): harness logic, metrics, verdict rules and checkout-independent manifest checks.

Everything here runs on whatever engine the checkout has (including a merge with later engine work such as FOR-26)
and nothing is skipped. Checks that need the frozen FOR-24 engine live in ``historical/for25`` and run in an
isolated checkout of the pinned commit (see docs/starvation_debt_verification.md and the historical-for25 CI job).
"""
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


def fake_row(map_id, matchup, seed, seat, digest, *, zombie=0, missing=0, reason="timeout", variant="A0", trace="t"):
    side = {"missing": missing, "flags": {"zombie": zombie, "chronic": 0, "forgiving": 0}, "debt_identity_holds": True}
    return {"map_id": map_id, "matchup": matchup, "seed": seed, "seat": seat, "trajectory_sha256": digest,
            "sides": {"first": dict(side), "second": dict(side, flags={"zombie": 0, "chronic": 0, "forgiving": 0})},
            "terminal": {"reason": reason, "terminal_tick": 2999, "winner": "draw"}, "variant": variant,
            "events_trace_sha256": f"{trace}-{seat}", "status": "complete", "last600_combat_events": 0, "adults_end": 1}


def a0_block(map_id, matchup, zombie, deficit=True):
    return {"map_id": map_id, "matchup": matchup, "zombie": zombie, "deficit": deficit,
            "zombie_windows": 2 if zombie else 0, "seeds": [1], "rows": []}


def comparison(support=6, remaining=3, z_v=2, z_0=4, c_v=1, c_0=1, false=0):
    return {"support_S": support, "r3_efficacy": bool(support >= 6 and remaining <= support // 2 and z_v <= z_0 // 2),
            "r4_safety": bool(not false and c_v - c_0 <= max(3, -(-c_0 // 4)))}


def verdict(blocks, comps, integrity=True):
    return sd.decide_verdict(blocks, comps, r1=True, integrity_ok=integrity)


SHORT = sd.schedule(("A0", "A3a"), ("R0-T0",), (("turtle", "balanced"),), (7,))


def tampered_copy(out, tmp_path, name, transform):
    target = tmp_path / "copy"
    target.mkdir()
    for f in out.iterdir():
        (target / f.name).write_bytes(f.read_bytes())
    path = target / name
    path.write_bytes(transform(path.read_bytes()))
    return target


def forge_manifest(out, tmp_path, name, mutate):
    """A copy of the run whose artifacts are untouched and whose manifest has one field changed."""
    sub = tmp_path / name
    sub.mkdir()

    def transform(b):
        manifest = json.loads(b)
        mutate(manifest)
        return canonical_dumps(manifest) + b"\n"

    return tampered_copy(out, sub, "manifest.json", transform)


def set_path(*path, value):
    def mutate(manifest):
        node = manifest
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
    return mutate


def delete_key(key):
    return lambda manifest: manifest.pop(key)


FORGERIES = {
    "rules_version": set_path("versions", "rules_version", value="forged-rules"),
    "balance_version": set_path("versions", "balance_version", value="forged-balance"),
    "config_R0-T0": set_path("configs", "R0-T0", value="0" * 64),
    "policy_parameters": set_path("policy_parameter_sha256", "turtle", value="1" * 64),
    "variants": set_path("variants", value=["A0"]),
    "maps": set_path("maps", value=["R0-T0"]),
    "pairs": set_path("pairs", value=[["turtle", "balanced"]]),
    "seeds": set_path("seeds", value=[1, 2, 3, 4, 5, 6]),
    "seeds_sha256": set_path("seeds_sha256", value="2" * 64),
    "seats": set_path("seats", value=["first_A"]),
    "budget": set_path("budget", "main", value=1),
    "expected_rows": set_path("expected_rows", value=4320),
    "protocol_id": set_path("protocol_id", value="formicmind.starvation-debt/v0"),
    "schema": set_path("schema", value="other"),
    "base_commit": set_path("base_commit", value="3" * 40),
    "protocol_sha256": set_path("protocol_sha256", value="4" * 64),
    "erratum_sha256": set_path("erratum_sha256", value="5" * 64),
    "base_engine_fingerprint": set_path("base_engine_fingerprint", value="6" * 64),
    "engine_fingerprint": set_path("engine_fingerprint_excluding_harness", value="7" * 64),
    "protected_files": set_path("protected_file_sha256", "docs/gate1_stalling_protocol.md", value="8" * 64),
    "window_columns": set_path("window_columns", value=["k"]),
    "checkpoint_ticks": set_path("checkpoint_ticks", value=[0]),
    "extra_key": set_path("surprise", value=True),
    "missing_versions": delete_key("versions"),
    "missing_seeds": delete_key("seeds"),
}


def pure_manifest():
    """A well-formed manifest built only from frozen values (no dependence on the checkout's engine or versions)."""
    manifest = sd.expected_frozen_manifest(len(SHORT), "a" * 40)
    manifest.update({"artifacts": {"matches.jsonl": "0" * 64}, "calls": {"main": 1, "anchor": 0},
                     "source_tree_dirty": False, "package_fingerprint": "e" * 64})
    return manifest


def head_commit():
    return sd._git("rev-parse", "HEAD")


PUBLISHED = sd._ROOT / "results" / "starvation_debt"


def test_the_protocol_and_erratum_are_frozen_and_the_default_rule_is_not_replaced():
    assert sd.protocol_sha256() == sd.PROTOCOL_SHA256
    assert sd._file_sha(sd._ROOT / sd.ERRATUM_PATH) == sd.ERRATUM_SHA256
    assert tick_module.upkeep is economy_module.upkeep


def test_a_moved_engine_is_refused_explicitly_not_silently_accepted(monkeypatch):
    monkeypatch.setattr(sd, "BASE_ENGINE_FINGERPRINT", "0" * 64)
    with pytest.raises(sd.StarvationDebtError, match="engine sources differ from the frozen base"):
        sd._check_static_inputs()


def test_the_schedule_has_exactly_4320_rows_and_the_frozen_axes():
    rows = sd.schedule()
    assert len(rows) == sd.EXPECTED_ROWS == 4320 == 6 * 6 * 10 * 6 * 2
    assert len(set(rows)) == 4320
    assert len(sd.PAIRS) == 10 and sd.SEEDS == tuple(range(20270301, 20270307))
    assert sd.VARIANTS == ("A0", "A1a", "A1b", "A2", "A3a", "A3b")
    assert sd.MAIN_BUDGET == 12_960_000 and sd.ANCHOR_BUDGET == 2_160_000
    assert sd.hashlib.sha256(canonical_dumps(list(sd.SEEDS))).hexdigest() == sd.SEEDS_SHA256
    assert {row[0] for row in rows} == set(ALL) and {row[5] for row in rows} == {False, True}


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


@pytest.mark.parametrize("name", sorted(FORGERIES))
def test_the_field_check_rejects_every_forgery_on_any_tree(name):
    manifest = json.loads(canonical_dumps(pure_manifest()))
    assert sd.check_manifest_fields(manifest, len(SHORT))                       # honest manifest passes
    FORGERIES[name](manifest)
    with pytest.raises(sd.StarvationDebtError, match="deterministic fields differ"):
        sd.check_manifest_fields(manifest, len(SHORT))


def test_the_field_check_ignores_exactly_the_provenance_and_derived_keys():
    manifest = json.loads(canonical_dumps(pure_manifest()))
    manifest.update({"python": "2.7", "platform": "Plan9", "commands": {}, "source_commit": "b" * 40,
                     "package_fingerprint": "c" * 64, "source_tree_dirty": True, "calls": {"main": 9},
                     "artifacts": {"matches.jsonl": "d" * 64}})
    assert sd.check_manifest_fields(manifest, len(SHORT))
    del manifest["python"], manifest["source_tree_dirty"]                       # absent provenance is fine too
    assert sd.check_manifest_fields(manifest, len(SHORT))


def test_every_deterministic_manifest_key_is_covered_by_a_forgery_or_an_explicit_exemption():
    manifest = pure_manifest()
    exempt = sd.PROVENANCE_KEYS | sd.DERIVED_KEYS
    covered = {"versions", "configs", "policy_parameter_sha256", "variants", "maps", "pairs", "seeds", "seeds_sha256",
               "seats", "budget", "expected_rows", "protocol_id", "schema", "base_commit", "protocol_sha256",
               "erratum_sha256", "base_engine_fingerprint", "engine_fingerprint_excluding_harness",
               "protected_file_sha256", "window_columns", "checkpoint_ticks", "episode_columns", "checkpoint_columns"}
    assert set(manifest) - exempt - covered == set()


def test_the_core_hash_covers_the_variant_rules_and_the_frozen_constants(monkeypatch):
    before = sd.experiment_core_sha256()
    monkeypatch.setattr(sd, "A3_STREAK", {"A3a": 4, "A3b": 10})
    assert sd.experiment_core_sha256() != before
    monkeypatch.undo()
    monkeypatch.setattr(sd, "CHRONIC_DEFICIT_TICKS", 31)
    assert sd.experiment_core_sha256() != before


def test_the_published_manifest_passes_the_deterministic_field_checks():
    manifest = json.loads((PUBLISHED / "manifest.json").read_text(encoding="utf-8"))
    assert len(sd.check_manifest_fields(manifest, sd.EXPECTED_ROWS)) >= 20


def test_the_published_manifest_keeps_its_true_generator_provenance_and_artifact_hashes():
    manifest = json.loads((PUBLISHED / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_commit"] == sd.PUBLISHED_GENERATOR_COMMIT
    assert manifest["package_fingerprint"] == sd.PUBLISHED_GENERATOR_FINGERPRINT
    assert "source_tree_dirty" not in manifest                      # the published manifest predates the field
    assert manifest["engine_fingerprint_excluding_harness"] == sd.BASE_ENGINE_FINGERPRINT
    for name, digest in manifest["artifacts"].items():
        assert sd._sha(sd._lf((PUBLISHED / name).read_bytes())) == digest
    if sd._commit_available(sd.PUBLISHED_GENERATOR_COMMIT):          # absent in a shallow clone: not authenticated
        assert sd.check_generator_binding(manifest) == "bound"
    assert sd.experiment_core_sha256() == sd.GENERATOR_CORE_SHA256


def test_the_protected_files_of_earlier_studies_are_unchanged_in_the_current_tree():
    assert sd._protected_hashes() == sd.FROZEN_PROTECTED_FILE_SHA256


def test_the_frozen_expectations_match_the_published_manifest():
    manifest = json.loads((PUBLISHED / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["versions"] == sd.FROZEN_VERSIONS
    assert manifest["protected_file_sha256"] == sd.FROZEN_PROTECTED_FILE_SHA256
    assert manifest["protocol_sha256"] == sd.PROTOCOL_SHA256 and manifest["erratum_sha256"] == sd.ERRATUM_SHA256
    assert manifest["engine_fingerprint_excluding_harness"] == sd.BASE_ENGINE_FINGERPRINT == manifest["base_engine_fingerprint"]


def test_the_manifest_field_check_does_not_depend_on_the_current_tree(monkeypatch):
    manifest = json.loads((PUBLISHED / "manifest.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(sd, "source_fingerprint", lambda exclude_harness=False: "0" * 64)
    monkeypatch.setattr(sd, "_versions", lambda: {"rules_version": "future"})
    monkeypatch.setattr(sd, "protocol_sha256", lambda: "0" * 64)
    monkeypatch.setattr(sd, "_protected_hashes", lambda: {})
    assert len(sd.check_manifest_fields(manifest, sd.EXPECTED_ROWS)) >= 20


def copy_published(tmp_path):
    target = tmp_path / "published"
    target.mkdir()
    for f in PUBLISHED.iterdir():
        (target / f.name).write_bytes(f.read_bytes())
    return target


def test_manifest_only_verification_of_the_published_results_works_on_any_tree():
    result = sd.verify(PUBLISHED, manifest_only=True)
    assert result["ok"] is True and result["mode"] == "manifest_only" and result["rows"] == sd.EXPECTED_ROWS
    assert result["reproducibility"] == "not_verified_in_this_mode" and result["verification_calls"] is None
    assert "versions" in result["deterministic_fields_checked"]


def test_manifest_only_verification_rejects_forged_fields_and_altered_artifacts(tmp_path):
    target = copy_published(tmp_path)
    original = (target / "manifest.json").read_bytes()
    for name in ("rules_version", "config_R0-T0", "variants", "seeds_sha256", "policy_parameters", "protocol_id"):
        manifest = json.loads(original)
        FORGERIES[name](manifest)
        (target / "manifest.json").write_bytes(canonical_dumps(manifest) + b"\n")
        with pytest.raises(sd.StarvationDebtError, match="deterministic fields differ"):
            sd.verify(target, manifest_only=True)
    (target / "manifest.json").write_bytes(original)
    assert sd.verify(target, manifest_only=True)["ok"] is True
    (target / "summary.json").write_bytes((target / "summary.json").read_bytes() + b" ")
    with pytest.raises(sd.StarvationDebtError, match="does not match its manifest hash"):
        sd.verify(target, manifest_only=True)


def test_full_verification_refuses_explicitly_when_the_engine_is_not_the_frozen_one(monkeypatch):
    real = sd.source_fingerprint
    monkeypatch.setattr(sd, "source_fingerprint",
                        lambda exclude_harness=False: "0" * 64 if exclude_harness else real())
    with pytest.raises(sd.StarvationDebtError, match="historical|isolated checkout"):
        sd.verify(PUBLISHED)
    assert tick_module.upkeep is economy_module.upkeep


def test_a_changed_result_defining_core_is_rejected_before_anything_else(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_bytes(canonical_dumps(pure_manifest()) + b"\n")
    assert sd.experiment_core_sha256() == sd.GENERATOR_CORE_SHA256
    monkeypatch.setattr(sd, "GENERATOR_CORE_SHA256", "0" * 64)
    with pytest.raises(sd.StarvationDebtError, match="core differs"):
        sd.verify(tmp_path, 1, _schedule=SHORT, manifest_only=True)


def test_the_historical_fingerprint_is_bound_to_its_commit_when_the_commit_is_available(tmp_path):
    commit = head_commit()
    honest = {"source_commit": commit, "package_fingerprint": sd.fingerprint_at_commit(commit)}
    assert sd.check_generator_binding(honest) == "bound"
    assert sd.check_generator_binding({**honest, "source_tree_dirty": True}) == "not_checked_dirty_worktree"
    assert sd.check_generator_binding({**honest, "source_commit": "c" * 40}) == "commit_unavailable"
    with pytest.raises(sd.StarvationDebtError, match="does not match the sources"):
        sd.check_generator_binding({**honest, "package_fingerprint": "0" * 64})
    for bad in ({"source_commit": "HEAD", "package_fingerprint": "0" * 64},
                {"source_commit": commit, "package_fingerprint": "xyz"}, {}):
        with pytest.raises(sd.StarvationDebtError):
            sd.check_generator_binding(bad)
    lie = pure_manifest()
    lie["source_commit"], lie["package_fingerprint"] = commit, "0" * 64
    lie.pop("source_tree_dirty", None)
    (tmp_path / "manifest.json").write_bytes(canonical_dumps(lie) + b"\n")
    with pytest.raises(sd.StarvationDebtError, match="does not match the sources"):
        sd.verify(tmp_path, 1, _schedule=SHORT, manifest_only=True)
