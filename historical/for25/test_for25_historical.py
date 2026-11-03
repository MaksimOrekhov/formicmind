"""FOR-25 (historical): checks that need the frozen FOR-24 engine, run in an isolated checkout of the pinned commit.

Executed by the ``historical-for25`` CI job (and locally on a base-equivalent tree):
    python -m pytest historical/for25 -q -ra
Outside that environment these tests fail loudly; they are never skipped. They are not collected by the default
``pytest`` run (``testpaths = ["tests"]``).
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


def test_the_default_config_and_map_hashes_are_untouched():
    for name, config in sd._configs().items():
        assert config_hash(config) == sd.CONFIG_SHA256[name]
    assert default_config().balance.starvation_unit_cost == 5


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


def test_tenths_of_upkeep_are_carried_and_sum_exactly():
    got = sd._tenths_rounding()
    assert got["dues"] == [0, 0, 1, 0, 1, 0, 0, 1, 0, 1] and got["sum"] == 4 and got["remainder_end"] == 0


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


def test_verification_accepts_artifacts_converted_to_crlf_by_a_windows_checkout(short_run, tmp_path):
    out, _ = short_run
    target = tmp_path / "crlf"
    target.mkdir()
    for f in out.iterdir():
        (target / f.name).write_bytes(f.read_bytes().replace(b"\n", b"\r\n"))
    assert sd.verify(target, 1, _schedule=SHORT)["ok"] is True


def test_run_refuses_a_non_empty_directory(short_run):
    out, _ = short_run
    with pytest.raises(sd.StarvationDebtError):
        sd.run(out, 1, _schedule=SHORT)


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


@pytest.mark.parametrize("name", sorted(FORGERIES))
def test_verification_rejects_every_forged_deterministic_manifest_field(short_run, tmp_path, name):
    out, _ = short_run
    target = forge_manifest(out, tmp_path, "f", FORGERIES[name])
    artifacts_before = {f.name: f.read_bytes() for f in out.iterdir() if f.name != "manifest.json"}
    assert artifacts_before == {f.name: f.read_bytes() for f in target.iterdir() if f.name != "manifest.json"}
    with pytest.raises(sd.StarvationDebtError, match="deterministic fields differ|row count"):
        sd.verify(target, 1, _schedule=SHORT)


def test_environment_and_historical_commit_fields_are_provenance_not_compared(short_run, tmp_path):
    out, manifest = short_run
    assert {"source_commit", "package_fingerprint", "python", "platform", "commands"} <= sd.PROVENANCE_KEYS

    def history(m):
        m["python"], m["platform"] = "2.7.0", "Plan9"
        m["commands"] = {"run": "historic"}
        m["source_commit"] = "a" * 40        # a commit this clone does not have
        m["package_fingerprint"] = "b" * 64  # historical harness fingerprint: not the verifier's own
        m.pop("source_tree_dirty", None)

    target = forge_manifest(out, tmp_path, "history", history)
    record = tmp_path / "record.json"
    result = sd.verify(target, 1, _schedule=SHORT, record=record)
    assert result["ok"] is True and result["generator_binding"] == "commit_unavailable"
    rec = json.loads(record.read_text(encoding="utf-8"))
    assert rec["generator"]["source_commit"] == "a" * 40 and rec["generator"]["package_fingerprint"] == "b" * 64
    assert rec["verifier"]["package_fingerprint"] == sd.source_fingerprint()
    assert rec["verifier"]["same_package_as_generator"] is False
    assert rec["verifier"]["core_sha256"] == rec["generator"]["core_sha256"] == sd.GENERATOR_CORE_SHA256
    assert set(rec["provenance_fields_not_compared"]) == set(sd.PROVENANCE_KEYS)
    assert "versions" in rec["deterministic_fields_checked"]
    assert "python" not in rec["deterministic_fields_checked"]


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


def test_the_checkout_is_the_frozen_historical_environment():
    assert sd.source_fingerprint(exclude_harness=True) == sd.BASE_ENGINE_FINGERPRINT
    assert sd._versions() == sd.FROZEN_VERSIONS
    assert sd._protected_hashes() == sd.FROZEN_PROTECTED_FILE_SHA256
    sd._check_static_inputs()                                    # configs, policies, seeds, protocol and erratum


def test_a_short_manifest_only_check_agrees_with_the_full_regeneration(short_run):
    out, _ = short_run
    assert sd.verify(out, 1, _schedule=SHORT, manifest_only=True)["mode"] == "manifest_only"
    assert sd.verify(out, 1, _schedule=SHORT)["mode"] == "full_regeneration"


def test_the_historical_fingerprint_binding_is_checked_end_to_end(short_run, tmp_path):
    out, _ = short_run
    commit = sd._git("rev-parse", "HEAD")

    def lie(m):
        m["source_commit"], m["package_fingerprint"] = commit, "0" * 64
        m.pop("source_tree_dirty", None)

    with pytest.raises(sd.StarvationDebtError, match="does not match the sources"):
        sd.verify(forge_manifest(out, tmp_path, "lie", lie), 1, _schedule=SHORT)
