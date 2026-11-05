"""FOR-38: effective-sample audit, pilot plan/config validity, pilot controllers and budget guards."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from formicmind.baselines import baseline_spec
from formicmind.ml import diversity as dv
from formicmind.ml.cli import main as cli_main
from formicmind.ml.dataset import DatasetError
from formicmind.ml.quality_dataset import collect_quality_dataset, load_quality_dataset
from formicmind.observation import check_observation_config
from formicmind.serialization import config_hash


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("diversity")
    path = root / "quality.jsonl"
    collect_quality_dataset(path, train_seeds=[7101, 7102, 7103], validation_seeds=[7201, 7202],
                            heldout_seeds=[7301, 7302], max_ticks=120)
    manifest, examples = load_quality_dataset(path)
    return {"root": root, "path": path, "manifest": manifest, "examples": examples}


# ----------------------------------------------------------------------------- effective independent units

def test_audit_counts_hierarchy_and_shows_seeds_and_sides_add_no_independent_games(tiny):
    report = dv.effective_sample_audit(tiny["manifest"], tiny["examples"])
    hierarchy = report["hierarchy"]
    assert hierarchy["match_records"] == tiny["manifest"]["match_count"]
    assert hierarchy["retained_decisions"] == len(tiny["examples"])
    assert hierarchy["unique_source_trajectories"] <= hierarchy["seed_context_pairs"] < hierarchy["match_records"]
    # the second seat reproduces the first seat exactly (own-frame observation, mirror-valid maps)
    assert hierarchy["seed_context_pairs_with_identical_trajectory_on_both_sides"] == hierarchy["seed_context_pairs"]
    deterministic = [c for c in report["contexts"] if not c["seed_sensitive"]]
    assert deterministic, "turtle/rush contexts must collapse to one trajectory"
    for context in deterministic:
        assert context["unique_source_trajectories"] == 1
        assert context["matches"] >= 2
        assert context["matches_per_unique_trajectory"] == context["matches"]
    assert report["interpretation"]["rules_consume_randomness"] is False


def test_audit_explains_where_source_decisions_go(tiny):
    report = dv.effective_sample_audit(tiny["manifest"], tiny["examples"], include_coverage=False)
    dropped = report["decision_drop_causes"]
    assert dropped["total_inside_surviving_trajectories"] == (
        dropped.get("position_0_shared_start_state", 0) + dropped.get("later_position", 0)
    )
    assert report["hierarchy"]["unique_source_decisions"] >= report["hierarchy"]["retained_decisions"]
    for split in ("train", "validation", "heldout"):
        info = report["splits"][split]
        assert info["retained_decisions"] <= info["source_decisions"]
        assert 0 < info["decision_retention"] <= 1


def test_audit_flags_contexts_emptied_by_global_dedup():
    manifest = {
        "dataset_sha256": "x", "deduplication": {"duplicate_trajectories_removed": 1, "duplicate_decisions_removed": 0},
        "matches": [
            {"match_id": "t", "split": "train", "context_id": "c", "seed": 1, "teacher_side": 0, "map_id": "classic",
             "opponent_id": "turtle:v2", "trajectory_sha256": "A", "trajectory_decision_sha256s": ["h1", "h2"],
             "source_decision_count": 2, "retained": True, "trajectory_retained": True, "reason": "timeout", "final_tick": 3000},
            {"match_id": "v", "split": "validation", "context_id": "c", "seed": 2, "teacher_side": 0, "map_id": "classic",
             "opponent_id": "turtle:v2", "trajectory_sha256": "A", "trajectory_decision_sha256s": ["h1", "h2"],
             "source_decision_count": 2, "retained": False, "trajectory_retained": False, "reason": "timeout", "final_tick": 3000},
        ],
    }
    ex = type("E", (), {})
    train_rows = []
    for h in ("h1", "h2"):
        row = ex()
        row.split, row.context_id, row.match_id, row.decision_sha256, row.observation = "train", "c", "t", h, (0.0,)
        train_rows.append(row)
    report = dv.effective_sample_audit(manifest, train_rows, include_coverage=False)
    assert report["splits"]["validation"]["contexts_emptied"] == ["c"]
    assert report["splits"]["validation"]["decision_retention"] == 0.0
    assert report["hierarchy"]["unique_source_trajectories"] == 1
    assert report["contexts"][1]["emptied_by_global_dedup"] is True


def test_resimulation_reproduces_every_trajectory_fingerprint(tiny):
    sequences = dv.resimulate_unique_trajectories(tiny["manifest"])
    assert len(sequences) == len({m["trajectory_sha256"] for m in tiny["manifest"]["matches"]})
    for match in tiny["manifest"]["matches"]:
        assert len(sequences[match["trajectory_sha256"]]) == match["source_decision_count"]
    report = dv.effective_sample_audit(tiny["manifest"], tiny["examples"], action_sequences=sequences,
                                       include_coverage=False)
    summary = report["teacher_action_sequences"]
    assert summary["unique_lay_army_avoid_sequences"] <= summary["unique_full_action_sequences"] <= len(sequences)


def test_resimulation_detects_a_changed_fingerprint(tiny):
    broken = json.loads(json.dumps(tiny["manifest"]))
    broken["matches"][0]["trajectory_sha256"] = "f" * 64
    with pytest.raises(DatasetError, match="does not reproduce"):
        dv.resimulate_unique_trajectories(broken)


def test_nearest_neighbor_distances():
    reference = [(0.0, 0.0), (3.0, 4.0)]
    assert dv.nearest_neighbor_distances([(0.0, 0.0), (3.0, 3.0)], reference) == [0.0, 1.0]
    assert dv.nearest_neighbor_distances(reference, reference, exclude_self=True) == [5.0, 5.0]


# ----------------------------------------------------------------------------- maps and plan

def test_pilot_maps_are_valid_distinct_and_keep_balance_and_rules():
    classic = dv.pilot_config("classic")
    hashes = {}
    for map_id in (*dv.EXISTING_MAPS, *dv.PILOT_MAP_EDITS):
        config = dv.pilot_config(map_id)
        check_observation_config(config)
        assert config.rules_version == classic.rules_version
        assert config.balance_version == classic.balance_version
        assert config.balance == classic.balance
        assert config.t_max == classic.t_max
        hashes[map_id] = config_hash(config)
    assert len(set(hashes.values())) == len(hashes), "every pilot topology must be a different configuration"
    with pytest.raises(DatasetError, match="unknown FOR-38 pilot map_id"):
        dv.pilot_config("moon")


def test_pilot_plan_matches_the_preregistered_budget():
    plan = dv.plan_pilot()
    assert len(plan) == 275 <= dv.PILOT_MAX_MATCHES
    assert Counter(p.arm for p in plan) == {"A": 29, "B": 78, "C": 144, "D": 24}
    assert sum(p.control for p in plan) == 5
    assert len({p.match_id for p in plan}) == len(plan)
    seeds = {p.seed for p in plan}
    assert min(seeds) >= 380_000_001 and max(seeds) <= 380_000_999
    assert {p.seed for p in plan if p.seed_sensitive} == set(dv.PILOT_SEEDS_RANDOM) | set(dv.PILOT_SEEDS_PERTURB)
    forbidden = set(range(370_000_001, 373_000_005))  # all FOR-37 seed ranges
    assert not seeds & forbidden


def test_pilot_budget_guard(monkeypatch):
    monkeypatch.setattr(dv, "PILOT_MAX_MATCHES", 10)
    with pytest.raises(DatasetError, match="preregistered maximum"):
        dv.plan_pilot()


def test_split_assignment_is_deterministic_and_sixty_twenty_twenty():
    units = {"A": [f"A|u{i}" for i in range(10)], "B": ["B|x", "B|y", "B|z"]}
    first = dv.assign_pilot_splits(units)
    assert first == dv.assign_pilot_splits({"B": list(reversed(units["B"])), "A": list(reversed(units["A"]))})
    counts = Counter(first[u] for u in units["A"])
    assert counts == {"train": 6, "validation": 2, "heldout": 2}
    assert Counter(first[u] for u in units["B"]) == {"train": 2, "validation": 1}  # ceil(60%), ceil(20%), rest


# ----------------------------------------------------------------------------- controllers

def _spec(arm="C", map_id="classic", opponent="turtle:v2", behavior="teacher", seed=380_000_001, side=0, sensitive=False):
    return dv.PilotMatchSpec(arm, map_id, opponent, behavior, seed, side, sensitive)


def test_labels_are_teacher_logits_even_when_the_played_action_is_perturbed():
    rows, record = dv.run_pilot_match(_spec(behavior="eps1.00", sensitive=True), max_ticks=90)
    assert record["substituted_decisions"] == len(rows) == 3
    from formicmind.actions import ActionMaskV2, mask_from_flat
    from formicmind.decoder import decode_action_logits

    for row in rows:
        mask = mask_from_flat(tuple(row["mask"]))
        decoded = decode_action_logits(row["teacher_logits"], mask)
        assert row["teacher_action"]["forage_share_q"] == list(decoded.forage_share_q)
        assert row["teacher_action"]["lay_order"] == decoded.lay_order
    plain, _ = dv.run_pilot_match(_spec(behavior="teacher"), max_ticks=90)
    assert rows[0]["decision_sha256"] == plain[0]["decision_sha256"]  # tick 0 state is the same before any substitution
    assert [r["decision_sha256"] for r in rows] != [r["decision_sha256"] for r in plain]  # later states diverge


def test_deterministic_arm_ignores_seed_and_side_but_perturbed_arm_does_not():
    a, _ = dv.run_pilot_match(_spec(arm="A", seed=380_000_001), max_ticks=300)
    b, _ = dv.run_pilot_match(_spec(arm="A", seed=380_000_002, side=1), max_ticks=300)
    assert [r["decision_sha256"] for r in a] == [r["decision_sha256"] for r in b]
    c1, _ = dv.run_pilot_match(_spec(behavior="eps0.25", seed=380_000_201, sensitive=True), max_ticks=600)
    c2, _ = dv.run_pilot_match(_spec(behavior="eps0.25", seed=380_000_202, sensitive=True), max_ticks=600)
    again, _ = dv.run_pilot_match(_spec(behavior="eps0.25", seed=380_000_201, sensitive=True), max_ticks=600)
    assert [r["decision_sha256"] for r in c1] == [r["decision_sha256"] for r in again]
    assert [r["decision_sha256"] for r in c1] != [r["decision_sha256"] for r in c2]


def test_scripted_opening_plays_the_script_then_hands_over():
    rows, record = dv.run_pilot_match(_spec(arm="D", behavior="script:workers@2"), max_ticks=150)
    assert record["substituted_decisions"] == 2
    assert len(rows) == 5
    assert all(s.army_choice == 0 and sum(s.forage_share_q) == 65535 for s in dv.OPENING_SCRIPTS.values())


def test_switching_opponent_follows_first_baseline_then_second():
    calls = []

    class Marker:
        def __init__(self, tag):
            self.tag = tag

        def __call__(self, observation, mask, rng):
            calls.append(self.tag)
            return [0.0] * 17

    policy = dv._SwitchingPolicy(Marker("first"), Marker("second"), 2)
    for _ in range(4):
        policy(None, None, None)
    assert calls == ["first", "first", "second", "second"]
    built, policy_id = dv._build_opponent("turtle:v2>rush:v2@6")
    assert isinstance(built, dv._SwitchingPolicy) and policy_id == "switch-turtle-v2-to-rush-v2-at6"
    plain, plain_id = dv._build_opponent("rush:v2")
    assert plain_id == baseline_spec("rush:v2").policy_id and not isinstance(plain, dv._SwitchingPolicy)


# ----------------------------------------------------------------------------- end to end (tiny plan)

def test_pilot_end_to_end_with_a_tiny_plan(tiny, tmp_path):
    plan = [
        _spec("A", "fast_wings", "turtle:v2"), _spec("A", "slow_center", "rush:v2"),
        _spec("A", "wide_ring", "rush:v2"),
        dv.PilotMatchSpec("A", "fast_wings", "turtle:v2", "teacher", 380_000_002, 1, False, control=True),
    ]
    plan += [_spec("C", "classic", "turtle:v2", "eps0.25", 380_000_201 + i, sensitive=True) for i in range(4)]
    output = tmp_path / "pilot.json"
    rows_output = tmp_path / "rows.jsonl"
    report = dv.run_diversity_pilot(tiny["path"], plan=plan, max_ticks=150, output=output, rows_output=rows_output)
    assert report["schema"] == dv.PILOT_SCHEMA
    assert report["budget"] == {"planned_matches": 8, "max_matches": 400, "max_ticks": 150,
                                "executed_matches": 8, "within_budget": True}
    assert report["controls"]["extra_seeds_and_sides_identical_to_main_trajectory"] is True
    assert set(report["arms"]) == {"A", "C"}
    assert report["arms"]["A"]["matches"] == 3
    assert report["arms"]["C"]["unique_trajectories"] >= 1
    assert report["totals"]["retained_after_pilot_dedup"] == report["totals"]["unique_decisions"]
    assert report["for37_reference"]["validation_retention"] < 1
    assert json.loads(output.read_text(encoding="utf-8")) == json.loads(json.dumps(report))
    assert len(rows_output.read_text(encoding="utf-8").splitlines()) == report["totals"]["retained_after_pilot_dedup"]
    first = json.loads(rows_output.read_text(encoding="utf-8").splitlines()[0])
    assert "state" not in first and "observation" in first  # same policy-boundary fields as FOR-37, no hidden state


def test_pilot_refuses_an_over_budget_plan(tiny):
    plan = [_spec("A", "fast_wings", "rush:v2")] * (dv.PILOT_MAX_MATCHES + 1)
    with pytest.raises(DatasetError, match="preregistered maximum"):
        dv.run_diversity_pilot(tiny["path"], plan=plan, max_ticks=10)


def test_pilot_does_not_touch_the_for37_inputs(tiny):
    before = (tiny["path"].read_bytes(), tiny["path"].with_name("quality.manifest.json").read_bytes())
    dv.run_diversity_pilot(tiny["path"], plan=[_spec("A", "fast_wings", "rush:v2")], max_ticks=60)
    assert before == (tiny["path"].read_bytes(), tiny["path"].with_name("quality.manifest.json").read_bytes())


def test_cli_audit_diversity_writes_json(tiny, tmp_path, capsys):
    output = tmp_path / "audit.json"
    assert cli_main(["audit-diversity", "--dataset", str(tiny["path"]), "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema"] == dv.AUDIT_SCHEMA
    assert json.loads(capsys.readouterr().out)["hierarchy"]["match_records"] == tiny["manifest"]["match_count"]


def test_cli_fails_cleanly_on_a_missing_dataset(tmp_path, capsys):
    assert cli_main(["audit-diversity", "--dataset", str(tmp_path / "nope.jsonl"), "--output", str(tmp_path / "o.json")]) == 2
    assert "formicmind-ml" in capsys.readouterr().err


# ----------------------------------------------------------------------------- published FOR-38 results stay consistent

def test_published_for38_json_matches_frozen_inputs_and_budget():
    from pathlib import Path

    from formicmind.ml.diagnostics import FROZEN_FOR37_SHA256

    docs = Path(__file__).resolve().parents[1] / "docs" / "ml_for38"
    diagnostics = json.loads((docs / "diagnostics.json").read_text(encoding="utf-8"))
    audit = json.loads((docs / "effective_samples.json").read_text(encoding="utf-8"))
    pilot = json.loads((docs / "pilot.json").read_text(encoding="utf-8"))
    dataset = FROZEN_FOR37_SHA256["quality.jsonl"]
    assert diagnostics["provenance"]["dataset_sha256"] == dataset == audit["dataset_sha256"] == pilot["dataset_for37_sha256"]
    assert diagnostics["provenance"]["weights_sha256"] == FROZEN_FOR37_SHA256["checkpoint/weights.pt"]
    assert all(info["match"] for info in diagnostics["provenance"]["frozen_artifacts"].values())
    assert diagnostics["splits"]["heldout"]["action_accuracy"]["forage_share_exact"] == 0.0
    assert audit["hierarchy"]["unique_source_decisions"] == audit["hierarchy"]["retained_decisions"] == 964
    assert pilot["budget"]["within_budget"] and pilot["budget"]["executed_matches"] == 275 <= dv.PILOT_MAX_MATCHES
    assert pilot["controls"]["extra_seeds_and_sides_identical_to_main_trajectory"] is True
    assert pilot["arms"]["A"]["meets_preregistered_rule"] is True
    assert pilot["arms"]["D"]["meets_preregistered_rule"] is False
