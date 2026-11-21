"""FOR-52: the committed scout (stage 3) artifacts - zero-shot failure, the single bounded pilot, its honest gates, lineage of the weights, the
verifiable replay in which scouts physically find nodes, and a real short run under ``scout-v1``."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

from formicmind.eco.cli import main
from formicmind.eco.replay_file import verify_replay
from formicmind.eco.rl.checkpoint import load_model, sha256_file
from formicmind.eco.rl.learned import register_model, variant_for

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "ecosystem_examples" / "for50_nest_pilot" / "model"
OUT = ROOT / "docs" / "ecosystem_examples" / "for52_scout"
MODEL = OUT / "model"
SOURCE_SHA = "1b4f8740a1b5ea3d5c2e5c9012c9dd23959c84eca25640f8b33c21a782c55d44"


def _json(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def test_the_pilot_started_from_the_for50_weights_and_kept_the_registered_budget_and_hyperparameters():
    source = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    assert source["weights_sha256"] == SOURCE_SHA == sha256_file(SOURCE / "weights.pt")
    provenance = _json("provenance.json")
    assert provenance["initial_weights"]["sha256"] == SOURCE_SHA and provenance["single_pilot"] is True and provenance["stop_rule_fired"] is False
    _model, manifest = load_model(MODEL)           # refuses a changed weights file, other observation / action versions or a foreign schema
    assert manifest["weights_sha256"] == provenance["final_weights"]["sha256"] == sha256_file(MODEL / "weights.pt") != SOURCE_SHA
    assert manifest["curriculum"] == "scout-v1" and manifest["reward_version"] == 3 and manifest["parameters"] == source["parameters"] == 430693
    config = manifest["training"]["config"]
    assert manifest["training"]["iterations"] == 80 and config["worlds_per_iteration"] == 16 and config["scenario"] == "moderate" and config["seed"] == 1
    assert config["ppo"] == source["training"]["config"]["ppo"] and config["reward_scale"] == source["training"]["config"]["reward_scale"] == 10.0
    assert config["network"] == source["training"]["config"]["network"] and config["ticks"] == 6000 and config["colonies"] == 8
    assert _json("config.json")["curriculum"] == "scout-v1" and _json("config.json")["iterations"] == 80
    assert len((OUT / "metrics.jsonl").read_text(encoding="utf-8").splitlines()) == 80
    assert not (OUT / "latest").exists()           # only the final model is kept; the resume checkpoint is not part of the evidence


def test_zero_shot_failed_the_economy_gates_but_scouting_was_physical():
    zero, gates = _json("zero_shot_scout_moderate.json"), _json("gates_zero_shot.json")
    assert zero["curriculum"] == "scout-v1" and zero["model"]["weights_sha256"] == SOURCE_SHA and zero["scenario"] == "moderate"
    assert gates["all_gates_pass_greedy"] is False and gates["all_gates_pass_sample"] is False
    greedy, peaceful = zero["learned_greedy"], zero["heuristic:peaceful"]
    assert greedy["functional_survival"] == 0.0 and greedy["workers_mean"] < 2 and greedy["worker_eggs_laid"] == 0
    assert greedy["share_scouted_new_node"] == 1.0 and greedy["scout_arrivals"] > 10 and greedy["combat"] == 0.0
    assert peaceful["functional_survival"] > 0.9           # the world itself was fine: the untrained scout heads wrecked the economy


def test_the_pilot_restored_the_economy_but_did_not_learn_to_scout_and_the_gates_say_so():
    namespace = runpy.run_path(str(OUT / "scripts" / "gates.py"), run_name="gates")
    final = _json("eval_final.json")
    assert final["curriculum"] == "scout-v1" and final["scenario"] == "moderate" and final["seeds"] == list(range(90000, 90008)) and final["colonies"] == 8
    gates = _json("gates_final.json")
    assert gates == namespace["main"](OUT / "eval_final.json") and gates["all_gates_pass_greedy"] is False and gates["all_gates_pass_sample"] is False
    for name in ("learned_greedy", "learned"):
        failed = {gate for gate, (_value, passed) in gates[name].items() if not passed}
        assert failed == {"share_colonies_scouting_new_node_ge_0.50", "unique_new_nodes_by_scouts_per_colony_ge_2"}       # only scouting fails
    greedy, sampled, peaceful = final["learned_greedy"], final["learned"], final["heuristic:peaceful"]
    assert greedy["functional_survival"] == 1.0 and greedy["workers_mean"] > 20 and greedy["worker_eggs_laid"] > 50 and greedy["reward"] > peaceful["reward"]
    assert greedy["scout_orders"] == greedy["scouts_sent"] == greedy["discovered_by_scout"] == 0.0         # the judged policy never sends a scout
    assert 0 < sampled["discovered_by_scout"] < 0.5 and sampled["share_scouted_new_node"] < 0.5 and sampled["scout_arrivals"] > 50     # scouts walk to known nodes
    for name in ("learned_greedy", "learned"):
        assert final[name]["dispatch_attack"] == final[name]["combat"] == final[name]["contact"] == 0.0
        assert final[name]["discovered_by_scout"] + final[name]["discovered_by_forage"] + final[name]["discovered_mixed"] \
            + final[name]["discovered_other"] == final[name]["discover"]             # the attribution never double counts
    for name in ("eval_0020.json", "eval_0040.json", "eval_0060.json"):
        assert _json(name)["curriculum"] == "scout-v1" and _json(name)["learned"]["functional_survival"] > 0.9        # the stop rule had nothing to stop


def test_the_economy_and_the_old_profile_are_not_worse_than_before_the_scout_stage():
    under_nest, scout = _json("final_under_nest_v1_moderate.json"), _json("eval_final.json")
    before = json.loads((ROOT / "docs" / "ecosystem_examples" / "for51_moderate_transfer" / "zero_shot_moderate.json").read_text(encoding="utf-8"))
    assert under_nest["curriculum"] == "nest-v1" and under_nest["learned_greedy"]["functional_survival"] == 1.0
    assert under_nest["learned_greedy"]["workers_mean"] >= before["learned_greedy"]["workers_mean"] - 1           # no loss against FOR-51
    for key in ("functional_survival", "workers_mean", "workers_end", "worker_eggs_laid", "delivered", "reward"):       # greedy never scouts: same numbers
        assert under_nest["learned_greedy"][key] == scout["learned_greedy"][key]
    abundance = _json("final_abundance_scout_v1.json")
    assert abundance["scenario"] == "abundance" and abundance["learned_greedy"]["functional_survival"] == 1.0 and abundance["learned_greedy"]["workers_mean"] > 29


def test_the_committed_replay_is_reproduced_byte_for_byte_and_shows_scouts_finding_nodes():
    _model, manifest = register_model(MODEL)
    raw = (OUT / "replay_moderate_seed90002_sample.jsonl").read_bytes()
    document = verify_replay(raw)
    assert document.policy.curriculum == "scout-v1" and document.header["seed"] == "90002"
    assert {p.variant for p in document.policy.policies} == {variant_for(manifest, "sample")}
    assert [o["status"] for o in document.result["colony_outcomes"]] == ["alive"] * 8 and document.result["ticks_completed"] == 6000
    events = [json.loads(line) for line in raw.decode("utf-8").splitlines() if '"kind"' in line]
    assert sum(1 for e in events if e.get("kind") == "discover") == 5 and not any(e.get("kind") in ("combat", "contact") for e in events)


def test_a_short_real_run_of_the_pilot_model_keeps_the_scout_profile(tmp_path):
    out = tmp_path / "e.json"
    assert main(["eval", "--model", str(MODEL), "--scenario", "moderate", "--worlds", "1", "--ticks", "600", "--no-baselines", "--workers", "1",
                 "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["curriculum"] == "scout-v1" and report["model"]["trained_curriculum"] == "scout-v1"
    for name in ("learned", "learned_greedy"):
        assert report[name]["dispatch_attack"] == 0.0 and report[name]["combat"] == 0.0 and report[name]["functional_survival"] > 0.0
