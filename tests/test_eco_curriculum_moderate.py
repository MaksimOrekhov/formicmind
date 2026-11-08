"""FOR-51: the FOR-50 nest model transferred to ``moderate`` without any further training - lineage of the weights, the same ``nest-v1`` masks,
the committed zero-shot numbers and gates, the verifiable replay, and a real short run on ``moderate``."""

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
OUT = ROOT / "docs" / "ecosystem_examples" / "for51_moderate_transfer"
SOURCE_SHA = "1b4f8740a1b5ea3d"


def _json(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def test_the_evaluated_weights_are_the_committed_for50_model_and_nothing_was_trained():
    model, manifest = load_model(SOURCE)         # refuses a changed weights file, other observation / action versions or a foreign schema
    assert manifest["weights_sha256"].startswith(SOURCE_SHA) and manifest["weights_sha256"] == sha256_file(SOURCE / "weights.pt")
    assert manifest["reward_version"] == 3 and manifest["curriculum"] == "nest-v1" and manifest["training"]["iterations"] == 80
    for name in ("zero_shot_moderate.json", "source_abundance.json"):
        report = _json(name)
        assert report["model"]["weights_sha256"] == manifest["weights_sha256"] and report["curriculum"] == "nest-v1" and report["reward_version"] == 3
        assert report["model"]["trained_curriculum"] == "nest-v1" and report["seeds"] == list(range(90000, 90008)) and report["colonies"] == 8
    assert _json("zero_shot_moderate.json")["scenario"] == "moderate" and _json("source_abundance.json")["scenario"] == "abundance"
    assert not (OUT / "model").exists()          # the gates were met zero-shot: no fine-tune, hence no second model
    assert manifest["training"]["config"]["scenario"] == "abundance"


def test_the_committed_gates_follow_from_the_committed_reports():
    namespace = runpy.run_path(str(OUT / "scripts" / "gates.py"), run_name="gates")          # recomputes from the reports (prints the JSON)
    assert namespace["out"] == _json("gates_moderate.json")
    gates = _json("gates_moderate.json")
    assert gates["all_gates_pass_greedy"] is True and gates["abundance_reproduces_for50_eval"] is True
    for name in ("learned_greedy", "learned"):            # the sampled policy is reported next to the judged greedy one, and passes as well
        assert all(passed for _value, passed in gates["moderate"][name].values())
    zero = _json("zero_shot_moderate.json")
    greedy, sampled, peaceful = zero["learned_greedy"], zero["learned"], zero["heuristic:peaceful"]
    assert zero["colonies"] == 8 and greedy["colonies"] == 64
    assert greedy["functional_survival"] == 1.0 and greedy["workers_mean"] > 19 and greedy["workers_end"] > 22 and greedy["worker_eggs_laid"] > 70
    assert greedy["reward"] > peaceful["reward"] > 0.5 and abs(greedy["reward"] - 0.589) < 0.001
    assert sampled["functional_survival"] < 1.0 and sampled["terminal_reasons"] == {"collapse": 4, "functional_extinction": 1}
    assert greedy["dispatch_attack"] == greedy["contact"] == greedy["combat"] == 0.0           # zero by the mask, not an achievement


def test_the_model_still_does_what_it_did_on_abundance():
    """No training happened, so this is a reproduction check: the same model on the FOR-50 maps gives the FOR-50 numbers exactly."""
    again, nest = _json("source_abundance.json"), json.loads((ROOT / "docs" / "ecosystem_examples" / "for50_nest_pilot" / "eval_final.json").read_text(encoding="utf-8"))
    for name in ("learned_greedy", "learned"):
        assert again[name] == nest[name]


def test_the_committed_moderate_replay_is_reproduced_byte_for_byte_with_the_source_model():
    model, manifest = register_model(SOURCE)
    document = verify_replay((OUT / "replay_moderate_seed90000_greedy.jsonl").read_bytes())
    assert document.policy.curriculum == "nest-v1" and document.header["seed"] == "90000"
    assert {p.variant for p in document.policy.policies} == {variant_for(manifest, "greedy")}
    assert [o["status"] for o in document.result["colony_outcomes"]] == ["alive"] * 8 and document.result["ticks_completed"] == 6000
    assert document.result["event_count"] == 0                      # no scouts and no squads: nothing to discover or fight


def test_a_short_real_run_of_the_source_model_on_moderate_keeps_the_profile(tmp_path):
    out = tmp_path / "e.json"
    code = main(["eval", "--model", str(SOURCE), "--scenario", "moderate", "--worlds", "1", "--ticks", "300", "--no-baselines", "--workers", "1",
                 "--out", str(out)])
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["scenario"] == "moderate" and report["curriculum"] == "nest-v1" and report["model"]["trained_curriculum"] == "nest-v1"
    for name in ("learned", "learned_greedy"):
        assert report[name]["dispatch_attack"] == 0.0 and report[name]["contact"] == 0.0 and report[name]["functional_survival"] > 0.0
