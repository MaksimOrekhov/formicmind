"""FOR-51 stage-2 gates from the committed zero-shot reports (no model, no torch needed). Usage: python gates.py > gates_moderate.json"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
moderate = json.loads((HERE / "zero_shot_moderate.json").read_text(encoding="utf-8"))
abundance = json.loads((HERE / "source_abundance.json").read_text(encoding="utf-8"))
nest = json.loads((HERE.parent / "for50_nest_pilot" / "eval_final.json").read_text(encoding="utf-8"))
peaceful = moderate["heuristic:peaceful"]["reward"]


def gates(row: dict) -> dict:
    return {
        "functional_survival_ge_0.80": [row["functional_survival"], row["functional_survival"] >= 0.8],
        "workers_mean_ge_10": [row["workers_mean"], row["workers_mean"] >= 10],
        "workers_end_ge_5": [row["workers_end"], row["workers_end"] >= 5],
        "worker_eggs_ge_50": [row["worker_eggs_laid"], row["worker_eggs_laid"] >= 50],
        "reward_ge_peaceful_minus_0.15": [row["reward"], row["reward"] >= peaceful - 0.15],
    }


out = {"peaceful_moderate_reward": peaceful, "moderate": {n: gates(moderate[n]) for n in ("learned_greedy", "learned")},
       "abundance_reproduces_for50_eval": all(abundance[n][k] == nest[n][k] for n in ("learned_greedy", "learned")
                                                for k in ("reward", "functional_survival", "workers_mean", "workers_end", "worker_eggs_laid", "delivered"))}
out["all_gates_pass_greedy"] = all(v[1] for v in out["moderate"]["learned_greedy"].values()) and out["abundance_reproduces_for50_eval"]
print(json.dumps(out, indent=2, sort_keys=True))
