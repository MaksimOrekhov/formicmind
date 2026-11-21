"""FOR-52 stage-3 (scout) gates from a committed evaluation report (no model, no torch needed).

Usage: python gates.py <report.json> > gates.json      (default: ../zero_shot_scout_moderate.json)

The thresholds below were fixed in the commit that precedes the first evaluation (docs/ecosystem_v1_curriculum_scout.md, "Фиксация до запуска") and are
not tuned afterwards. The greedy policy is judged; the sampled one is reported next to it. ``discovered_by_scout`` is the strict count (a node first
visited while only scout cohorts of the colony stood on it); nodes where a scout and a forager arrived in the same tick are kept apart (``discovered_mixed``)
and give no credit to either.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
GATES = {                                                      # name: (key, comparison, threshold)
    "functional_survival_ge_0.80": ("functional_survival", ">=", 0.80),
    "workers_mean_ge_10": ("workers_mean", ">=", 10),
    "workers_end_ge_5": ("workers_end", ">=", 5),
    "worker_eggs_ge_50": ("worker_eggs_laid", ">=", 50),
    "share_colonies_scouting_new_node_ge_0.50": ("share_scouted_new_node", ">=", 0.50),
    "unique_new_nodes_by_scouts_per_colony_ge_2": ("discovered_by_scout", ">=", 2.0),
    "no_attacks": ("dispatch_attack", "==", 0.0),
    "no_combat": ("combat", "==", 0.0),
}


def passes(value: float, op: str, threshold: float) -> bool:
    return value >= threshold if op == ">=" else value == threshold


def gates(row: dict) -> dict:
    return {name: [row[key], passes(row[key], op, threshold)] for name, (key, op, threshold) in GATES.items()}


def main(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    out = {"report": path.name, "thresholds": {name: [key, op, threshold] for name, (key, op, threshold) in GATES.items()},
           "learned_greedy": gates(report["learned_greedy"]), "learned": gates(report["learned"])}
    out["all_gates_pass_greedy"] = all(passed for _value, passed in out["learned_greedy"].values())
    out["all_gates_pass_sample"] = all(passed for _value, passed in out["learned"].values())
    return out


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "zero_shot_scout_moderate.json"), indent=2, sort_keys=True))
