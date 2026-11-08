"""Finalise the early-stopped reward-v3 pilot and evaluate it once on the held-out maps 90000..90007 under reward v3 (no training).

usage: python final_eval_v3.py PILOT_DIR PILOT1_MODEL PILOT2_MODEL
"""
import json
import sys
import time
from pathlib import Path

import torch

from formicmind.eco.rl.checkpoint import load_model, save_model
from formicmind.eco.rl.evaluate import evaluation_suite
from formicmind.eco.rl.network import ActorCriticE1, NetworkConfig
from formicmind.eco.rl.train import TrainConfig, _Pool

KEYS = ("reward", "reward_v2", "reward_v1", "physical_survival", "functional_survival", "functional_extinction", "hollow", "workers_end", "adults_end",
        "workers_mean", "adults_mean", "workers_peak", "brood_end", "eggs_laid", "delivered", "discover", "contact", "dispatch_attack", "combat")


def main() -> None:
    out, pilot1_dir, pilot2_dir = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    latest, manifest = load_model(out / "latest")
    iteration = manifest["training"]["iteration"]
    rows = [json.loads(x) for x in (out / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) >= iteration, (len(rows), iteration)
    save_model(out / "final", latest, training={
        "iterations": iteration, "planned_iterations": 120, "stopped_early": True,
        "stop_rule": "iteration-80 evaluation: functional survival 0%, ~1 egg per colony, no adults at the horizon (rule of the owner's FOR-49 v3 decision)",
        "config": {k: v for k, v in cfg.items() if k != "reward_version"}, "wall_seconds": rows[iteration - 1]["elapsed_s"]})
    config = TrainConfig(eval_worlds=8, workers=8, ticks=6000)
    torch.manual_seed(1)
    untrained = ActorCriticE1(NetworkConfig())
    pilot1, _ = load_model(pilot1_dir)
    pilot2, _ = load_model(pilot2_dir)
    pool = _Pool(8)
    t0 = time.perf_counter()
    try:
        report = evaluation_suite(latest, config, pool, label=f"final (reward v3 pilot stopped early at iteration {iteration})", baselines=True,
                                  greedy=True, extra={"untrained_init": untrained, "pilot1_model_reward_v1": pilot1, "pilot2_model_reward_v2": pilot2})
    finally:
        pool.close()
    report["eval_wall_s"] = time.perf_counter() - t0
    (out / "eval_final.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    for name, row in report.items():
        if isinstance(row, dict) and "reward" in row:
            print(name.ljust(26), " ".join(f"{k}={row[k]:.3f}" for k in KEYS), row.get("terminal_reasons"))


if __name__ == "__main__":
    main()
