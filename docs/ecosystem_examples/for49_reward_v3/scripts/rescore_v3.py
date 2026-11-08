"""Offline re-scoring (NO training) of the historical policies under reward v3 on the held-out maps 90000..90007.

usage: python rescore_v3.py OUT.json PILOT1_MODEL PILOT2_MODEL
The natural world is played (nobody is retired); the v3 terminal only shows in the scores. reward = v3; reward_v1 / reward_v2 are the older rewards of
the same run and are not on the same scale.
"""
import json
import sys
import time
from pathlib import Path

import torch

from formicmind.eco.rl.checkpoint import load_model
from formicmind.eco.rl.evaluate import evaluation_suite
from formicmind.eco.rl.network import ActorCriticE1, NetworkConfig
from formicmind.eco.rl.train import TrainConfig, _Pool

KEYS = ("reward", "reward_v2", "reward_v1", "physical_survival", "functional_survival", "functional_extinction", "hollow", "workers_end", "adults_end",
        "workers_mean", "adults_mean", "brood_end", "eggs_laid", "delivered", "discover", "contact", "dispatch_attack", "combat")


def main() -> None:
    out, pilot1_dir, pilot2_dir = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    pilot1, _ = load_model(pilot1_dir)
    pilot2, _ = load_model(pilot2_dir)
    torch.manual_seed(1)
    untrained = ActorCriticE1(NetworkConfig())
    config = TrainConfig(eval_worlds=8, workers=8, ticks=6000)
    pool = _Pool(8)
    t0 = time.perf_counter()
    try:
        report = evaluation_suite(pilot2, config, pool, label="offline re-scoring under reward v3 (no training)", baselines=True, greedy=False,
                                  extra={"pilot1_model_reward_v1": pilot1, "untrained_init": untrained})
    finally:
        pool.close()
    report["learned"] = report.pop("learned")          # this entry is the reward-v2 pilot model (iteration 89)
    report["note"] = ("'learned' is the reward-v2 pilot model (stopped at iteration 89); 'pilot1_model_reward_v1' is the first pilot's model. "
                      "All scored by reward v3; reward_v1 / reward_v2 are the older rewards of the same runs, not comparable in scale.")
    report["eval_wall_s"] = time.perf_counter() - t0
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    for name, row in report.items():
        if isinstance(row, dict) and "reward" in row:
            print(name.ljust(26), " ".join(f"{k}={row[k]:.3f}" for k in KEYS), row.get("terminal_reasons"))


if __name__ == "__main__":
    main()
