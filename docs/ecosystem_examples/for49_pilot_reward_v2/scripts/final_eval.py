"""Finalise the stopped reward-v2 pilot and evaluate it once on the held-out maps 90000..90007 (no training)."""
import json
import sys
import time
from pathlib import Path

import torch

from formicmind.eco.rl.checkpoint import load_model, save_model
from formicmind.eco.rl.evaluate import evaluation_suite
from formicmind.eco.rl.network import ActorCriticE1, NetworkConfig
from formicmind.eco.rl.train import TrainConfig, _Pool

OUT = Path(sys.argv[1])
PILOT1 = Path(sys.argv[2])


def main() -> None:
    cfg = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
    latest, manifest = load_model(OUT / "latest")
    iteration = manifest["training"]["iteration"]
    rows = [json.loads(x) for x in (OUT / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    save_model(OUT / "final", latest, training={"iterations": iteration, "planned_iterations": 120, "stopped_by_owner": True,
                                                 "config": {k: v for k, v in cfg.items() if k != "reward_version"},
                                                 "wall_seconds": rows[-1]["elapsed_s"]})
    config = TrainConfig(eval_worlds=8, workers=8, ticks=6000)
    torch.manual_seed(1)
    untrained = ActorCriticE1(NetworkConfig())
    pilot1, _ = load_model(PILOT1)
    pool = _Pool(8)
    t0 = time.perf_counter()
    try:
        report = evaluation_suite(latest, config, pool, label=f"final (reward v2 pilot stopped at iteration {iteration})", baselines=True,
                                  greedy=True, extra={"untrained_init": untrained, "pilot1_reward_v1_model": pilot1})
    finally:
        pool.close()
    report["eval_wall_s"] = time.perf_counter() - t0
    (OUT / "eval_final.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    keys = ("reward", "reward_v1", "survival", "hollow", "workers_end", "workers_mean", "workers_peak", "brood_end", "eggs_laid", "delivered",
            "discover", "contact", "dispatch_attack", "combat")
    for name, row in report.items():
        if isinstance(row, dict):
            print(name.ljust(26), " ".join(f"{k}={row[k]:.3f}" for k in keys))


if __name__ == "__main__":
    main()
