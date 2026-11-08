"""Where does the hollow strategy lose to the peaceful one? Mean viability (reward v2) and cumulative discounted return per 500-tick window."""
import io
import json
import sys

import torch

from formicmind.eco.config import default_config
from formicmind.eco.rl.checkpoint import load_model
from formicmind.eco.rl.evaluate import _controller, _weights
from formicmind.eco.rl.play import play_world
from formicmind.eco.viability import Q, gamma_tick

model, _ = load_model(sys.argv[1])
config = default_config(8, "moderate")
W = 500
out = {}
for name, kind in (("learned_v2", "learned-sample"), ("peaceful", "heuristic:peaceful")):
    rows = []
    for seed in range(90000, 90004):
        job = {"kind": kind, "weights": _weights(model) if kind.startswith("learned") else b"", "network": model.config.to_dict() if kind.startswith("learned") else {}}
        res = play_world(config, seed, _controller(job, config), 6000)
        for c in range(8):
            v = res.tracker.traces[c].v_q
            v = v + [0] * (6000 - len(v))
            rows.append(v)
    n = len(rows)
    means = [sum(sum(r[i:i + W]) for r in rows) / (n * W * Q) for i in range(0, 6000, W)]
    g = gamma_tick()
    disc_total = sum(sum(x * g ** t for t, x in enumerate(r)) for r in rows) / (n * Q * 6000)
    out[name] = {"window_mean_v": means, "cum_reward": [sum(means[: k + 1]) * W / 6000 for k in range(len(means))], "disc_return": disc_total}
    print(name, "mean v per 500 ticks:", " ".join(f"{m:.2f}" for m in means))
    print(name, "cumulative reward   :", " ".join(f"{m:.3f}" for m in out[name]["cum_reward"]), "disc", round(disc_total, 4))
print(json.dumps(out))
