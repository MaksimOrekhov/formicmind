"""Argument definitions of the learning commands (torch-free, so ``python -m formicmind.eco --help`` never imports PyTorch)."""

from __future__ import annotations

from ..curriculum import CURRICULUM_PROFILES

SCENARIOS = ("scarcity", "moderate", "abundance")


def add_rl_commands(sub, add_world_args) -> None:
    t = sub.add_parser("train", help="PPO training of the mirrored colonies on real E1 worlds (needs PyTorch)")
    t.add_argument("--out", required=True, help="output directory: config.json, metrics.jsonl, latest/, final/, eval_final.json")
    t.add_argument("--colonies", type=int, default=8, choices=(6, 7, 8))
    t.add_argument("--scenario", default="moderate", choices=SCENARIOS)
    t.add_argument("--iterations", type=int, default=40)
    t.add_argument("--worlds", type=int, default=8, help="complete worlds played per iteration (each gives `colonies` trajectories)")
    t.add_argument("--ticks", type=int, default=6000, help="world ticks per episode (< 6000 truncates and bootstraps with the critic)")
    t.add_argument("--workers", type=int, default=1, help="rollout worker processes (1 = in this process)")
    t.add_argument("--seed", type=int, default=1, help="network initialisation, policy streams, minibatch order")
    t.add_argument("--train-seed0", type=int, default=200000, help="first training map seed; every world of every iteration gets a new one")
    t.add_argument("--eval-seed0", type=int, default=90000, help="first held-out evaluation map seed")
    t.add_argument("--eval-worlds", type=int, default=4, help="held-out worlds of the final (and periodic) evaluation; 0 disables it")
    t.add_argument("--eval-every", type=int, default=0, help="periodic evaluation of the learned policy every N iterations (0 = final only)")
    t.add_argument("--lr", type=float, default=3.0e-4)
    t.add_argument("--epochs", type=int, default=4)
    t.add_argument("--minibatch-seqs", type=int, default=16, help="colony trajectories per PPO minibatch")
    t.add_argument("--reward-scale", type=float, default=10.0)
    t.add_argument("--init", default=None, help="start from the weights of this model directory instead of a fresh network")
    t.add_argument("--curriculum", default="none", choices=CURRICULUM_PROFILES,
                   help="action-mask profile of every rollout and evaluation (FOR-50): nest-v1 = workers only, no scouts, no squad orders (a diagnostic stage, not a free agent)")

    e = sub.add_parser("eval", help="evaluate a trained model next to scripted baselines on held-out maps; optionally export a viewer replay")
    e.add_argument("--model", required=True, help="model directory (weights.pt + manifest.json)")
    e.add_argument("--colonies", type=int, default=8, choices=(6, 7, 8))
    e.add_argument("--scenario", default="moderate", choices=SCENARIOS)
    e.add_argument("--worlds", type=int, default=4)
    e.add_argument("--seed0", type=int, default=90000, help="first held-out map seed")
    e.add_argument("--ticks", type=int, default=6000)
    e.add_argument("--workers", type=int, default=1)
    e.add_argument("--no-baselines", action="store_true", help="evaluate only the learned policy")
    e.add_argument("--curriculum", default=None, choices=CURRICULUM_PROFILES,
                   help="action-mask profile of the evaluation and of the replay (default: the profile the model was trained under; `none` tests the transfer to the full action space)")
    e.add_argument("--out", default=None, help="write the evaluation report (JSON) here")
    e.add_argument("--replay-out", default=None, help="also export one evaluation replay of the learned policy for the E1 viewer")
    e.add_argument("--replay-seed", type=int, default=None, help="map seed of the replay (default: the first held-out seed)")
    e.add_argument("--replay-mode", default="sample", choices=("sample", "greedy"))
    e.add_argument("--replay-ticks", type=int, default=6000)
    e.add_argument("--frame-every", type=int, default=40)
    e.add_argument("--policy-seed", type=int, default=1, help="root seed of the colonies' private random streams in the replay")
    e.add_argument("--decisions", default=None, help="also write the decision records (tick, colony, observation hash, action) as JSONL")
    e.add_argument("--verify", action="store_true", help="re-execute the exported replay with the model and demand identical bytes")
