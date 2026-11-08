"""Handlers of ``python -m formicmind.eco train | eval`` and the model loading shared with ``policy-run`` / ``replay-verify`` (needs PyTorch)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from ..config import default_config
from ..policy import PolicyRunSpec, PolicySpec
from ..replay_file import export_policy_replay, read_replay, verify_replay
from .checkpoint import load_model
from .evaluate import evaluation_suite
from .learned import register_model, variant_for
from .ppo import PPOConfig
from .train import TrainConfig, _Pool, train


def load_models(paths: list[str]) -> None:
    for path in paths or []:
        register_model(path)


def cmd_train(args) -> int:
    config = TrainConfig(
        colonies=args.colonies, scenario=args.scenario, ticks=args.ticks, iterations=args.iterations, worlds_per_iteration=args.worlds,
        workers=args.workers, seed=args.seed, train_seed0=args.train_seed0, eval_seed0=args.eval_seed0, eval_worlds=args.eval_worlds,
        eval_every=args.eval_every, reward_scale=args.reward_scale,
        ppo=PPOConfig(lr=args.lr, epochs=args.epochs, seqs_per_minibatch=args.minibatch_seqs))
    train(config, args.out, init_model=args.init, log=lambda text: (sys.stderr.write(text + "\n"), sys.stderr.flush()))
    return 0


def cmd_eval(args) -> int:
    model, manifest = load_model(args.model)
    config = TrainConfig(colonies=args.colonies, scenario=args.scenario, ticks=args.ticks, workers=args.workers,
                         eval_seed0=args.seed0, eval_worlds=args.worlds)
    pool = _Pool(args.workers)
    try:
        report = evaluation_suite(model, config, pool, label=str(args.model), baselines=not args.no_baselines, greedy=True)
    finally:
        pool.close()
    report["model"] = {"weights_sha256": manifest["weights_sha256"], "path": str(args.model)}
    for name in (k for k in report if isinstance(report[k], dict) and "reward" in report[k]):
        row = report[name]
        sys.stderr.write(f"{name:22s} reward={row['reward']:.4f} phys={row['physical_survival']:.2f} func={row['functional_survival']:.2f} hollow={row['hollow']:.2f} workers={row['workers_end']:.1f} "
                         f"eggs={row['eggs_laid']:.0f} brood={row['brood_end']:.1f} delivered={row['delivered']:.0f} contacts={row['contact']:.1f} attacks={row['dispatch_attack']:.2f}\n")
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8", newline="\n")
    else:
        sys.stdout.write(text)
    if args.replay_out:
        register_model(args.model)
        replay_config = default_config(args.colonies, args.scenario)
        seed = args.replay_seed if args.replay_seed is not None else args.seed0
        variant = variant_for(manifest, args.replay_mode)
        spec = PolicyRunSpec(args.policy_seed, tuple(PolicySpec("learned", variant) for _ in range(replay_config.n_colonies)))
        started = time.perf_counter()
        replay, decisions = export_policy_replay(replay_config, seed, spec, args.replay_ticks, args.frame_every)
        Path(args.replay_out).write_bytes(replay)
        if args.decisions:
            Path(args.decisions).write_bytes(decisions)
        result = read_replay(replay).result
        sys.stderr.write(f"wrote {args.replay_out}: {len(replay)} bytes, decisions {result['decision_count']}, events {result['event_count']}, "
                         f"statuses={[o['status'] for o in result['colony_outcomes']]} ({time.perf_counter() - started:.1f} s)\n")
        if args.verify:
            verify_replay(replay, decisions if args.decisions else None)
            sys.stderr.write("verified: strict read OK and re-execution with the model reproduces the replay byte for byte\n")
    return 0
