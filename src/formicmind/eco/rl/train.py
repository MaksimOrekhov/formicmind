"""The first real PPO training of Ecosystem v1 colonies (FOR-49): mirror mode, real worlds, fresh maps every iteration.

Each iteration plays ``worlds_per_iteration`` complete E1 worlds (``colonies`` colonies each, all driven by the one network with a private memory
and random stream per colony), possibly in worker processes, merges the recorded colony trajectories and does one PPO update. Maps are never
reused (map seed = ``train_seed0 + iteration * worlds + j``) and the evaluation maps (``eval_seed0 ...``) are disjoint from them.
"""

from __future__ import annotations

import copy
import io
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from ..config import default_config
from ..viability import GAMMA_30, LAMBDA_30, REWARD_VERSION
from .checkpoint import load_model, save_model
from .network import ActorCriticE1, NetworkConfig
from .ppo import PPOConfig, merge_episodes, ppo_update
from .rollout import Episode, collect_episode


@dataclass(frozen=True, slots=True)
class TrainConfig:
    colonies: int = 8
    scenario: str = "moderate"
    ticks: int = 6000                 # world ticks per training episode (the horizon is 6000; shorter = truncated with a critic bootstrap)
    iterations: int = 40
    worlds_per_iteration: int = 8
    workers: int = 1
    seed: int = 1
    train_seed0: int = 200000
    eval_seed0: int = 90000
    eval_worlds: int = 4
    eval_every: int = 0               # 0: only a final evaluation
    reward_scale: float = 10.0        # the trainer multiplies the (tiny, 1/H-scaled) interval rewards; advantages are normalised anyway
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def _policy_seed(config: TrainConfig, index: int) -> int:
    return (config.seed * 1_000_003 + index) % (1 << 63)


# ------------------------------------------------------------------------------------------------ worker functions (module level: picklable)


def _collect_job(job: dict) -> bytes:
    torch.set_num_threads(1)
    model = ActorCriticE1(NetworkConfig(**job["network"]))
    model.load_state_dict(torch.load(io.BytesIO(job["weights"]), weights_only=True))
    config = default_config(job["colonies"], job["scenario"])
    episode = collect_episode(config, job["seed"], model, job["policy_seed"], job["ticks"], reward_scale=job["reward_scale"],
                              gamma30=GAMMA_30, lambda30=LAMBDA_30)
    buffer = io.BytesIO()
    torch.save(episode, buffer)
    return buffer.getvalue()


def _weights_bytes(model: ActorCriticE1) -> bytes:
    buffer = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, buffer)
    return buffer.getvalue()


class _Pool:
    """``workers <= 1``: run in this process; otherwise a persistent spawn pool (Windows-safe)."""

    def __init__(self, workers: int) -> None:
        self.workers = workers
        self.executor = None
        self._start()

    def _start(self) -> None:
        if self.workers > 1:
            self.executor = ProcessPoolExecutor(max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"))

    def map(self, function, jobs: list):
        if self.executor is None:
            return [function(job) for job in jobs]
        for attempt in range(3):
            try:
                return list(self.executor.map(function, jobs))
            except (BrokenProcessPool, MemoryError, RuntimeError) as exc:     # a worker lost to memory pressure: the jobs are deterministic, run them again
                if attempt == 2:
                    raise
                sys.stderr.write(f"worker failure ({type(exc).__name__}: {str(exc)[:80]}); restarting the pool and repeating the jobs\n")
                self.executor.shutdown(wait=False, cancel_futures=True)
                self._start()

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown()


def collect(pool: _Pool, model: ActorCriticE1, config: TrainConfig, iteration: int) -> list[Episode]:
    weights = _weights_bytes(model)
    jobs = []
    for j in range(config.worlds_per_iteration):
        index = iteration * config.worlds_per_iteration + j
        jobs.append({"weights": weights, "network": config.network.to_dict(), "colonies": config.colonies, "scenario": config.scenario,
                     "seed": config.train_seed0 + index, "policy_seed": _policy_seed(config, index), "ticks": config.ticks,
                     "reward_scale": config.reward_scale})
    return [torch.load(io.BytesIO(blob), weights_only=False) for blob in pool.map(_collect_job, jobs)]


def episode_summary(episodes: list[Episode]) -> dict:
    rows = [row for ep in episodes for row in ep.metrics]
    n = len(rows)
    return {
        "reward": sum(r["reward"] for r in rows) / n, "mean_viability": sum(r["mean_viability"] for r in rows) / n,
        "survival": sum(1 for r in rows if r["survived"]) / n, "hollow": sum(1 for r in rows if r["hollow"]) / n,
        "workers_end": sum(r["workers_end"] for r in rows) / n,
        "delivered": sum(r["delivered"] for r in rows) / n, "contacts": sum(r["contact"] for r in rows) / n,
        "attacks": sum(r["dispatch_attack"] for r in rows) / n, "discovered": sum(r["discover"] for r in rows) / n,
        "steps": sum(ep.batch.size for ep in episodes), "sim_wall_mean": sum(ep.wall for ep in episodes) / len(episodes),
    }


def train(config: TrainConfig, out_dir: str | Path, *, init_model: str | Path | None = None, log=print) -> dict:
    """Run the training; returns the final summary. Writes ``config.json``, ``metrics.jsonl`` and the models ``latest/`` and ``final/``."""
    from .evaluate import evaluation_suite

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    if init_model:
        model, _ = load_model(init_model)
    else:
        model = ActorCriticE1(config.network)
    untrained = None if init_model else copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.ppo.lr, eps=1e-5)
    generator = torch.Generator().manual_seed(config.seed)
    (out / "config.json").write_text(json.dumps({**config.to_dict(), "reward_version": REWARD_VERSION}, indent=2, sort_keys=True) + "\n",
                                       encoding="utf-8", newline="\n")
    pool = _Pool(config.workers)
    started = time.perf_counter()
    metrics_path = out / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    first_eval = None
    try:
        if config.eval_every:
            first_eval = evaluation_suite(model, config, pool, label="iteration 0 (untrained)", baselines=False)
            log(f"eval iteration 0: {json.dumps(first_eval['learned'], sort_keys=True)}")
        for iteration in range(config.iterations):
            t0 = time.perf_counter()
            episodes = collect(pool, model, config, iteration)
            t1 = time.perf_counter()
            data = merge_episodes(episodes)
            stats = ppo_update(model, optimizer, data, config.ppo, generator)
            t2 = time.perf_counter()
            summary = episode_summary(episodes)
            del episodes, data                       # the next collection must not share memory with this iteration's trajectories
            row = {"iteration": iteration + 1, **summary, **{f"ppo_{k}": v for k, v in stats.items()},
                   "collect_s": t1 - t0, "update_s": t2 - t1, "elapsed_s": time.perf_counter() - started}
            with open(metrics_path, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            log(f"it {iteration + 1:3d}/{config.iterations} reward={row['reward']:.4f} surv={row['survival']:.2f} hollow={row['hollow']:.2f} workers={row['workers_end']:.1f} "
                f"deliv={row['delivered']:.0f} ent={row['ppo_entropy']:.2f} kl={row['ppo_approx_kl']:.4f} vloss={row['ppo_value_loss']:.4f} "
                f"ev={row['ppo_explained_variance']:.2f} idev={row['ppo_identity_ratio_max_dev']:.1e} collect={row['collect_s']:.0f}s "
                f"update={row['update_s']:.0f}s")
            save_model(out / "latest", model, training={"iteration": iteration + 1, "config": config.to_dict()})
            if config.eval_every and (iteration + 1) % config.eval_every == 0 and iteration + 1 < config.iterations:
                report = evaluation_suite(model, config, pool, label=f"iteration {iteration + 1}", baselines=False)
                (out / f"eval_{iteration + 1:04d}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
                log(f"eval iteration {iteration + 1}: {json.dumps(report['learned'], sort_keys=True)}")
        manifest = save_model(out / "final", model, training={"iterations": config.iterations, "config": config.to_dict(),
                                                              "wall_seconds": time.perf_counter() - started})
        log(f"saved {out / 'final'} (weights sha256 {manifest['weights_sha256'][:16]})")
        final_eval = None
        if config.eval_worlds:
            final_eval = evaluation_suite(model, config, pool, label="final", baselines=True, greedy=True,
                                          extra={"untrained_init": untrained} if untrained is not None else None)
            (out / "eval_final.json").write_text(json.dumps(final_eval, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
            log("final evaluation on held-out maps:")
            for name in (k for k in final_eval if isinstance(final_eval[k], dict)):
                row = final_eval[name]
                log(f"  {name:22s} reward={row['reward']:.4f} survival={row['survival']:.2f} hollow={row['hollow']:.2f} workers={row['workers_end']:.1f} "
                    f"brood={row['brood_end']:.1f} eggs={row['eggs_laid']:.0f} delivered={row['delivered']:.0f} contacts={row['contact']:.1f} "
                    f"attacks={row['dispatch_attack']:.2f} combat={row['combat']:.1f}")
    finally:
        pool.close()
    return {"manifest": manifest, "first_eval": first_eval, "final_eval": final_eval, "wall_seconds": time.perf_counter() - started}
