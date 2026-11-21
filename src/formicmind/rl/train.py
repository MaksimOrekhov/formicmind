"""The live loop: decision -> full match -> reward -> PPO update -> new match (FOR-42, phase 2).

One iteration:

1. ``plan_matches`` (seeded) picks maps, seats and opponents (current weights or a frozen snapshot of an earlier
   iteration; baselines only if ``baseline_sparring > 0``).
2. CPU workers (``spawn``) play the full matches with the **current** actor weights, sampling stochastic latent actions
   from their own PCG32 streams and recording observation, mask, latent, ``old_logp`` and reward per decision.
3. The GPU computes values, GAE, and runs PPO-Clip epochs on exactly this fresh batch, which is then discarded
   (no stale trajectories are ever reused). Frozen opponents contribute no training data.
4. Every ``snapshot_every`` iterations the actor is frozen into the pool; every ``checkpoint_every`` a resumable
   checkpoint is written; every ``eval_every`` an independent evaluation runs (never fed back into training).

Every random choice is a function of ``(run_seed, iteration)``, so a resumed run continues exactly where it stopped.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import torch

from .checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from .config import RunConfig
from .evaluate import headline, run_evaluation
from .network import ActorCritic, actor_state_to_bytes
from .parallel import MatchExecutor
from .pool import SnapshotInfo, SnapshotPool, atomic_write_json, plan_matches, thin_active
from .ppo import gae_padded, ppo_update
from .rollout import decode_trajectory

METRICS_NAME = "metrics.jsonl"


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda was requested but CUDA is not available")
    if requested not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    return torch.device(requested)


def build_batch(
    results: list[dict[str, Any]], model: ActorCritic, device: torch.device, gamma: float, lam: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Recorded trajectories -> one flat on-policy batch with GAE advantages and returns."""
    episodes: list[dict[str, torch.Tensor]] = []
    bootstraps: list[torch.Tensor | None] = []
    for result in results:
        for seat in result["seats"]:
            traj = seat.get("trajectory")
            if traj is None or traj["steps"] == 0:
                continue
            episodes.append(decode_trajectory(traj))
            bootstraps.append(
                torch.frombuffer(bytearray(traj["bootstrap_obs"]), dtype=torch.float32) if traj["truncated"] else None)
    if not episodes:
        raise RuntimeError("no recorded trajectories in this iteration")
    lengths = torch.tensor([e["obs"].shape[0] for e in episodes], dtype=torch.long, device=device)
    obs_all = torch.cat([e["obs"] for e in episodes]).to(device)
    model.eval()
    with torch.no_grad():
        values_all = torch.cat([model.value(chunk) for chunk in obs_all.split(16384)])
        last_values = torch.zeros(len(episodes), device=device)
        for i, boot in enumerate(bootstraps):
            if boot is not None:
                last_values[i] = model.value(boot.to(device).unsqueeze(0))[0]
    pieces = values_all.split(lengths.tolist())
    values_pad = torch.nn.utils.rnn.pad_sequence(list(pieces), batch_first=True)
    rewards_pad = torch.nn.utils.rnn.pad_sequence([e["reward"].to(device) for e in episodes], batch_first=True)
    adv_pad = gae_padded(rewards_pad, values_pad, lengths, last_values, gamma, lam)
    valid = torch.arange(values_pad.shape[1], device=device)[None, :] < lengths[:, None]
    adv = adv_pad[valid]
    ret = adv + values_all
    batch = {
        "obs": obs_all,
        "mask": torch.cat([e["mask"] for e in episodes]).to(device),
        "ints": torch.cat([e["ints"] for e in episodes]).to(device),
        "z": torch.cat([e["z"] for e in episodes]).to(device),
        "old_logp": torch.cat([e["logp"] for e in episodes]).to(device),
        "adv": adv,
        "ret": ret,
    }
    info = {
        "trajectories": len(episodes), "samples": int(lengths.sum()),
        "reward_sum_mean": float(rewards_pad.sum(1).mean()), "value_mean": float(values_all.mean()),
        "return_mean": float(ret.mean()), "return_std": float(ret.std()) if ret.numel() > 1 else 0.0,
        "executed_forage_valid": True,
    }
    return batch, info


def summarize_matches(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Outcome and behaviour statistics of one collection phase (training matches only)."""
    by_kind: dict[str, Counter] = {}
    reasons: Counter = Counter()
    lay, army, k_hist, support = [0] * 3, [0] * 7, [0] * 3, [0] * 4
    avoid = decisions = 0
    home_share = ent_d = ent_s = inference = 0.0
    ticks = sim = wall = 0.0
    seat_a_wins = mirror_decisive = 0
    for r in results:
        tag = r["tag"]
        kind = tag["opponent_kind"]
        counter = by_kind.setdefault(kind, Counter())
        counter["matches"] += 1
        reasons[r["reason"] or "truncated"] += 1
        ticks += r["ticks_executed"]
        sim += r["sim_seconds"]
        wall += r["wall_seconds"]
        if r["truncated"]:
            counter["truncated"] += 1
        elif kind == "current":
            counter["decisive" if r["winner"] != "draw" else "draw"] += 1
            if r["winner"] == "A":
                seat_a_wins += 1
            if r["winner"] != "draw":
                mirror_decisive += 1
        else:
            learner = r["seats"][tag["learner_seat"]]
            counter["win" if learner["score"] == 1.0 else "draw" if learner["score"] == 0.5 else "loss"] += 1
        for seat in r["seats"]:
            stats = seat.get("stats")
            if stats is None:
                continue
            if "trajectory" not in seat:
                continue
            decisions += stats["decisions"]
            lay = [a + b for a, b in zip(lay, stats["lay"])]
            army = [a + b for a, b in zip(army, stats["army"])]
            k_hist = [a + b for a, b in zip(k_hist, stats["k_hist"])]
            support = [a + b for a, b in zip(support, stats["forage_support"])]
            avoid += stats["avoid"]
            home_share += stats["home_share_sum"]
            ent_d += stats["entropy_discrete_sum"]
            ent_s += stats["entropy_share_sum"]
            inference += stats["inference_seconds"]
    n = max(decisions, 1)
    return {
        "matches": len(results), "decisions": decisions, "ticks": int(ticks), "sim_seconds": sim,
        "worker_wall_seconds": wall, "inference_seconds": inference,
        "by_opponent": {k: dict(v) for k, v in sorted(by_kind.items())},
        "reasons": dict(sorted(reasons.items())),
        "mirror_seat_a_win_share": seat_a_wins / mirror_decisive if mirror_decisive else None,
        "lay_fraction": [v / n for v in lay], "army_fraction": [v / n for v in army],
        "k_fraction": [v / n for v in k_hist], "executed_support_fraction": [v / n for v in support[1:]],
        "avoid_rate": avoid / n, "mean_home_share": home_share / n,
        "sample_entropy_discrete": ent_d / n, "sample_entropy_share": ent_s / n,
    }


def _write_metric(run_dir: Path, record: dict[str, Any]) -> None:
    with open(run_dir / METRICS_NAME, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=True) + "\n")


def _truncate_metrics(run_dir: Path, keep_through_iteration: int) -> None:
    path = run_dir / METRICS_NAME
    if not path.exists():
        return
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and json.loads(line).get("iteration", 0) <= keep_through_iteration:
            kept.append(line)
    path.write_text("".join(line + "\n" for line in kept), encoding="utf-8")


def _flat_params(model: ActorCritic) -> torch.Tensor:
    return torch.cat([p.detach().flatten().clone() for p in model.parameters()])


def _actor_digest(model: ActorCritic) -> str:
    import hashlib
    return hashlib.sha256(actor_state_to_bytes(model.actor_state_dict())).hexdigest()


def train(
    cfg: RunConfig, run_dir: str | Path, *, resume: bool = False, log: Callable[[str], None] = print,
    executor: MatchExecutor | None = None,
) -> dict[str, Any]:
    cfg.validate()
    run_dir = Path(run_dir)
    device = resolve_device(cfg.device)
    pool = SnapshotPool(run_dir / "snapshots")
    snapshots: list[SnapshotInfo] = []
    counters = {"matches": 0, "decisions": 0, "ticks": 0, "samples": 0, "wall_seconds": 0.0, "updates": 0}
    start_iteration = 0
    config_path = run_dir / "config.json"

    if resume:
        checkpoint_dir = latest_checkpoint(run_dir)
        if checkpoint_dir is None:
            raise RuntimeError(f"nothing to resume in {run_dir}")
        saved = RunConfig.from_json(json.loads(config_path.read_text(encoding="utf-8")))
        if saved.identity_sha256() != cfg.identity_sha256():
            raise RuntimeError("the configuration differs from the run being resumed in a learning-relevant field")
        model, manifest, optimizer_state = load_checkpoint(checkpoint_dir, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
        optimizer.load_state_dict(optimizer_state)
        start_iteration = manifest["iteration"]
        counters.update(manifest["counters"])
        snapshots = [pool.load_info(i["snapshot_id"]) for i in manifest["snapshots"]]
        _truncate_metrics(run_dir, start_iteration)
        log(f"resumed {run_dir} at iteration {start_iteration} (matches so far {counters['matches']})")
    else:
        if config_path.exists():
            raise RuntimeError(f"{run_dir} already holds a run; use --resume or another directory")
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(cfg.run_seed)
        model = ActorCritic(cfg.hidden_sizes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
        atomic_write_json(config_path, cfg.to_json())
        snapshots = [pool.add(0, 0, model.actor_state_dict())]  # the untrained initial policy is the first frozen opponent
        save_checkpoint(run_dir, 0, model, optimizer, _meta(cfg, counters, snapshots))
        log(f"new run {run_dir}: params={model.parameter_count()} device={device} workers={cfg.workers}")

    own_executor = executor is None
    executor = executor or MatchExecutor(cfg.workers, cfg.chunk_size)
    ppo_cfg = cfg.ppo()
    run_started = time.perf_counter()
    initial_actor_digest = _actor_digest(model)
    summary: dict[str, Any] = {"initial_actor_sha256": initial_actor_digest}
    completed_iteration = saved_iteration = start_iteration
    try:
        for iteration in range(start_iteration, cfg.iterations):
            if cfg.max_matches and counters["matches"] >= cfg.max_matches:
                break
            if cfg.max_wall_seconds and time.perf_counter() - run_started >= cfg.max_wall_seconds:
                log("wall-clock budget reached")
                break
            t_iter = time.perf_counter()
            active = thin_active(snapshots, cfg.max_active_snapshots)
            count = cfg.matches_per_iteration
            if cfg.max_matches:
                count = min(count, cfg.max_matches - counters["matches"])
            specs = plan_matches(
                cfg.run_seed, iteration, count, active, p_current=cfg.p_current,
                baseline_sparring=cfg.baseline_sparring, gen_map_fraction=cfg.gen_map_fraction,
                shaping_coef=cfg.shaping_coef, gamma=cfg.gamma, max_ticks=cfg.max_ticks,
            )
            refs: dict[str, dict[str, Any]] = {"current": {"bytes": actor_state_to_bytes(model.actor_state_dict())}}
            for snap in active:
                refs[snap.ref] = {"path": snap.path, "sha256": snap.sha256}
            results = executor.run(specs, refs)
            t_collect = time.perf_counter() - t_iter

            t0 = time.perf_counter()
            batch, batch_info = build_batch(results, model, device, cfg.gamma, cfg.lam)
            t_gae = time.perf_counter() - t0
            before = _flat_params(model)
            generator = torch.Generator().manual_seed(cfg.run_seed * 1_000_003 + iteration)
            t0 = time.perf_counter()
            stats = ppo_update(model, optimizer, batch, ppo_cfg, generator)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_update = time.perf_counter() - t0
            delta = float(torch.linalg.vector_norm(_flat_params(model) - before))

            summary_m = summarize_matches(results)
            counters["matches"] += summary_m["matches"]
            counters["decisions"] += summary_m["decisions"]
            counters["ticks"] += summary_m["ticks"]
            counters["samples"] += batch_info["samples"]
            counters["updates"] += int(stats["minibatch_updates"])
            t_total = time.perf_counter() - t_iter
            counters["wall_seconds"] += t_total
            completed = completed_iteration = iteration + 1
            record = {
                "kind": "train", "iteration": completed, "matches_total": counters["matches"],
                "timing": {
                    "iteration_seconds": t_total, "collect_seconds": t_collect, "gae_seconds": t_gae,
                    "update_seconds": t_update,
                    "episodes_per_second": summary_m["matches"] / t_collect if t_collect else None,
                    "policy_decisions_per_second": summary_m["decisions"] / t_collect if t_collect else None,
                    "ticks_per_second": summary_m["ticks"] / t_collect if t_collect else None,
                },
                "batch": batch_info, "ppo": stats, "matches": summary_m, "param_delta_l2": delta,
                "actor_sha256": _actor_digest(model)[:16], "active_snapshots": [s.snapshot_id for s in active],
            }
            if device.type == "cuda":
                record["cuda_max_memory_mb"] = torch.cuda.max_memory_allocated() / 1e6
            _write_metric(run_dir, record)
            log(_line(record))

            did_snapshot = False
            if cfg.snapshot_every and completed % cfg.snapshot_every == 0:
                snapshots.append(pool.add(len(snapshots), completed, model.actor_state_dict()))
                did_snapshot = True
            last = completed == cfg.iterations or (
                cfg.max_matches and counters["matches"] >= cfg.max_matches)
            if did_snapshot or last or (cfg.checkpoint_every and completed % cfg.checkpoint_every == 0):
                save_checkpoint(run_dir, completed, model, optimizer, _meta(cfg, counters, snapshots))
                saved_iteration = completed
            if cfg.eval_every and (completed % cfg.eval_every == 0 or last):
                _evaluate(cfg, executor, model, snapshots, pool, run_dir, completed, log, cfg.eval_suite)
        if completed_iteration > saved_iteration:
            save_checkpoint(run_dir, completed_iteration, model, optimizer, _meta(cfg, counters, snapshots))
    finally:
        if own_executor:
            executor.close()
    summary.update(
        final_actor_sha256=_actor_digest(model), counters=counters,
        actor_changed=_actor_digest(model) != initial_actor_digest,
    )
    atomic_write_json(run_dir / "summary.json", summary)
    return summary


def _evaluate(
    cfg: RunConfig, executor: MatchExecutor, model: ActorCritic, snapshots: list[SnapshotInfo], pool: SnapshotPool,
    run_dir: Path, iteration: int, log: Callable[[str], None], suite: str,
) -> dict[str, Any]:
    opponents: dict[str, dict[str, Any]] = {"init": {"path": snapshots[0].path, "sha256": snapshots[0].sha256}}
    if suite == "full" and len(snapshots) > 2:
        for snap in (snapshots[len(snapshots) // 2], snapshots[-1]):
            if snap.snapshot_id != 0:
                opponents[snap.ref] = {"path": snap.path, "sha256": snap.sha256}
    report = run_evaluation(
        executor, {"bytes": actor_state_to_bytes(model.actor_state_dict())}, opponents, suite, iteration=iteration)
    atomic_write_json(run_dir / "eval" / f"eval_{iteration:06d}.json", report)
    _write_metric(run_dir, {"kind": "eval", "iteration": iteration, "suite": suite, "headline": headline(report),
                            "matches": report["matches"], "wall_seconds": report["wall_seconds"]})
    parts = " ".join(f"{k.split(':')[0]}={v['score']:.2f}" if v["score"] is not None else f"{k}=-"
                     for k, v in headline(report).items())
    log(f"  eval[{suite}] it={iteration} score: {parts}")
    return report


def _meta(cfg: RunConfig, counters: dict[str, Any], snapshots: list[SnapshotInfo]) -> dict[str, Any]:
    return {
        "config_identity_sha256": cfg.identity_sha256(), "run_seed": cfg.run_seed, "counters": dict(counters),
        "snapshots": [s.to_json() for s in snapshots],
    }


def _line(record: dict[str, Any]) -> str:
    p, m, t = record["ppo"], record["matches"], record["timing"]
    return (
        f"it {record['iteration']:4d} matches={record['matches_total']:6d} "
        f"samples={record['batch']['samples']:6d} kl={p['approx_kl']:.4f} clip={p['clipfrac']:.3f} "
        f"ent={p['entropy_discrete']:.3f} vloss={p['value_loss']:.4f} ev={p['explained_variance']:.2f} "
        f"dW={record['param_delta_l2']:.3f} | {t['episodes_per_second']:.1f} ep/s {t['policy_decisions_per_second']:.0f} dec/s "
        f"collect={t['collect_seconds']:.1f}s upd={t['update_seconds']:.1f}s {m['by_opponent']}"
    )


__all__ = ["build_batch", "resolve_device", "summarize_matches", "train"]
