"""Collection-side performance measurement and profile (FOR-42): profiler first, optimise only the measured bottleneck."""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import torch

from .network import ActorCritic, actor_state_to_bytes
from .parallel import MatchExecutor, execute_chunk
from .pool import atomic_write_json, plan_matches


def _gpu_sampler(stop: threading.Event, samples: list[float]) -> None:
    while not stop.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            samples.append(float(out.stdout.strip().splitlines()[0]))
        except Exception:  # noqa: BLE001 - monitoring must never break the benchmark
            return
        stop.wait(0.5)


def profile_single_process(matches: int = 8) -> dict[str, Any]:
    """Where does one worker spend its time? engine ticks vs policy inference vs the rest."""
    import cProfile
    import io
    import pstats

    torch.manual_seed(0)
    model = ActorCritic().eval()
    refs = {"current": {"bytes": actor_state_to_bytes(model.actor_state_dict())}}
    specs = plan_matches(1, 0, matches, [], p_current=1.0, baseline_sparring=0.0, gen_map_fraction=0.5,
                         shaping_coef=0.0, gamma=0.997, max_ticks=None)
    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiler.enable()
    results = execute_chunk((specs, refs))
    profiler.disable()
    wall = time.perf_counter() - started
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(14)
    inference = sum(seat["stats"]["inference_seconds"] for r in results for seat in r["seats"] if "stats" in seat)
    decisions = sum(seat["stats"]["decisions"] for r in results for seat in r["seats"] if "stats" in seat)
    ticks = sum(r["ticks_executed"] for r in results)
    return {
        "matches": matches, "wall_seconds": wall, "ticks": ticks, "decisions": decisions,
        "inference_seconds": inference, "inference_fraction": inference / wall if wall else None,
        "engine_fraction_estimate": 1.0 - (inference / wall if wall else 0.0),
        "ms_per_decision_inference": 1000.0 * inference / max(decisions, 1),
        "top_cumulative": stream.getvalue().splitlines()[:40],
    }


def long_match_throughput(workers: int, matches: int, chunk_size: int) -> dict[str, Any]:
    """Full-length (T_max = 3000 ticks, nobody dies early) matches: the upper bound of per-match engine cost."""
    specs = []
    for index in range(matches):
        a = {"kind": "baseline", "ref": "balanced:v2", "replicate": 0}
        b = {"kind": "baseline", "ref": "turtle:v2", "replicate": 0}
        specs.append({"match_id": index, "seed": 5_000_000_000 + index, "map_id": "classic", "seats": [a, b], "tag": {}})
    with MatchExecutor(workers, chunk_size) as executor:
        executor.run(specs[:workers * chunk_size], {})
        started = time.perf_counter()
        results = executor.run(specs, {})
        wall = time.perf_counter() - started
    ticks = sum(r["ticks_executed"] for r in results)
    return {"matches": matches, "wall_seconds": wall, "episodes_per_second": matches / wall,
            "ticks_per_second": ticks / wall, "mean_ticks": ticks / matches,
            "engine_ms_per_tick_per_worker": 1000.0 * sum(r["sim_seconds"] for r in results) / ticks}


def gpu_update_capacity(samples: int = 60000) -> dict[str, Any]:
    """PPO update throughput on synthetic on-policy data: CPU versus the GPU, to size the learner for larger batches."""
    import copy
    from ..rng import Pcg32
    from .distribution import evaluate_latent, sample_latent
    from .ppo import PPOConfig, ppo_update

    torch.manual_seed(0)
    base = ActorCritic()
    obs = torch.rand(samples, 223)
    mask_row = [True] * 3 + [True] * 6 + [True] * 7 + [True, False] + [True, True] * 6
    with torch.no_grad():
        heads = base.policy(obs)
    rng = Pcg32.seeded(1, 1)
    ints, zs = [], []
    log_std = base.log_std.tolist()
    for row in heads:
        i, z = sample_latent(row.tolist(), log_std, mask_row, rng)
        ints.append(i)
        zs.append(z)
    mask = torch.tensor([mask_row] * samples)
    ints_t, z_t = torch.tensor(ints), torch.tensor(zs)
    with torch.no_grad():
        old = evaluate_latent(heads, base.log_std.detach(), mask, ints_t, z_t).log_prob
    report: dict[str, Any] = {"samples": samples}
    cfg = PPOConfig(minibatch=4096, epochs=4, target_kl=0.0)
    for device_name in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        device = torch.device(device_name)
        model = copy.deepcopy(base).to(device)
        batch = {"obs": obs.to(device), "mask": mask.to(device), "ints": ints_t.to(device), "z": z_t.to(device),
                 "old_logp": old.to(device), "adv": torch.randn(samples, device=device), "ret": torch.randn(samples, device=device)}
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        ppo_update(model, optimizer, batch, cfg, torch.Generator().manual_seed(0))  # warm-up
        if device.type == "cuda":
            torch.cuda.synchronize()
        stop, gpu = threading.Event(), []
        sampler = threading.Thread(target=_gpu_sampler, args=(stop, gpu), daemon=True)
        if device.type == "cuda":
            sampler.start()
        started = time.perf_counter()
        ppo_update(model, optimizer, batch, cfg, torch.Generator().manual_seed(1))
        if device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - started
        stop.set()
        report[device_name] = {"update_seconds": wall, "sample_epochs_per_second": samples * cfg.epochs / wall,
                               "gpu_utilization_percent_samples": gpu[:]}
    return report


def run_benchmark(*, workers: int, matches: int, chunk_size: int, out: Path | None) -> dict[str, Any]:
    torch.manual_seed(0)
    model = ActorCritic().eval()
    refs = {"current": {"bytes": actor_state_to_bytes(model.actor_state_dict())}}
    specs = plan_matches(7, 0, matches, [], p_current=1.0, baseline_sparring=0.0, gen_map_fraction=0.5,
                         shaping_coef=0.0, gamma=0.997, max_ticks=None)
    stop, gpu = threading.Event(), []
    sampler = threading.Thread(target=_gpu_sampler, args=(stop, gpu), daemon=True)
    sampler.start()
    with MatchExecutor(workers, chunk_size) as executor:
        executor.run(specs[:max(workers * chunk_size, 1)], refs)  # warm-up: spawn + import + first inference
        started = time.perf_counter()
        results = executor.run(specs, refs)
        wall = time.perf_counter() - started
    stop.set()
    decisions = sum(seat["stats"]["decisions"] for r in results for seat in r["seats"] if "stats" in seat)
    ticks = sum(r["ticks_executed"] for r in results)
    report = {
        "workers": workers, "chunk_size": chunk_size, "matches": len(results), "wall_seconds": wall,
        "episodes_per_second": len(results) / wall, "policy_decisions_per_second": decisions / wall,
        "ticks_per_second": ticks / wall, "decisions": decisions, "ticks": ticks,
        "worker_sim_seconds": sum(r["sim_seconds"] for r in results),
        "worker_wall_seconds": sum(r["wall_seconds"] for r in results),
        "gpu_utilization_during_collection_percent": gpu[:], "cuda_available": torch.cuda.is_available(),
        "profile_single_process": profile_single_process(),
        "long_matches_balanced_vs_turtle": long_match_throughput(workers, max(workers * 4, 40), chunk_size),
        "ppo_update_capacity": gpu_update_capacity(),
    }
    if out is not None:
        atomic_write_json(out, report)
    return report


__all__ = ["profile_single_process", "run_benchmark"]
