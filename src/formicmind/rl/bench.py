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
    }
    if out is not None:
        atomic_write_json(out, report)
    return report


__all__ = ["profile_single_process", "run_benchmark"]
