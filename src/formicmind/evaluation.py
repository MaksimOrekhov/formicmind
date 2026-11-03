"""Deterministic L0 episode evaluation and end-to-end benchmark harness.

All simulation paths call the existing :func:`step_tick`. The benchmark discards each
tick's events as soon as the tick returns and computes no state/checkpoint hashes inside
its timed window.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .baselines import baseline_names, baseline_spec, make_baseline
from .match import _register_streams
from .match_smoke import SCENARIOS as SCENARIO_FIXTURES
from .rng import policy_stream_id
from .reset import reset
from .replay.writer import describe_controller
from .rules import SOLDIER, WORKER, default_config
from .serialization import checkpoint_hash, config_hash, state_hash
from .tick import Controller, PolicyController, ScriptedController, step_tick
from .versions import (
    ACTION_VERSION,
    BALANCE_VERSION,
    DECODER_VERSION,
    OBSERVATION_VERSION,
    REPLAY_FORMAT,
    REPLAY_SCHEMA_VERSION,
    RNG_VERSION,
    RULES_VERSION,
    SERIALIZATION_VERSION,
    STREAM_DERIVATION_VERSION,
)

U64_MAX = (1 << 64) - 1
MANIFEST_SCHEMA = "formicmind.eval-manifest/v1"
RESULT_SCHEMA = "formicmind.eval-results/v1"
GATE0_CANDIDATE_TICKS_PER_SECOND = 5_000
TECHNICAL_SCENARIO = "technical"
SCENARIOS = (TECHNICAL_SCENARIO, *SCENARIO_FIXTURES.keys())
SCENARIOS_BY_NAME = SCENARIO_FIXTURES


class EvaluationError(ValueError):
    """A manifest or evaluation request does not match the supported L0 contract."""


@dataclass(frozen=True, slots=True)
class ControllerPair:
    controllers: tuple[Controller, Controller]
    ids: tuple[str, str]
    script_hashes: tuple[str | None, str | None]
    descriptors: tuple[dict[str, Any], dict[str, Any]]

    def swapped(self) -> "ControllerPair":
        """Swap seats while keeping each controller's role-keyed RNG stream attached."""
        return ControllerPair(
            tuple(reversed(self.controllers)), tuple(reversed(self.ids)),
            tuple(reversed(self.script_hashes)), tuple(reversed(self.descriptors)),
        )  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class EpisodeSummary:
    seed: int
    outcome: Any
    ticks_executed: int
    state_tick: int
    queen_hp: tuple[int, int]
    food: tuple[int, int]
    workers: tuple[int, int]
    soldiers: tuple[int, int]
    brood: tuple[int, int]
    config_hash: str | None
    state_hash: str | None
    checkpoint_hash: str | None

    @property
    def winner(self) -> str | None:
        return None if self.outcome is None else self.outcome.winner.value

    @property
    def reason(self) -> str | None:
        return None if self.outcome is None else self.outcome.reason.value

    @property
    def terminal_tick(self) -> int | None:
        return None if self.outcome is None else self.outcome.terminal_tick

    @property
    def tick_completed(self) -> bool | None:
        return None if self.outcome is None else self.outcome.tick_completed

    @property
    def score(self) -> tuple[float, float] | None:
        return None if self.outcome is None else self.outcome.score

    @property
    def truncated(self) -> bool:
        return self.outcome is None

    def to_record(self, *, episode_index: int, pair_id: str | None, assignment: tuple[str, str]) -> dict[str, Any]:
        score = self.score
        return {
            "episode_index": episode_index,
            "pair_id": pair_id,
            "seed": self.seed,
            "seat_assignment": {"A": assignment[0], "B": assignment[1]},
            "winner": self.winner,
            "reason": self.reason,
            "terminal_tick": self.terminal_tick,
            "state_tick": self.state_tick,
            "tick_completed": self.tick_completed,
            "ticks_executed": self.ticks_executed,
            "truncated": self.truncated,
            "score": None if score is None else {"A": score[0], "B": score[1]},
            "queen_hp": {"A": self.queen_hp[0], "B": self.queen_hp[1]},
            "food": {"A": self.food[0], "B": self.food[1]},
            "workers": {"A": self.workers[0], "B": self.workers[1]},
            "soldiers": {"A": self.soldiers[0], "B": self.soldiers[1]},
            "brood": {"A": self.brood[0], "B": self.brood[1]},
            "config_hash": self.config_hash,
            "state_hash": self.state_hash,
            "checkpoint_hash": self.checkpoint_hash,
        }


def validate_seed(seed: int, name: str = "seed") -> int:
    if type(seed) is not int or not 0 <= seed <= U64_MAX:
        raise EvaluationError(f"{name} must be a u64 integer (0..{U64_MAX}).")
    return seed


def validate_scenario(scenario: str) -> str:
    if scenario not in SCENARIOS:
        raise EvaluationError(f"scenario must be one of: {', '.join(SCENARIOS)}.")
    return scenario


def controller_pair(
    scenario: str | None = TECHNICAL_SCENARIO,
    *, a: str = "scripted", b: str = "scripted", swapped: bool = False,
) -> ControllerPair:
    """Build scripted fixtures or one of the fixed FOR-19 policies for each seat."""
    if a == "scripted" and b == "scripted":
        if scenario is None:
            raise EvaluationError("scripted controllers require a scripted scenario.")
        validate_scenario(scenario)
        if scenario == TECHNICAL_SCENARIO:
            scripts: Mapping[str, Mapping[int, Any]] = {"script_a": {}, "script_b": {}}
        else:
            fixture = SCENARIOS_BY_NAME[scenario]
            scripts = {"script_a": fixture.script_a, "script_b": fixture.script_b}
        labels = ("script_a", "script_b")
        controllers = tuple(ScriptedController(scripts[label]) for label in labels)
        ids = tuple(f"scripted:{scenario}:{label}" for label in labels)
        hashes = tuple(describe_controller(controller).ref.removeprefix("script-sha256:")
                       for controller in controllers)
        descriptors = tuple({
            "id": ids[index], "kind": "scripted", "scenario": scenario, "script_sha256": hashes[index],
        } for index in range(2))
        pair = ControllerPair(controllers, ids, hashes, descriptors)  # type: ignore[arg-type]
        return pair.swapped() if swapped else pair

    if scenario is not None:
        raise EvaluationError("--scenario is supported only with --a scripted --b scripted.")
    if a == "scripted" or b == "scripted":
        raise EvaluationError("scripted fixtures cannot be mixed with strategic policies.")
    try:
        spec_a, spec_b = baseline_spec(a), baseline_spec(b)
    except ValueError as exc:
        raise EvaluationError(str(exc)) from exc
    replicate_by_seat: list[int] = []
    used_replicates: dict[str, int] = {}
    for spec in (spec_a, spec_b):
        replicate = used_replicates.get(spec.policy_id, 0)
        used_replicates[spec.policy_id] = replicate + 1
        replicate_by_seat.append(replicate)
    specs = (spec_a, spec_b)
    controllers: tuple[Controller, Controller] = tuple(
        PolicyController(make_baseline(spec.name), policy_stream_id(spec.policy_id, replicate_by_seat[index]))
        for index, spec in enumerate(specs)
    )  # type: ignore[assignment]
    ids = tuple(f"{spec.full_id}#replicate-{replicate_by_seat[index]}" for index, spec in enumerate(specs))
    descriptors = tuple({
        "id": spec.full_id,
        "kind": "policy",
        "policy_id": spec.policy_id,
        "version": spec.version,
        "parameters": spec.parameter_object(),
        "parameter_sha256": spec.digest(),
        "stream_id": policy_stream_id(spec.policy_id, replicate_by_seat[index]),
        "replicate": replicate_by_seat[index],
    } for index, spec in enumerate(specs))
    pair = ControllerPair(controllers, ids, (None, None), descriptors)  # type: ignore[arg-type]
    return pair.swapped() if swapped else pair


def execute_episode(
    seed: int,
    pair: ControllerPair,
    *,
    config=None,
    max_ticks: int | None = None,
    include_hashes: bool = True,
    on_phase_timing: Callable[[Any, int], None] | None = None,
) -> EpisodeSummary:
    """Run one episode without retaining its per-tick events.

    ``max_ticks`` is a harness cutoff. The number of executed ticks counts calls to
    ``step_tick``, including an incomplete mid-tick terminal.
    """
    validate_seed(seed)
    config = default_config() if config is None else config
    if max_ticks is not None and (type(max_ticks) is not int or not 0 <= max_ticks <= config.t_max):
        raise EvaluationError(f"max_ticks must be within 0..T_max ({config.t_max}).")
    limit = config.t_max if max_ticks is None else max_ticks
    state, run = reset(config, seed)
    _register_streams(run, pair.controllers, ())
    executed = 0
    while run.outcome is None and executed < limit and state.tick < config.t_max:
        tick_result = step_tick(state, run, pair.controllers, on_phase_timing=on_phase_timing)
        executed += 1
        del tick_result  # this tick's events are deliberately not accumulated

    colonies = state.colonies
    worker_counts = tuple(sum(unit.caste == WORKER for unit in colony.units) for colony in colonies)
    soldier_counts = tuple(sum(unit.caste == SOLDIER for unit in colony.units) for colony in colonies)
    hashes = (config_hash(config), state_hash(state), checkpoint_hash(state, run)) if include_hashes else (
        None, None, None)
    return EpisodeSummary(
        seed=seed,
        outcome=run.outcome,
        ticks_executed=executed,
        state_tick=state.tick,
        queen_hp=(colonies[0].queen_hp, colonies[1].queen_hp),
        food=(colonies[0].food, colonies[1].food),
        workers=worker_counts,  # type: ignore[arg-type]
        soldiers=soldier_counts,  # type: ignore[arg-type]
        brood=(len(colonies[0].brood), len(colonies[1].brood)),
        config_hash=hashes[0], state_hash=hashes[1], checkpoint_hash=hashes[2],
    )


def _mean(values: list[float | int]) -> float | None:
    return None if not values else statistics.fmean(values)


def aggregate_evaluation(records: list[dict[str, Any]]) -> dict[str, Any]:
    finished = [record for record in records if not record["truncated"]]
    scores_a = [record["score"]["A"] for record in finished]
    scores_b = [record["score"]["B"] for record in finished]
    reasons = Counter(record["reason"] for record in finished)
    winners = Counter(record["winner"] for record in finished)
    lengths = [record["ticks_executed"] for record in records]
    finished_lengths = [record["ticks_executed"] for record in finished]
    return {
        "attempted": len(records),
        "finished": len(finished),
        "truncated": len(records) - len(finished),
        "wins_A": winners["A"],
        "wins_B": winners["B"],
        "draws": winners["draw"],
        "terminal_reasons": {key: reasons[key] for key in sorted(reasons)},
        "ticks_executed": sum(lengths),
        "mean_episode_ticks": _mean(lengths),
        "median_episode_ticks": None if not lengths else statistics.median(lengths),
        "mean_finished_episode_ticks": _mean(finished_lengths),
        "mean_score_finished_only": {
            "A": _mean(scores_a), "B": _mean(scores_b),
        } if finished else None,
    }


def evaluate_episodes(
    *,
    episodes: int,
    seed_start: int,
    scenario: str | None = TECHNICAL_SCENARIO,
    a: str = "scripted",
    b: str = "scripted",
    paired_seat: bool = False,
    max_ticks: int | None = None,
    config=None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if type(episodes) is not int or episodes <= 0:
        raise EvaluationError("episodes must be a positive integer.")
    validate_seed(seed_start, "seed_start")
    if a == "scripted" and b == "scripted":
        if scenario is None:
            raise EvaluationError("scripted controllers require a scripted scenario.")
        validate_scenario(scenario)
    elif scenario is not None:
        raise EvaluationError("--scenario is supported only with --a scripted --b scripted.")
    run_count = episodes * (2 if paired_seat else 1)
    if seed_start + episodes - 1 > U64_MAX:
        raise EvaluationError("seed_start + episodes - 1 exceeds the u64 seed range.")
    base_pair = controller_pair(scenario, a=a, b=b)
    records = []
    for index in range(episodes):
        seed = seed_start + index
        sides = (False, True) if paired_seat else (False,)
        for side_index, swapped in enumerate(sides):
            pair = base_pair if not swapped else base_pair.swapped()
            summary = execute_episode(seed, pair, config=config, max_ticks=max_ticks)
            pair_id = f"seed-{seed}" if paired_seat else None
            row = summary.to_record(episode_index=len(records), pair_id=pair_id, assignment=pair.ids)
            row["controller_config"] = result_controller_config(pair)
            records.append(row)
    aggregate = aggregate_evaluation(records)
    aggregate["seed_pairs"] = episodes if paired_seat else 0
    aggregate["requested_episodes"] = episodes
    aggregate["paired_seat"] = paired_seat
    aggregate["attempted"] = run_count
    return records, aggregate


def nearest_rank(values: list[float], percentile: float) -> float:
    """Nearest-rank percentile (for 5 runs p10 is the minimum); percentile is 0..1."""
    if not values or not 0 < percentile <= 1:
        raise EvaluationError("nearest-rank needs values and a percentile in (0, 1].")
    ordered = sorted(values)
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def _run_tick_budget(
    *, ticks: int, seed_start: int, pair: ControllerPair, config, on_phase_timing=None,
) -> dict[str, Any]:
    if ticks < 0:
        raise EvaluationError("tick budget cannot be negative.")
    if ticks and seed_start + ticks - 1 > U64_MAX:
        raise EvaluationError("seed range may overflow u64 while resetting episodes in this tick budget.")
    completed_ticks = 0
    episode_index = 0
    episodes = 0
    full_terminals = 0
    boundary_cutoffs = 0
    reasons: Counter[str] = Counter()
    winners: Counter[str] = Counter()
    worker_total = 0
    soldier_total = 0
    brood_total = 0
    while completed_ticks < ticks:
        seed = seed_start + episode_index
        remaining = ticks - completed_ticks
        summary = execute_episode(
            seed, pair, config=config, max_ticks=min(remaining, config.t_max),
            include_hashes=False, on_phase_timing=on_phase_timing,
        )
        if summary.ticks_executed <= 0:
            raise EvaluationError("a benchmark episode executed no step_tick calls.")
        completed_ticks += summary.ticks_executed
        episodes += 1
        episode_index += 1
        worker_total += sum(summary.workers)
        soldier_total += sum(summary.soldiers)
        brood_total += sum(summary.brood)
        if summary.outcome is None:
            boundary_cutoffs += 1
        else:
            full_terminals += 1
            reasons[summary.reason or "unknown"] += 1
            winners[summary.winner or "unknown"] += 1
    if completed_ticks != ticks:
        raise EvaluationError(f"benchmark tick budget mismatch: wanted {ticks}, executed {completed_ticks}.")
    return {
        "ticks": completed_ticks,
        "episodes": episodes,
        "full_terminal_episodes": full_terminals,
        "benchmark_boundary_cutoffs": boundary_cutoffs,
        "terminal_reasons": {key: reasons[key] for key in sorted(reasons)},
        "winners": {key: winners[key] for key in sorted(winners)},
        "mean_ticks_per_episode": completed_ticks / episodes if episodes else None,
        "mean_workers_per_colony_at_episode_end": worker_total / (episodes * 2) if episodes else None,
        "mean_soldiers_per_colony_at_episode_end": soldier_total / (episodes * 2) if episodes else None,
        "mean_brood_per_colony_at_episode_end": brood_total / (episodes * 2) if episodes else None,
    }


class SingleCoreAffinity:
    """Best-effort process affinity with restoration on normal and exceptional exits."""

    def __init__(self, core_index: int | None = None):
        if core_index is not None and (type(core_index) is not int or core_index < 0):
            raise EvaluationError("pin_core must be a non-negative logical CPU index or null.")
        self.requested_core = core_index
        self._restore = None
        self.report: dict[str, Any] = {
            "request": "auto" if core_index is None else core_index,
            "affinity_applied": False,
            "selected_core": None,
            "prior_affinity": None,
            "active_affinity": None,
            "restored": None,
            "status": "not_applied",
            "reason": None,
        }

    def __enter__(self) -> "SingleCoreAffinity":
        try:
            if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
                prior = set(os.sched_getaffinity(0))
                core = min(prior) if self.requested_core is None and prior else self.requested_core
                if core is None or core not in prior:
                    self.report["reason"] = f"logical CPU {core} is not in the process affinity mask"
                    return self
                os.sched_setaffinity(0, {core})
                self._restore = lambda: os.sched_setaffinity(0, prior)
                self.report.update(
                    affinity_applied=True, selected_core=core,
                    prior_affinity=sorted(prior), active_affinity=[core], status="applied",
                )
                return self
            if os.name == "nt":
                self._apply_windows()
                return self
            self.report["reason"] = f"process affinity is unsupported on {platform.system() or 'this platform'}"
        except (OSError, AttributeError) as exc:
            self.report["reason"] = f"{type(exc).__name__}: {exc}"
        return self

    def _apply_windows(self) -> None:
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        ulong_ptr = ctypes.c_size_t
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetProcessAffinityMask.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(ulong_ptr), ctypes.POINTER(ulong_ptr),
        ]
        kernel32.GetProcessAffinityMask.restype = wintypes.BOOL
        kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ulong_ptr]
        kernel32.SetProcessAffinityMask.restype = wintypes.BOOL
        process = kernel32.GetCurrentProcess()
        process_mask, system_mask = ulong_ptr(), ulong_ptr()
        if not kernel32.GetProcessAffinityMask(process, ctypes.byref(process_mask), ctypes.byref(system_mask)):
            raise ctypes.WinError(ctypes.get_last_error())
        prior_mask = int(process_mask.value)
        allowed = [index for index in range(ctypes.sizeof(ulong_ptr) * 8) if prior_mask & (1 << index)]
        core = min(allowed) if self.requested_core is None and allowed else self.requested_core
        if core is None or core not in allowed:
            self.report["reason"] = f"logical CPU {core} is not in the process affinity mask"
            return
        target_mask = ulong_ptr(1 << core)
        if not kernel32.SetProcessAffinityMask(process, target_mask):
            raise ctypes.WinError(ctypes.get_last_error())

        def restore() -> None:
            if not kernel32.SetProcessAffinityMask(process, ulong_ptr(prior_mask)):
                raise ctypes.WinError(ctypes.get_last_error())

        self._restore = restore
        self.report.update(
            affinity_applied=True, selected_core=core,
            prior_affinity=f"0x{prior_mask:x}", active_affinity=[core], status="applied",
        )

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self._restore is None:
            self.report["restored"] = False if self.report["affinity_applied"] else None
            return False
        try:
            self._restore()
            self.report["restored"] = True
            self.report["status"] = "applied_and_restored"
        except OSError as restore_error:
            self.report["restored"] = False
            self.report["status"] = "restore_failed"
            self.report["restore_error"] = f"{type(restore_error).__name__}: {restore_error}"
        return False


def system_profile() -> dict[str, Any]:
    processor = platform.processor() or platform.uname().processor or "unknown"
    clock = time.get_clock_info("perf_counter")
    return {
        "os": platform.platform(),
        "os_name": platform.system(),
        "os_release": platform.release(),
        "processor": processor,
        "architecture": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "perf_counter": {
            "name": "time.perf_counter_ns",
            "monotonic": clock.monotonic,
            "adjustable": clock.adjustable,
            "resolution_seconds": clock.resolution,
        },
    }


def run_benchmark(
    *,
    runs: int = 5,
    ticks_per_run: int = 100_000,
    warmup_ticks: int = 10_000,
    seed_start: int,
    scenario: str | None = TECHNICAL_SCENARIO,
    a: str = "scripted",
    b: str = "scripted",
    pin_core: int | None = None,
    phase_timing_ticks: int = 0,
    config=None,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = default_config() if config is None else config
    if type(runs) is not int or runs <= 0:
        raise EvaluationError("runs must be a positive integer.")
    if type(ticks_per_run) is not int or ticks_per_run <= 0:
        raise EvaluationError("ticks_per_run must be a positive integer.")
    if type(warmup_ticks) is not int or warmup_ticks < 0:
        raise EvaluationError("warmup_ticks must be a non-negative integer.")
    if type(phase_timing_ticks) is not int or phase_timing_ticks < 0:
        raise EvaluationError("phase_timing_ticks must be a non-negative integer.")
    validate_seed(seed_start, "seed_start")
    if a == "scripted" and b == "scripted":
        if scenario is None:
            raise EvaluationError("scripted controllers require a scripted scenario.")
        validate_scenario(scenario)
    elif scenario is not None:
        raise EvaluationError("--scenario is supported only with --a scripted --b scripted.")
    if ticks_per_run and seed_start + ticks_per_run - 1 > U64_MAX:
        raise EvaluationError("seed_start + ticks_per_run - 1 exceeds the u64 seed range.")
    if warmup_ticks and seed_start + warmup_ticks - 1 > U64_MAX:
        raise EvaluationError("seed_start + warmup_ticks - 1 exceeds the u64 seed range.")
    if phase_timing_ticks and seed_start + phase_timing_ticks - 1 > U64_MAX:
        raise EvaluationError("seed_start + phase_timing_ticks - 1 exceeds the u64 seed range.")
    if pin_core is not None and (type(pin_core) is not int or pin_core < 0):
        raise EvaluationError("pin_core must be a non-negative logical CPU index or null.")

    pair = controller_pair(scenario, a=a, b=b)
    samples: list[dict[str, Any]] = []
    diagnostic = None
    affinity = SingleCoreAffinity(pin_core)
    with affinity:
        warmup_started = time.perf_counter_ns()
        warmup = _run_tick_budget(ticks=warmup_ticks, seed_start=seed_start, pair=pair, config=config)
        warmup_elapsed_ns = time.perf_counter_ns() - warmup_started

        for sample_index in range(runs):
            started_ns = time.perf_counter_ns()
            sample_stats = _run_tick_budget(
                ticks=ticks_per_run, seed_start=seed_start, pair=pair, config=config,
            )
            elapsed_ns = max(1, time.perf_counter_ns() - started_ns)
            samples.append({
                "sample_index": sample_index,
                "seed_start": seed_start,
                "seed_schedule_repeated": True,
                **sample_stats,
                "elapsed_ns": elapsed_ns,
                "elapsed_seconds": elapsed_ns / 1_000_000_000,
                "ticks_per_second": ticks_per_run * 1_000_000_000 / elapsed_ns,
                "completed_episodes_per_hour": (
                    sample_stats["full_terminal_episodes"] * 3_600_000_000_000 / elapsed_ns
                ),
                "episode_attempts_per_hour": sample_stats["episodes"] * 3_600_000_000_000 / elapsed_ns,
            })

        if phase_timing_ticks:
            phase_ns: Counter[str] = Counter()
            phase_calls: Counter[str] = Counter()

            def note_phase(phase, elapsed_ns: int) -> None:
                name = phase.name
                phase_calls[name] += 1
                phase_ns[name] += elapsed_ns

            diagnostic_started = time.perf_counter_ns()
            diagnostic_stats = _run_tick_budget(
                ticks=phase_timing_ticks, seed_start=seed_start, pair=pair, config=config,
                on_phase_timing=note_phase,
            )
            diagnostic_elapsed = max(1, time.perf_counter_ns() - diagnostic_started)
            diagnostic = {
                "ticks": phase_timing_ticks,
                "elapsed_ns": diagnostic_elapsed,
                "episodes": diagnostic_stats["episodes"],
                "phase_timings": {
                    name: {
                        "calls": phase_calls[name],
                        "cumulative_ns": phase_ns[name],
                        "percent_of_diagnostic_wall_time": phase_ns[name] * 100 / diagnostic_elapsed,
                    }
                    for name in sorted(phase_calls)
                },
                "excluded_from_throughput_samples": True,
                "timing_note": (
                    "Phase callbacks and per-phase clock reads add overhead. These diagnostics run after the main "
                    "samples, are not subtracted from them, and are not a throughput metric. Percentages use the "
                    "whole diagnostic wall time; phase totals do not include harness/orchestrator gaps."
                ),
            }

    rates = [sample["ticks_per_second"] for sample in samples]
    p10 = nearest_rank(rates, 0.10)
    p50 = statistics.median(rates)
    profile = system_profile()
    protocol_qualified = runs == 5 and ticks_per_run == 100_000 and warmup_ticks == 10_000
    windows_qualified = (
        protocol_qualified
        and profile["os_name"] == "Windows"
        and bool(profile["processor"] and profile["processor"] != "unknown")
        and affinity.report["affinity_applied"] is True
        and affinity.report["restored"] is True
    )
    generated_manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "command": "benchmark",
        "config": _config_descriptor(config),
        "versions": contract_versions(),
        "controllers": controller_descriptors(scenario, a=a, b=b),
        "parameters": {
            "scenario": scenario,
            **({} if a == "scripted" and b == "scripted" else {"a": a, "b": b}),
            "runs": runs,
            "ticks_per_run": ticks_per_run,
            "warmup_ticks": warmup_ticks,
            "seed_start": seed_start,
            "seed_schedule": "seed_start + episode_index; reset to the same schedule for warmup and every sample",
            "pin_core": pin_core,
            "pin_core_policy": "auto on supported platforms when pin_core is null",
            "phase_timing_ticks": phase_timing_ticks,
        },
        "git_commit": current_git_commit(),
        "git_worktree_dirty": current_git_worktree_dirty(),
    }
    manifest = generated_manifest if manifest is None else manifest
    measurement = {
        "system_profile": profile,
        "affinity": affinity.report,
        "warmup": {"ticks": warmup_ticks, "elapsed_ns": warmup_elapsed_ns, **warmup},
        "samples": samples,
        "throughput": {
            "ticks_per_second_p10": p10,
            "ticks_per_second_p50": p50,
            "p10_method": "nearest-rank ceil(0.10 * n); with five runs this is the minimum sample",
            "p50_method": "median of per-run ticks/second",
            "candidate_threshold_ticks_per_second": GATE0_CANDIDATE_TICKS_PER_SECOND,
            "qualified_for_windows_single_core_comparison": windows_qualified,
            "candidate_threshold_met": p10 >= GATE0_CANDIDATE_TICKS_PER_SECOND if windows_qualified else None,
            "gate0_status": "MEASURED_CANDIDATE_ONLY" if windows_qualified else "NOT MEASURED",
            "interpretation": "A throughput comparison is not a balance result or a Gate 0 pass by itself.",
        },
    }
    if diagnostic is not None:
        measurement["phase_diagnostics"] = diagnostic
    return make_result_document("benchmark", manifest, measurement=measurement)


def contract_versions() -> dict[str, str]:
    return {
        "rules": RULES_VERSION,
        "balance": BALANCE_VERSION,
        "observation": OBSERVATION_VERSION,
        "action": ACTION_VERSION,
        "decoder": DECODER_VERSION,
        "rng": RNG_VERSION,
        "stream_derivation": STREAM_DERIVATION_VERSION,
        "serialization": SERIALIZATION_VERSION,
        "replay_format": REPLAY_FORMAT,
        "replay_schema": REPLAY_SCHEMA_VERSION,
    }


def _config_descriptor(config) -> dict[str, Any]:
    return {
        "id": "default_config",
        "config_hash": config_hash(config),
        "t_max": config.t_max,
        "policy_interval_k": config.policy_interval_k,
    }


def controller_descriptors(
    scenario: str | None = TECHNICAL_SCENARIO, *, a: str = "scripted", b: str = "scripted",
) -> dict[str, dict[str, Any]]:
    pair = controller_pair(scenario, a=a, b=b)
    return {side: pair.descriptors[index] for index, side in enumerate(("A", "B"))}


def result_controller_config(pair: ControllerPair) -> dict[str, dict[str, Any]]:
    """Keep the historical scripted result shape; include full descriptors for policies."""
    if all(descriptor["kind"] == "scripted" for descriptor in pair.descriptors):
        return {
            side: {
                "id": pair.ids[index], "kind": "scripted", "script_sha256": pair.script_hashes[index],
            }
            for index, side in enumerate(("A", "B"))
        }
    return {side: pair.descriptors[index] for index, side in enumerate(("A", "B"))}


def current_git_commit() -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, timeout=1,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and len(value) == 40 else None


def current_git_worktree_dirty() -> bool | None:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            ["git", "diff", "--quiet", "HEAD"], cwd=repo_root, capture_output=True, timeout=1, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode == 0:
        return False
    return True if completed.returncode == 1 else None


def build_manifest(command: str, config, parameters: dict[str, Any]) -> dict[str, Any]:
    if command not in {"run", "eval", "benchmark"}:
        raise EvaluationError(f"unsupported manifest command: {command!r}")
    scenario = parameters.get("scenario")
    a, b = parameters.get("a", "scripted"), parameters.get("b", "scripted")
    if a == "scripted" and b == "scripted":
        validate_scenario(scenario)
    elif scenario is not None:
        raise EvaluationError("--scenario is supported only with --a scripted --b scripted.")
    return {
        "schema_version": MANIFEST_SCHEMA,
        "command": command,
        "config": _config_descriptor(config),
        "versions": contract_versions(),
        "controllers": controller_descriptors(scenario, a=a, b=b),
        "parameters": parameters,
        "git_commit": current_git_commit(),
        "git_worktree_dirty": current_git_worktree_dirty(),
    }


def manifest_digest(manifest: dict[str, Any]) -> str:
    payload = json.dumps(manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_result_document(command: str, manifest: dict[str, Any], *, results=None, aggregate=None, measurement=None):
    doc: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "command": command,
        "manifest_sha256": manifest_digest(manifest),
        "manifest": manifest,
    }
    if results is not None:
        doc["results"] = results
    if aggregate is not None:
        doc["aggregate"] = aggregate
    if measurement is not None:
        doc["measurement"] = measurement
    return doc


def load_manifest(path: str | os.PathLike[str], expected_command: str, config) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as file:
            manifest = json.load(file)
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"manifest JSON is invalid: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise EvaluationError(f"manifest schema_version must be {MANIFEST_SCHEMA!r}.")
    if manifest.get("command") != expected_command:
        raise EvaluationError(f"manifest command must be {expected_command!r}.")
    if manifest.get("config") != _config_descriptor(config):
        raise EvaluationError("manifest config/hash/T_max/policy interval differs from the current default config.")
    if manifest.get("versions") != contract_versions():
        raise EvaluationError("manifest contract versions differ from this simulator build.")
    params = manifest.get("parameters")
    if not isinstance(params, dict):
        raise EvaluationError("manifest parameters must be an object.")
    scenario = params.get("scenario")
    a, b = params.get("a", "scripted"), params.get("b", "scripted")
    try:
        if a == "scripted" and b == "scripted":
            validate_scenario(scenario)
        elif scenario is not None:
            raise EvaluationError("manifest scenario cannot be combined with strategic policies.")
        expected_controllers = controller_descriptors(scenario, a=a, b=b)
    except (ValueError, TypeError) as exc:
        raise EvaluationError(f"manifest controller selection is invalid: {exc}") from exc
    if manifest.get("controllers") != expected_controllers:
        raise EvaluationError("manifest controller configuration/script hashes do not match the declared controllers.")
    requested_commit = manifest.get("git_commit")
    local_commit = current_git_commit()
    if requested_commit is not None and local_commit is not None and requested_commit != local_commit:
        raise EvaluationError("manifest git_commit differs from this checkout; regenerate it for the current build.")
    requested_dirty = manifest.get("git_worktree_dirty")
    local_dirty = current_git_worktree_dirty()
    if requested_dirty is not None and local_dirty is not None and requested_dirty != local_dirty:
        raise EvaluationError("manifest git_worktree_dirty differs from this checkout; regenerate the manifest.")
    _validate_manifest_parameters(expected_command, params, config)
    return manifest


def _validate_manifest_parameters(command: str, params: dict[str, Any], config) -> None:
    scripted = "a" not in params and "b" not in params
    expected_keys = {
        "run": {"scenario", "seed", "max_ticks"},
        "eval": {"scenario", "episodes", "seed_start", "paired_seat", "max_ticks", "seed_schedule"},
        "benchmark": {
            "scenario", "runs", "ticks_per_run", "warmup_ticks", "seed_start", "seed_schedule",
            "pin_core", "pin_core_policy", "phase_timing_ticks",
        },
    }[command]
    if not scripted:
        if params.get("a") not in baseline_names() or params.get("b") not in baseline_names():
            raise EvaluationError("manifest a/b must name supported baseline policies.")
        expected_keys = {*expected_keys, "a", "b"}
    if set(params) != expected_keys:
        raise EvaluationError(f"manifest {command} parameters must contain exactly: {', '.join(sorted(expected_keys))}.")
    if command == "run":
        validate_seed(params.get("seed"))
        _validate_max_ticks(params.get("max_ticks"), config)
    elif command == "eval":
        episodes = params.get("episodes")
        paired = params.get("paired_seat")
        seed_start = validate_seed(params.get("seed_start"), "seed_start")
        if type(episodes) is not int or episodes <= 0 or type(paired) is not bool:
            raise EvaluationError("manifest eval parameters need positive episodes and boolean paired_seat.")
        expected_schedule = "seed_start + episode_index; one seed per pair" if paired else "seed_start + episode_index"
        if params.get("seed_schedule") != expected_schedule:
            raise EvaluationError("manifest eval seed_schedule does not match paired_seat.")
        if seed_start + episodes - 1 > U64_MAX:
            raise EvaluationError("manifest seed_start + episodes - 1 exceeds u64.")
        _validate_max_ticks(params.get("max_ticks"), config)
    elif command == "benchmark":
        runs, ticks, warmup, phase_ticks = (
            params.get("runs"), params.get("ticks_per_run"), params.get("warmup_ticks"),
            params.get("phase_timing_ticks"),
        )
        seed_start = validate_seed(params.get("seed_start"), "seed_start")
        if type(runs) is not int or runs <= 0 or type(ticks) is not int or ticks <= 0:
            raise EvaluationError("manifest benchmark needs positive runs and ticks_per_run.")
        if type(warmup) is not int or warmup < 0 or type(phase_ticks) is not int or phase_ticks < 0:
            raise EvaluationError("manifest benchmark warmup/phase_timing_ticks must be non-negative integers.")
        if seed_start + max(ticks, warmup, phase_ticks) - 1 > U64_MAX:
            raise EvaluationError("manifest benchmark seed range exceeds u64.")
        if params.get("seed_schedule") != (
            "seed_start + episode_index; reset to the same schedule for warmup and every sample"
        ):
            raise EvaluationError("manifest benchmark seed_schedule is not supported.")
        if params.get("pin_core_policy") != "auto on supported platforms when pin_core is null":
            raise EvaluationError("manifest benchmark pin_core_policy is not supported.")
        pin_core = params.get("pin_core")
        if pin_core is not None and (type(pin_core) is not int or pin_core < 0):
            raise EvaluationError("manifest pin_core must be a non-negative integer or null.")


def _validate_max_ticks(max_ticks: Any, config) -> None:
    if max_ticks is not None and (type(max_ticks) is not int or not 0 <= max_ticks <= config.t_max):
        raise EvaluationError(f"max_ticks must be null or within 0..T_max ({config.t_max}).")


def write_json(path: str | os.PathLike[str], value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    target.write_text(content, encoding="utf-8")
