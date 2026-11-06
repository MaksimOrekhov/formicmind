"""Independent evaluation (FOR-42): fixed opponents, held-out maps, both seats; never used to update weights.

The suite is a pure function of its name, so every checkpoint of a run is measured on the **same** matches (comparable
curves). Evaluation seeds are derived from the match label into a range disjoint from every training seed, and the
maps are the held-out split of ``maps.py`` (plus ``classic``, reported separately as in-distribution).
``greedy`` plays the mode of the learned distribution, ``sample`` plays the stochastic policy that was trained.
"""

from __future__ import annotations

import hashlib
import time
from collections import Counter
from typing import Any

from .maps import map_split
from .parallel import MatchExecutor
from .pool import EVAL_SEED_BASE, TRAIN_SEED_BASE

BASELINE_OPPONENTS = ("random:v2", "turtle:v2", "rush:v2", "balanced:v2", "raider:v2")
EVAL_SCHEMA = "formicmind.rl-eval/v1"

SUITES: dict[str, dict[str, Any]] = {
    "tiny": {"baselines": ("random:v2", "turtle:v2"), "maps": ("classic",), "reps": 1, "modes": ("sample",)},
    "quick": {"baselines": BASELINE_OPPONENTS, "maps": ("classic", "dense", "chain", "gen:1000000", "gen:1000001"),
              "reps": 1, "modes": ("sample",)},
    "full": {"baselines": BASELINE_OPPONENTS,
             "maps": ("classic", "dense", "far_flank", "chain", *(f"gen:{1_000_000 + i}" for i in range(6))),
             "reps": 2, "modes": ("sample", "greedy")},
}


def _eval_seed(label: str) -> int:
    digest = hashlib.sha256(f"formicmind-rl-eval-{label}".encode()).hexdigest()
    seed = EVAL_SEED_BASE + int(digest[:8], 16)
    assert seed >= TRAIN_SEED_BASE + 10**9
    return seed


def build_suite(name: str, rl_opponents: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Specs for ``name``; ``rl_opponents`` maps a label (e.g. ``init``, ``snap:0003``) to a weight source."""
    if name not in SUITES:
        raise ValueError(f"unknown eval suite {name!r}; choose from {sorted(SUITES)}")
    cfg = SUITES[name]
    opponents: list[tuple[str, str]] = [("baseline", b) for b in cfg["baselines"]]
    opponents += [("rl", label) for label in sorted(rl_opponents or {})]
    specs: list[dict[str, Any]] = []
    seen: set[int] = set()
    for mode in cfg["modes"]:
        reps = cfg["reps"] if mode == "sample" else 1
        for kind, opponent in opponents:
            for map_id in cfg["maps"]:
                for learner_seat in (0, 1):
                    for rep in range(reps):
                        label = f"{mode}|{opponent}|{map_id}|{learner_seat}|{rep}"
                        seed = _eval_seed(label)
                        if seed in seen:
                            raise ValueError(f"evaluation seed collision for {label}")
                        seen.add(seed)
                        learner = {"kind": "rl", "ref": "eval", "stream": ("rl-eval", 0), "greedy": mode == "greedy"}
                        if kind == "baseline":
                            other = {"kind": "baseline", "ref": opponent, "replicate": 0}
                        else:
                            policy_id = "rl-ref-" + opponent.replace(":", "-").replace("_", "-")
                            other = {"kind": "rl", "ref": opponent, "stream": (policy_id, 0), "greedy": False}
                        seats = [learner, other] if learner_seat == 0 else [other, learner]
                        specs.append({
                            "match_id": len(specs), "seed": seed, "map_id": map_id, "seats": seats,
                            "tag": {"opponent": opponent, "mode": mode, "learner_seat": learner_seat, "rep": rep,
                                    "map_split": "indist" if map_id == "classic" else map_split(map_id)},
                            "max_ticks": None,
                        })
    return specs


def _blank() -> dict[str, Any]:
    return {"n": 0, "wins": 0, "draws": 0, "losses": 0, "truncated": 0, "reasons": Counter(), "ticks": 0.0,
            "soldiers": 0.0, "workers": 0.0, "food": 0.0, "decisions": 0, "lay": [0, 0, 0], "army": [0] * 7,
            "avoid": 0, "k": [0, 0, 0], "home_share": 0.0}


def _accumulate(bucket: dict[str, Any], result: dict[str, Any], learner_side: int) -> None:
    learner = result["seats"][learner_side]
    bucket["n"] += 1
    if result["truncated"]:
        bucket["truncated"] += 1
    else:
        score = learner["score"]
        bucket["wins" if score == 1.0 else "draws" if score == 0.5 else "losses"] += 1
        bucket["reasons"][result["reason"]] += 1
    bucket["ticks"] += result["terminal_tick"] if result["terminal_tick"] is not None else result["ticks_executed"]
    final = result["final"]
    bucket["soldiers"] += final["soldiers"][learner_side]
    bucket["workers"] += final["workers"][learner_side]
    bucket["food"] += final["food"][learner_side]
    stats = learner["stats"]
    bucket["decisions"] += stats["decisions"]
    for i, v in enumerate(stats["lay"]):
        bucket["lay"][i] += v
    for i, v in enumerate(stats["army"]):
        bucket["army"][i] += v
    for i, v in enumerate(stats["k_hist"]):
        bucket["k"][i] += v
    bucket["avoid"] += stats["avoid"]
    bucket["home_share"] += stats["home_share_sum"]


def _finish(bucket: dict[str, Any]) -> dict[str, Any]:
    n = max(bucket["n"], 1)
    decisions = max(bucket["decisions"], 1)
    finished = bucket["wins"] + bucket["draws"] + bucket["losses"]
    return {
        "n": bucket["n"], "wins": bucket["wins"], "draws": bucket["draws"], "losses": bucket["losses"],
        "truncated": bucket["truncated"],
        "score_mean": (bucket["wins"] + 0.5 * bucket["draws"]) / finished if finished else None,
        "win_rate": bucket["wins"] / finished if finished else None,
        "reasons": dict(sorted(bucket["reasons"].items())),
        "mean_ticks": bucket["ticks"] / n,
        "mean_final_soldiers": bucket["soldiers"] / n, "mean_final_workers": bucket["workers"] / n,
        "mean_final_food": bucket["food"] / n,
        "lay_fraction": [v / decisions for v in bucket["lay"]],
        "army_fraction": [v / decisions for v in bucket["army"]],
        "k_fraction": [v / decisions for v in bucket["k"]],
        "avoid_rate": bucket["avoid"] / decisions, "mean_home_share": bucket["home_share"] / decisions,
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Win/draw/loss and behaviour statistics from the learner's point of view, per opponent, mode and map split."""
    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    for result in results:
        tag = result["tag"]
        side = tag["learner_seat"]
        for split in ("all", tag["map_split"]):
            key = (tag["opponent"], tag["mode"], split)
            _accumulate(buckets.setdefault(key, _blank()), result, side)
    table: dict[str, Any] = {}
    for (opponent, mode, split), bucket in sorted(buckets.items()):
        table.setdefault(opponent, {}).setdefault(mode, {})[split] = _finish(bucket)
    return table


def run_evaluation(
    executor: MatchExecutor, learner_source: dict[str, Any], rl_opponents: dict[str, dict[str, Any]] | None,
    suite: str, *, iteration: int | None = None,
) -> dict[str, Any]:
    specs = build_suite(suite, rl_opponents)
    refs = {"eval": learner_source, **(rl_opponents or {})}
    started = time.perf_counter()
    results = executor.run(specs, refs)
    wall = time.perf_counter() - started
    return {
        "schema": EVAL_SCHEMA, "suite": suite, "iteration": iteration, "matches": len(results),
        "wall_seconds": wall, "matches_per_second": len(results) / wall if wall else None,
        "truncated_matches": sum(1 for r in results if r["truncated"]),
        "results": aggregate(results),
        "matches_detail": [
            {"seed": r["seed"], "map": r["map_id"], **r["tag"], "winner": r["winner"], "reason": r["reason"],
             "terminal_tick": r["terminal_tick"], "truncated": r["truncated"]} for r in results
        ],
    }


def headline(report: dict[str, Any], mode: str = "sample") -> dict[str, Any]:
    """Compact per-opponent score (all maps) for logs."""
    out: dict[str, Any] = {}
    for opponent, modes in report["results"].items():
        entry = modes.get(mode, {}).get("all")
        if entry:
            out[opponent] = {"score": entry["score_mean"], "w": entry["wins"], "d": entry["draws"], "l": entry["losses"]}
    return out


__all__ = ["BASELINE_OPPONENTS", "EVAL_SCHEMA", "SUITES", "aggregate", "build_suite", "headline", "run_evaluation"]
