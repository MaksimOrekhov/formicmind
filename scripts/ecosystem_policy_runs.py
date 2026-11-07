"""FOR-48 evidence: reproducible headless worlds whose eight colonies are decided by independent ObservationE1 -> ActionE1 policies.

Runs each world twice from scratch and records the hashes (state, events, decisions) of both runs, so that "same seed, same hashes" is a
checked fact of the committed JSON and not a claim. Nothing is tuned and nothing is a benchmark of strategies: random bots are not expected to
survive, the heuristic adapter is an interface check.

    python scripts/ecosystem_policy_runs.py --out docs/ecosystem_examples/for48_policy_runs.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formicmind.eco import default_config  # noqa: E402
from formicmind.eco.policy import PolicyRunSpec, PolicySpec, spec_to_canonical  # noqa: E402
from formicmind.eco.policy_run import run_policies  # noqa: E402

WORLDS = (
    # (map seed, scenario, policy seed, description, per-colony policies)
    (50004, "moderate", 7, "all random_valid", ["random_valid"] * 8),
    (50011, "moderate", 11, "fair heuristic economy, scouts, one raider, one random",
     ["heuristic:explorer", "heuristic:explorer", "heuristic:peaceful", "heuristic:raider", "heuristic:explorer", "heuristic:peaceful",
      "random_valid", "heuristic:explorer"]),
    (50012, "moderate", 5, "three aggressors (attack path through ActionE1), raider, peaceful, explorers, one random",
     ["heuristic:explorer", "heuristic:aggressor", "heuristic:explorer", "heuristic:aggressor", "heuristic:peaceful", "heuristic:aggressor",
      "random_valid", "heuristic:raider"]),
    (50031, "abundance", 3, "same mix on an abundant map",
     ["heuristic:explorer", "heuristic:aggressor", "heuristic:explorer", "heuristic:raider", "heuristic:peaceful", "heuristic:explorer",
      "random_valid", "heuristic:explorer"]),
)


def spec_of(items: list[str], seed: int) -> PolicyRunSpec:
    out = []
    for item in items:
        kind, _, variant = item.partition(":")
        out.append(PolicySpec(kind, variant))
    return PolicyRunSpec(seed, tuple(out))


def run_world(map_seed: int, scenario: str, policy_seed: int, description: str, items: list[str], ticks: int) -> dict:
    config = default_config(8, scenario)
    spec = spec_of(items, policy_seed)
    runs = []
    for _ in range(2):
        started = time.perf_counter()
        result = run_policies(config, map_seed, spec, ticks)
        elapsed = time.perf_counter() - started
        runs.append((result, elapsed))
    first, second = runs[0][0], runs[1][0]
    counts = first.events.counts()
    return {
        "map_seed": str(map_seed), "scenario": scenario, "colonies": 8, "ticks": first.state.tick, "description": description,
        "policy": spec_to_canonical(spec),
        "statuses": [colony.status for colony in first.state.colonies],
        "event_counts": counts, "decisions": first.decisions.count,
        "alert_decisions": sum(1 for r in first.decisions.records if r["kind"] == "alert"),
        "hashes": {"state": first.state_hash, "events": first.events_hash, "decisions": first.decisions_hash},
        "reproduced": (first.state_hash, first.events_hash, first.decisions_hash) == (second.state_hash, second.events_hash, second.decisions_hash),
        "wall_seconds": [round(elapsed, 2) for _result, elapsed in runs],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ticks", type=int, default=3000)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    report = {"schema": "formicmind.eco.policy-runs/1", "worlds": [run_world(*world, args.ticks) for world in WORLDS]}
    text = json.dumps(report, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    else:
        sys.stdout.write(text)
    return 0 if all(world["reproduced"] for world in report["worlds"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
