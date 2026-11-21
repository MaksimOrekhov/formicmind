"""FOR-53 fixed-world scouting preflight. Evaluates existing policies on CPU."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from formicmind.eco.config import ConfigE1, default_config
from formicmind.eco.engine import TickRecord
from formicmind.eco.model import MapE1, WorldStateE1
from formicmind.eco.policy import ObservationHeuristicPolicy
from formicmind.eco.policy_run import PolicyController
from formicmind.eco.rl.checkpoint import load_model
from formicmind.eco.rl.evaluate import EVAL_POLICY_SEED, _controller, _weights
from formicmind.eco.rl.play import colony_metrics, play_world, summarize
from formicmind.eco.rl.train import _Pool
from formicmind.eco.serialization import state_hash
from formicmind.eco.versions import ACTION_VERSION, ECO_RULES_VERSION, OBSERVATION_VERSION
from formicmind.eco.viability import REWARD_VERSION


COLONIES = 8
TICKS = 6000
SEEDS = tuple(range(90000, 90008))
PEACEFUL = "heuristic:peaceful"
EXPLORER = "heuristic:explorer"
BOUNDED = "peaceful+bounded_scout"
EXAMPLES = Path(__file__).resolve().parents[1] / "docs" / "ecosystem_examples"
MODELS = {
    "FOR-50": ("for50_nest_pilot/model", "1b4f8740a1b5ea3d5c2e5c9012c9dd23959c84eca25640f8b33c21a782c55d44"),
    "FOR-52": ("for52_scout/model", "49e942b0c8dbd48198bcfcd1e3db3931abd8b1459f237a8c70799ed7ab6d471d"),
}


class BoundedScout(ObservationHeuristicPolicy):
    """FOR-53 diagnostic: peaceful with one scout on the nearest short frontier trip."""

    def __init__(self, config: ConfigE1) -> None:
        super().__init__(config, "peaceful")
        self.name = BOUNDED

    def act(self, obs, masks):
        economy = super().act(obs, masks)
        facts = obs.facts
        if (facts.scouts_out or facts.workers < 12 or facts.food < 30
                or not facts.idle_workers_home or not masks.scout_n[1]):
            return economy
        candidates = ((distance, slot) for slot, (_, visited, _, distance) in enumerate(obs.node_ints)
                      if masks.scout_target[slot] and not visited and 0 <= distance <= 15)
        target = min(candidates, default=None)
        if target is None:
            return economy
        return replace(economy, scout_n=1, scout_targets=(target[1],))


@dataclass
class ScoutPatch:
    discovered_tick: int
    first_harvest_tick: int | None = None
    first_delivery_tick: int | None = None
    harvested: int = 0
    delivered: int = 0


class ScoutFoodTrace:
    """Join first visits to later physical harvest and nest receipts for one world."""

    def __init__(self, colonies: int) -> None:
        self.patches: list[dict[int, ScoutPatch]] = [{} for _ in range(colonies)]
        self.junctions = [0] * colonies
        self.worker_ticks = [0] * colonies
        self.peak_active = [0] * colonies
        self.map_hash = ""

    def observe(self, state: WorldStateE1, world_map: MapE1, record: TickRecord) -> None:
        self.map_hash = world_map.map_hash
        for colony, node, source in record.discoveries:
            if source != "scout":
                continue
            if world_map.nodes[node].patch:
                self.patches[colony][node] = ScoutPatch(record.tick)
            elif world_map.nodes[node].kind == "junction":
                self.junctions[colony] += 1
        flows = record.forage_flows
        assert flows is not None
        for (colony, node), amount in flows.harvested.items():
            patch = self.patches[colony].get(node)
            if patch is not None and record.tick > patch.discovered_tick:
                patch.harvested += amount
                if patch.first_harvest_tick is None:
                    patch.first_harvest_tick = record.tick
        for (colony, node), amount in flows.delivered.items():
            patch = self.patches[colony].get(node)
            if patch is not None and patch.first_harvest_tick is not None and record.tick > patch.first_harvest_tick:
                patch.delivered += amount
                if patch.first_delivery_tick is None:
                    patch.first_delivery_tick = record.tick
        active = [0] * len(self.worker_ticks)
        for cohort in state.cohorts:
            if cohort.mission.kind == "scout" and cohort.caste == "worker":
                active[cohort.colony_id] += cohort.count
        for colony, count in enumerate(active):
            self.worker_ticks[colony] += count
            self.peak_active[colony] = max(self.peak_active[colony], count)

    def metrics(self, colony: int) -> dict:
        patches = self.patches[colony]
        return {
            "junctions_discovered_by_scout": self.junctions[colony],
            "patches_discovered_by_scout": len(patches),
            "scout_patches_harvested": sum(patch.harvested > 0 for patch in patches.values()),
            "scout_patches_delivered": sum(patch.delivered > 0 for patch in patches.values()),
            "food_harvested_from_scout_patches": sum(patch.harvested for patch in patches.values()),
            "food_delivered_from_scout_patches": sum(patch.delivered for patch in patches.values()),
            "scout_worker_ticks": self.worker_ticks[colony],
            "scouts_active_peak": self.peak_active[colony],
            "scout_patches": [{"node": node, **vars(patch)} for node, patch in sorted(patches.items())],
        }


SCOUT_TOTALS = (
    "junctions_discovered_by_scout", "patches_discovered_by_scout", "scout_patches_harvested", "scout_patches_delivered",
    "food_harvested_from_scout_patches", "food_delivered_from_scout_patches", "scout_worker_ticks",
)
PAIR_METRICS = ("reward", "functional_survival", "physical_survival", "delivered", "workers_mean", "workers_end", "worker_eggs_laid", "food_end")


def summary(runs: list[dict]) -> dict:
    rows = [run["colonies"] for run in runs]
    flattened = [row for world in rows for row in world]
    return {
        **summarize(rows),
        "totals": {key: sum(row[key] for row in flattened) for key in SCOUT_TOTALS},
        "colonies_scouted_new_node": sum(row["discovered_by_scout"] > 0 for row in flattened),
        "colonies_delivered_from_scout_patch": sum(row["scout_patches_delivered"] > 0 for row in flattened),
        "scouts_active_peak": max(row["scouts_active_peak"] for row in flattened),
    }


def evaluate_world(job: dict) -> dict:
    torch.set_num_threads(1)
    config = default_config(COLONIES, job["scenario"])
    if job["kind"] == BOUNDED:
        controller = PolicyController([BoundedScout(config) for _ in range(COLONIES)], curriculum=job["curriculum"])
    else:
        controller = _controller(job, config)
    trace = ScoutFoodTrace(COLONIES)
    result = play_world(config, job["seed"], controller, TICKS, observe_tick=trace.observe)
    colonies = [{**row, **trace.metrics(row["colony"])} for row in colony_metrics(result)]
    if job["kind"] == BOUNDED and max(row["scouts_active_peak"] for row in colonies) > 1:
        raise AssertionError("bounded scout exceeded its active-worker limit")
    return {"seed": job["seed"], "ticks_played": result.ticks, "map_hash": trace.map_hash,
            "state_hash": state_hash(result.state), "events_hash": result.events.head,
            "decisions_hash": controller.log.head, "colonies": colonies}


def paired_delta(scout: dict, peaceful: dict) -> dict:
    return {key: scout[key] - peaceful[key] for key in PAIR_METRICS}


def score_gate(policies: dict) -> dict:
    peaceful = policies[PEACEFUL]
    checks = {}
    for name in (EXPLORER, BOUNDED):
        scout = policies[name]
        pairs = []
        for test, control in zip(scout["worlds"], peaceful["worlds"], strict=True):
            if (test["seed"], test["map_hash"]) != (control["seed"], control["map_hash"]):
                raise ValueError("scout and peaceful must play the same maps in the same order")
            pairs.append({"seed": test["seed"], **paired_delta(test["summary"], control["summary"])})
        row = scout["summary"]
        delta = paired_delta(row, peaceful["summary"])
        criteria = {
            "at_least_half_scout_discovered": row["share_scouted_new_node"] >= 0.5,
            "has_delivery_from_scout_patch": row["totals"]["food_delivered_from_scout_patches"] > 0,
            "economic_gain": delta["reward"] > 0 or (delta["functional_survival"] >= 0.10 and delta["reward"] >= -0.05),
        }
        checks[name] = {"passed": all(criteria.values()), "criteria": criteria, "delta": delta, "paired_world_deltas": pairs}
    return {"passed": any(check["passed"] for check in checks.values()), "scout_controls": checks}


def learned_jobs(model_dir: Path, expected_sha: str, curriculum: str) -> list[dict]:
    model, manifest = load_model(model_dir)
    if manifest["weights_sha256"] != expected_sha or manifest["reward_version"] != REWARD_VERSION:
        raise ValueError(f"unexpected weight lineage or reward version in {model_dir}")
    weights = _weights(model)
    return [{"kind": f"learned-{mode}", "curriculum": curriculum, "weights": weights,
             "network": model.config.to_dict(), "model_sha256": expected_sha} for mode in ("greedy", "sample")]


def evaluate(policies: dict[str, dict], workers: int, scenario: str = "severe") -> dict:
    jobs = [{**policy, "scenario": scenario, "seed": seed} for policy in policies.values() for seed in SEEDS]
    pool = _Pool(workers)
    try:
        runs = iter(pool.map(evaluate_world, jobs))
    finally:
        pool.close()
    results = {}
    for name, policy in policies.items():
        worlds = [next(runs) for _ in SEEDS]
        for world in worlds:
            world["summary"] = summary([world])
        results[name] = {"curriculum": policy["curriculum"], "model_sha256": policy.get("model_sha256"),
                         "summary": summary(worlds), "worlds": worlds}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    torch.set_num_threads(1)
    policies = {name: {"kind": name, "curriculum": "scout-v1"} for name in (PEACEFUL, EXPLORER, "random_valid", BOUNDED)}
    for name, (path, sha) in MODELS.items():
        profiles = ("scout-v1", "nest-v1") if name == "FOR-50" else ("scout-v1",)
        for profile in profiles:
            for job in learned_jobs(EXAMPLES / path, sha, profile):
                policies[f"{name}:{job['kind'][len('learned-'):]}:{profile}"] = job
    results = evaluate(policies, args.workers)
    report = {"scenario": "severe", "seeds": SEEDS, "colonies_per_world": COLONIES, "ticks": TICKS,
              "policy_seed": EVAL_POLICY_SEED, "eco_rules_version": ECO_RULES_VERSION,
              "observation_version": OBSERVATION_VERSION, "action_version": ACTION_VERSION,
              "reward_version": REWARD_VERSION, "python": platform.python_version(), "torch": torch.__version__,
              "policies": results, "gate": score_gate(results)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
