"""FOR-54 phase A: fixed-world evaluation of the first scouting-to-food cycle."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from formicmind.eco.config import default_config
from formicmind.eco.events import MISSION_CODES
from formicmind.eco.knowledge import UNKNOWN_FOOD_V1
from formicmind.eco.policy import BOUNDED_UNKNOWN_FOOD, PolicyRunSpec, PolicySpec, build_policies
from formicmind.eco.policy_run import PolicyController
from formicmind.eco.replay_file import export_policy_replay, verify_replay
from formicmind.eco.rl.play import colony_metrics, play_world, summarize
from formicmind.eco.serialization import state_hash
from formicmind.eco.versions import ACTION_VERSION, ECO_RULES_VERSION, OBSERVATION_VERSION
from formicmind.eco.viability import REWARD_VERSION
from scripts.ecosystem_scout_value import ScoutFoodTrace


SEEDS = tuple(range(90000, 90008))
COLONIES = 8
HORIZON = 6000
POLICY_SEED = 1
SCENARIO = "moderate"
CONTROLS = {"no_scout": ("peaceful", "nest-v1"), "bounded": (BOUNDED_UNKNOWN_FOOD, "scout-v1")}


class BootstrapTrace(ScoutFoodTrace):
    def __init__(self, colonies: int) -> None:
        super().__init__(colonies)
        self.first_order: dict[int, int] = {}
        self.first_arrival: dict[int, int] = {}
        self.first_empty_food: dict[int, int] = {}
        self.first_starvation: dict[int, int] = {}

    def observe(self, state, world_map, record) -> None:
        super().observe(state, world_map, record)
        for event in record.events:
            if event.kind == "dispatch" and event.a == MISSION_CODES["scout"]:
                self.first_order.setdefault(event.colony, record.tick)
        for colony, _, _ in record.scout_arrivals:
            self.first_arrival.setdefault(colony, record.tick)
        for colony in state.colonies:
            if colony.food == 0:
                self.first_empty_food.setdefault(colony.colony_id, record.tick)
            if colony.starvation_debt > 0:
                self.first_starvation.setdefault(colony.colony_id, record.tick)

    def metrics(self, colony: int) -> dict:
        row = super().metrics(colony)
        patches = self.patches[colony].values()
        first_delivery = min((patch.first_delivery_tick for patch in patches if patch.first_delivery_tick is not None), default=None)
        return {**row, "first_scout_order_tick": self.first_order.get(colony), "first_scout_arrival_tick": self.first_arrival.get(colony),
                "first_scout_patch_tick": min((patch.discovered_tick for patch in patches), default=None),
                "first_scout_patch_delivery_tick": first_delivery, "first_empty_food_tick": self.first_empty_food.get(colony),
                "first_starvation_debt_tick": self.first_starvation.get(colony)}


def spec_for(variant: str, curriculum: str) -> PolicyRunSpec:
    return PolicyRunSpec(POLICY_SEED, (PolicySpec("heuristic", variant),) * COLONIES, curriculum)


def run_control(job: tuple[int, str, str]) -> dict:
    seed, variant, curriculum = job
    config = default_config(COLONIES, SCENARIO)
    spec = spec_for(variant, curriculum)
    policies = build_policies(spec, config, initial_knowledge=UNKNOWN_FOOD_V1)
    controller = PolicyController(policies, curriculum=curriculum, initial_knowledge=UNKNOWN_FOOD_V1)
    trace = BootstrapTrace(COLONIES)
    play = play_world(config, seed, controller, HORIZON, observe_tick=trace.observe, initial_knowledge=UNKNOWN_FOOD_V1)
    rows = []
    for row in colony_metrics(play):
        colony = row["colony"]
        row.update(trace.metrics(colony))
        delivery = row["first_scout_patch_delivery_tick"]
        outcome = play.state.outcomes[colony]
        physical_death = outcome.terminal_tick if outcome.finished else None
        row["physical_death_tick"] = physical_death
        row["physical_death_before_delivery"] = physical_death is not None and (delivery is None or physical_death < delivery)
        row["functional_death_before_delivery"] = row["death_tick"] is not None and (delivery is None or row["death_tick"] < delivery)
        rows.append(row)
    return {"seed": seed, "map_hash": trace.map_hash, "ticks_played": play.ticks, "state_hash": state_hash(play.state),
            "events_hash": play.events.head, "decisions_hash": controller.log.head, "colonies": rows}


def summarize_control(runs: list[dict]) -> dict:
    rows = [row for run in runs for row in run["colonies"]]
    counts = ("physical_death_before_delivery", "functional_death_before_delivery", "junctions_discovered_by_scout",
              "patches_discovered_by_scout", "scout_patches_harvested", "scout_patches_delivered", "food_delivered_from_scout_patches")
    return {**summarize([run["colonies"] for run in runs]), **{key: sum(row[key] for row in rows) for key in counts},
            "colonies_with_delivery_from_scout_patch": sum(row["scout_patches_delivered"] > 0 for row in rows),
            "colonies_with_tick0_scout": sum(row["first_scout_order_tick"] == 0 for row in rows),
            "scouts_active_peak": max(row["scouts_active_peak"] for row in rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", default=8, type=int)
    args = parser.parse_args()
    jobs = [(seed, *control) for control in CONTROLS.values() for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        runs = list(pool.map(run_control, jobs))
    results = {}
    for index, (name, (variant, curriculum)) in enumerate(CONTROLS.items()):
        worlds = runs[index * len(SEEDS):(index + 1) * len(SEEDS)]
        results[name] = {"variant": variant, "curriculum": curriculum, "worlds": worlds, "summary": summarize_control(worlds)}
    control, scout = (results[name]["summary"] for name in CONTROLS)
    if [r["map_hash"] for r in results["no_scout"]["worlds"]] != [r["map_hash"] for r in results["bounded"]["worlds"]]:
        raise AssertionError("paired controls must use the same maps")
    if scout["colonies_with_tick0_scout"] != COLONIES * len(SEEDS) or scout["scouts_active_peak"] != 1:
        raise AssertionError("the bounded scout must deploy at tick zero with at most one active scout")
    gate = {"no_scout_delivery_zero": control["delivered"] == 0,
            "delivery_from_scout_patch_at_least_half": scout["colonies_with_delivery_from_scout_patch"] >= COLONIES * len(SEEDS) / 2,
            "functional_survival_at_least_half": scout["functional_survival"] >= 0.5}
    world = next((r for r in results["bounded"]["worlds"] if any(row["scout_patches_delivered"] for row in r["colonies"])),
                 results["bounded"]["worlds"][0])
    config = default_config(COLONIES, SCENARIO)
    replay, decisions = export_policy_replay(config, world["seed"], spec_for(*CONTROLS["bounded"]), HORIZON, config.decision_k,
                                             initial_knowledge=UNKNOWN_FOOD_V1)
    verified = verify_replay(replay, decisions)
    if verified.result["final_state_hash"] != world["state_hash"] or verified.result["decisions_hash"] != world["decisions_hash"]:
        raise AssertionError("verified replay must reproduce the measured world")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"replay_seed{world['seed']}.jsonl").write_bytes(replay)
    report = {"scenario": SCENARIO, "initial_knowledge": UNKNOWN_FOOD_V1, "seeds": SEEDS, "policy_seed": POLICY_SEED,
              "colonies_per_world": COLONIES, "horizon": HORIZON, "decision_k": config.decision_k,
              "start_food": config.economy.start_food, "start_workers": config.economy.start_workers,
              "reward_version": REWARD_VERSION, "eco_rules_version": ECO_RULES_VERSION, "observation_version": OBSERVATION_VERSION,
              "action_version": ACTION_VERSION, "controls": results, "gate": {**gate, "passed": all(gate.values())},
              "replay_seed": world["seed"], "replay_chain_head": verified.chain_head}
    (args.out / "preflight.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"gate": report["gate"], "no_scout": control, "bounded": scout}, sort_keys=True))


if __name__ == "__main__":
    main()
