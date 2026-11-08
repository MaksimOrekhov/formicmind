"""What does the reward-v3 network actually do? Per-window behaviour of the pilot-3 model next to heuristic:peaceful on the held-out maps (no training).

usage: python behavior_v3.py MODEL_DIR OUT.json
"""
import json
import sys
from pathlib import Path

from formicmind.eco.config import default_config
from formicmind.eco.engine import RouteTable, step_world_tick
from formicmind.eco.initial import build_world
from formicmind.eco.rl.checkpoint import load_model
from formicmind.eco.rl.evaluate import _controller, _weights
from formicmind.eco.viability import ViabilityTracker

W, H = 500, 6000
model, _ = load_model(sys.argv[1])
config = default_config(8, "moderate")
seeds = list(range(90000, 90004))
result = {}
for name, kind in (("v3_model", "learned-sample"), ("peaceful", "heuristic:peaceful")):
    n_win = H // W
    acc = {k: [0.0] * n_win for k in ("workers", "food", "eggs", "alive_v3", "forage_decisions", "egg_decisions", "decisions", "scouts")}
    first_action = None
    death_ticks, last_egg_tick, first_egg_tick = [], [], []
    count = 0
    for seed in seeds:
        job = {"kind": kind, "weights": _weights(model) if kind.startswith("learned") else b"",
               "network": model.config.to_dict() if kind.startswith("learned") else {}}
        controller = _controller(job, config)
        world = build_world(config, seed)
        routes = RouteTable(world.world_map)
        tracker = ViabilityTracker(world.state)
        seen_brood = [set() for _ in range(8)]
        eggs_ticks = [[] for _ in range(8)]
        for _ in range(H):
            if world.state.world_outcome.finished:
                break
            record = step_world_tick(world.state, world.run, config, world.world_map, controller, routes=routes, validate_input=world.state.tick == 0)
            tracker.update(world.state, record)
            t = record.tick
            w = min(n_win - 1, t // W)
            for colony in world.state.colonies:
                c = colony.colony_id
                for b in colony.brood:
                    if b.cohort_id not in seen_brood[c]:
                        seen_brood[c].add(b.cohort_id)
                        eggs_ticks[c].append(t)
                        acc["eggs"][w] += 1
            if t % 10 == 0:
                workers = [0] * 8
                for cohort in world.state.cohorts:
                    if cohort.caste == "worker":
                        workers[cohort.colony_id] += cohort.count
                for colony in world.state.colonies:
                    acc["workers"][w] += workers[colony.colony_id] / (W // 10)
                    acc["food"][w] += colony.food / (W // 10)
                    acc["alive_v3"][w] += (tracker.traces[colony.colony_id].death_tick is None) / (W // 10)
        for rec in controller.log.records:
            w = min(n_win - 1, rec["tick"] // W)
            action = rec["action"]
            if first_action is None:
                first_action = action
            acc["decisions"][w] += 1
            acc["egg_decisions"][w] += 1 if action.get("egg_level", action.get("egg", 0)) else 0
        for c in range(8):
            trace = tracker.traces[c]
            death_ticks.append(None if trace.death_tick is None else trace.death_tick - 1)
            if eggs_ticks[c]:
                first_egg_tick.append(min(eggs_ticks[c]))
                last_egg_tick.append(max(eggs_ticks[c]))
        count += 8
    out = {k: [v / count for v in vals] for k, vals in acc.items() if k not in ("eggs", "decisions", "egg_decisions", "forage_decisions", "scouts")}
    out["eggs_per_colony"] = [v / count for v in acc["eggs"]]
    out["share_decisions_with_egg_level_gt0"] = [e / d if d else None for e, d in zip(acc["egg_decisions"], acc["decisions"])]
    alive = [d for d in death_ticks if d is not None]
    out["episode_end_tick"] = {"n_terminated": len(alive), "n_colonies": count, "min": min(alive) if alive else None,
                               "median": sorted(alive)[len(alive) // 2] if alive else None, "max": max(alive) if alive else None}
    out["colonies_with_any_egg"] = len(first_egg_tick)
    out["first_egg_tick_median"] = sorted(first_egg_tick)[len(first_egg_tick) // 2] if first_egg_tick else None
    out["last_egg_tick_median"] = sorted(last_egg_tick)[len(last_egg_tick) // 2] if last_egg_tick else None
    out["sample_action_keys"] = sorted(first_action) if isinstance(first_action, dict) else str(first_action)[:200]
    result[name] = out
    print(name, json.dumps({k: (v if not isinstance(v, list) else [round(x, 2) if isinstance(x, (int, float)) else x for x in v]) for k, v in out.items()}))
Path(sys.argv[2]).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
