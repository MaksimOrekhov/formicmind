import json, sys
from formicmind.eco.config import default_config
from formicmind.eco.engine import RouteTable, step_world_tick
from formicmind.eco.initial import build_world
from formicmind.eco.rl.checkpoint import load_model
from formicmind.eco.rl.evaluate import _controller, _weights
from formicmind.eco.viability import ViabilityTracker
model, _ = load_model(sys.argv[1])
config = default_config(8, "moderate")
job = {"kind": "learned-sample", "weights": _weights(model), "network": model.config.to_dict()}
caste_count = {"worker": 0, "soldier": 0}
ends, soldier_end, worker_end = [], [], []
per_colony = []
for seed in range(90000, 90008):
    controller = _controller(job, config)
    world = build_world(config, seed)
    routes = RouteTable(world.world_map)
    tracker = ViabilityTracker(world.state)
    seen = [dict() for _ in range(8)]
    for _ in range(6000):
        if world.state.world_outcome.finished: break
        r = step_world_tick(world.state, world.run, config, world.world_map, controller, routes=routes, validate_input=world.state.tick == 0)
        tracker.update(world.state, r)
        for c in world.state.colonies:
            for b in c.brood:
                seen[c.colony_id].setdefault(b.cohort_id, (b.caste, b.laid_tick))
    for c in range(8):
        eggs = list(seen[c].values())
        kinds = tuple(sorted(k for k, _ in eggs))
        end = tracker.traces[c].death_tick
        per_colony.append((kinds, end))
        for k, _ in eggs: caste_count[k] += 1
from collections import Counter
print("eggs by caste (64 colonies):", caste_count)
print("egg set -> colonies, median episode end:")
by = {}
for kinds, end in per_colony:
    by.setdefault(kinds, []).append(end)
for kinds, ends in sorted(by.items()):
    ends = sorted(e for e in ends if e)
    print(kinds, len(ends), "median end", ends[len(ends)//2] if ends else None, "range", (ends[0], ends[-1]) if ends else None)
