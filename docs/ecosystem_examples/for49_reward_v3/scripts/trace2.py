import sys
from formicmind.eco.config import default_config
from formicmind.eco.engine import RouteTable, step_world_tick
from formicmind.eco.initial import build_world
from formicmind.eco.rl.checkpoint import load_model
from formicmind.eco.rl.evaluate import _controller, _weights
from formicmind.eco.viability import ViabilityTracker
model, _ = load_model(sys.argv[1])
config = default_config(8, "moderate")
job = {"kind": "learned-sample", "weights": _weights(model), "network": model.config.to_dict()}
controller = _controller(job, config)
for seed in (90000, 90001):
    world = build_world(config, seed)
    routes = RouteTable(world.world_map)
    tracker = ViabilityTracker(world.state)
    prev = None
    print("seed", seed)
    for _ in range(4000):
        if world.state.world_outcome.finished: break
        r = step_world_tick(world.state, world.run, config, world.world_map, controller, routes=routes, validate_input=world.state.tick == 0)
        tracker.update(world.state, r)
        t = world.state.tick
        c = world.state.colonies[0]
        w = sum(x.count for x in world.state.cohorts if x.colony_id == 0 and x.caste == "worker")
        s = sum(x.count for x in world.state.cohorts if x.colony_id == 0 and x.caste == "soldier")
        key = (w, s, len(c.brood), c.status, tracker.terminal_reason(0))
        if key != prev and t > 1000:
            print(t, "food", c.food, "workers", w, "soldiers", s, "brood", len(c.brood), c.status, tracker.terminal_reason(0), "old_age_w", r.old_age_workers, "old_age_s", r.old_age_soldiers, "starv_w", r.starvation_workers, "starv_s", r.starvation_soldiers)
            prev = key
