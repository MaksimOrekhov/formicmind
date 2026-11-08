"""Offline re-scoring (no training): when would a 'functional extinction' terminal (no adults and no worker brood, whatever the store holds) fire,
and what would the reward of the existing policies be with it? Same held-out maps 90000..90007, sampled policies, reward v2 trace."""
import statistics
import sys

from formicmind.eco.config import default_config
from formicmind.eco.engine import RouteTable, step_world_tick
from formicmind.eco.initial import build_world
from formicmind.eco.rl.checkpoint import load_model
from formicmind.eco.rl.evaluate import _controller, _weights
from formicmind.eco.viability import DEATH_PENALTY, Q, ViabilityTracker

model, _ = load_model(sys.argv[1])
config = default_config(8, "moderate")
H = 6000
for name, kind in (("learned_v2", "learned-sample"), ("peaceful", "heuristic:peaceful"), ("explorer", "heuristic:explorer")):
    rescored, old, ticks_fire, fired, real_dead = [], [], [], 0, 0
    for seed in range(90000, 90008):
        job = {"kind": kind, "weights": _weights(model) if kind.startswith("learned") else b"",
               "network": model.config.to_dict() if kind.startswith("learned") else {}}
        controller = _controller(job, config)
        world = build_world(config, seed)
        routes = RouteTable(world.world_map)
        tracker = ViabilityTracker(world.state)
        extinct = [None] * 8
        for _ in range(H):
            record = step_world_tick(world.state, world.run, config, world.world_map, controller, routes=routes, validate_input=world.state.tick == 0)
            tracker.update(world.state, record)
            adults = [0] * 8
            for cohort in world.state.cohorts:
                adults[cohort.colony_id] += cohort.count
            for colony in world.state.colonies:
                c = colony.colony_id
                if extinct[c] is None and colony.status == "alive" and adults[c] == 0 and not any(b.caste == "worker" for b in colony.brood):
                    extinct[c] = world.state.tick
            if world.state.world_outcome.finished:
                break
        for c in range(8):
            trace = tracker.traces[c]
            dead = trace.death_tick is not None
            old.append(sum(trace.v_q) / (Q * H) - (DEATH_PENALTY if dead else 0.0))
            if extinct[c] is not None and not dead:
                fired += 1
                ticks_fire.append(extinct[c])
                rescored.append(sum(trace.v_q[: extinct[c]]) / (Q * H) - DEATH_PENALTY)
            else:
                real_dead += dead
                rescored.append(old[-1])
    print(f"{name:11s} colonies={len(old)} real_deaths={real_dead} functional_extinction_only={fired}"
          f" fire_tick median={statistics.median(ticks_fire) if ticks_fire else None} range={(min(ticks_fire), max(ticks_fire)) if ticks_fire else None}"
          f"  reward_v2={sum(old) / len(old):.3f} -> with terminal={sum(rescored) / len(rescored):.3f}")
