"""FOR-53 attribution, no change to world execution, and the one-scout control."""

from dataclasses import replace

from formicmind.eco.action import action_masks, validate_action
from formicmind.eco.config import default_config
from formicmind.eco.curriculum import apply_curriculum_masks
from formicmind.eco.engine import ForageFlows, TickRecord
from formicmind.eco.initial import build_world
from formicmind.eco.observation import build_observation
from formicmind.eco.policy import ObservationHeuristicPolicy
from formicmind.eco.policy_run import PolicyController
from formicmind.eco.rl.play import play_world
from scripts.ecosystem_scout_value import BoundedScout, ScoutFoodTrace


def test_forage_flows_match_real_food_totals_without_changing_execution():
    config = default_config()

    def play(observe_tick=None):
        controller = PolicyController([ObservationHeuristicPolicy(config) for _ in range(config.n_colonies)], curriculum="scout-v1")
        return play_world(config, 42, controller, 300, observe_tick=observe_tick), controller.log.head

    flows_seen = set()

    def observe(_state, world_map, record):
        flows = record.forage_flows
        assert sum(flows.harvested.values()) == record.harvested
        assert sum(flows.delivered.values()) == record.delivered
        for key in flows.delivered:
            assert key in flows_seen
            assert world_map.nodes[key[1]].patch
        flows_seen.update(flows.harvested)

    observed, observed_decisions = play(observe)
    control, control_decisions = play()
    assert flows_seen and sum(observed.delivered) > 0
    assert observed.state == control.state
    assert observed.events.head == control.events.head and observed_decisions == control_decisions


def test_attribution_requires_an_exclusive_visit_and_later_harvest_and_delivery_by_its_colony():
    config = default_config()
    world = build_world(config, 42)
    patches = (node.node_id for node in world.world_map.nodes if node.patch)
    primary, secondary = next(patches), next(patches)
    junction = next(node.node_id for node in world.world_map.nodes if node.kind == "junction")
    trace = ScoutFoodTrace(config.n_colonies)

    def observe(tick, visits=(), harvests=None, deliveries=None):
        record = TickRecord(tick, discoveries=list(visits), forage_flows=ForageFlows(harvests or {}, deliveries or {}))
        trace.observe(world.state, world.world_map, record)

    observe(1, harvests={(0, primary): 10})
    observe(2, [(0, primary, "scout"), (0, secondary, "scout"), (1, primary, "mixed"), (0, junction, "scout")],
            harvests={(0, primary): 5})
    observe(3, harvests={(0, primary): 4, (0, secondary): 7, (1, primary): 5}, deliveries={(0, primary): 10})
    observe(4, deliveries={(0, primary): 3, (1, primary): 5})

    row = trace.metrics(0)
    assert row["junctions_discovered_by_scout"] == 1
    assert row["patches_discovered_by_scout"] == row["scout_patches_harvested"] == 2
    assert row["scout_patches_delivered"] == 1 and row["food_delivered_from_scout_patches"] == 3
    assert row["scout_patches"][0] == {"node": primary, "discovered_tick": 2, "first_harvest_tick": 3,
                                       "first_delivery_tick": 4, "harvested": 4, "delivered": 3}
    assert trace.metrics(1)["patches_discovered_by_scout"] == 0


def test_bounded_scout_preserves_peaceful_orders_and_respects_private_idle_and_active_counts():
    config = default_config()
    world = build_world(config, 42)
    colony = world.state.colonies[0]
    cohorts = [c for c in world.state.cohorts if c.colony_id == 0]
    original = build_observation(colony, cohorts, 0, config, decision_index=0, ticks_since_last_decision=0)
    obs = replace(original, facts=replace(original.facts, workers=12))
    masks = apply_curriculum_masks(obs, action_masks(obs), "scout-v1")
    policy = BoundedScout(config)
    action = policy.act(obs, masks)
    assert action.scout_n == 1
    validate_action(action, masks)
    assert replace(action, scout_n=0, scout_targets=()) == ObservationHeuristicPolicy(config).act(obs, masks)

    for facts_change in ({"scouts_out": 1}, {"idle_workers_home": 0}, {"food": 29}):
        blocked = replace(obs, facts=replace(obs.facts, **facts_change))
        assert policy.act(blocked, action_masks(blocked)).scout_n == 0
    assert policy.act(obs, apply_curriculum_masks(obs, action_masks(obs), "nest-v1")).scout_n == 0
