from copy import deepcopy
from dataclasses import replace

import pytest

from formicmind.eco.config import default_config
from formicmind.eco.engine import RouteTable, split_cohort, step_world_tick
from formicmind.eco.fixtures import fixture_controller
from formicmind.eco.initial import build_world
from formicmind.eco.model import BroodRecord, Location, Mission
from formicmind.eco.serialization import checkpoint_hash, state_hash
from formicmind.eco.validate import InvariantError, validate_state


def _run_step(world, routes=None, controller=None):
    return step_world_tick(
        world.state, world.run, world.config, world.world_map, controller,
        routes=routes, validate_input=world.state.tick == 0,
    )


def test_tick_moves_cohorts_before_food_can_be_delivered():
    world = build_world(default_config(), 42)
    routes = RouteTable(world.world_map)
    initial_food = sum(colony.food for colony in world.state.colonies)

    record = _run_step(world, routes, fixture_controller("peaceful_fair"))

    assert record.tick == 0
    assert record.delivered == 0
    assert record.harvested == 0
    assert sum(colony.food for colony in world.state.colonies) == initial_food - record.upkeep_paid
    assert any(cohort.loc.kind == "edge" for cohort in world.state.cohorts)
    assert world.state.edge_flow
    assert all(cohort.mission.kind not in ("attack", "guard", "withdraw") for cohort in world.state.cohorts)


def test_shared_patch_harvest_is_pro_rata_and_cohort_partition_independent():
    config = default_config()
    world = build_world(config, 42)
    routes = RouteTable(world.world_map)
    shared = next(
        node for node in world.world_map.nodes if node.patch
        and sum(routes.distance(nest, node.node_id) <= config.map_params.contest_radius
                for nest in world.world_map.nests) > 1
    )

    outcomes = []
    for split in (False, True):
        current = deepcopy(world)
        patch = next(item for item in current.state.patches if item.node_id == shared.node_id)
        patch.stock = shared.patch[0].site_cap
        for colony_id in (0, 1):
            worker = next(cohort for cohort in current.state.cohorts
                          if cohort.colony_id == colony_id and cohort.caste == "worker")
            worker.loc = Location("node", shared.node_id, -1, -1, 0, 0)
            worker.mission = Mission("forage", shared.node_id, -1, "work", 0)
            if split and colony_id == 0:
                half = split_cohort(current.state, worker, worker.count // 2)
                half.loc = Location("node", shared.node_id, -1, -1, 0, 0)
                half.mission = Mission("forage", shared.node_id, -1, "work", 0)
        record = _run_step(current, routes)
        loaded_by_colony = {
            colony_id: sum(cohort.load for cohort in current.state.cohorts if cohort.colony_id == colony_id)
            for colony_id in (0, 1)
        }
        assert record.harvested == shared.patch[0].site_cap
        assert loaded_by_colony == {0: shared.patch[0].site_cap // 2, 1: shared.patch[0].site_cap // 2}
        outcomes.append(loaded_by_colony)

    assert outcomes[0] == outcomes[1]


def test_overflow_cargo_waits_at_nest_and_unloads_as_storage_frees():
    world = build_world(default_config(), 43)
    routes = RouteTable(world.world_map)
    colony = world.state.colonies[0]
    colony.food = world.config.economy.storage_cap - 2
    worker = next(cohort for cohort in world.state.cohorts
                  if cohort.colony_id == 0 and cohort.caste == "worker")
    worker.load = 16
    worker.loc = Location("node", world.world_map.nests[0], -1, -1, 0, 0)
    worker.mission = Mission("forage", world.world_map.nodes[0].node_id, -1, "return", 0)

    first = _run_step(world, routes)
    assert first.delivered == 2
    assert worker.load == 14
    second = _run_step(world, routes)
    assert second.delivered == 5
    assert worker.load == 9


def test_eggs_hatch_and_enter_the_age_ledger():
    config = default_config()
    config = replace(config, economy=replace(config.economy, worker_hatch_delay=1))
    world = build_world(config, 17)
    colony = world.state.colonies[0]
    colony.orders.egg_level = 3

    laid = _run_step(world, RouteTable(world.world_map))
    assert laid.laid_workers == 1
    assert len(colony.brood) == 1
    assert colony.food < config.economy.start_food

    colony.orders.egg_level = 0
    hatched = _run_step(world, RouteTable(world.world_map))
    assert hatched.hatched_workers == 1
    assert sum(cohort.count for cohort in world.state.cohorts
               if cohort.colony_id == 0 and cohort.caste == "worker") == config.economy.start_workers + 1
    assert sum(bucket.count for bucket in colony.ages.workers) == config.economy.start_workers + 1


def test_old_age_expiry_removes_adults_at_the_bucket_boundary():
    config = default_config()
    config = replace(config, economy=replace(config.economy, lifespan_worker=1, start_age_spread_permille=0))
    world = build_world(config, 21)
    routes = RouteTable(world.world_map)

    assert _run_step(world, routes).old_age_workers == 0
    expired = _run_step(world, routes)

    assert expired.old_age_workers == config.n_colonies * config.economy.start_workers
    assert not any(cohort.colony_id == 0 and cohort.caste == "worker" for cohort in world.state.cohorts)
    assert sum(bucket.count for bucket in world.state.colonies[0].ages.workers) == 0


def test_starvation_debt_kills_soldiers_first_and_updates_ages():
    config = default_config()
    config = replace(config, economy=replace(config.economy, start_food=0))
    world = build_world(config, 33)
    routes = RouteTable(world.world_map)

    first = _run_step(world, routes)
    second = _run_step(world, routes)

    assert first.upkeep_missing > 0
    assert first.starvation_soldiers == config.n_colonies
    assert first.starvation_workers == 0
    assert second.starvation_soldiers == config.n_colonies
    assert second.starvation_workers == 0
    colony = world.state.colonies[0]
    assert sum(bucket.count for bucket in colony.ages.soldiers) == sum(
        cohort.count for cohort in world.state.cohorts if cohort.colony_id == 0 and cohort.caste == "soldier"
    )


def test_midpoint_checkpoint_resume_matches_continuous_run():
    config = default_config()
    world = build_world(config, 83)
    routes = RouteTable(world.world_map)
    policy = fixture_controller("peaceful_fair")
    for _ in range(15):
        _run_step(world, routes, policy)
    resumed = deepcopy(world)

    for _ in range(15):
        _run_step(world, routes, policy)
        _run_step(resumed, routes, policy)

    assert world.state.tick == resumed.state.tick == 30
    assert state_hash(world.state) == state_hash(resumed.state)
    assert checkpoint_hash(world.state, world.run) == checkpoint_hash(resumed.state, resumed.run)


# ---------------------------------------------------------------------------------------- RFC V-7 soft lane capacity


def _depart_setup(num_cohorts_sizes):
    """Colony 0 workers, split into cohorts of the given sizes, all at the nest and sent over one wide edge."""
    config = default_config()
    config = replace(config, economy=replace(config.economy, start_workers=24))
    world = build_world(config, 42)
    nest = world.world_map.nests[0]
    edge = next(ed for ed in world.world_map.edges
                if nest in (ed.a, ed.b) and ed.lane_cap == world.config.economy.lane_cap)
    target = edge.b if edge.a == nest else edge.a
    workers = next(c for c in world.state.cohorts if c.colony_id == 0 and c.caste == "worker")
    assert workers.count >= sum(num_cohorts_sizes)
    rest = workers.count - sum(num_cohorts_sizes)
    if rest:
        split_cohort(world.state, workers, rest)  # the remainder stays idle at home (higher cohort id)
    parts = [workers]
    for size in num_cohorts_sizes[1:]:  # the original keeps the first size; later splits get ascending ids
        parts.append(split_cohort(world.state, workers, size))
    assert [c.count for c in parts] == list(num_cohorts_sizes)
    assert [c.cohort_id for c in parts] == sorted(c.cohort_id for c in parts)
    for cohort in parts:
        cohort.mission = Mission("forage", target, -1, "go", 0)
    return world, edge, parts


def test_soft_lane_capacity_delays_later_cohorts_instead_of_holding_them_at_the_node():
    world, edge, (first, second) = _depart_setup([8, 2])
    assert edge.lane_cap == 8
    routes = RouteTable(world.world_map)
    for other in world.state.cohorts:  # keep every other cohort of the world at home
        if other not in (first, second):
            other.mission = Mission("none", -1, -1, "idle", 0)

    _run_step(world, routes)
    assert first.loc.kind == second.loc.kind == "edge"       # nothing is queued at the node
    assert first.loc.arrive_tick - first.loc.depart_tick == edge.length
    assert second.loc.arrive_tick - second.loc.depart_tick == edge.length + 1   # floor(8 / 8)
    assert second.loc.depart_tick == first.loc.depart_tick
    assert first.count == 8 and second.count == 2             # no cohort was split to fit the lane
    assert world.state.edge_flow[0].in_flight_units == 10     # more than lane_cap is in flight at once


def test_soft_lane_capacity_multi_cohort_arrivals_are_staggered_by_floor_in_flight_over_cap():
    world, edge, (a, b, c) = _depart_setup([8, 8, 4])
    routes = RouteTable(world.world_map)
    for other in world.state.cohorts:
        if other not in (a, b, c):
            other.mission = Mission("none", -1, -1, "idle", 0)

    _run_step(world, routes)
    delays = [x.loc.arrive_tick - x.loc.depart_tick - edge.length for x in (a, b, c)]
    assert delays == [0, 1, 2]                                # in flight 0, 8, 16 at the moment of departure
    assert world.state.edge_flow[0].in_flight_units == 20

    # A cohort that reached the target turns round in the same tick, so its return trip departs at the arrival tick.
    target = edge.b if edge.a == world.world_map.nests[0] else edge.a
    start_tick = world.state.tick - 1
    arrivals = {}
    while len(arrivals) < 3:
        _run_step(world, routes)
        for name, cohort in (("a", a), ("b", b), ("c", c)):
            if cohort.loc.kind == "edge" and cohort.loc.edge_from == target:
                arrivals.setdefault(name, cohort.loc.depart_tick)
        assert world.state.tick < 100
    assert arrivals["a"] == start_tick + edge.length
    assert arrivals["b"] == arrivals["a"] + 1 and arrivals["c"] == arrivals["a"] + 2


def test_a_cohort_larger_than_lane_cap_departs_whole_and_the_edge_trip_cannot_beat_the_edge_length():
    world, edge, (big,) = _depart_setup([10])
    routes = RouteTable(world.world_map)
    for other in world.state.cohorts:
        if other is not big:
            other.mission = Mission("none", -1, -1, "idle", 0)

    _run_step(world, routes)
    assert big.count == 10 > edge.lane_cap
    assert big.loc.arrive_tick - big.loc.depart_tick == edge.length  # it only sees what is already in flight

    big.loc = replace(big.loc, arrive_tick=big.loc.depart_tick + edge.length - 1)
    with pytest.raises(InvariantError, match="shorter than the edge length"):
        validate_state(world.state, world.config, world.world_map)


def test_soft_lane_capacity_checkpoint_resume_is_exact_with_cohorts_in_flight():
    world, edge, parts = _depart_setup([8, 2])
    routes = RouteTable(world.world_map)
    _run_step(world, routes)
    resumed = deepcopy(world)
    for _ in range(edge.length + 3):
        _run_step(world, routes)
        _run_step(resumed, routes)
    assert checkpoint_hash(world.state, world.run) == checkpoint_hash(resumed.state, resumed.run)


# ------------------------------------------------------------------------------- contracts section 9.5 terminal rules


def _emptied_colony(brood=(), food=0, queen_hp=None):
    world = build_world(default_config(), 5)
    state = world.state
    colony = state.colonies[0]
    state.cohorts = [c for c in state.cohorts if c.colony_id != 0]
    colony.ages.workers, colony.ages.soldiers = [], []
    colony.brood = list(brood)
    colony.food = food
    colony.orders.egg_level = 0
    if queen_hp is not None:
        colony.queen_hp = queen_hp
        colony.status = "alive" if queen_hp > 0 else "orphan"
    state.next_cohort_id = max(state.next_cohort_id, 1000)
    return world, colony


def test_empty_colony_with_a_living_queen_and_enough_food_for_a_worker_egg_is_not_terminal():
    world, colony = _emptied_colony(food=default_config().economy.worker_egg_cost)
    _run_step(world, RouteTable(world.world_map))
    assert colony.status == "alive" and colony.queen_hp > 0
    assert not world.state.outcomes[0].finished


def test_empty_colony_without_food_for_a_worker_egg_collapses_and_its_food_becomes_a_pile():
    cost = default_config().economy.worker_egg_cost
    world, colony = _emptied_colony(food=cost - 1)
    nest = world.world_map.nests[0]
    _run_step(world, RouteTable(world.world_map))
    assert colony.status == "extinct" and colony.food == 0
    outcome = world.state.outcomes[0]
    assert (outcome.finished, outcome.reason, outcome.terminal_tick) == (True, "collapse", 0)
    assert next(p.stock for p in world.state.piles if p.node_id == nest) == cost - 1
    assert not world.state.world_outcome.finished           # the other seven colonies go on


def test_soldier_only_brood_does_not_prevent_collapse_when_no_worker_egg_is_affordable():
    cost = default_config().economy.worker_egg_cost
    world, colony = _emptied_colony(brood=[BroodRecord(900, "soldier", 0, 30)], food=cost - 1)
    _run_step(world, RouteTable(world.world_map))
    assert colony.status == "extinct" and colony.brood == []
    assert world.state.outcomes[0].reason == "collapse"


def test_worker_brood_keeps_a_poor_empty_colony_alive_and_it_hatches():
    world, colony = _emptied_colony(brood=[BroodRecord(900, "worker", -1, 0)], food=0)
    _run_step(world, RouteTable(world.world_map))
    assert colony.status == "alive" and not world.state.outcomes[0].finished
    assert any(c.colony_id == 0 and c.caste == "worker" for c in world.state.cohorts)


def test_orphan_extinction_stays_distinct_from_economic_collapse():
    world, colony = _emptied_colony(food=0, queen_hp=0)
    _run_step(world, RouteTable(world.world_map))
    assert colony.status == "extinct"
    assert world.state.outcomes[0].reason == "no_adults_or_brood"
