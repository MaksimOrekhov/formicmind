from copy import deepcopy
from dataclasses import replace

from formicmind.eco.config import default_config
from formicmind.eco.engine import RouteTable, split_cohort, step_world_tick
from formicmind.eco.fixtures import fixture_controller
from formicmind.eco.initial import build_world
from formicmind.eco.model import Location, Mission
from formicmind.eco.serialization import checkpoint_hash, state_hash


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
