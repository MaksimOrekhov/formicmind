"""E1-c: fog of war, scouting, contacts, orders, local combat, queens and orphans, events (FOR-46 acceptance)."""

from copy import deepcopy
from dataclasses import replace

import pytest

from formicmind.eco.cohorts import remove_cohort_members, split_cohort
from formicmind.eco.combat import queen_terminal, resolve_combat
from formicmind.eco.config import default_config
from formicmind.eco.engine import RouteTable, TickRecord, step_world_tick
from formicmind.eco.events import EventLog
from formicmind.eco.fixtures import fixture_controller
from formicmind.eco.initial import build_world
from formicmind.eco.memory import bucket, known_next_hop, sense
from formicmind.eco.model import BroodRecord, CohortE1, Location, Mission
from formicmind.eco.orders import (
    IllegalOrderError, attack_target_is_legal, order_attack, order_guard, order_scout, order_withdraw,
)
from formicmind.eco.serialization import checkpoint_hash, state_from_bytes, state_hash, state_to_bytes
from formicmind.eco.simulation import run_simulation
from formicmind.eco.validate import InvariantError, validate_state


def _config(**economy):
    """Default config without upkeep (these tests are about contact and war, and no controller feeds the colonies) unless overridden."""
    config = default_config()
    base = {"start_food": 500, "worker_upkeep_tenths": 0, "soldier_upkeep_tenths": 0}
    return replace(config, economy=replace(config.economy, **{**base, **economy}))


def _world(seed=42, **economy):
    world = build_world(_config(**economy), seed)
    return world, RouteTable(world.world_map)


def _step(world, routes, controller=None, validate=False):
    return step_world_tick(world.state, world.run, world.config, world.world_map, controller, routes=routes,
                           validate_input=validate)


def _path(world_map, routes, a, b):
    nodes = [a]
    while nodes[-1] != b:
        nodes.append(routes.next_hop(nodes[-1], b))
    return nodes


def _learn_route(world, routes, colony_id, target_node):
    """Make a colony *physically* know the way to ``target_node``: a worker stands on every node of the route while the real sensor runs."""
    state, world_map = world.state, world.world_map
    worker = next(c for c in state.cohorts if c.colony_id == colony_id and c.caste == "worker")
    home = world_map.nests[colony_id]
    for node in _path(world_map, routes, home, target_node):
        worker.loc = Location("node", node, -1, -1, 0, 0)
        sense(state, world_map, [])
    worker.loc = Location("node", home, -1, -1, 0, 0)


def _slot(world, colony_id, node_id):
    return next(i for i, k in enumerate(world.state.colonies[colony_id].memory.nodes) if k.node_id == node_id)


def _foreign_slot(world, colony_id, other):
    return next(i for i, r in enumerate(world.state.colonies[colony_id].memory.foreign) if r.colony_id == other)


# ------------------------------------------------------------------------------------------------ fog of war and scouting


def test_a_colony_starts_without_any_foreign_knowledge():
    world, _ = _world()
    for colony in world.state.colonies:
        memory = colony.memory
        assert memory.foreign == [] and memory.sightings == [] and memory.events == []
        known_nests = [k for k in memory.nodes if k.kind == "nest"]
        assert [k.node_id for k in known_nests] == [world.world_map.nests[colony.colony_id]]
        assert all(k.nest_owner_slot == -1 for k in memory.nodes)


def test_attacking_an_undiscovered_nest_is_rejected_and_changes_nothing():
    world, routes = _world(start_soldiers=4)
    before = state_hash(world.state)
    memory = world.state.colonies[0].memory
    for node_slot, colony_slot in ((0, 0), (len(memory.nodes) + 3, 0), (1, 5), (-1, -1)):
        with pytest.raises(IllegalOrderError):
            order_attack(world.state, world.config, world.world_map, 0, node_slot, colony_slot)
    # the world id of another colony's nest is not an address
    other_nest = world.world_map.nests[1]
    assert other_nest not in {k.node_id for k in memory.nodes}
    assert state_hash(world.state) == before


def test_visiting_a_foreign_nest_makes_the_attack_order_possible():
    world, routes = _world(start_soldiers=4)
    target = world.world_map.nests[1]
    assert not attack_target_is_legal(world.state, 0, 1, 0)
    _learn_route(world, routes, 0, target)
    memory = world.state.colonies[0].memory
    colony_slot = _foreign_slot(world, 0, 1)
    node_slot = _slot(world, 0, target)
    assert memory.foreign[colony_slot].nest_node_slot == node_slot
    assert memory.nodes[node_slot].nest_owner_slot == colony_slot and memory.nodes[node_slot].queen_seen == 1
    assert attack_target_is_legal(world.state, 0, node_slot, colony_slot)
    assert order_attack(world.state, world.config, world.world_map, 0, node_slot, colony_slot)
    validate_state(world.state, world.config, world.world_map)


def test_a_scout_walks_the_known_edges_and_the_map_is_revealed_only_on_arrival():
    world, routes = _world(seed=7, start_workers=12)
    state, world_map = world.state, world.world_map
    memory = state.colonies[0].memory
    home = world_map.nests[0]
    frontier = sorted((routes.distance(home, k.node_id), s) for s, k in enumerate(memory.nodes) if not k.visited)
    _, slot = frontier[-1]
    target = memory.nodes[slot].node_id
    known_before = len(memory.nodes)
    ids = order_scout(state, world.config, world_map, 0, slot, 1)
    scout = next(c for c in state.cohorts if c.cohort_id == ids[0])
    assert scout.mission.kind == "scout" and scout.loc.kind == "node"
    eta = routes.distance(home, target)
    arrived = None
    for _ in range(eta + 12):
        _step(world, routes)
        assert not memory.nodes[slot].visited or arrived is not None or scout.loc.kind == "node"
        if memory.nodes[slot].visited and arrived is None:
            arrived = state.tick - 1
            assert scout.loc.kind == "node" and scout.loc.node_id == target  # standing there, not teleported
    assert arrived is not None and arrived >= eta - 1                        # it took the physical travel time
    assert len(memory.nodes) >= known_before
    visited = [k for k in memory.nodes if k.visited]
    assert any(k.node_id == target for k in visited)
    # every node the colony has ever seen is the end of an edge incident to a node it visited
    adjacent = {x for e in world_map.edges for x in (e.a, e.b) if e.a in {v.node_id for v in visited} or e.b in {v.node_id for v in visited}}
    assert {k.node_id for k in memory.nodes} <= adjacent
    assert all(k.kind == "unknown" and k.last_stock_tick == -1 for k in memory.nodes if not k.visited)
    # after the look-around the scout comes home and is idle again
    for _ in range(80):
        _step(world, routes)
    assert scout.mission.kind == "home" and scout.loc.kind == "node" and scout.loc.node_id == home
    validate_state(state, world.config, world_map)


def test_scouting_reveals_neither_foreign_stores_nor_unobserved_cohorts():
    world, routes = _world()
    target = world.world_map.nests[1]
    foreign_food = world.state.colonies[1].food = 333
    _learn_route(world, routes, 0, target)
    memory = world.state.colonies[0].memory
    record = memory.foreign[_foreign_slot(world, 0, 1)]
    # only a strength bucket of what stood in the node was perceived; there is no field for stores or queen hit points
    assert record.last_seen_workers_bucket == bucket(sum(c.count for c in world.state.cohorts
                                                          if c.colony_id == 1 and c.caste == "worker"))
    fields = {f for f in record.__slots__}
    assert not {"food", "queen_hp", "x", "y"} & fields
    assert 333 not in {getattr(record, f) for f in fields}
    # colonies the scout did not meet are not in the memory at all
    assert [r.colony_id for r in memory.foreign] == [1]
    other_nests = {n for i, n in enumerate(world.world_map.nests) if i not in (0, 1)}
    assert not other_nests & {k.node_id for k in memory.nodes if k.visited}
    assert foreign_food == 333


def test_known_routing_ignores_edges_the_colony_has_not_discovered():
    world, routes = _world()
    memory = world.state.colonies[0].memory
    home = world.world_map.nests[0]
    far = max((k.node_id for k in memory.nodes if not k.visited), key=lambda n: routes.distance(home, n))
    hop = known_next_hop(memory, home, far)
    known_pairs = {(memory.nodes[e.slot_a].node_id, memory.nodes[e.slot_b].node_id) for e in memory.edges}
    known_pairs |= {(b, a) for a, b in known_pairs}
    assert (home, hop) in known_pairs
    assert known_next_hop(memory, home, world.world_map.nests[1]) == -1  # a node it has never seen has no known way


# ------------------------------------------------------------------------------------------------ order legality


def test_illegal_orders_fail_closed_without_touching_the_state():
    world, routes = _world(start_soldiers=4)
    state, config, world_map = world.state, world.config, world.world_map
    before = (state_hash(state), checkpoint_hash(state, world.run))
    memory = state.colonies[0].memory
    visited_far = next(s for s, k in enumerate(memory.nodes) if k.visited and s != 0)
    unvisited = next(s for s, k in enumerate(memory.nodes) if not k.visited)
    bad = [
        lambda: order_scout(state, config, world_map, 0, 0),                      # own nest
        lambda: order_scout(state, config, world_map, 0, visited_far),            # seen too recently
        lambda: order_scout(state, config, world_map, 0, 10_000),                 # not in memory
        lambda: order_scout(state, config, world_map, 0, unvisited, 99),          # too many workers
        lambda: order_scout(state, config, world_map, 0, unvisited, 0),
        lambda: order_scout(state, config, world_map, 99, unvisited),             # no such colony
        lambda: order_guard(state, config, world_map, 0, unvisited),              # guard only a visited node
        lambda: order_guard(state, config, world_map, 0, visited_far, 9),         # bad size
        lambda: order_guard(state, config, world_map, 0, visited_far, 1, 5),      # bad squad index
        lambda: order_withdraw(state, config, world_map, 0),                      # nobody to recall
    ]
    for call in bad:
        with pytest.raises(IllegalOrderError):
            call()
    state.colonies[0].status = "orphan"
    state.colonies[0].queen_hp = 0
    with pytest.raises(IllegalOrderError):
        order_scout(state, config, world_map, 0, unvisited)
    state.colonies[0].status, state.colonies[0].queen_hp = "alive", config.economy.queen_hp
    assert (state_hash(state), checkpoint_hash(state, world.run)) == before


def test_not_enough_idle_units_is_an_illegal_order():
    world, routes = _world(start_workers=2, start_soldiers=0)
    memory = world.state.colonies[0].memory
    unvisited = next(s for s, k in enumerate(memory.nodes) if not k.visited)
    order_scout(world.state, world.config, world.world_map, 0, unvisited, 2)
    with pytest.raises(IllegalOrderError):
        order_scout(world.state, world.config, world.world_map, 0, unvisited, 1)  # all workers are already away
    with pytest.raises(IllegalOrderError):
        order_guard(world.state, world.config, world.world_map, 0, 0)            # there are no soldiers at all


# ------------------------------------------------------------------------------------------------ peaceful meetings


def _put(world, colony_id, caste, node, mission=None):
    cohort = next(c for c in world.state.cohorts if c.colony_id == colony_id and c.caste == caste
                  and c.loc.kind == "node" and c.mission.kind == "home")
    cohort.loc = Location("node", node, -1, -1, 0, 0)
    if mission:
        cohort.mission = mission
    return cohort


def test_two_colonies_sharing_a_patch_keep_harvesting_in_peace_until_an_attack_order():
    world, routes = _world(start_soldiers=4)
    state, world_map = world.state, world.world_map
    shared = next(n for n in world_map.nodes if n.patch
                  and routes.distance(world_map.nests[0], n.node_id) <= 12 and routes.distance(world_map.nests[1], n.node_id) <= 14)
    for colony in (0, 1):
        _put(world, colony, "worker", shared.node_id, Mission("forage", shared.node_id, -1, "work", 0))
        _put(world, colony, "soldier", shared.node_id, Mission("guard", shared.node_id, -1, "work", 0))
    events = EventLog()
    harvested = 0
    for _ in range(10):
        record = _step(world, routes)
        events.extend(record.events)
        harvested += record.harvested
    assert harvested > 0
    counts = events.counts()
    assert counts["combat"] == 0 and counts["death"] == 0 and counts["queen_damage"] == 0
    assert counts["contact"] == 2                                                           # both noticed the other, nobody attacked
    assert state.colonies[0].memory.foreign[0].they_attacked_q == 0
    # one explicit ATTACK order is the only thing that starts a fight
    _learn_route(world, routes, 0, shared.node_id)
    slot = _slot(world, 0, shared.node_id)
    cslot = _foreign_slot(world, 0, 1)
    assert attack_target_is_legal(state, 0, slot, cslot)                                    # a fresh sighting is enough
    attackers = next(c for c in state.cohorts if c.colony_id == 0 and c.caste == "soldier")
    attackers.mission = Mission("attack", shared.node_id, 1, "work", 0)
    attackers.loc = Location("node", shared.node_id, -1, -1, 0, 0)
    record = _step(world, routes)
    assert any(e.kind == "combat" for e in record.events)


# ------------------------------------------------------------------------------------------------ combat vs an independent reference


def _arena(attackers, soldiers, workers, seed=11):
    """Colony 0 attacks colony 1 at colony 1's nest; every other colony is elsewhere."""
    world, routes = _world(seed, start_workers=max(workers, 1), start_soldiers=max(attackers, soldiers, 1))
    state, world_map = world.state, world.world_map
    nest = world_map.nests[1]
    remove_cohort_members(state, world.config, world_map, 1, "soldier", max(attackers, soldiers, 1) - soldiers, book_age_deaths=True)
    remove_cohort_members(state, world.config, world_map, 0, "soldier", max(attackers, soldiers, 1) - attackers, book_age_deaths=True)
    remove_cohort_members(state, world.config, world_map, 1, "worker", max(workers, 1) - workers, book_age_deaths=True)
    for cohort in state.cohorts:
        if cohort.colony_id == 0 and cohort.caste == "soldier":
            cohort.loc = Location("node", nest, -1, -1, 0, 0)
            cohort.mission = Mission("attack", nest, 1, "work", 0)
    validate_state(state, world.config, world_map)
    return world, routes


def _reference(attackers, soldiers, workers, config):
    """Unit-by-unit L0 combat (docs/combat_terminal.md) for one attacker colony and one defender at the defender's nest."""
    e = config.economy
    a = [e.soldier_hp] * attackers
    ds = [e.soldier_hp] * soldiers
    dw = [e.worker_hp] * workers
    queen = e.queen_hp
    history = []
    for _ in range(80):
        if not a or (not ds and not dw and queen == 0):
            break
        dmg_a = 2 * len(a)
        dmg_d = (2 * len(ds)) * e.home_defense_num // e.home_defense_den
        if ds:
            tier = ds
        elif dw:
            tier = dw
        else:
            tier = None

        def pour(units, damage):
            for index in range(len(units)):
                take = min(damage, units[index])
                units[index] -= take
                damage -= take
            return [u for u in units if u > 0]

        if tier is ds:
            ds = pour(ds, dmg_a)
        elif tier is dw:
            dw = pour(dw, dmg_a)
        else:
            queen = max(0, queen - dmg_a)
        a = pour(a, dmg_d)
        history.append((len(a), len(ds), len(dw), queen))
    return history


@pytest.mark.parametrize("attackers,soldiers,workers", [(6, 2, 3), (8, 4, 6), (4, 4, 2), (12, 3, 1), (5, 0, 4), (3, 0, 0), (9, 5, 0)])
def test_cohort_combat_matches_the_unit_level_l0_reference_tick_by_tick(attackers, soldiers, workers):
    world, _ = _arena(attackers, soldiers, workers)
    expected = _reference(attackers, soldiers, workers, world.config)
    state = world.state
    for tick, want in enumerate(expected):
        state.tick = tick
        record = TickRecord(tick)
        queen_terminal(state, world.config, world.world_map,
                       resolve_combat(state, world.config, world.world_map, record.events, record), record.events)
        got = (sum(c.count for c in state.cohorts if c.colony_id == 0 and c.caste == "soldier"),
               sum(c.count for c in state.cohorts if c.colony_id == 1 and c.caste == "soldier"),
               sum(c.count for c in state.cohorts if c.colony_id == 1 and c.caste == "worker"),
               state.colonies[1].queen_hp)
        assert got == want, (tick, got, want)
        validate_state(state, world.config, world.world_map)


def test_the_queen_is_only_hit_when_no_soldier_or_worker_of_hers_is_home_and_dies_into_an_orphan():
    world, routes = _arena(8, 0, 0)
    state = world.state
    first = TickRecord(0)
    resolve_combat(state, world.config, world.world_map, first.events, first)
    assert state.colonies[1].queen_hp == world.config.economy.queen_hp - 16   # nobody to shield her
    # with a worker at home she is shielded
    world, routes = _arena(8, 0, 1)
    record = TickRecord(0)
    resolve_combat(world.state, world.config, world.world_map, record.events, record)
    assert world.state.colonies[1].queen_hp == world.config.economy.queen_hp
    # kill her
    world, routes = _arena(8, 0, 0)
    state = world.state
    state.colonies[1].queen_hp = 16
    record = TickRecord(0)
    queen_terminal(state, world.config, world.world_map,
                   resolve_combat(state, world.config, world.world_map, record.events, record), record.events)
    colony = state.colonies[1]
    assert colony.queen_hp == 0 and colony.status == "orphan"
    assert (state.outcomes[1].finished, state.outcomes[1].reason, state.outcomes[1].terminal_tick) == (True, "queen_death", 0)
    kinds = [e.kind for e in record.events]
    assert "queen_dead" in kinds
    dead = next(e for e in record.events if e.kind == "queen_dead")
    assert (dead.colony, dead.other) == (1, 0)
    assert colony.memory.nodes[0].queen_seen == 2


def test_simultaneous_blows_a_dying_soldier_still_strikes_and_a_third_colony_is_untouched():
    world, _ = _arena(3, 3, 0)
    state = world.state
    nest = world.world_map.nests[1]
    bystander = next(c for c in state.cohorts if c.colony_id == 2 and c.caste == "soldier")
    bystander.loc = Location("node", nest, -1, -1, 0, 0)
    bystander.mission = Mission("guard", nest, -1, "work", 0)
    hp_before = bystander.hp_total
    # 3 v 3 at the defender's nest: attackers deal 6, defenders deal 6*5//4 = 7 (floor of the sum), both from the pre-combat state
    record = TickRecord(0)
    resolve_combat(state, world.config, world.world_map, record.events, record)
    attackers = [c for c in state.cohorts if c.colony_id == 0 and c.caste == "soldier"]
    defenders = [c for c in state.cohorts if c.colony_id == 1 and c.caste == "soldier"]
    assert sum(c.hp_total for c in attackers) == 3 * 30 - 7
    assert sum(c.hp_total for c in defenders) == 3 * 30 - 6
    assert bystander.hp_total == hp_before
    assert {(e.colony, e.other) for e in record.events if e.kind == "combat"} == {(0, 1), (1, 0)}


def test_equal_fights_are_symmetric_no_priority_by_colony_id():
    results = []
    for first, second in ((0, 1), (1, 0)):
        world, routes = _world(3, start_workers=4, start_soldiers=5)
        state = world.state
        node = world.world_map.nests[5]   # a node that is nobody's battlefield advantage: a third colony's empty nest
        remove_cohort_members(state, world.config, world.world_map, 5, "soldier", 5, book_age_deaths=True)
        remove_cohort_members(state, world.config, world.world_map, 5, "worker", 4, book_age_deaths=True)
        for me, other in ((first, second), (second, first)):
            for cohort in state.cohorts:
                if cohort.colony_id == me and cohort.caste == "soldier":
                    cohort.loc = Location("node", node, -1, -1, 0, 0)
                    cohort.mission = Mission("attack", node, other, "work", 0)
        for _ in range(4):
            record = TickRecord(state.tick)
            resolve_combat(state, world.config, world.world_map, record.events, record)
        results.append(tuple(sum(c.hp_total for c in state.cohorts if c.colony_id == cid and c.caste == "soldier") for cid in (0, 1)))
    assert results[0] == results[1] and results[0][0] == results[0][1]                     # same blows, same losses


def test_splitting_the_attackers_into_cohorts_does_not_change_the_fight():
    outcomes = []
    for parts in (1, 3):
        world, _ = _arena(9, 4, 5)
        state = world.state
        if parts == 3:
            cohort = next(c for c in state.cohorts if c.colony_id == 0 and c.caste == "soldier")
            for size in (2, 3):
                split_cohort(state, cohort, size)
        for tick in range(6):
            state.tick = tick
            record = TickRecord(tick)
            resolve_combat(state, world.config, world.world_map, record.events, record)
        outcomes.append((sum(c.hp_total for c in state.cohorts if c.colony_id == 0),
                         sum(c.hp_total for c in state.cohorts if c.colony_id == 1), state.colonies[1].queen_hp))
    assert outcomes[0] == outcomes[1]


def test_dead_workers_drop_their_cargo_and_the_ledger_and_food_stay_exact():
    world, _ = _arena(10, 0, 4)
    state = world.state
    for cohort in state.cohorts:
        if cohort.colony_id == 1 and cohort.caste == "worker":
            cohort.load = cohort.count * 5
    food_before = (sum(p.stock for p in state.piles) + sum(c.load for c in state.cohorts))
    for tick in range(6):
        state.tick = tick
        record = TickRecord(tick)
        resolve_combat(state, world.config, world.world_map, record.events, record)
    assert sum(c.count for c in state.cohorts if c.colony_id == 1 and c.caste == "worker") == 0
    assert sum(p.stock for p in state.piles) + sum(c.load for c in state.cohorts) == food_before
    assert next(p.stock for p in state.piles if p.node_id == world.world_map.nests[1]) == 20
    validate_state(state, world.config, world.world_map)  # the age ledger still equals the cohorts


def test_a_cohort_that_is_not_attacking_never_starts_a_fight_and_edges_have_no_combat():
    world, routes = _arena(6, 6, 3)
    state = world.state
    for cohort in state.cohorts:
        if cohort.colony_id == 0 and cohort.caste == "soldier":
            cohort.mission = Mission("guard", cohort.loc.node_id, -1, "work", 0)
    record = TickRecord(0)
    resolve_combat(state, world.config, world.world_map, record.events, record)
    assert not record.events
    # attackers on an edge are not in the node
    for cohort in state.cohorts:
        if cohort.colony_id == 0 and cohort.caste == "soldier":
            cohort.mission = Mission("attack", world.world_map.nests[1], 1, "work", 0)
            neighbor = world.world_map.edges[0]
            cohort.loc = Location("edge", -1, neighbor.a, neighbor.b, 0, 5)
    resolve_combat(state, world.config, world.world_map, record.events, record)
    assert not record.events


# ------------------------------------------------------------------------------------------------ movement, withdrawal, standing orders


def _raid_world(attackers=6, defenders=2, seed=5):
    """Colony 0 knows the way to colony 1's nest; its idle soldiers are ready. Returns (world, routes, target node, slots)."""
    world, routes = _world(seed, start_workers=8, start_soldiers=max(attackers, defenders))
    remove_cohort_members(world.state, world.config, world.world_map, 1, "soldier", max(attackers, defenders) - defenders,
                          book_age_deaths=True)
    remove_cohort_members(world.state, world.config, world.world_map, 0, "soldier", max(attackers, defenders) - attackers,
                          book_age_deaths=True)
    target = world.world_map.nests[1]
    _learn_route(world, routes, 0, target)
    return world, routes, target, (_slot(world, 0, target), _foreign_slot(world, 0, 1))


def test_an_attack_squad_walks_to_the_nest_fights_there_and_only_there():
    world, routes, target, (node_slot, colony_slot) = _raid_world()
    state = world.state
    events = EventLog()
    ids = order_attack(state, world.config, world.world_map, 0, node_slot, colony_slot)
    squad = [c for c in state.cohorts if c.cohort_id in ids]
    assert squad and all(c.mission.kind == "attack" and c.mission.target_colony == 1 for c in squad)
    eta = routes.distance(world.world_map.nests[0], target)
    first_combat = None
    for _ in range(eta + 60):
        record = _step(world, routes)
        events.extend(record.events)
        for item in record.events:
            if item.kind == "combat" and first_combat is None:
                first_combat = item.tick
        for event in record.events:
            if event.kind == "combat":
                assert event.node == target                                              # never anywhere else
    assert first_combat is not None and first_combat >= eta - 1                        # not before the walk was possible
    counts = events.counts()
    assert counts["combat"] > 0 and counts["death"] > 0
    validate_state(state, world.config, world.world_map)


def test_withdrawal_goes_home_over_known_edges_and_the_enemy_does_not_pursue():
    world, routes, target, (node_slot, colony_slot) = _raid_world(attackers=10, defenders=2)
    state = world.state
    order_attack(state, world.config, world.world_map, 0, node_slot, colony_slot)
    for _ in range(routes.distance(world.world_map.nests[0], target) + 3):
        _step(world, routes)
    assert any(c.mission.kind == "attack" for c in state.cohorts if c.colony_id == 0)
    order_withdraw(state, world.config, world.world_map, 0)
    assert all(c.mission.kind in ("withdraw", "home") for c in state.cohorts if c.colony_id == 0 and c.caste == "soldier")
    after = EventLog()
    for _ in range(80):
        after.extend(_step(world, routes).events)
    counts = after.counts()
    assert counts["combat"] == 0 and counts["death"] == 0                                # no pursuit, no more blows once they leave
    home = world.world_map.nests[0]
    soldiers = [c for c in state.cohorts if c.colony_id == 0 and c.caste == "soldier"]
    assert soldiers and all(c.mission.kind == "home" and c.loc.kind == "node" and c.loc.node_id == home for c in soldiers)
    assert all(c.mission.kind != "attack" for c in state.cohorts)


def test_orphans_pull_their_soldiers_back_and_keep_living_by_standing_orders():
    world, routes, target, (node_slot, colony_slot) = _raid_world(attackers=12, defenders=0)
    state = world.state
    # kill the queen of colony 1 through the real mechanism
    order_attack(state, world.config, world.world_map, 0, node_slot, colony_slot)
    state.colonies[1].queen_hp = 24
    controller = fixture_controller("peaceful_fair")
    log = EventLog()
    for _ in range(routes.distance(world.world_map.nests[0], target) + 40):
        record = _step(world, routes, controller)
        log.extend(record.events)
    colony = state.colonies[1]
    assert colony.status == "orphan" and colony.queen_hp == 0
    assert state.outcomes[1].reason == "queen_death" and not state.world_outcome.finished
    laid_after = sum(1 for _ in colony.brood)
    lay_tick = colony.next_lay_tick
    for _ in range(100):
        _step(world, routes, controller)
    assert colony.next_lay_tick == lay_tick and len(colony.brood) <= laid_after           # no new eggs, ever
    assert sum(c.count for c in state.cohorts if c.colony_id == 1) > 0                  # the orphans are still alive
    assert [e.kind for e in log.events].count("queen_dead") == 1
    validate_state(state, world.config, world.world_map)


def test_orphan_brood_hatches_but_the_colony_dies_out_and_its_food_becomes_a_pile():
    world, routes = _world(2, worker_upkeep_tenths=4, soldier_upkeep_tenths=8)
    state = world.state
    colony = state.colonies[3]
    colony.queen_hp, colony.status = 0, "orphan"
    state.outcomes[3].finished, state.outcomes[3].reason, state.outcomes[3].terminal_tick = True, "queen_death", 0
    colony.brood.append(BroodRecord(state.next_cohort_id, "worker", -1, 0))
    state.next_cohort_id += 1
    colony.food = 7
    remove_cohort_members(state, world.config, world.world_map, 3, "worker", 10, book_age_deaths=True)
    remove_cohort_members(state, world.config, world.world_map, 3, "soldier", 2, book_age_deaths=True)
    record = _step(world, routes)
    assert record.hatched_workers == 1                                                   # paid brood still hatches
    assert colony.status == "orphan"
    for _ in range(400):
        if colony.status == "extinct":
            break
        _step(world, routes)
    assert colony.status == "extinct" and colony.food == 0
    assert state.outcomes[3].reason == "queen_death"                                     # written once
    assert not state.world_outcome.finished                                              # seven others are alive


def test_the_world_goes_on_with_one_colony_left_and_ends_only_when_no_colony_is_alive():
    world, routes = _world(2)
    state = world.state
    for colony in state.colonies[1:]:
        colony.queen_hp, colony.status = 0, "orphan"
        state.outcomes[colony.colony_id].finished = True
        state.outcomes[colony.colony_id].reason = "queen_death"
        state.outcomes[colony.colony_id].terminal_tick = 0
    _step(world, routes)
    assert not state.world_outcome.finished
    state.colonies[0].queen_hp, state.colonies[0].status = 0, "orphan"
    state.outcomes[0].finished, state.outcomes[0].reason, state.outcomes[0].terminal_tick = True, "queen_death", 1
    _step(world, routes)
    assert state.world_outcome.finished and state.world_outcome.reason == "no_alive_colonies"


# ------------------------------------------------------------------------------------------------ extra decisions in a fight


def test_colonies_in_a_fight_get_an_extra_decision_every_alert_interval():
    world, routes, target, (node_slot, colony_slot) = _raid_world(attackers=12, defenders=4)
    state = world.state
    calls = []

    def controller(st, config, world_map, route_table):
        calls.append((st.tick, tuple(sorted(route_table.alert))))

    order_attack(state, world.config, world.world_map, 0, node_slot, colony_slot)
    for _ in range(routes.distance(world.world_map.nests[0], target) + 40):
        _step(world, routes, controller)
    planned = [t for t, alert in calls if not alert]
    alerts = [(t, alert) for t, alert in calls if alert]
    assert all(t % world.config.decision_k == 0 for t in planned)
    assert alerts and all(t % world.config.alert_interval == 0 and t % world.config.decision_k != 0 for t, _ in alerts)
    assert all(set(alert) <= {0, 1} for _, alert in alerts)                               # only the fighting colonies


# ------------------------------------------------------------------------------------------------ roles, determinism, checkpoints, serialization


def test_peaceful_fixtures_never_fight_in_a_full_eight_colony_run():
    for fixture in ("peaceful_fair", "greedy_scarcity"):
        result = run_simulation(default_config(8, "moderate"), 50003, fixture, ticks=1500, collect_timeline=False,
                                include_map=False)
        inter = result.report["interaction"]
        assert inter["events_by_kind"]["combat"] == 0 and inter["events_by_kind"]["dispatch"] == 0
        assert inter["queen_deaths"] == [] and inter["combat_deaths"] == {"workers": 0, "soldiers": 0}
        assert inter["colony_pairs_in_contact"] > 0                                       # they did meet, in peace
        assert result.report["food_ledger"]["difference"] == 0


def test_a_raider_scouts_finds_a_nest_and_attacks_by_its_own_decision():
    result = run_simulation(default_config(8, "moderate"), 50005, "peaceful_fair", ticks=2200, collect_timeline=False,
                            include_map=False, roles={5: "raider"}, include_events=True)
    inter = result.report["interaction"]
    timeline = result.report["event_timeline"]
    found = next(i for i, line in enumerate(timeline) if "nest_found: colony 5" in line)
    dispatch = next(i for i, line in enumerate(timeline) if "on attack" in line)
    assert found < dispatch                                                               # it attacks what it found, after finding it
    assert inter["attacks_dispatched"] and all(a["attacker"] == 5 for a in inter["attacks_dispatched"])
    assert inter["combat_ticks"] > 0 and inter["combat_deaths"]["soldiers"] + inter["combat_deaths"]["workers"] > 0
    assert any(q["colony"] == 3 and q["killer"] == 5 for q in inter["queen_deaths"])
    assert result.report["food_ledger"]["difference"] == 0
    # nobody else attacked anyone
    assert {e["colony"] for e in inter["queen_deaths"]} == {3}


def test_the_scenario_is_reproducible_and_the_event_hash_is_stable():
    args = (default_config(8, "moderate"), 50005, "peaceful_fair")
    kwargs = dict(ticks=1500, collect_timeline=False, include_map=False, roles={5: "raider"}, include_events=True)
    a, b = run_simulation(*args, **kwargs), run_simulation(*args, **kwargs)
    assert a.report["interaction"]["events_hash"] == b.report["interaction"]["events_hash"]
    assert a.report["hashes"] == b.report["hashes"] and a.report["event_timeline"] == b.report["event_timeline"]
    other = run_simulation(*args, **{**kwargs, "roles": {5: "scout"}})
    assert other.report["interaction"]["events_hash"] != a.report["interaction"]["events_hash"]


def test_checkpoint_resume_in_a_world_with_contacts_and_combat_matches_the_continuous_run():
    config = default_config(8, "moderate")
    controller = fixture_controller("peaceful_fair", {5: "raider"})
    world, routes = build_world(config, 50005), None
    routes = RouteTable(world.world_map)
    log = EventLog()
    for _ in range(1670):                                                                 # the first fight is under way
        log.extend(_step(world, routes, controller, validate=world.state.tick == 0).events)
    fighting = any(e.kind == "combat" for e in log.events)
    assert fighting
    resumed = deepcopy(world)
    resumed_log = EventLog(list(log.events), log.head)
    for _ in range(330):
        log.extend(_step(world, routes, controller).events)
        resumed_log.extend(_step(resumed, RouteTable(resumed.world_map), fixture_controller("peaceful_fair", {5: "raider"})).events)
    assert state_hash(world.state) == state_hash(resumed.state)
    assert checkpoint_hash(world.state, world.run) == checkpoint_hash(resumed.state, resumed.run)
    assert log.head == resumed_log.head and log.events == resumed_log.events
    assert world.state.tick == 2000


def test_a_state_with_contacts_serializes_strictly_and_corrupt_memory_is_rejected():
    result = run_simulation(default_config(8, "moderate"), 50005, "peaceful_fair", ticks=600, collect_timeline=False,
                            include_map=False, roles={5: "raider"})
    world = build_world(default_config(8, "moderate"), 50005)
    state = result.state
    assert any(colony.memory.foreign for colony in state.colonies)
    data = state_to_bytes(state)
    assert state_hash(state_from_bytes(data, world.config, world.world_map)) == state_hash(state)
    validate_state(state, world.config, world.world_map)
    bad = deepcopy(state)
    colony = next(c for c in bad.colonies if len(c.memory.foreign) >= 1)
    colony.memory.foreign[0].colony_id = colony.colony_id               # a colony cannot "meet" itself
    with pytest.raises(InvariantError):
        validate_state(bad, world.config, world.world_map)
    bad = deepcopy(state)
    colony = next(c for c in bad.colonies if len(c.memory.foreign) >= 1)
    colony.memory.nodes[-1].visited = False
    colony.memory.nodes[-1].kind = "unknown"
    colony.memory.nodes[-1].queen_seen = 1                              # an unvisited node must not carry information
    with pytest.raises(InvariantError):
        validate_state(bad, world.config, world.world_map)
    bad = deepcopy(state)
    cohort = next(c for c in bad.cohorts if c.caste == "worker")
    cohort.mission = Mission("attack", 0, 0 if cohort.colony_id else 1, "go", 0)
    with pytest.raises(InvariantError):
        validate_state(bad, world.config, world.world_map)               # only soldiers attack


def test_turtle_and_retreat_roles_run_and_the_retreat_raider_can_break_off():
    turtle = run_simulation(default_config(8, "moderate"), 50004, "peaceful_fair", ticks=1800, collect_timeline=False,
                            include_map=False, roles={3: "turtle"}, include_events=True)
    assert turtle.report["interaction"]["events_by_kind"]["combat"] == 0               # a turtle never starts anything
    assert turtle.report["food_ledger"]["difference"] == 0
    retreat = run_simulation(default_config(8, "moderate"), 50004, "peaceful_fair", ticks=3600, collect_timeline=False,
                             include_map=False, roles={3: "retreat"}, include_events=True)
    assert retreat.report["food_ledger"]["difference"] == 0
    assert all(a["attacker"] == 3 for a in retreat.report["interaction"]["attacks_dispatched"])


def test_the_run_command_accepts_roles_and_writes_a_timeline(tmp_path):
    from formicmind.eco.cli import main
    timeline = tmp_path / "timeline.txt"
    out = tmp_path / "run.json"
    code = main(["run", "--seed", "50005", "--ticks", "700", "--role", "5=raider", "--no-time-series", "--omit-map",
                 "--timeline", str(timeline), "--out", str(out), "--compact"])
    assert code == 0
    import json
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["roles"] == {"5": "raider"} and "event_timeline" not in report
    lines = timeline.read_text(encoding="utf-8").splitlines()
    assert len(lines) == report["interaction"]["event_count"] and all(line.startswith("t=") for line in lines)
    assert main(["run", "--seed", "1", "--ticks", "5", "--role", "9=raider", "--omit-map"]) == 3
    assert main(["run", "--seed", "1", "--ticks", "5", "--role", "oops", "--omit-map"]) == 3


def test_the_published_raid_scenario_is_reproduced_exactly_and_the_orphan_dies_out_naturally():
    """docs/ecosystem_examples/for46_raid_seed50004*.json: colony 3 scouts, finds a nest, raids; colony 0 loses its queen and dies out ~0.9 L later."""
    import json
    from pathlib import Path
    published = json.loads((Path(__file__).resolve().parents[1] / "docs" / "ecosystem_examples" / "for46_raid_seed50004.json")
                           .read_text(encoding="utf-8"))
    result = run_simulation(default_config(8, "moderate"), 50004, "peaceful_fair", ticks=6000, sample_interval=500,
                            include_map=False, roles={3: "raider"})
    inter = result.report["interaction"]
    assert inter["events_hash"] == published["interaction"]["events_hash"]
    assert result.report["hashes"] == published["hashes"]
    assert [(q["colony"], q["tick"], q["killer"]) for q in inter["queen_deaths"]] == [(0, 2621, 3)]
    assert [(e["colony"], e["tick"]) for e in inter["extinctions"]] == [(0, 4400)]
    lifespan = default_config().economy.lifespan_worker
    assert 0.5 * lifespan <= 4400 - 2621 <= 1.1 * lifespan                              # contracts C13: 0.5-1.1 L after the queen's death
    assert {a["attacker"] for a in inter["attacks_dispatched"]} == {3}
    assert result.report["outcome"]["reason"] == "horizon" and result.report["food_ledger"]["difference"] == 0  # the world outlived the death
    assert result.state.outcomes[0].reason == "queen_death"
