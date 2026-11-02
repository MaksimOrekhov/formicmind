"""Exact FOR-1 v0.2.1 initial state, topology and side mirror (hand-written expectations)."""

import dataclasses
import heapq

import pytest

from formicmind import default_config, initial_state, reset
from formicmind.invariants import InvariantError, validate_config, validate_state
from formicmind.rules import MIRROR, NODE_NAMES, local_to_world
from formicmind.serialization import state_hash, state_to_canonical
from formicmind.versions import BALANCE_VERSION, RULES_VERSION

from conftest import OTHER_TEST_SEED, TEST_SEED

A, N1, N2, C, N4, N5, B = range(7)

# FOR-1 v0.2.1 section 2: (u, v, ticks), written independently of rules.py.
EXPECTED_EDGES = {
    (A, N1): 2, (N4, B): 2, (A, N2): 3, (N5, B): 3,
    (N1, C): 2, (C, N4): 2, (N2, C): 3, (C, N5): 3, (N2, N5): 4,
}


def shortest(edges, src, dst, banned=()):
    adj = {}
    for (u, v), w in edges.items():
        adj.setdefault(u, []).append((v, w))
        adj.setdefault(v, []).append((u, w))
    dist, heap = {src: 0}, [(0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == dst:
            return d
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if v in banned:
                continue
            if d + w < dist.get(v, 1 << 30):
                dist[v] = d + w
                heapq.heappush(heap, (d + w, v))
    return None


def test_topology_matches_spec(config):
    got = {(e.a, e.b): e.length for e in config.topology.edges}
    assert got == {tuple(sorted(k)): v for k, v in EXPECTED_EDGES.items()}
    assert [n.name for n in config.topology.nodes] == ["A", "N1", "N2", "C", "N4", "N5", "B"]
    assert [n.home_side for n in config.topology.nodes] == [0, -1, -1, -1, -1, -1, 1]


def test_topology_route_costs_match_spec(config):
    edges = {(e.a, e.b): e.length for e in config.topology.edges}
    assert shortest(edges, A, B) == 8            # A-N1-C-N4-B
    assert shortest(edges, A, B, banned={C}) == 10  # A-N2-N5-B


def test_mirror_is_an_involution_and_maps_topology_onto_itself(config):
    assert MIRROR == (B, N4, N5, C, N1, N2, A)
    assert all(MIRROR[MIRROR[i]] == i for i in range(7))
    edges = {(e.a, e.b): e.length for e in config.topology.edges}
    for (u, v), w in edges.items():
        assert edges[tuple(sorted((MIRROR[u], MIRROR[v])))] == w
    res = config.balance.resources
    assert all(res[i] == dataclasses.replace(res[MIRROR[i]], node_id=i) for i in range(7))


def test_resource_table_matches_spec(config):
    got = [(r.start_food, r.capacity, r.regen, r.max_harvest) for r in config.balance.resources]
    assert got == [
        (0, 0, 0, 0), (60, 80, 8, 8), (40, 60, 8, 8), (80, 120, 12, 12),
        (60, 80, 8, 8), (40, 60, 8, 8), (0, 0, 0, 0),
    ]


def test_config_versions_are_prototype_not_frozen_balance(config):
    assert config.rules_version == RULES_VERSION == "l0-proto-FOR1-v0.2.1"
    assert config.balance_version == BALANCE_VERSION
    assert "unvalidated" in config.balance_version
    assert (config.policy_interval_k, config.t_max) == (30, 3000)
    b = config.balance
    assert (b.eggs_per_lay, b.lay_interval_ticks) == (1, 10)
    assert (b.worker_egg_cost, b.soldier_egg_cost) == (10, 20)
    assert (b.worker_hatch_delay, b.soldier_hatch_delay) == (20, 30)
    assert (b.worker_upkeep_tenths, b.soldier_upkeep_tenths) == (4, 8)


def expected_colony(side):
    base = 16 * side
    home = A if side == 0 else B
    unit = lambda uid, caste, hp: {  # noqa: E731
        "unit_id": uid, "caste": caste, "hp": hp, "node": home,
        "from_node": -1, "to_node": -1, "arrival_tick": -1, "target_node": home,
    }
    return {
        "side": side, "food": 60, "queen_hp": 240, "next_lay_tick": 0,
        "upkeep_remainder10": 0, "starvation_debt": 0,
        "lay_order": 0, "forage_q": [65535, 0, 0, 0, 0, 0], "army_choice": 0, "avoid_center": False,
        "units": [unit(base + i, 0, 20) for i in range(1, 13)]
        + [unit(base + i, 1, 30) for i in range(13, 17)],
        "brood": [],
        "intel": {
            "nodes": [{"seen_tick": -1, "food_last": 0, "enemy_workers_last": 0,
                       "enemy_soldiers_last": 0}] * 7,
            "enemy_queen_hp_last": 0, "enemy_queen_seen_tick": -1,
        },
    }


def test_initial_state_is_exactly_the_for1_start(config):
    state, _ = reset(config, TEST_SEED)
    canon = state_to_canonical(state)
    assert canon["tick"] == 0
    assert canon["next_unit_id"] == 33
    assert canon["node_food"] == [0, 60, 40, 80, 60, 40, 0]
    assert canon["colonies"] == [expected_colony(0), expected_colony(1)]


def test_initial_invariants(config):
    state, _ = reset(config, TEST_SEED)
    validate_state(state)
    for colony in state.colonies:
        assert len(colony.units) == 16
        assert sum(1 for u in colony.units if u.caste == 0) == 12
        assert sum(1 for u in colony.units if u.caste == 1) == 4
        assert all(u.node == colony.side * 6 and u.target_node == u.node for u in colony.units)
        assert colony.brood == []
    ids = [u.unit_id for c in state.colonies for u in c.units]
    assert ids == list(range(1, 33)) and len(set(ids)) == 32
    # Upkeep sanity from FOR-1 section 3.2: 12*4 + 4*8 = 80 tenths = 8 food per tick.
    assert 12 * config.balance.worker_upkeep_tenths + 4 * config.balance.soldier_upkeep_tenths == 80


def test_side_b_is_the_mirror_of_side_a(config):
    state, _ = reset(config, TEST_SEED)
    a, b = state.colonies
    scalars = ("food", "queen_hp", "next_lay_tick", "upkeep_remainder10", "starvation_debt",
               "lay_order", "forage_q", "army_choice", "avoid_center")
    assert all(getattr(a, k) == getattr(b, k) for k in scalars)  # orders are in the own-side frame
    assert len(a.units) == len(b.units)
    for ua, ub in zip(a.units, b.units):
        assert ub.unit_id == ua.unit_id + 16
        assert (ub.caste, ub.hp) == (ua.caste, ua.hp)
        assert ub.node == MIRROR[ua.node] and ub.target_node == MIRROR[ua.target_node]
    assert [local_to_world(1, k) for k in range(7)] == list(MIRROR)
    assert [local_to_world(0, k) for k in range(7)] == list(range(7))
    assert state.node_food == [state.node_food[MIRROR[i]] for i in range(7)]


def test_reset_is_repeatable_and_does_not_mutate_config(config):
    snapshot = dataclasses.asdict(config)
    h1 = state_hash(reset(config, TEST_SEED)[0])
    h2 = state_hash(reset(default_config(), TEST_SEED)[0])
    assert h1 == h2
    assert dataclasses.asdict(config) == snapshot


def test_rules_consume_no_randomness_so_state_is_seed_independent(config):
    assert state_hash(reset(config, TEST_SEED)[0]) == state_hash(reset(config, OTHER_TEST_SEED)[0])


def test_initial_state_has_no_run_state(config):
    assert initial_state(config).tick == 0
    state, run = reset(config, TEST_SEED)
    assert not hasattr(state, "streams") and "world" in run.streams


def test_unit_id_allocator(config):
    state, _ = reset(config, TEST_SEED)
    assert state.allocate_unit_id() == 33
    assert state.allocate_unit_id() == 34
    assert state.next_unit_id == 35


def test_clone_is_independent(config):
    state, _ = reset(config, TEST_SEED)
    copy = state.clone()
    assert state_hash(copy) == state_hash(state)
    copy.colonies[0].units[0].hp = 1
    copy.colonies[1].intel.nodes[3].seen_tick = 0
    copy.colonies[0].forage_q[0] = 0
    copy.node_food[1] = 0
    copy.next_unit_id += 5
    assert state_hash(copy) != state_hash(state)
    assert state.colonies[0].units[0].hp == 20
    assert state.colonies[1].intel.nodes[3].seen_tick == -1
    assert state.colonies[0].forage_q[0] == 65535
    assert state.node_food[1] == 60 and state.next_unit_id == 33


def test_intel_memories_are_private_per_side(config):
    state, _ = reset(config, TEST_SEED)
    a, b = state.colonies
    assert a.intel is not b.intel and a.intel.nodes is not b.intel.nodes
    a.intel.nodes[N2].seen_tick = 0
    assert b.intel.nodes[N2].seen_tick == -1


def test_validate_state_catches_corruption(config):
    def broken(mutate):
        s = reset(config, TEST_SEED)[0]
        mutate(s)
        with pytest.raises(InvariantError):
            validate_state(s)

    broken(lambda s: setattr(s.colonies[0], "food", -1))
    broken(lambda s: setattr(s.colonies[0].units[0], "hp", 21))
    broken(lambda s: setattr(s.colonies[0].units[12], "hp", 31))
    broken(lambda s: setattr(s.colonies[1], "queen_hp", 241))
    broken(lambda s: s.node_food.__setitem__(N1, 81))
    broken(lambda s: setattr(s.colonies[0], "forage_q", [1, 0, 0, 0, 0, 0]))
    broken(lambda s: setattr(s.colonies[0], "forage_q", [16384, 16384, 16384, 16383, 0, 0]))  # support 4
    broken(lambda s: setattr(s.colonies[0].units[1], "unit_id", 1))  # duplicate id
    broken(lambda s: setattr(s, "next_unit_id", 10))
    broken(lambda s: setattr(s.colonies[0], "upkeep_remainder10", 10))
    broken(lambda s: setattr(s.colonies[0], "avoid_center", True))  # with garrison_home
    broken(lambda s: setattr(s.colonies[0].units[0], "node", -1))  # transit fields missing


def test_validate_config_catches_broken_config(config):
    topo = config.topology
    asym = dataclasses.replace(
        topo, edges=tuple(dataclasses.replace(e, length=9) if (e.a, e.b) == (0, 1) else e for e in topo.edges)
    )
    with pytest.raises(InvariantError):
        validate_config(dataclasses.replace(config, topology=asym))
    with pytest.raises(InvariantError):
        validate_config(dataclasses.replace(config, rules_version="other"))
    with pytest.raises(InvariantError):
        validate_config(dataclasses.replace(
            config, balance=dataclasses.replace(config.balance, eggs_per_lay=0)))


def test_node_names_constant(config):
    assert NODE_NAMES == tuple(n.name for n in config.topology.nodes)
