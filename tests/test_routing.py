"""Canonical shortest paths: costs, side-local tie-break (routing errata R1), avoid_center, mirror."""

import itertools

import pytest

from formicmind import default_config
from formicmind.routing import (
    RoutingError,
    edge_length,
    next_hop,
    path_cost,
    shortest_path,
    shortest_path_world_id,
    side_local_rank,
)
from formicmind.rules import MIRROR, local_to_world

A, N1, N2, C, N4, N5, B = range(7)
TOPOLOGY = default_config().topology
PAIRS = [(u, v) for u in range(7) for v in range(7) if u != v]
SIDES = (0, 1)


def route(side, u, v, avoid=False):
    return shortest_path(TOPOLOGY, u, v, side=side, avoid_center=avoid)


def mirror_path(path):
    return tuple(MIRROR[n] for n in path)


def floyd(avoid_center):
    inf = 10**9
    nodes = [n for n in range(7) if not (avoid_center and n == C)]
    d = {(u, v): (0 if u == v else inf) for u in nodes for v in nodes}
    for e in TOPOLOGY.edges:
        if e.a in nodes and e.b in nodes:
            d[(e.a, e.b)] = d[(e.b, e.a)] = e.length
    for k, i, j in itertools.product(nodes, nodes, nodes):
        d[(i, j)] = min(d[(i, j)], d[(i, k)] + d[(k, j)])
    return d


def test_spec_route_costs_a_to_b_and_mirrored_b_to_a():
    through_c, around = route(0, A, B), route(0, A, B, True)
    assert (through_c, path_cost(TOPOLOGY, through_c)) == ((A, N1, C, N4, B), 8)
    assert (around, path_cost(TOPOLOGY, around)) == ((A, N2, N5, B), 10)
    back, back_around = route(1, B, A), route(1, B, A, True)
    assert (back, path_cost(TOPOLOGY, back)) == ((B, N4, C, N1, A), 8)
    assert (back_around, path_cost(TOPOLOGY, back_around)) == ((B, N5, N2, A), 10)


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("avoid", [False, True])
def test_every_path_is_a_cheapest_simple_path(side, avoid):
    dist = floyd(avoid)
    for u, v in PAIRS:
        if avoid and C in (u, v):
            continue
        path = route(side, u, v, avoid)
        assert path[0] == u and path[-1] == v and len(set(path)) == len(path)
        assert path_cost(TOPOLOGY, path) == dist[(u, v)]
        if avoid:
            assert C not in path


def test_side_local_rank_is_the_sides_canonical_order():
    assert side_local_rank(0) == (0, 1, 2, 3, 4, 5, 6)
    assert side_local_rank(1) == (6, 4, 5, 3, 1, 2, 0)
    for side in SIDES:
        rank = side_local_rank(side)
        assert [rank[local_to_world(side, k)] for k in range(7)] == list(range(7))


def test_equal_cost_paths_use_the_movers_local_order_not_world_ids():
    # N1-A-N2 and N1-C-N2 both cost 5. Side A prefers home (local 0) -> via A.
    assert path_cost(TOPOLOGY, (N1, A, N2)) == path_cost(TOPOLOGY, (N1, C, N2)) == 5
    assert route(0, N1, N2) == (N1, A, N2)
    # the mirrored move N4 -> N5 is, for side B, local N1 -> N2: it also goes through its own home
    assert path_cost(TOPOLOGY, (N4, B, N5)) == path_cost(TOPOLOGY, (N4, C, N5)) == 5
    assert route(1, N4, N5) == (N4, B, N5)
    # the old world-id rule sent N4 -> N5 through C (not mirror-symmetric): R1 replaced it
    assert shortest_path_world_id(TOPOLOGY, N4, N5) == (N4, C, N5)
    # cost 7 both ways, A-N2-N5 vs A-N1-C-N5: local [0,1,3,5] < [0,2,5]
    assert path_cost(TOPOLOGY, (A, N2, N5)) == path_cost(TOPOLOGY, (A, N1, C, N5)) == 7
    assert route(0, A, N5) == (A, N1, C, N5) and route(1, B, N2) == (B, N4, C, N2)


@pytest.mark.parametrize("avoid", [False, True])
def test_every_route_is_mirror_symmetric_for_all_side_src_dst_avoid_combinations(avoid):
    """route_B(mirror(u), mirror(v)) == mirror(route_A(u, v)), and the other way round."""
    checked = 0
    for u, v in PAIRS:
        if avoid and v == C:
            continue
        a = route(0, u, v, avoid)
        b = route(1, MIRROR[u], MIRROR[v], avoid)
        assert mirror_path(a) == b
        assert mirror_path(route(1, u, v, avoid)) == route(0, MIRROR[u], MIRROR[v], avoid)
        checked += 1
    assert checked == (42 if not avoid else 36)


def test_the_superseded_world_id_rule_was_asymmetric_for_exactly_six_ordered_pairs():
    asymmetric = set()
    for avoid in (False, True):
        for u, v in PAIRS:
            if avoid and v == C:
                continue
            a = shortest_path_world_id(TOPOLOGY, u, v, avoid_center=avoid)
            b = shortest_path_world_id(TOPOLOGY, MIRROR[u], MIRROR[v], avoid_center=avoid)
            if mirror_path(a) != b:
                asymmetric.add((avoid, u, v))
    assert asymmetric == {
        (False, N1, N2), (False, N2, N1), (False, N4, N5), (False, N5, N4),
        (False, N2, B), (False, N5, A),
    }
    # R1 differs from the old rule only where an equal-cost tie exists
    differing = {(s, u, v, a) for s in SIDES for a in (False, True) for u, v in PAIRS
                 if not (a and v == C) and route(s, u, v, a) != shortest_path_world_id(
                     TOPOLOGY, u, v, avoid_center=a)}
    assert differing and all(
        path_cost(TOPOLOGY, route(s, u, v, a)) == path_cost(TOPOLOGY, shortest_path_world_id(
            TOPOLOGY, u, v, avoid_center=a)) for s, u, v, a in differing)


@pytest.mark.parametrize("side", SIDES)
def test_avoid_center_never_bans_the_current_node(side):
    assert route(side, C, B, True)[0] == C and route(side, C, A, True)[0] == C
    for v in (A, N1, N2, N4, N5, B):
        path = route(side, C, v, True)
        assert path[0] == C and C not in path[1:]


@pytest.mark.parametrize("side", SIDES)
def test_avoid_center_makes_c_an_invalid_destination_and_never_an_intermediate(side):
    with pytest.raises(RoutingError):
        route(side, A, C, True)
    assert route(side, A, C) == (A, N1, C)
    for u, v in PAIRS:
        if v != C:
            assert C not in route(side, u, v, True)[1:-1]


def test_degenerate_and_invalid_inputs():
    assert route(0, N2, N2) == (N2,)
    with pytest.raises(RoutingError):
        next_hop(TOPOLOGY, N2, N2, side=0)
    for bad in ((7, 0), (0, -1)):
        with pytest.raises(RoutingError):
            route(0, *bad)
    for bad_side in (2, -1, None):
        with pytest.raises(RoutingError):
            shortest_path(TOPOLOGY, A, B, side=bad_side)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        shortest_path(TOPOLOGY, A, B)  # type: ignore[call-arg]  # side is mandatory: no silent default
    with pytest.raises(RoutingError):
        edge_length(TOPOLOGY, A, B)


def test_next_hop_and_edge_lengths():
    assert next_hop(TOPOLOGY, A, B, side=0) == N1
    assert next_hop(TOPOLOGY, A, B, side=0, avoid_center=True) == N2
    assert next_hop(TOPOLOGY, N4, N5, side=1) == B and next_hop(TOPOLOGY, N1, N2, side=0) == A
    assert [edge_length(TOPOLOGY, a, b) for a, b in ((A, N1), (N1, A), (A, N2), (N2, N5), (C, N5))] == [2, 2, 3, 4, 3]


@pytest.mark.parametrize("side", SIDES)
def test_suffix_of_a_route_is_the_route_from_the_intermediate_node(side):
    # re-routing at every node must reproduce the original path (stateless routing is consistent)
    for avoid in (False, True):
        for u, v in PAIRS:
            if avoid and v == C:
                continue
            path = route(side, u, v, avoid)
            for i in range(len(path) - 1):
                assert route(side, path[i], v, avoid) == path[i:]
