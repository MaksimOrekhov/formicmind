"""Canonical shortest paths: costs, tie-break, avoid_center, mirror property."""

import itertools

import pytest

from formicmind import default_config
from formicmind.routing import (
    RoutingError,
    edge_length,
    next_hop,
    path_cost,
    shortest_path,
    side_local_rank,
)
from formicmind.rules import MIRROR

A, N1, N2, C, N4, N5, B = range(7)
TOPOLOGY = default_config().topology
PAIRS = [(u, v) for u in range(7) for v in range(7) if u != v]


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
    through_c = shortest_path(TOPOLOGY, A, B)
    around = shortest_path(TOPOLOGY, A, B, avoid_center=True)
    assert (through_c, path_cost(TOPOLOGY, through_c)) == ((A, N1, C, N4, B), 8)
    assert (around, path_cost(TOPOLOGY, around)) == ((A, N2, N5, B), 10)
    back = shortest_path(TOPOLOGY, B, A)
    back_around = shortest_path(TOPOLOGY, B, A, avoid_center=True)
    assert (back, path_cost(TOPOLOGY, back)) == ((B, N4, C, N1, A), 8)
    assert (back_around, path_cost(TOPOLOGY, back_around)) == ((B, N5, N2, A), 10)


@pytest.mark.parametrize("avoid", [False, True])
def test_every_path_is_a_cheapest_simple_path(avoid):
    dist = floyd(avoid)
    for u, v in PAIRS:
        if avoid and C in (u, v):
            continue
        path = shortest_path(TOPOLOGY, u, v, avoid_center=avoid)
        assert path[0] == u and path[-1] == v and len(set(path)) == len(path)
        assert path_cost(TOPOLOGY, path) == dist[(u, v)]
        if avoid:
            assert C not in path


def test_equal_cost_paths_use_lexicographic_world_ids():
    # cost 5 both ways: N1-A-N2 and N1-C-N2 -> via A (0 < 3)
    assert path_cost(TOPOLOGY, (N1, A, N2)) == path_cost(TOPOLOGY, (N1, C, N2)) == 5
    assert shortest_path(TOPOLOGY, N1, N2) == (N1, A, N2)
    # the mirrored pair: N4-B-N5 vs N4-C-N5 -> via C (3 < 6)
    assert shortest_path(TOPOLOGY, N4, N5) == (N4, C, N5)
    # cost 7 both ways: A-N2-N5 vs A-N1-C-N5 -> [0,1,3,5] < [0,2,5]
    assert path_cost(TOPOLOGY, (A, N2, N5)) == path_cost(TOPOLOGY, (A, N1, C, N5)) == 7
    assert shortest_path(TOPOLOGY, A, N5) == (A, N1, C, N5)
    assert shortest_path(TOPOLOGY, B, N2) == (B, N4, C, N2)


def test_world_id_tie_break_is_not_mirror_symmetric_for_exactly_six_ordered_pairs():
    """Fixture for the FOR-11 report: literal FOR-1/FOR-2 tie-break vs side symmetry."""
    asymmetric = set()
    for avoid in (False, True):
        for u, v in PAIRS:
            if avoid and v == C:
                continue
            path = shortest_path(TOPOLOGY, u, v, avoid_center=avoid)
            mirrored = shortest_path(TOPOLOGY, MIRROR[u], MIRROR[v], avoid_center=avoid)
            if tuple(MIRROR[n] for n in path) != mirrored:
                asymmetric.add((avoid, u, v))
    assert asymmetric == {
        (False, N1, N2), (False, N2, N1), (False, N4, N5), (False, N5, N4),
        (False, N2, B), (False, N5, A),
    }


def test_side_local_rank_tie_break_is_mirror_symmetric_everywhere():
    for avoid in (False, True):
        for u, v in PAIRS:
            if avoid and v == C:
                continue
            a = shortest_path(TOPOLOGY, u, v, avoid_center=avoid, rank=side_local_rank(0))
            b = shortest_path(TOPOLOGY, MIRROR[u], MIRROR[v], avoid_center=avoid, rank=side_local_rank(1))
            assert tuple(MIRROR[n] for n in a) == b
    # and it differs from the literal rule exactly where that one is asymmetric
    assert shortest_path(TOPOLOGY, N4, N5, rank=side_local_rank(1)) == (N4, B, N5)


def test_avoid_center_never_bans_the_current_node():
    assert shortest_path(TOPOLOGY, C, B, avoid_center=True) == (C, N4, B)
    assert shortest_path(TOPOLOGY, C, A, avoid_center=True) == (C, N1, A)
    for v in (A, N1, N2, N4, N5, B):
        path = shortest_path(TOPOLOGY, C, v, avoid_center=True)
        assert path[0] == C and C not in path[1:]


def test_avoid_center_makes_c_an_invalid_destination_and_never_an_intermediate():
    with pytest.raises(RoutingError):
        shortest_path(TOPOLOGY, A, C, avoid_center=True)
    assert shortest_path(TOPOLOGY, A, C) == (A, N1, C)
    for u, v in PAIRS:
        if v != C:
            assert C not in shortest_path(TOPOLOGY, u, v, avoid_center=True)[1:-1]


def test_degenerate_and_invalid_inputs():
    assert shortest_path(TOPOLOGY, N2, N2) == (N2,)
    with pytest.raises(RoutingError):
        next_hop(TOPOLOGY, N2, N2)
    for bad in ((7, 0), (0, -1)):
        with pytest.raises(RoutingError):
            shortest_path(TOPOLOGY, *bad)
    with pytest.raises(RoutingError):
        edge_length(TOPOLOGY, A, B)


def test_next_hop_and_edge_lengths():
    assert next_hop(TOPOLOGY, A, B) == N1 and next_hop(TOPOLOGY, A, B, avoid_center=True) == N2
    assert [edge_length(TOPOLOGY, a, b) for a, b in ((A, N1), (N1, A), (A, N2), (N2, N5), (C, N5))] == [2, 2, 3, 4, 3]


def test_suffix_of_a_route_is_the_route_from_the_intermediate_node():
    # re-routing at every node must reproduce the original path (stateless routing is consistent)
    for avoid in (False, True):
        for u, v in PAIRS:
            if avoid and v == C:
                continue
            path = shortest_path(TOPOLOGY, u, v, avoid_center=avoid)
            for i in range(len(path) - 1):
                assert shortest_path(TOPOLOGY, path[i], v, avoid_center=avoid) == path[i:]
