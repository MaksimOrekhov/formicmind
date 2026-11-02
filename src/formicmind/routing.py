"""Canonical shortest paths on the public topology (FOR-1 v0.2.1 errata A, FOR-2 section 2).

* Cost is the sum of edge lengths in ticks.
* Among equal-cost paths the lexicographically smallest sequence of node ids wins.
  The engine uses **world** node ids, as FOR-1/FOR-2 literally state. That tie-break is
  *not* mirror-symmetric for a few equal-cost pairs (e.g. N1->N2 goes through A, but the
  mirrored N4->N5 goes through C); see ``docs/tick_movement.md``. ``side_local_rank``
  implements the symmetric alternative (ties by the side's own canonical order) for
  comparison tests; the engine does not use it.
* ``avoid_center`` forbids C as an *intermediate or final* node. It never forbids the
  unit's current node: a unit standing in C may leave through C's edges.
"""

from __future__ import annotations

import heapq
from functools import lru_cache

from .config import Topology
from .rules import NODE_C, NODE_COUNT, local_to_world

ROUTE_TIE_BREAK = "world-node-id-lexicographic"


class RoutingError(ValueError):
    """No admissible route (e.g. destination C with avoid_center)."""


@lru_cache(maxsize=None)
def _adjacency(topology: Topology) -> tuple[tuple[tuple[int, int], ...], ...]:
    adj: list[list[tuple[int, int]]] = [[] for _ in range(NODE_COUNT)]
    for edge in topology.edges:
        adj[edge.a].append((edge.b, edge.length))
        adj[edge.b].append((edge.a, edge.length))
    return tuple(tuple(sorted(neigh)) for neigh in adj)


@lru_cache(maxsize=None)
def _distance_to(topology: Topology, dst: int, banned: frozenset[int]) -> dict[int, int]:
    adj = _adjacency(topology)
    dist = {dst: 0}
    heap = [(0, dst)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if v in banned:
                continue
            nd = d + w
            if nd < dist.get(v, 1 << 60):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def side_local_rank(side: int) -> tuple[int, ...]:
    """Tie-break rank = index in the side's own canonical node order."""
    return tuple(local_to_world(side, n) for n in range(NODE_COUNT))  # an involution


def shortest_path(
    topology: Topology,
    src: int,
    dst: int,
    *,
    avoid_center: bool = False,
    rank: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    """Node sequence ``src .. dst`` (both included); ``(src,)`` when ``src == dst``."""
    if not (0 <= src < NODE_COUNT and 0 <= dst < NODE_COUNT):
        raise RoutingError(f"bad node ids {src}, {dst}")
    banned = frozenset({NODE_C}) if avoid_center else frozenset()
    if dst in banned:
        raise RoutingError("destination C is not reachable with avoid_center")
    dist = _distance_to(topology, dst, banned - {src})
    if src not in dist:
        raise RoutingError(f"no route from {src} to {dst}")
    adj = _adjacency(topology)
    key = (lambda n: n) if rank is None else (lambda n: rank[n])
    path = [src]
    while path[-1] != dst:
        cur = path[-1]
        candidates = [n for n, w in adj[cur] if n in dist and w + dist[n] == dist[cur]]
        path.append(min(candidates, key=key))
    return tuple(path)


def path_cost(topology: Topology, path: tuple[int, ...]) -> int:
    length = {(e.a, e.b): e.length for e in topology.edges}
    return sum(length[(min(a, b), max(a, b))] for a, b in zip(path, path[1:]))


def next_hop(
    topology: Topology, src: int, dst: int, *, avoid_center: bool = False
) -> int:
    path = shortest_path(topology, src, dst, avoid_center=avoid_center)
    if len(path) < 2:
        raise RoutingError("already at the destination")
    return path[1]


def edge_length(topology: Topology, a: int, b: int) -> int:
    for edge in topology.edges:
        if (edge.a, edge.b) == (min(a, b), max(a, b)):
            return edge.length
    raise RoutingError(f"no edge between {a} and {b}")
