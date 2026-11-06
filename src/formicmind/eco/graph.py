"""Small integer graph toolbox used by the map generator and the fairness report (no third-party packages)."""

from __future__ import annotations

import heapq
from collections.abc import Sequence

INF = 1 << 60


def ceil_sqrt(value: int) -> int:
    """Smallest integer ``r`` with ``r * r >= value`` (``value >= 0``)."""
    from math import isqrt

    return 0 if value <= 0 else isqrt(value - 1) + 1


def adjacency(n: int, edges: Sequence[tuple[int, int, int]]) -> list[list[tuple[int, int]]]:
    """``adj[u] = [(v, length), ...]`` for undirected ``(a, b, length)`` edges."""
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for a, b, w in edges:
        adj[a].append((b, w))
        adj[b].append((a, w))
    return adj


def dijkstra(adj: Sequence[Sequence[tuple[int, int]]], source: int) -> list[int]:
    """Shortest path lengths in ticks from ``source`` (``INF`` where unreachable)."""
    dist = [INF] * len(adj)
    dist[source] = 0
    heap = [(0, source)]
    pop, push = heapq.heappop, heapq.heappush
    while heap:
        d, u = pop(heap)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                push(heap, (nd, v))
    return dist


def components(adj: Sequence[Sequence[tuple[int, int]]]) -> int:
    seen = [False] * len(adj)
    count = 0
    for start in range(len(adj)):
        if seen[start]:
            continue
        count += 1
        seen[start] = True
        stack = [start]
        while stack:
            u = stack.pop()
            for v, _ in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
    return count


def bridges_with_subtree_weight(adj: Sequence[Sequence[tuple[int, int]]], weight: Sequence[int]) -> list[tuple[int, int, int]]:
    """Bridges ``(u, v, w)`` of a connected graph, where ``w`` is the total ``weight`` on the DFS-child side.

    Iterative Tarjan low-link; a bridge separates the nodes of the child subtree from the rest, so ``w`` and ``total - w``
    are the weights of the two sides (used to count nests on each side: a *chokepoint* separates nests from nests)."""
    n = len(adj)
    tin = [-1] * n
    low = [0] * n
    sub = list(weight)
    timer = 0
    found: list[tuple[int, int, int]] = []
    tin[0] = low[0] = timer
    timer += 1
    stack = [(0, -1, 0)]
    while stack:
        u, parent, idx = stack.pop()
        if idx < len(adj[u]):
            stack.append((u, parent, idx + 1))
            v = adj[u][idx][0]
            if v == parent:
                continue
            if tin[v] == -1:
                tin[v] = low[v] = timer
                timer += 1
                stack.append((v, u, 0))
            else:
                low[u] = min(low[u], tin[v])
        elif parent != -1:
            low[parent] = min(low[parent], low[u])
            sub[parent] += sub[u]
            if low[u] > tin[parent]:
                found.append((parent, u, sub[u]))
    return found


class UnionFind:
    __slots__ = ("parent",)

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[ra] = rb
        return True


def _orient(ax: int, ay: int, bx: int, by: int, cx: int, cy: int) -> int:
    value = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    return (value > 0) - (value < 0)


def segments_cross(p: tuple[int, int], q: tuple[int, int], r: tuple[int, int], s: tuple[int, int]) -> bool:
    """True if segments ``pq`` and ``rs`` intersect in more than a shared endpoint (exact integer arithmetic)."""
    if p == r or p == s or q == r or q == s:
        return False  # edges sharing a node meet only there (nodes have distinct coordinates)
    o1 = _orient(*p, *q, *r)
    o2 = _orient(*p, *q, *s)
    o3 = _orient(*r, *s, *p)
    o4 = _orient(*r, *s, *q)
    if o1 != o2 and o3 != o4:
        return True

    def on_segment(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> bool:
        return min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])

    return (
        (o1 == 0 and on_segment(p, q, r))
        or (o2 == 0 and on_segment(p, q, s))
        or (o3 == 0 and on_segment(r, s, p))
        or (o4 == 0 and on_segment(r, s, q))
    )
