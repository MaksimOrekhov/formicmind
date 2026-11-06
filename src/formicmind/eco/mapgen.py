"""Deterministic generator of the shared Ecosystem v1 map (RFC section 5.2, contracts section 3).

Algorithm ``eco-mapgen-v1`` (integers only, one RNG stream ``map``, no floating point, so the result is identical on every platform):

1. *Nests*: Mitchell best-candidate sampling in the inner square, then an integer repulsion relaxation towards a spacing of
   ``min_nest_path - 1`` ticks (irregular, never mirrored).
2. *Junction points*: a few around every nest, one where neighbours meet (Gabriel midpoints, Delaunay circumcentres), then dart
   throwing with a minimum separation up to ``n_nodes`` nodes. All of them are plain junctions without any label.
3. *Edges*: Euclidean minimum spanning tree (connected, planar) + edges to the ``edge_k`` nearest neighbours that cross nothing
   (degree <= ``max_degree``); nests get a second edge if they would be dead ends. Length = ``clamp(ceil(dist / units_per_tick))`` ticks.
4. *Roles by path length* (not by geometry): shortest paths from every nest are computed on the finished graph; then
   ``near`` patches are junctions 3-5 ticks from exactly one nest, ``border`` patches are junctions 5-14 ticks from two neighbours,
   ``far`` patches are the remaining contested junctions (two or more nests within 14 ticks, preferring a close third nest). Candidates must not be dead ends. Border and far patches are chosen
   greedily for the colonies that currently have the least shared supply (coarse buckets of 1.5 food/tick), then the candidate whose
   second-nearest nest is closest wins; this keeps access balanced without fixing who gets what.
5. *Chokepoints*: bridges that separate nests from nests get the narrow lane capacity.
6. *Shuffle*: node ids and colony ids are drawn from the RNG, so neither the id order nor the colony order carries role information.
7. *Acceptance*: ``fairness.evaluate`` must pass every check (path limits, contested patches, access fairness, chokepoint limits).
   Otherwise the whole layout is rejected and re-drawn from the same stream (at most ``max_attempts``); the number of attempts is
   recorded in the map, so the result stays a pure function of ``(config, map_seed)``.

There is no ownership, no preset attack route and no scarcity timer in the map: only geometry, path lengths and patch parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isqrt

from ..rng import Pcg32, new_stream
from .config import ConfigE1, config_hash, patch_parameters
from .fairness import evaluate
from .graph import UnionFind, adjacency, bridges_with_subtree_weight, ceil_sqrt, dijkstra, segments_cross
from .model import EdgeE1, GenerationInfo, MapE1, NodeE1, PatchSpec
from .serialization import seal_map
from .versions import GENERATOR_ALGORITHM, MAP_SCHEMA, MAP_STREAM_ID


class MapGenerationError(RuntimeError):
    """No acceptable layout was found within ``max_attempts``."""


@dataclass(slots=True)
class GenerationResult:
    world_map: MapE1
    attempts: int
    rejections: list[tuple[int, str]]  # (attempt number, reason) for every rejected layout
    rng: Pcg32  # the map stream after generation (kept for the checkpoint)


class _Reject(Exception):
    pass


# ------------------------------------------------------------------------------------------------ layout


def _place_nests(rng: Pcg32, n: int, side: int, units: int, min_path: int) -> list[tuple[int, int]]:
    margin = side // 12
    lo, span = margin, side - 2 * margin + 1
    nests: list[tuple[int, int]] = []
    for k in range(n):  # Mitchell best-candidate: a well spread random start
        best, best_d = (0, 0), -1
        for _ in range(1 if k == 0 else 32):
            x, y = lo + rng.below(span), lo + rng.below(span)
            d = min(((x - a) ** 2 + (y - b) ** 2 for a, b in nests), default=1 << 60)
            if d > best_d:
                best, best_d = (x, y), d
        nests.append(best)
    # integer repulsion relaxation towards a spacing of (min_path - 1) ticks: pushes crowded pairs apart, keeps the layout irregular
    pts = [[x, y] for x, y in nests]
    target = (min_path - 1) * units
    hi = side - margin
    for _ in range(40):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx, dy = pts[j][0] - pts[i][0], pts[j][1] - pts[i][1]
                d2 = dx * dx + dy * dy
                if d2 >= target * target:
                    continue
                if d2 == 0:
                    dx, dy, d2 = 1, 0, 1
                d = isqrt(d2) or 1
                push = (target - d) // 2 + 1
                ox, oy = dx * push // d, dy * push // d
                pts[i][0] = min(hi, max(lo, pts[i][0] - ox))
                pts[i][1] = min(hi, max(lo, pts[i][1] - oy))
                pts[j][0] = min(hi, max(lo, pts[j][0] + ox))
                pts[j][1] = min(hi, max(lo, pts[j][1] + oy))
                moved = True
        if not moved:
            break
    nests = [(x, y) for x, y in pts]
    floor = ((min_path - 3) * units) ** 2
    if min(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 for i, a in enumerate(nests) for b in nests[i + 1:]), default=floor) < floor:
        raise _Reject("nest_spacing")
    return nests


def _free(placed: list[tuple[int, int]], x: int, y: int, sep2: int) -> bool:
    for px, py in placed:
        if (px - x) ** 2 + (py - y) ** 2 < sep2:
            return False
    return True


def _select_patches(rng: Pcg32, config: ConfigE1, pts: list[tuple[int, int]], adj: list, nest_ids: list[int],
                    degree: list[int]) -> dict[int, str]:
    """Assign ``near`` / ``border`` / ``far`` zones to junction nodes by shortest-path distances from the nests."""
    p = config.map_params
    n_col = len(nest_ids)
    n = len(pts)
    units = p.units_per_tick
    dist = [dijkstra(adj, nest) for nest in nest_ids]
    nest_set = set(nest_ids)
    order = list(range(n))
    for i in range(n - 1, 0, -1):  # a random tie-break order, fixed for this attempt
        j = rng.below(i + 1)
        order[i], order[j] = order[j], order[i]
    candidates = [v for v in order if v not in nest_set and degree[v] >= 2]
    ranked = {v: sorted((dist[c][v], c) for c in range(n_col)) for v in candidates}  # (path, colony) from the nearest nest outwards
    zones: dict[int, str] = {}
    chosen_pts: list[tuple[int, int]] = []

    def separated(v: int, sep: int) -> bool:
        x, y = pts[v]
        return all((x - cx) ** 2 + (y - cy) ** 2 >= sep * sep for cx, cy in chosen_pts)

    def take(v: int, zone: str) -> None:
        zones[v] = zone
        chosen_pts.append(pts[v])

    # near: 3-5 ticks from one nest, at least 4 ticks closer to it than to any other nest
    target = (p.near_min_path + p.near_max_path) // 2
    for c in range(n_col):
        pool = []
        for v in candidates:
            d = dist[c][v]
            if p.near_min_path <= d <= p.near_max_path - 1 and v not in zones:
                rest = min(dist[o][v] for o in range(n_col) if o != c)
                if rest - d >= 4:
                    pool.append((abs(d - target), v))
        pool.sort(key=lambda t: t[0])  # stable: equal distances keep the random order
        picked = 0
        for _, v in pool:
            if picked == p.near_per_nest:
                break
            if separated(v, 2 * units):
                take(v, "near")
                picked += 1
        if picked < p.near_per_nest:
            raise _Reject("near_candidates")

    load = [0] * n_col  # shared supply per colony so far, in food/tick x 1000
    regen_border = patch_parameters(config, "border")[1]
    regen_far = patch_parameters(config, "far")[1]

    def pick(zone: str, count: int, reach_radius: int, min_near: int, regen: int, sep: int) -> None:
        for _ in range(count):
            best = None
            for v in candidates:
                if v in zones:
                    continue
                ds = ranked[v]
                if ds[0][0] < min_near:
                    continue
                reach = [c for d, c in ds if d <= reach_radius]
                if len(reach) < 2:
                    continue
                if zone == "border" and (ds[1][0] - ds[0][0] > 5 or ds[2][0] < ds[1][0] + 2):
                    continue  # a border patch belongs to exactly two close neighbours
                if not separated(v, sep):
                    continue
                share = [c for d, c in ds if d <= p.economic_radius] or reach
                score = (sum(load[c] for c in share) * 1000 // len(share) // 1500, ds[1][0], -len(share), ds[2][0])
                if best is None or score < best[0]:
                    best = (score, v, share)
            if best is None:
                raise _Reject(f"{zone}_candidates")
            take(best[1], zone)
            for c in best[2]:
                load[c] += regen * 1000 // len(best[2])

    pick("border", p.n_border, p.contest_radius, 5, regen_border, 2 * units)
    pick("far", p.n_far, p.contest_radius, 5, regen_far, 2 * units)
    return zones


def _surroundings(rng: Pcg32, nests: list[tuple[int, int]], per_nest: int, placed: list[tuple[int, int]], units: int,
                  side: int) -> list[tuple[int, int]]:
    """A few plain junction points 2.2-4.8 ticks around every nest (the neighbourhood a nest is born into)."""
    rmin2, rmax = (11 * units // 5) ** 2, 24 * units // 5
    sep2 = (8 * units // 5) ** 2
    out = []
    for nx, ny in nests:
        for _ in range(per_nest):
            for _try in range(60):
                dx, dy = rng.below(2 * rmax + 1) - rmax, rng.below(2 * rmax + 1) - rmax
                x, y = nx + dx, ny + dy
                if rmin2 <= dx * dx + dy * dy <= rmax * rmax and 0 < x < side and 0 < y < side and _free(placed, x, y, sep2):
                    placed.append((x, y))
                    out.append((x, y))
                    break
    return out


def _connectors(rng: Pcg32, nests: list[tuple[int, int]], placed: list[tuple[int, int]], units: int,
                side: int) -> list[tuple[int, int]]:
    """Plain junction points where neighbours meet: one near the midpoint of every Gabriel nest pair and one at the circumcentre of every
    Delaunay triple whose circumradius is 6-13 ticks. They are ordinary junctions (no label); roles are assigned later by path length."""
    n = len(nests)
    sep2 = (8 * units // 5) ** 2
    out: list[tuple[int, int]] = []

    def put(cx: int, cy: int) -> None:
        for _try in range(25):
            x = cx + rng.below(units + 1) - units // 2
            y = cy + rng.below(units + 1) - units // 2
            if 0 < x < side and 0 < y < side and _free(placed, x, y, sep2):
                placed.append((x, y))
                out.append((x, y))
                return

    for i in range(n):
        for j in range(i + 1, n):
            d2 = (nests[i][0] - nests[j][0]) ** 2 + (nests[i][1] - nests[j][1]) ** 2
            mx2, my2 = nests[i][0] + nests[j][0], nests[i][1] + nests[j][1]
            if d2 <= (19 * units) ** 2 and all((2 * nests[k][0] - mx2) ** 2 + (2 * nests[k][1] - my2) ** 2 >= d2
                                                for k in range(n) if k != i and k != j):
                put(mx2 // 2, my2 // 2)
    for i in range(n):
        ax, ay = nests[i]
        for j in range(i + 1, n):
            bx, by = nests[j]
            for k in range(j + 1, n):
                cx, cy = nests[k]
                d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
                if d == 0:
                    continue
                a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
                ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) // d
                uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) // d
                r2 = (ux - ax) ** 2 + (uy - ay) ** 2
                if not ((6 * units) ** 2 <= r2 <= (13 * units) ** 2 and 0 < ux < side and 0 < uy < side):
                    continue
                if any((nests[m][0] - ux) ** 2 + (nests[m][1] - uy) ** 2 < r2 for m in range(n) if m not in (i, j, k)):
                    continue
                put(ux, uy)
    return out


def _junctions(rng: Pcg32, need: int, placed: list[tuple[int, int]], sep: int, side: int) -> list[tuple[int, int]]:
    cells: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for x, y in placed:
        cells.setdefault((x // sep, y // sep), []).append((x, y))
    sep2 = sep * sep
    out = []
    span = side - 40
    for _ in range(need):
        for _try in range(80):
            x, y = 20 + rng.below(span), 20 + rng.below(span)
            cx, cy = x // sep, y // sep
            ok = True
            for gx in (cx - 1, cx, cx + 1):
                for gy in (cy - 1, cy, cy + 1):
                    for px, py in cells.get((gx, gy), ()):
                        if (px - x) ** 2 + (py - y) ** 2 < sep2:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                cells.setdefault((cx, cy), []).append((x, y))
                out.append((x, y))
                break
        else:
            raise _Reject("junction_placement")
    return out


# ------------------------------------------------------------------------------------------------ edges


class _EdgeGrid:
    """Exact planarity filter: a new segment is compared only with the segments that share a grid cell with its bounding box."""

    def __init__(self, cell: int) -> None:
        self.cell = cell
        self.cells: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}

    def _span(self, a: tuple[int, int], b: tuple[int, int]):
        c = self.cell
        return range(min(a[0], b[0]) // c, max(a[0], b[0]) // c + 1), range(min(a[1], b[1]) // c, max(a[1], b[1]) // c + 1)

    def add(self, a: tuple[int, int], b: tuple[int, int], ia: int, ib: int) -> None:
        xs, ys = self._span(a, b)
        for gx in xs:
            for gy in ys:
                self.cells.setdefault((gx, gy), []).append((ia, ib, 0, 0))

    def crosses(self, pts: list[tuple[int, int]], ia: int, ib: int) -> bool:
        a, b = pts[ia], pts[ib]
        xs, ys = self._span(a, b)
        for gx in xs:
            for gy in ys:
                for ja, jb, _, _ in self.cells.get((gx, gy), ()):
                    if segments_cross(a, b, pts[ja], pts[jb]):
                        return True
        return False


def _build_edges(rng: Pcg32, pts: list[tuple[int, int]], kinds: list[int], p, units: int) -> list[tuple[int, int, int]]:
    """Returns ``(a, b, length_ticks)`` with ``a < b`` (creation indices)."""
    n = len(pts)
    max_len = p.max_edge_ticks * units
    for reach in (2, 4, 8):  # candidate pairs up to this many longest-edge lengths; widen only if the points do not connect
        cap2 = (reach * max_len) ** 2
        pairs = []
        for i in range(n):
            xi, yi = pts[i]
            for j in range(i + 1, n):
                dx, dy = xi - pts[j][0], yi - pts[j][1]
                d2 = dx * dx + dy * dy
                if d2 <= cap2:
                    pairs.append((d2, i, j))
        pairs.sort()
        uf = UnionFind(n)
        mst = [pair for pair in pairs if uf.union(pair[1], pair[2])]
        if len(mst) == n - 1:
            break
    else:
        raise _Reject("disconnected")
    uf = UnionFind(n)
    degree = [0] * n
    present: set[tuple[int, int]] = set()
    grid = _EdgeGrid(max_len)
    chosen: list[tuple[int, int, int]] = []

    def add(d2: int, i: int, j: int) -> None:
        present.add((i, j))
        degree[i] += 1
        degree[j] += 1
        grid.add(pts[i], pts[j], i, j)
        length = ceil_sqrt(d2)
        ticks = max(p.min_edge_ticks, min(p.max_edge_ticks, -(-length // units)))
        chosen.append((i, j, ticks))

    for d2, i, j in mst:  # Euclidean MST: connected and crossing-free
        uf.union(i, j)
        add(d2, i, j)
    seen_count = [0] * n
    cap2 = max_len * max_len
    for d2, i, j in pairs:  # k-nearest-neighbour extras
        seen_count[i] += 1
        seen_count[j] += 1
        if d2 > cap2 or (i, j) in present:
            continue
        if seen_count[i] > p.edge_k and seen_count[j] > p.edge_k:
            continue
        if degree[i] >= p.max_degree or degree[j] >= p.max_degree or grid.crosses(pts, i, j):
            continue
        add(d2, i, j)
    for v in range(n):  # nests and patches are never dead ends
        if kinds[v] != 0 and degree[v] < 2:
            options = sorted(((pts[v][0] - pts[w][0]) ** 2 + (pts[v][1] - pts[w][1]) ** 2, w) for w in range(n) if w != v)
            for d2, w in options:
                a, b = (v, w) if v < w else (w, v)
                if (a, b) in present or degree[w] >= p.max_degree or d2 > (2 * max_len) ** 2 or grid.crosses(pts, a, b):
                    continue
                add(d2, a, b)
                break
            else:
                raise _Reject("dead_end")
    return chosen


# ------------------------------------------------------------------------------------------------ assembly


def _attempt(config: ConfigE1, rng: Pcg32, seed: int, attempt: int) -> MapE1:
    p = config.map_params
    n_col = config.n_colonies
    units = p.units_per_tick
    side = p.world_side
    nests = _place_nests(rng, n_col, side, units, p.min_nest_path)
    placed = list(nests)
    around = _surroundings(rng, nests, p.surroundings_per_nest, placed, units, side)
    around += _connectors(rng, nests, placed, units, side)
    junctions = around + _junctions(rng, p.n_nodes - n_col - len(around), placed, p.junction_min_sep, side)
    pts = list(nests) + junctions
    n = len(pts)
    kinds = [2] * n_col + [0] * (n - n_col)  # 2 nest, 1 patch, 0 junction
    raw_edges = _build_edges(rng, pts, kinds, p, units)
    adj = adjacency(n, raw_edges)
    degree = [len(a) for a in adj]
    zones = _select_patches(rng, config, pts, adj, list(range(n_col)), degree)
    for v in zones:
        kinds[v] = 1

    nest_weight = [1 if kinds[v] == 2 else 0 for v in range(n)]
    choke = {(min(u, v), max(u, v)) for u, v, side_w in bridges_with_subtree_weight(adj, nest_weight) if 0 < side_w < n_col}

    # random ids: shuffle node ids and colony ids so that neither order carries role information
    new_id = list(range(n))
    for i in range(n - 1, 0, -1):
        j = rng.below(i + 1)
        new_id[i], new_id[j] = new_id[j], new_id[i]
    colony_nest = list(range(n_col))
    for i in range(n_col - 1, 0, -1):
        j = rng.below(i + 1)
        colony_nest[i], colony_nest[j] = colony_nest[j], colony_nest[i]
    colony_of_nest = {nest: c for c, nest in enumerate(colony_nest)}

    nodes: list[NodeE1 | None] = [None] * n
    for v in range(n):
        spec: tuple[PatchSpec, ...] = ()
        kind, nest_of = "junction", -1
        if kinds[v] == 2:
            kind, nest_of = "nest", colony_of_nest[v]
        elif kinds[v] == 1:
            kind = "patch"
            cap, regen, site = patch_parameters(config, zones[v])
            spec = (PatchSpec(zones[v], cap, regen, site, cap),)  # type: ignore[arg-type]
        nodes[new_id[v]] = NodeE1(new_id[v], kind, nest_of, spec, pts[v][0], pts[v][1])  # type: ignore[arg-type]
    edges = []
    for a, b, ticks in raw_edges:
        na, nb = sorted((new_id[a], new_id[b]))
        narrow = (min(a, b), max(a, b)) in choke
        edges.append(EdgeE1(na, nb, ticks, config.economy.narrow_lane_cap if narrow else config.economy.lane_cap, narrow))
    edges.sort(key=lambda ed: (ed.a, ed.b))
    nests_by_colony = tuple(new_id[colony_nest[c]] for c in range(n_col))
    return MapE1(MAP_SCHEMA, seed, config_hash(config), n_col, tuple(nodes), tuple(edges),  # type: ignore[arg-type]
                 nests_by_colony, GenerationInfo(GENERATOR_ALGORITHM, attempt), "")


def generate(config: ConfigE1, map_seed: int) -> GenerationResult:
    """Generate and verify a map. Pure function of ``(config, map_seed)``; raises ``MapGenerationError`` if nothing passes."""
    if type(map_seed) is not int or not 0 <= map_seed < 1 << 64:
        raise ValueError("map_seed must be a u64")
    rng = new_stream(map_seed, MAP_STREAM_ID)
    rejections: list[tuple[int, str]] = []
    for attempt in range(1, config.map_params.max_attempts + 1):
        try:
            candidate = _attempt(config, rng, map_seed, attempt)
        except _Reject as reason:
            rejections.append((attempt, str(reason)))
            continue
        report = evaluate(config, candidate)
        if report["all_checks_passed"]:
            return GenerationResult(seal_map(candidate), attempt, rejections, rng)
        rejections.append((attempt, ",".join(c["name"] for c in report["checks"] if not c["ok"])))
    raise MapGenerationError(
        f"no acceptable map for seed {map_seed} in {config.map_params.max_attempts} attempts: {rejections[-3:]}")


def generate_map(config: ConfigE1, map_seed: int) -> MapE1:
    return generate(config, map_seed).world_map


def permute_colonies(world_map: MapE1, permutation: list[int] | tuple[int, ...]) -> MapE1:
    """Relabel colonies: new colony ``permutation[c]`` is old colony ``c``. Only ids change (nest_of and the order of ``nests``)."""
    n = world_map.n_colonies
    if sorted(permutation) != list(range(n)):
        raise ValueError("permutation must be a permutation of range(n_colonies)")
    new_nests = [0] * n
    for old, new in enumerate(permutation):
        new_nests[new] = world_map.nests[old]
    nodes = tuple(
        replace(node, nest_of=permutation[node.nest_of]) if node.kind == "nest" else node for node in world_map.nodes
    )
    return seal_map(replace(world_map, nodes=nodes, nests=tuple(new_nests)))
