"""Independent verification and fairness report of a generated map (RFC section 5.2, economy sections 2-4, 9).

``evaluate(config, map)`` recomputes everything from the *finished* ``MapE1`` (graph, patch specs, nests); it never looks at generator
internals, so the generator accepts a layout only if this module says every check passes, and a tester can recompute the same numbers
from the JSON alone. All values are integers (distances in ticks, fractions in per-mille), so the report is canonical-hashable.

Terminology: *near* patches were generated close to one nest, *border* between two neighbours, *far* at the junction of three
(``PatchSpec.zone``). The zone is a generation label, not ownership: access is measured only by path length in ticks.
"""

from __future__ import annotations

from math import isqrt

from ..canonical import canonical_dumps, domain_hash
from .config import ConfigE1
from .graph import INF, adjacency, bridges_with_subtree_weight, components, dijkstra
from .model import MapE1
from .versions import HASH_DOMAIN_REPORT, REPORT_SCHEMA


def forager_rate_permille(distance: int, load: int, harvest_rate: int) -> int:
    """Average delivery rate of one forager, per-mille food/tick: ``load / (2d + load / h)`` (economy section 2)."""
    return 1000 * load // (2 * distance + -(-load // harvest_rate))


def cv_permille(values: list[int]) -> int:
    """Coefficient of variation ``std / mean`` in per-mille, exact integer arithmetic (0 for a zero mean)."""
    n = len(values)
    s = sum(values)
    if n == 0 or s == 0:
        return 0
    q = sum(v * v for v in values)
    return isqrt(((n * q - s * s) * 1_000_000) // (s * s))


def _stats(values: list[int]) -> dict:
    n = len(values)
    return {"min": min(values), "max": max(values), "mean_x1000": sum(values) * 1000 // n, "cv_permille": cv_permille(values)}


def _check(name: str, value: int, limit: int, ok: bool, kind: str) -> dict:
    return {"name": name, "value": value, "limit": limit, "relation": kind, "ok": bool(ok)}


def evaluate(config: ConfigE1, world_map: MapE1) -> dict:
    p = config.map_params
    e = config.economy
    n_nodes = len(world_map.nodes)
    n_col = world_map.n_colonies
    edges = [(ed.a, ed.b, ed.length) for ed in world_map.edges]
    adj = adjacency(n_nodes, edges)
    connected = components(adj) == 1
    nests = list(world_map.nests)
    dist = [dijkstra(adj, nest) for nest in nests]

    patch_nodes = [node for node in world_map.nodes if node.patch]
    zone_count = {"near": 0, "border": 0, "far": 0}
    for node in patch_nodes:
        zone_count[node.patch[0].zone] += 1
    degree = [len(a) for a in adj]
    nest_set = set(nests)

    # reach: which colonies are within the economic radius of each patch
    reach_count: dict[int, int] = {}
    contest_count: dict[int, int] = {}
    for node in patch_nodes:
        reach_count[node.node_id] = sum(1 for c in range(n_col) if dist[c][node.node_id] <= p.economic_radius)
        contest_count[node.node_id] = sum(1 for c in range(n_col) if dist[c][node.node_id] <= p.contest_radius)

    # near-patch ownership by distance only (no ownership flag exists in the map)
    near_of: list[list[tuple[int, int]]] = [[] for _ in range(n_col)]  # (node_id, path)
    near_margin_min = INF
    near_path_min, near_path_max = INF, 0
    for node in patch_nodes:
        if node.patch[0].zone != "near":
            continue
        ds = sorted((dist[c][node.node_id], c) for c in range(n_col))
        owner_d, owner = ds[0]
        margin = ds[1][0] - owner_d
        near_margin_min = min(near_margin_min, margin)
        near_path_min, near_path_max = min(near_path_min, owner_d), max(near_path_max, owner_d)
        near_of[owner].append((node.node_id, owner_d))

    # chokepoints recomputed from the graph: a bridge with nests on both sides
    nest_weight = [1 if node.node_id in nest_set else 0 for node in world_map.nodes]
    expected_choke = set()
    for u, v, side in bridges_with_subtree_weight(adj, nest_weight) if connected else []:
        if 0 < side < n_col:
            expected_choke.add((min(u, v), max(u, v)))
    flagged = {(ed.a, ed.b) for ed in world_map.edges if ed.chokepoint}
    narrow = [ed for ed in world_map.edges if ed.lane_cap == e.narrow_lane_cap]
    lanes_consistent = flagged == expected_choke and all(
        (ed.lane_cap == e.narrow_lane_cap) == ed.chokepoint and ed.lane_cap in (e.lane_cap, e.narrow_lane_cap)
        for ed in world_map.edges
    )
    choke_edges = [ed for ed in world_map.edges if ed.chokepoint]

    phi_ref = forager_rate_permille(p.near_min_path, e.carry_load, e.harvest_rate)
    spec_by_node = {node.node_id: node.patch[0] for node in patch_nodes}
    colonies = []
    for c in range(n_col):
        d_c = dist[c]
        others = [d_c[nests[o]] for o in range(n_col) if o != c]
        near_access = sum(
            spec_by_node[nid].regen * forager_rate_permille(path, e.carry_load, e.harvest_rate) * 1000 // phi_ref
            for nid, path in near_of[c]
        )
        in_radius = [nid for nid in spec_by_node if d_c[nid] <= p.economic_radius]
        radius_supply = sum(spec_by_node[nid].regen * 1000 // reach_count[nid] for nid in in_radius)
        contested = sum(1 for nid in in_radius if reach_count[nid] >= 2)
        choke_near = sum(1 for ed in choke_edges if min(d_c[ed.a], d_c[ed.b]) <= p.choke_radius)
        colonies.append({
            "colony_id": c,
            "nest_degree": degree[nests[c]],
            "nearest_nest_path": min(others),
            "near_patch_paths": sorted(path for _, path in near_of[c]),
            "near_access_x1000": near_access,
            "patches_in_radius": len(in_radius),
            "contested_patches_in_radius": contested,
            "radius_supply_x1000": radius_supply,
            "chokepoints_within_radius": choke_near,
        })

    near_acc = [c["near_access_x1000"] for c in colonies]
    rad_sup = [c["radius_supply_x1000"] for c in colonies]
    nn = [c["nearest_nest_path"] for c in colonies]
    pair_min = min((dist[a][nests[b]] for a in range(n_col) for b in range(a + 1, n_col)), default=INF)
    contested_min = min((contest_count[node.node_id] for node in patch_nodes if node.patch[0].zone != "near"), default=2)
    narrow_share = len(narrow) * 1000 // max(1, len(world_map.edges))
    expected_patches = {"near": p.near_per_nest * n_col, "border": p.n_border, "far": p.n_far}
    nest_nodes_ok = (
        len(set(nests)) == n_col
        and all(world_map.nodes[nid].kind == "nest" and world_map.nodes[nid].nest_of == c for c, nid in enumerate(nests))
        and sum(1 for node in world_map.nodes if node.kind == "nest") == n_col
    )

    checks = [
        _check("connected", int(connected), 1, connected, "=="),
        _check("node_count_in_range", n_nodes, p.n_nodes, n_nodes == p.n_nodes and 80 <= n_nodes <= 120, "=="),
        _check("patch_counts_match_config", len(patch_nodes), config.patch_count(), zone_count == expected_patches, "=="),
        _check("nests_distinct_and_typed", int(nest_nodes_ok), 1, nest_nodes_ok, "=="),
        _check("nest_degree_at_least_2", min(degree[nid] for nid in nests), 2, all(degree[nid] >= 2 for nid in nests), ">="),
        _check("nest_pair_min_path", pair_min, p.min_nest_path, pair_min >= p.min_nest_path, ">="),
        _check("nearest_nest_max_path", max(nn), p.max_nest_nn_path, max(nn) <= p.max_nest_nn_path, "<="),
        _check("near_patch_min_path", near_path_min if near_path_min != INF else 0, p.near_min_path,
               near_path_min >= p.near_min_path, ">="),
        _check("near_patch_max_path", near_path_max, p.near_max_path, near_path_max <= p.near_max_path, "<="),
        _check("near_patch_owner_margin", near_margin_min if near_margin_min != INF else 0, 3, near_margin_min >= 3, ">="),
        _check("shared_patches_contested", contested_min, 2, contested_min >= 2, ">="),
        _check("near_access_cv_permille", cv_permille(near_acc), p.max_near_cv_permille,
               cv_permille(near_acc) <= p.max_near_cv_permille, "<="),
        _check("radius_supply_cv_permille", cv_permille(rad_sup), p.max_radius_cv_permille,
               cv_permille(rad_sup) <= p.max_radius_cv_permille, "<="),
        _check("chokepoints_near_nest_max", max(c["chokepoints_within_radius"] for c in colonies), p.max_choke_near,
               max(c["chokepoints_within_radius"] for c in colonies) <= p.max_choke_near, "<="),
        _check("narrow_edge_share_permille", narrow_share, p.max_narrow_share_permille,
               narrow_share <= p.max_narrow_share_permille, "<="),
        _check("edge_lengths_in_range", max((ed.length for ed in world_map.edges), default=0), p.max_edge_ticks,
               all(p.min_edge_ticks <= ed.length <= p.max_edge_ticks for ed in world_map.edges), "<="),
        _check("lanes_and_chokepoints_consistent", int(lanes_consistent), 1, lanes_consistent, "=="),
    ]

    budget = {}
    for zone in ("near", "border", "far"):
        specs = [n.patch[0] for n in patch_nodes if n.patch[0].zone == zone]
        budget[zone] = {
            "count": len(specs),
            "capacity": sum(s.capacity for s in specs),
            "regen": sum(s.regen for s in specs),
            "site_cap": sum(s.site_cap for s in specs),
            "start_stock": sum(s.start_stock for s in specs),
        }
    total_regen = sum(b["regen"] for b in budget.values())
    lengths = [ed.length for ed in world_map.edges]
    report = {
        "schema": REPORT_SCHEMA,
        "all_checks_passed": all(c["ok"] for c in checks),
        "checks": checks,
        "counts": {
            "colonies": n_col, "nodes": n_nodes, "edges": len(world_map.edges), "patches": len(patch_nodes),
            "junctions": sum(1 for n in world_map.nodes if n.kind == "junction"),
            "patches_by_zone": zone_count, "chokepoints": len(choke_edges), "narrow_edges": len(narrow),
        },
        "ranges": {
            "degree": {"min": min(degree), "max": max(degree), "mean_x100": sum(degree) * 100 // n_nodes},
            "edge_length": {"min": min(lengths), "max": max(lengths), "mean_x100": sum(lengths) * 100 // len(lengths),
                            "histogram": [lengths.count(v) for v in range(p.min_edge_ticks, p.max_edge_ticks + 1)],
                            "histogram_from": p.min_edge_ticks},
            "nest_pair_path": {"min": pair_min},
            "nearest_nest_path": _stats(nn),
        },
        "fairness": {
            "scenario": config.scenario,
            "near_access_x1000": _stats(near_acc),
            "radius_supply_x1000": _stats(rad_sup),
            "contested_patches_in_radius": _stats([c["contested_patches_in_radius"] for c in colonies]),
            "economic_radius_ticks": p.economic_radius,
        },
        "resource_budget": {
            "zones": budget,
            "total_regen": total_regen,
            "regen_per_colony_x1000": total_regen * 1000 // n_col,
            "total_capacity": sum(b["capacity"] for b in budget.values()),
        },
        "colonies": colonies,
    }
    return report


def report_hash(report: dict) -> str:
    return domain_hash(HASH_DOMAIN_REPORT, canonical_dumps(report))
