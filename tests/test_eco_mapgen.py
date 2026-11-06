"""FOR-44: the map generator on hundreds of seeds, sizes and scenarios, verified by code that does NOT use formicmind.eco.graph / fairness.

Seeds are a plain consecutive block (no cherry-picking); a single failure to generate, or one violated invariant, fails the test.
"""

from __future__ import annotations

import heapq
import itertools
import random
import time
from fractions import Fraction
from math import isqrt

import pytest

from formicmind.canonical import canonical_dumps
from formicmind.eco.codec import to_canonical
from formicmind.eco.config import SCENARIOS, SIZES, default_config, patch_parameters
from formicmind.eco.fairness import evaluate
from formicmind.eco.mapgen import MapGenerationError, generate, generate_map, permute_colonies
from formicmind.eco.serialization import compute_map_hash, map_to_bytes

SEED0 = 100_000
MAPS = 200  # per (size, scenario): 200 maps x 3 scenarios x 3 sizes = 1800 maps
INF = 10**9


# ------------------------------------------------------------------------------------------------ independent checkers


def dijkstra(n, edges, source):
    adj = [[] for _ in range(n)]
    for e in edges:
        adj[e.a].append((e.b, e.length))
        adj[e.b].append((e.a, e.length))
    dist = [INF] * n
    dist[source] = 0
    heap = [(0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(heap, (d + w, v))
    return dist


def connected_without(n, edges, skip=None):
    adj = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        if i != skip:
            adj[e.a].append(e.b)
            adj[e.b].append(e.a)
    seen = {0}
    stack = [0]
    while stack:
        for v in adj[stack.pop()]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == n


def cv_permille(values):
    n, mean = len(values), Fraction(sum(values), len(values))
    if mean == 0:
        return 0
    var = sum((Fraction(v) - mean) ** 2 for v in values) / n
    return (float(var) ** 0.5) / float(mean) * 1000


def phi_permille(d, load=16, h=2):
    return 1000 * load // (2 * d + -(-load // h))


def check_map(cfg, m, seed):
    """Every documented invariant, recomputed from the finished map with independent code. Returns the per-colony supply list."""
    p, e = cfg.map_params, cfg.economy
    tag = f"[{cfg.n_colonies} colonies, {cfg.scenario}, seed {seed}]"
    n = len(m.nodes)
    assert n == p.n_nodes and 80 <= n <= 120, tag
    assert [x.node_id for x in m.nodes] == list(range(n)), tag
    assert len({(x.x, x.y) for x in m.nodes}) == n, f"{tag} duplicate coordinates"
    pairs = [(x.a, x.b) for x in m.edges]
    assert pairs == sorted(set(pairs)) and all(a < b for a, b in pairs), f"{tag} edges not unique/sorted"
    assert all(2 <= x.length <= 6 and x.lane_cap in (8, 4) for x in m.edges), tag
    assert connected_without(n, m.edges), f"{tag} not connected"
    deg = [0] * n
    for a, b in pairs:
        deg[a] += 1
        deg[b] += 1
    assert max(deg) <= p.max_degree, tag
    nests = list(m.nests)
    assert len(set(nests)) == cfg.n_colonies and all(m.nodes[v].kind == "nest" and m.nodes[v].nest_of == c for c, v in enumerate(nests)), tag
    assert min(deg[v] for v in nests) >= 2, f"{tag} a nest is a dead end"

    dist = [dijkstra(n, m.edges, v) for v in nests]
    pair_min = min(dist[a][nests[b]] for a in range(len(nests)) for b in range(a + 1, len(nests)))
    assert pair_min >= 14, f"{tag} two nests only {pair_min} ticks apart"
    assert all(min(dist[c][nests[o]] for o in range(len(nests)) if o != c) <= 40 for c in range(len(nests))), tag

    patches = [x for x in m.nodes if x.patch]
    zones = {z: sum(1 for x in patches if x.patch[0].zone == z) for z in ("near", "border", "far")}
    assert zones == {"near": 2 * cfg.n_colonies, "border": cfg.n_colonies, "far": cfg.n_colonies}, f"{tag} {zones}"
    assert 24 <= len(patches) <= 40, tag
    assert sum(1 for x in m.nodes if x.kind == "nest") == cfg.n_colonies and sum(1 for x in m.nodes if x.kind == "patch") == len(patches), tag
    assert all(x.kind == "patch" or not x.patch for x in m.nodes) and all(not (x.kind == "nest" and x.patch) for x in m.nodes), tag

    owner_of = {}
    for x in patches:
        ds = sorted((dist[c][x.node_id], c) for c in range(cfg.n_colonies))
        if x.patch[0].zone == "near":
            assert 3 <= ds[0][0] <= 6 and ds[1][0] - ds[0][0] >= 3, f"{tag} near patch {x.node_id}: {ds[:2]}"
            owner_of[x.node_id] = ds[0][1]
        else:
            assert ds[1][0] <= 14, f"{tag} shared patch {x.node_id} is not contested: {ds[:3]}"
        assert deg[x.node_id] >= 2, f"{tag} patch {x.node_id} is a dead end"
    per_colony_near = [sum(1 for v in owner_of.values() if v == c) for c in range(cfg.n_colonies)]
    assert per_colony_near == [2] * cfg.n_colonies, f"{tag} {per_colony_near}"

    # resource budget: independently recomputed from the scenario table
    for x in patches:
        spec = x.patch[0]
        assert (spec.capacity, spec.regen, spec.site_cap) == patch_parameters(cfg, spec.zone) and spec.start_stock == spec.capacity, tag
        assert spec.site_cap >= spec.regen >= 1 and spec.capacity >= spec.regen, tag
    expected_regen = sum(patch_parameters(cfg, z)[1] * k for z, k in zones.items())
    assert sum(x.patch[0].regen for x in patches) == expected_regen, tag

    # chokepoints by brute force: removing a flagged edge splits the nests; no unflagged edge does
    narrow = {i for i, x in enumerate(m.edges) if x.lane_cap == 4}
    for i, x in enumerate(m.edges):
        splits_nests = False
        if not connected_without(n, m.edges, skip=i):
            reach = {nests[0]}
            adj = [[] for _ in range(n)]
            for j, y in enumerate(m.edges):
                if j != i:
                    adj[y.a].append(y.b)
                    adj[y.b].append(y.a)
            seen, stack = {nests[0]}, [nests[0]]
            while stack:
                for v in adj[stack.pop()]:
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
            splits_nests = any(v not in seen for v in nests)
        assert x.chokepoint == splits_nests and (i in narrow) == x.chokepoint, f"{tag} edge {x.a}-{x.b}"
    assert len(narrow) * 1000 <= p.max_narrow_share_permille * len(m.edges) + 999, tag
    for c in range(cfg.n_colonies):
        near = sum(1 for x in m.edges if x.chokepoint and min(dist[c][x.a], dist[c][x.b]) <= 8)
        assert near <= p.max_choke_near, f"{tag} colony {c} has {near} chokepoints within 8 ticks"

    # fairness, recomputed with exact fractions
    ref = phi_permille(3)
    access = []
    supply = []
    for c in range(cfg.n_colonies):
        access.append(sum(m.nodes[v].patch[0].regen * phi_permille(dist[c][v]) * 1000 // ref for v, o in owner_of.items() if o == c))
        total = Fraction(0)
        for x in patches:
            if dist[c][x.node_id] <= p.economic_radius:
                sharers = sum(1 for o in range(cfg.n_colonies) if dist[o][x.node_id] <= p.economic_radius)
                total += Fraction(x.patch[0].regen, sharers)
        supply.append(total)
    assert cv_permille(access) <= p.max_near_cv_permille + 1, f"{tag} near access CV {cv_permille(access):.1f}"
    assert cv_permille(supply) <= p.max_radius_cv_permille + 1, f"{tag} supply CV {cv_permille(supply):.1f}"
    assert 1 <= m.generation.attempts <= p.max_attempts, tag
    return supply


@pytest.fixture(scope="module")
def sweep():
    """200 maps for each of the 9 (size, scenario) combinations, generated once."""
    out = {}
    for n in SIZES:
        for scenario in SCENARIOS:
            cfg = default_config(n, scenario)
            out[(n, scenario)] = (cfg, [generate(cfg, SEED0 + i) for i in range(MAPS)])
    return out


# ------------------------------------------------------------------------------------------------ tests


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_generated_map_satisfies_every_invariant(sweep, size, scenario):
    cfg, results = sweep[(size, scenario)]
    assert len(results) == MAPS
    for i, result in enumerate(results):
        check_map(cfg, result.world_map, SEED0 + i)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_the_internal_report_agrees_with_the_independent_checks(sweep, size, scenario):
    cfg, results = sweep[(size, scenario)]
    for result in results[:25]:
        rep = evaluate(cfg, result.world_map)
        assert rep["all_checks_passed"] and all(c["ok"] for c in rep["checks"])
        canonical_dumps(rep)  # integers only


def test_acceptance_is_not_a_lottery(sweep):
    for (size, scenario), (cfg, results) in sweep.items():
        attempts = [r.attempts for r in results]
        assert max(attempts) < cfg.map_params.max_attempts, f"{size}/{scenario}: some map needed {max(attempts)} attempts"
        assert sum(attempts) / len(attempts) <= 4.0, f"{size}/{scenario}: mean attempts {sum(attempts) / len(attempts):.2f}"
        rejected = [reason for r in results for _, reason in r.rejections]
        assert all(reason for reason in rejected)


def test_total_regeneration_budget_matches_the_economy_document(sweep):
    # economy section 3: moderate, 8 colonies -> 16*5 + 8*6 + 8*8 = 192 food/tick, 24 per colony
    cfg, results = sweep[(8, "moderate")]
    rep = evaluate(cfg, results[0].world_map)
    assert rep["resource_budget"]["total_regen"] == 192 and rep["resource_budget"]["regen_per_colony_x1000"] == 24000
    # integer rounding of the scaled parameters (7.5 -> 8, 3.6 -> 4, 4.8 -> 5): 8*(2*8+9+12) = 296 -> 37 and 8*(2*3+4+5) = 120 -> 15
    for scenario, per_colony in (("abundance", 37000), ("severe", 15000)):
        cfg_s, res_s = sweep[(8, scenario)]
        assert evaluate(cfg_s, res_s[0].world_map)["resource_budget"]["regen_per_colony_x1000"] == per_colony
    for size in SIZES:  # the per-colony economy does not depend on the number of colonies
        cfg_n, res_n = sweep[(size, "moderate")]
        assert evaluate(cfg_n, res_n[0].world_map)["resource_budget"]["regen_per_colony_x1000"] == 24000


def test_the_same_seed_gives_byte_identical_maps_and_a_different_seed_a_different_one():
    cfg = default_config(8, "moderate")
    a, b = generate(cfg, 4242), generate(cfg, 4242)
    assert map_to_bytes(a.world_map) == map_to_bytes(b.world_map) and a.attempts == b.attempts and a.rejections == b.rejections
    assert a.rng == b.rng
    hashes = {generate_map(cfg, s).map_hash for s in range(500, 540)}
    assert len(hashes) == 40, "different seeds must give different maps"
    coords = {tuple((n.x, n.y) for n in generate_map(cfg, s).nodes) for s in range(500, 520)}
    assert len(coords) == 20


def test_generation_is_independent_of_python_hash_randomisation_and_run_order():
    cfg = default_config(7, "severe")
    first = generate_map(cfg, 99).map_hash
    for s in range(10, 14):
        generate_map(cfg, s)  # other work in between must not matter (no shared mutable state)
    assert generate_map(cfg, 99).map_hash == first


def test_changing_the_scenario_changes_only_the_resources_not_the_layout_rules():
    seed = 31337
    maps = {sc: generate_map(default_config(8, sc), seed) for sc in SCENARIOS}
    assert len({m.map_hash for m in maps.values()}) == 3
    for sc, m in maps.items():
        assert {n.patch[0].regen for n in m.nodes if n.patch} <= {patch_parameters(default_config(8, sc), z)[1] for z in ("near", "border", "far")}


def test_node_and_colony_ids_carry_no_role_information():
    cfg = default_config(8, "moderate")
    maps = [generate_map(cfg, SEED0 + i) for i in range(60)]
    nest_id_sets = {tuple(sorted(m.nests)) for m in maps}
    assert len(nest_id_sets) > 40, "nest node ids must not be a fixed block"
    assert sum(1 for m in maps if list(m.nests) == sorted(m.nests)) <= 3, "colony order must not follow node order"
    assert sum(1 for m in maps if all(m.nodes[i].kind == "nest" for i in range(8))) == 0
    near_ids = [tuple(sorted(n.node_id for n in m.nodes if n.patch and n.patch[0].zone == "near")) for m in maps]
    assert len(set(near_ids)) > 50 and all(ids != tuple(range(ids[0], ids[0] + len(ids))) for ids in near_ids)


def test_nothing_in_the_map_encodes_ownership_or_attack_routes():
    m = generate_map(default_config(8, "moderate"), 7)
    keys = set()

    def walk(value):
        if isinstance(value, dict):
            keys.update(value)
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    walk(to_canonical(m))
    forbidden = {"owner", "owned_by", "ownership", "territory", "route", "target", "attack", "war", "timer", "policy", "observation"}
    assert not (keys & forbidden), keys & forbidden
    assert keys == {"schema", "map_seed", "config_hash", "n_colonies", "nodes", "edges", "nests", "generation", "map_hash", "node_id",
                    "kind", "nest_of", "patch", "x", "y", "zone", "capacity", "regen", "site_cap", "start_stock", "a", "b", "length",
                    "lane_cap", "chokepoint", "algorithm", "attempts"}


def test_permuting_colony_ids_keeps_the_map_valid_and_the_fairness_unchanged():
    cfg = default_config(8, "moderate")
    base = generate_map(cfg, SEED0 + 3)
    rep = evaluate(cfg, base)
    rng = random.Random(5)  # test-side shuffling only
    for _ in range(5):
        perm = list(range(8))
        rng.shuffle(perm)
        moved = permute_colonies(base, perm)
        assert moved.map_hash == compute_map_hash(moved)
        assert (moved.nodes != base.nodes) == (perm != list(range(8)))
        moved_rep = evaluate(cfg, moved)
        assert moved_rep["all_checks_passed"]
        assert moved_rep["fairness"] == rep["fairness"] and moved_rep["counts"] == rep["counts"] and moved_rep["ranges"] == rep["ranges"]
        for old, new in enumerate(perm):  # colony `old` is now colony `new` with identical metrics
            a, b = dict(rep["colonies"][old]), dict(moved_rep["colonies"][new])
            a.pop("colony_id"), b.pop("colony_id")
            assert a == b
        assert moved.nests[perm[2]] == base.nests[2]
        inverse = [perm.index(i) for i in range(8)]
        assert permute_colonies(moved, inverse) == base
    with pytest.raises(ValueError):
        permute_colonies(base, [0, 1, 2, 3, 4, 5, 6, 6])
    with pytest.raises(ValueError):
        permute_colonies(base, [0, 1, 2])


def test_the_planar_layout_has_no_crossing_edges():
    from formicmind.eco.graph import segments_cross

    cfg = default_config(8, "moderate")
    for i in range(12):
        m = generate_map(cfg, SEED0 + i)
        seg = [((m.nodes[e.a].x, m.nodes[e.a].y), (m.nodes[e.b].x, m.nodes[e.b].y)) for e in m.edges]
        for (p, q), (r, s) in itertools.combinations(seg, 2):
            assert not segments_cross(p, q, r, s)


def test_chokepoints_exist_on_some_maps_and_are_narrow(sweep):
    cfg, results = sweep[(8, "moderate")]
    with_choke = [r for r in results if any(e.chokepoint for e in r.world_map.edges)]
    assert 0.05 * MAPS < len(with_choke) < 0.9 * MAPS
    for r in with_choke[:20]:
        assert all(e.lane_cap == 4 for e in r.world_map.edges if e.chokepoint)


def test_an_unsatisfiable_config_fails_loudly_and_quickly():
    from dataclasses import replace

    cfg = default_config(8, "moderate")
    hard = replace(cfg, map_params=replace(cfg.map_params, min_nest_path=40, max_nest_nn_path=45, max_attempts=3))
    t0 = time.perf_counter()
    with pytest.raises(MapGenerationError):
        generate(hard, 1)
    assert time.perf_counter() - t0 < 5.0


@pytest.mark.parametrize("seed", [-1, 1 << 64, 1.5, True, "7", None])
def test_invalid_seeds_are_rejected(seed):
    with pytest.raises((ValueError, TypeError)):
        generate(default_config(8, "moderate"), seed)


def test_generation_latency_stays_in_a_sane_range():
    """Not a benchmark (see docs): a loose guard against accidental quadratic blow-ups on slow CI machines."""
    cfg = default_config(8, "moderate")
    t0 = time.perf_counter()
    for i in range(20):
        generate_map(cfg, 9000 + i)
    mean = (time.perf_counter() - t0) / 20
    assert mean < 0.25, f"mean {mean * 1000:.0f} ms per map"


def test_isqrt_helpers_used_by_the_fairness_math_are_exact():
    from formicmind.eco.fairness import cv_permille as eco_cv

    for values in ([5, 5, 5], [1, 2, 3, 4], [100, 90, 110, 95], [0, 0, 7]):
        n, s, q = len(values), sum(values), sum(v * v for v in values)
        assert eco_cv(values) == (isqrt(((n * q - s * s) * 1_000_000) // (s * s)) if s else 0)
    assert eco_cv([]) == 0 and eco_cv([0, 0]) == 0
