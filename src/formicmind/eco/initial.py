"""Initial WorldStateE1 (tick 0, no simulation) and the one-call world builder (config + seed -> map, state, run state)."""

from __future__ import annotations

from dataclasses import dataclass

from ..rng import new_stream
from .config import ConfigE1, config_hash
from .graph import adjacency, dijkstra
from .mapgen import GenerationResult, generate
from .model import (
    AgeBucket, AgeLedger, CohortE1, ColonyMemoryE1, ColonyOrders, ColonyOutcome, ColonyStateE1, KnownEdge, KnownNode, Location,
    MapE1, Mission, PatchState, RollingE1, RunStateE1, SquadOrder, WindowStatsE1, WorldOutcome, WorldStateE1,
)
from .serialization import new_run_state
from .validate import validate_state
from .versions import MAP_STREAM_ID, STATE_SCHEMA, WORLD_STREAM_ID

Q = 65536  # fixed point of the rolling averages (1.0 == Q)


def _seed_ledger(count: int, lifespan: int, spread_permille: int, bucket: int) -> list[AgeBucket]:
    """Start adults are aged evenly over ``[0, spread * L)`` so that they do not all die in the same bucket (economy 7.1)."""
    counts: dict[int, int] = {}
    for i in range(count):
        age = (i * spread_permille * lifespan) // (1000 * count) if lifespan else 0
        start = (-age // bucket) * bucket  # floor division: the older bucket, as in scripts/ecosystem_budget.py
        counts[start] = counts.get(start, 0) + 1
    return [AgeBucket(start, counts[start]) for start in sorted(counts)]


def _initial_memory(config: ConfigE1, world_map: MapE1, colony: int, adj: list, dist: list[int]) -> ColonyMemoryE1:
    """Own nest and everything within ``initial_known_radius`` ticks is *visited* at tick 0; the other ends of the edges seen from there
    are *discovered* only (kind unknown). Nothing else of the map is present."""
    radius = config.economy.initial_known_radius
    visited = sorted((v for v, d in enumerate(dist) if d <= radius), key=lambda v: (dist[v], v))
    nest = world_map.nests[colony]
    assert visited[0] == nest
    slot = {v: i for i, v in enumerate(visited)}
    discovered: list[int] = []
    for v in visited:
        for w, _ in adj[v]:
            if w not in slot and w not in discovered:
                discovered.append(w)
    for w in sorted(discovered, key=lambda w: (min(dist[v] for v, _ in adj[w] if v in slot), w)):
        slot[w] = len(slot)
    order = sorted(slot, key=slot.get)  # type: ignore[arg-type]
    nodes = []
    for v in order:
        truth = world_map.nodes[v]
        if v in visited:
            if truth.kind == "nest" and v != nest:
                raise AssertionError("a foreign nest inside the initial known radius")
            capacity = truth.patch[0].capacity if truth.patch else 0
            stock = truth.patch[0].start_stock if truth.patch else 0
            own = v == nest
            nodes.append(KnownNode(v, truth.kind, True, 0, 0, capacity, stock, 0 if truth.patch else -1, 0, -1,
                                   1 if own else 0, 0 if own else -1, 0, 0))
        else:
            nodes.append(KnownNode(v, "unknown", False, 0, -1, 0, 0, -1, 0, -1, 0, -1, 0, 0))
    edges = []
    for ed in world_map.edges:
        if ed.a in slot and ed.b in slot and (ed.a in visited or ed.b in visited):
            a, b = sorted((slot[ed.a], slot[ed.b]))
            edges.append(KnownEdge(a, b, ed.length, ed.lane_cap, 0, False, 0))
    edges.sort(key=lambda e: (e.slot_a, e.slot_b))
    return ColonyMemoryE1(nodes=nodes, edges=edges)


def initial_world_state(config: ConfigE1, world_map: MapE1) -> WorldStateE1:
    e = config.economy
    n_col = config.n_colonies
    adj = adjacency(len(world_map.nodes), [(ed.a, ed.b, ed.length) for ed in world_map.edges])
    patches = [PatchState(node.node_id, node.patch[0].start_stock) for node in world_map.nodes if node.patch]
    colonies: list[ColonyStateE1] = []
    cohorts: list[CohortE1] = []
    next_id = 0
    for c in range(n_col):
        nest = world_map.nests[c]
        for caste, count, hp in (("worker", e.start_workers, e.worker_hp), ("soldier", e.start_soldiers, e.soldier_hp)):
            if count > 0:
                cohorts.append(CohortE1(next_id, c, caste, count, count * hp,  # type: ignore[arg-type]
                                        Location("node", nest, -1, -1, 0, 0), Mission("home", -1, -1, "idle", 0), 0))
                next_id += 1
        orders = ColonyOrders(0, 0, 0, 0, config.options.workers_flee_default, [], [], [],
                              [SquadOrder("none", 0, -1, -1) for _ in range(3)])
        ledger = AgeLedger(
            _seed_ledger(e.start_workers, e.lifespan_worker, e.start_age_spread_permille, e.age_bucket),
            _seed_ledger(e.start_soldiers, e.lifespan_soldier, e.start_age_spread_permille, e.age_bucket),
        )
        upkeep_tenths = e.worker_upkeep_tenths * e.start_workers + e.soldier_upkeep_tenths * e.start_soldiers
        rolling = RollingE1(0, upkeep_tenths * Q // 10, (2 * e.start_workers + e.start_soldiers) * Q // 2)
        memory = _initial_memory(config, world_map, c, adj, dijkstra(adj, nest))
        colonies.append(ColonyStateE1(c, "alive", e.start_food, e.queen_hp, 0, 0, 0, 0, 0, [], ledger, orders, memory,
                                      WindowStatsE1(), rolling))
    state = WorldStateE1(STATE_SCHEMA, config_hash(config), world_map.map_hash, 0, patches, [], colonies, cohorts, next_id, [],
                         [ColonyOutcome(False, "", -1) for _ in range(n_col)], WorldOutcome(False, "", -1))
    validate_state(state, config, world_map)
    return state


@dataclass(slots=True)
class World:
    config: ConfigE1
    world_map: MapE1
    state: WorldStateE1
    run: RunStateE1
    generation: GenerationResult


def build_world(config: ConfigE1, map_seed: int) -> World:
    """Generate the map for ``map_seed`` and create the tick-0 state and run state (the ``map`` stream is saved after generation)."""
    generation = generate(config, map_seed)
    state = initial_world_state(config, generation.world_map)
    run = new_run_state(map_seed, {MAP_STREAM_ID: generation.rng, WORLD_STREAM_ID: new_stream(map_seed, WORLD_STREAM_ID)})
    return World(config, generation.world_map, state, run, generation)
