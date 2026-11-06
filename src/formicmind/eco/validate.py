"""Semantic validation (invariants) of MapE1, WorldStateE1 and RunStateE1. Loaders call these after the strict schema decode."""

from __future__ import annotations

from ..canonical import CanonicalError
from ..rng import Pcg32, check_stream_id
from .config import ConfigE1, config_hash, patch_parameters
from .model import ColonyMemoryE1, MapE1, RunStateE1, WorldStateE1
from .versions import GENERATOR_ALGORITHM, MAP_SCHEMA, STATE_SCHEMA


class InvariantError(CanonicalError):
    """A schema-valid object violates an invariant of the contracts."""


def _need(cond: bool, message: str) -> None:
    if not cond:
        raise InvariantError(message)


# ------------------------------------------------------------------------------------------------ map


def validate_map(world_map: MapE1, config: ConfigE1) -> None:
    p, e = config.map_params, config.economy
    _need(world_map.schema == MAP_SCHEMA, f"unknown map schema {world_map.schema!r}")
    _need(world_map.config_hash == config_hash(config), "map was generated for a different config")
    _need(world_map.n_colonies == config.n_colonies, "n_colonies differs from the config")
    _need(world_map.generation.algorithm == GENERATOR_ALGORITHM, "unknown generator algorithm")
    _need(1 <= world_map.generation.attempts <= p.max_attempts, "generation attempts out of range")
    n = len(world_map.nodes)
    _need(p.n_nodes == n, "node count differs from the config")
    coords = set()
    nest_nodes = []
    patch_zone_count = {"near": 0, "border": 0, "far": 0}
    for index, node in enumerate(world_map.nodes):
        _need(node.node_id == index, "nodes must be listed by node_id 0..M-1")
        _need(0 <= node.x <= p.world_side and 0 <= node.y <= p.world_side, f"node {index} is outside the layout")
        _need((node.x, node.y) not in coords, f"node {index} shares coordinates with another node")
        coords.add((node.x, node.y))
        _need((node.kind == "patch") == (len(node.patch) == 1) and len(node.patch) <= 1, f"node {index}: patch spec mismatch")
        _need((node.kind == "nest") == (node.nest_of >= 0), f"node {index}: nest_of mismatch")
        if node.kind == "nest":
            nest_nodes.append(index)
            _need(0 <= node.nest_of < config.n_colonies, f"node {index}: nest_of out of range")
        if node.patch:
            spec = node.patch[0]
            cap, regen, site = patch_parameters(config, spec.zone)
            _need((spec.capacity, spec.regen, spec.site_cap) == (cap, regen, site),
                  f"node {index}: patch parameters do not match the {config.scenario} scenario")
            _need(spec.start_stock == spec.capacity, f"node {index}: patches start full")
            patch_zone_count[spec.zone] += 1
    _need(len(world_map.nests) == config.n_colonies, "one nest per colony required")
    _need(len(set(world_map.nests)) == config.n_colonies, "nests must be distinct nodes")
    for colony, node_id in enumerate(world_map.nests):
        _need(0 <= node_id < n and world_map.nodes[node_id].kind == "nest" and world_map.nodes[node_id].nest_of == colony,
              f"nests[{colony}] does not point to its nest node")
    _need(len(nest_nodes) == config.n_colonies, "wrong number of nest nodes")
    _need(patch_zone_count == {"near": p.near_per_nest * config.n_colonies, "border": p.n_border, "far": p.n_far},
          "patch zone counts differ from the config")
    last = (-1, -1)
    for edge in world_map.edges:
        _need(0 <= edge.a < edge.b < n, "edges need 0 <= a < b < M")
        _need((edge.a, edge.b) > last, "edges must be strictly sorted by (a, b)")
        last = (edge.a, edge.b)
        _need(p.min_edge_ticks <= edge.length <= p.max_edge_ticks, "edge length out of range")
        _need(edge.lane_cap in (e.lane_cap, e.narrow_lane_cap), "edge lane_cap must be the normal or the narrow value")


# ------------------------------------------------------------------------------------------------ state


def _validate_memory(memory: ColonyMemoryE1, colony_id: int, config: ConfigE1, world_map: MapE1, edge_index: dict) -> None:
    n = len(world_map.nodes)
    seen: set[int] = set()
    for slot, kn in enumerate(memory.nodes):
        _need(0 <= kn.node_id < n and kn.node_id not in seen, f"colony {colony_id}: memory node {slot} invalid or duplicated")
        seen.add(kn.node_id)
        truth = world_map.nodes[kn.node_id]
        if kn.visited:
            _need(kn.kind == truth.kind, f"colony {colony_id}: visited node {slot} has the wrong kind")
            _need(kn.discovered_tick >= 0 and kn.seen_tick >= 0, f"colony {colony_id}: visited node {slot} was never seen")
            if truth.patch:
                _need(kn.patch_capacity == truth.patch[0].capacity, f"colony {colony_id}: patch capacity mismatch")
            else:
                _need(kn.patch_capacity == 0, "only patches have a capacity")
        else:
            _need(kn.kind == "unknown" and kn.patch_capacity == 0 and kn.seen_tick == -1 and kn.last_stock_tick == -1,
                  f"colony {colony_id}: unvisited node {slot} must not carry information")
        _need(kn.queen_seen in (0, 1, 2), "queen_seen must be 0, 1 or 2")
        _need(-1 <= kn.nest_owner_slot < len(memory.foreign), "nest_owner_slot out of range")
    _need(bool(memory.nodes) and memory.nodes[0].node_id == world_map.nests[colony_id] and memory.nodes[0].visited,
          f"colony {colony_id}: memory slot 0 must be its own, visited nest")
    seen_edges = set()
    for ke in memory.edges:
        _need(0 <= ke.slot_a < len(memory.nodes) and 0 <= ke.slot_b < len(memory.nodes) and ke.slot_a != ke.slot_b,
              f"colony {colony_id}: memory edge slots out of range")
        a, b = memory.nodes[ke.slot_a].node_id, memory.nodes[ke.slot_b].node_id
        key = (min(a, b), max(a, b))
        _need(key in edge_index and key not in seen_edges, f"colony {colony_id}: memory edge does not exist or is duplicated")
        seen_edges.add(key)
        truth_edge = edge_index[key]
        _need((ke.length, ke.lane_cap) == (truth_edge.length, truth_edge.lane_cap), "memory edge differs from the map")
        _need(memory.nodes[ke.slot_a].visited or memory.nodes[ke.slot_b].visited,
              f"colony {colony_id}: an edge can only be known from a visited endpoint")
    _need(len(memory.foreign) <= config.n_colonies - 1, "too many foreign colony records")
    _need(len(memory.sightings) <= 32 and len(memory.events) <= 16, "memory ring buffers overflow")


def validate_state(state: WorldStateE1, config: ConfigE1, world_map: MapE1) -> None:
    e = config.economy
    _need(state.schema == STATE_SCHEMA, f"unknown state schema {state.schema!r}")
    _need(state.config_hash == config_hash(config), "state belongs to a different config")
    _need(state.map_hash == world_map.map_hash, "state belongs to a different map")
    _need(0 <= state.tick <= config.horizon, "tick out of range")
    n_col = config.n_colonies
    patch_nodes = [node for node in world_map.nodes if node.patch]
    _need([ps.node_id for ps in state.patches] == [node.node_id for node in patch_nodes], "patch states must cover every patch, sorted")
    for ps, node in zip(state.patches, patch_nodes):
        _need(0 <= ps.stock <= node.patch[0].capacity, f"patch {ps.node_id}: stock outside 0..capacity")
    last = -1
    for pile in state.piles:
        _need(pile.node_id > last and 0 <= pile.node_id < len(world_map.nodes) and pile.stock >= 0, "bad pile")
        last = pile.node_id
    edge_index = {(ed.a, ed.b): ed for ed in world_map.edges}
    _need(len(state.colonies) == n_col and len(state.outcomes) == n_col, "one colony state and one outcome per colony")
    _need(state.next_cohort_id >= 0, "bad next_cohort_id")

    adults = [{"worker": 0, "soldier": 0} for _ in range(n_col)]
    last = -1
    for cohort in state.cohorts:
        _need(cohort.cohort_id > last and cohort.cohort_id < state.next_cohort_id, "cohort ids must be unique, sorted, < next_cohort_id")
        last = cohort.cohort_id
        _need(0 <= cohort.colony_id < n_col, "cohort colony out of range")
        unit_hp = e.worker_hp if cohort.caste == "worker" else e.soldier_hp
        _need(cohort.count >= 1 and unit_hp * (cohort.count - 1) < cohort.hp_total <= unit_hp * cohort.count,
              f"cohort {cohort.cohort_id}: count must equal ceil(hp_total / unit_hp)")
        _need(cohort.load >= 0 and (cohort.caste == "worker" or cohort.load == 0) and cohort.load <= cohort.count * e.carry_load,
              f"cohort {cohort.cohort_id}: bad load")
        loc = cohort.loc
        if loc.kind == "node":
            _need(0 <= loc.node_id < len(world_map.nodes) and loc.edge_from == -1 and loc.edge_to == -1,
                  f"cohort {cohort.cohort_id}: bad node location")
        else:
            _need(loc.node_id == -1 and (min(loc.edge_from, loc.edge_to), max(loc.edge_from, loc.edge_to)) in edge_index
                  and loc.depart_tick < loc.arrive_tick, f"cohort {cohort.cohort_id}: bad edge location")
        _need(-1 <= cohort.mission.target_node < len(world_map.nodes), "mission target out of range")
        adults[cohort.colony_id][cohort.caste] += cohort.count

    for index, col in enumerate(state.colonies):
        _need(col.colony_id == index, "colonies must be listed by colony_id")
        _need(0 <= col.food <= e.storage_cap, f"colony {index}: food outside 0..storage_cap")
        _need(0 <= col.queen_hp <= e.queen_hp, f"colony {index}: bad queen_hp")
        _need((col.status == "alive") == (col.queen_hp > 0) or col.status == "extinct", f"colony {index}: status contradicts the queen")
        _need(min(col.lay_acc_q, col.soldier_acc_q, col.upkeep_remainder10, col.starvation_debt) >= 0, "negative accumulator")
        last = -1
        for br in col.brood:
            _need(br.cohort_id > last and br.cohort_id < state.next_cohort_id and br.hatch_tick > br.laid_tick, "bad brood record")
            last = br.cohort_id
        for caste, ledger, unit in (("worker", col.ages.workers, "workers"), ("soldier", col.ages.soldiers, "soldiers")):
            starts = [b.bucket_start for b in ledger]
            _need(all(b.count >= 1 for b in ledger) and starts == sorted(set(starts)), f"colony {index}: age ledger must be sorted and positive")
            _need(all(s % e.age_bucket == 0 for s in starts), f"colony {index}: age bucket starts must be multiples of age_bucket")
            _need(sum(b.count for b in ledger) == adults[index][caste], f"colony {index}: age ledger of {unit} differs from the cohorts")
        if col.status == "extinct":
            _need(adults[index]["worker"] + adults[index]["soldier"] == 0 and not col.brood, f"colony {index}: extinct but not empty")
        o = col.orders
        _need(0 <= o.egg_level <= 3 and 0 <= o.soldier_share_q <= 4 and 0 <= o.forage_frac_q <= 8 and 0 <= o.scout_n <= 4,
              f"colony {index}: order out of range")
        _need(len(o.squads) == 3 and len(o.forage_targets) <= 3 and len(o.scout_targets) <= 2, f"colony {index}: bad order shape")
        _need(len(o.forage_targets) == len(o.forage_shares_q) and len(set(o.forage_targets)) == len(o.forage_targets),
              f"colony {index}: forage targets must be distinct with one share each")
        _need((sum(o.forage_shares_q) == 16) if o.forage_targets else not o.forage_shares_q, f"colony {index}: shares must sum to 16")
        slots = len(col.memory.nodes)
        _need(all(0 <= t < slots for t in o.forage_targets + o.scout_targets)
              and all(-1 <= s.target_node_slot < slots and 0 <= s.size_q <= 4 for s in o.squads), "order targets must be known slots")
        _validate_memory(col.memory, index, config, world_map, edge_index)
        _need(min(col.rolling.income_ema_q, col.rolling.burn_ema_q, col.rolling.pop_ema_q) >= 0, "negative rolling average")
    for flow in state.edge_flow:
        _need((flow.a, flow.b) in edge_index and flow.direction in (0, 1) and flow.in_flight_units >= 0, "bad edge flow")
    for outcome in state.outcomes:
        _need(outcome.finished == bool(outcome.reason) and (outcome.terminal_tick >= 0) == outcome.finished, "bad colony outcome")
    wo = state.world_outcome
    _need(wo.finished == bool(wo.reason) and (wo.terminal_tick >= 0) == wo.finished, "bad world outcome")


# ------------------------------------------------------------------------------------------------ run


def validate_run_state(run: RunStateE1) -> None:
    ids = [s.stream_id for s in run.streams]
    _need(ids == sorted(set(ids)), "streams must be sorted and unique")
    for s in run.streams:
        check_stream_id(s.stream_id)
        try:
            Pcg32(s.state, s.inc)
        except ValueError as exc:
            raise InvariantError(f"stream {s.stream_id}: {exc}") from exc
