"""ObservationE1/v2: what one colony may know when it decides (contracts section 7).

The builder reads exactly three things and nothing else: the colony's **own** ``ColonyStateE1`` (vitals, orders, brood, age ledger and its private
``ColonyMemoryE1``), the colony's **own** cohorts, and public constants (the config, the tick). It receives no ``WorldStateE1``, no ``MapE1`` and no
other colony, so a hidden read is structurally impossible (``tests/test_eco_policy_leak.py`` also proves it with a poisoned world and counterfactual
worlds). World ids never leave the memory: every node and colony is a local **slot** (slot 0 of the nodes is the colony's own nest).

Shapes (variable length: only valid rows are stored, a batch loader pads):

* ``global_features``  40 floats, indices fixed by the version;
* ``node_features``    n x 24, ``node_ints`` n x 4 ``(kind_code, visited, age, path_ticks)`` with ``age``/``path_ticks`` -1 = never / no known path;
* ``edge_features``    m x 6, ``edge_slots`` m x 2;
* ``foreign_features`` f x 12 (f <= 7), ``foreign_nest_slots`` f ints (-1 = nest unknown);
* ``cohort_features``  c x 14 (own cohorts only, c <= 48), ``cohort_slots`` c x 2 ``(node_slot, target_slot)``;
* ``sighting_features`` s x 6 (s <= 32), ``sighting_ints`` s x 3 ``(node_slot, colony_slot, age)``;
* ``event_features``   e x 8 (e <= 16), ``event_ints`` e x 3 ``(code, node_slot, colony_slot)``;
* ``facts``            the integers the action masks and the decoder need (counts of units, food, alert flag ...), all own-state.

Every float is rounded to ``float32`` and computed with correctly rounded IEEE operations only (no ``libm`` call): ``log`` normalisation is the
piecewise-linear ``log2(1 + n)``. The observation therefore has one canonical byte form, ``observation_bytes``, and the same value on every platform.
"""

from __future__ import annotations

import heapq
import struct
from dataclasses import dataclass
from typing import Sequence

from ..canonical import domain_hash
from .config import ConfigE1
from .memory import (
    MEM_ATTACKED, MEM_OWN_QUEEN_DEAD, MEM_QUEEN_HIT, MEM_WE_LOST,
)
from .model import CohortE1, ColonyStateE1, RollingE1, WindowStatsE1
from .versions import OBSERVATION_VERSION

MAX_NODES = 128
MAX_EDGES = 320
MAX_FOREIGN = 7
MAX_COHORTS = 48
MAX_SIGHTINGS = 32
MAX_EVENTS = 16
GLOBAL_DIM = 40
NODE_DIM = 24
EDGE_DIM = 6
FOREIGN_DIM = 12
COHORT_DIM = 14
SIGHTING_DIM = 6
EVENT_DIM = 8

Q = 65536  # fixed point of the rolling averages (``initial.Q``)
KIND_JUNCTION, KIND_PATCH, KIND_OWN_NEST, KIND_FOREIGN_NEST, KIND_UNKNOWN = 0, 1, 2, 3, 4
MISSION_ORDER = ("home", "forage", "scout", "guard", "attack", "withdraw")
PHASE_VALUE = {"idle": 0, "go": 1, "work": 2, "return": 3}
OBSERVATION_HASH_DOMAIN = "formicmind/eco/observation/v1"
_THREAT_CODES = (MEM_ATTACKED, MEM_WE_LOST, MEM_QUEEN_HIT)
_COMBAT_CODES = (2, 3, 4, 5)  # as in ``engine._COMBAT_CODES``: attacked, we_killed, we_lost, queen_hit
_RECENT_TICKS = 150


class ObservationError(ValueError):
    """The colony cannot produce a valid observation (technical failure, never a defeat)."""


# ------------------------------------------------------------------------------------------------ number helpers (no libm)


def _f32(value: float) -> float:
    return struct.unpack(">f", struct.pack(">f", value))[0]


def _clip01(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


def _ratio(numerator: int, denominator: int) -> float:
    return _clip01(numerator / denominator) if denominator > 0 else 0.0


def _log2p1(n: int) -> float:
    """Piecewise-linear ``log2(1 + n)``: exact at powers of two, linear in between; only integer shifts and one division."""
    m = n + 1
    exponent = m.bit_length() - 1
    return exponent + (m - (1 << exponent)) / (1 << exponent)


def _lognorm(n: int, cap: int) -> float:
    return _clip01(_log2p1(max(0, n)) / _log2p1(cap))


# ------------------------------------------------------------------------------------------------ the observation


@dataclass(frozen=True, slots=True)
class ObservationFacts:
    """Integers of the colony's own state that the masks and the decoder need (they are also encoded, normalised, in the feature blocks)."""

    tick: int
    decision_index: int
    ticks_since_last_decision: int
    food: int
    burn_tenths: int          # upkeep per tick in tenths of food
    workers: int
    soldiers: int
    idle_workers_home: int    # workers at the own nest with no mission and no cargo: the pool every order draws on
    idle_soldiers_home: int
    scouts_out: int
    brood_workers: int
    brood_soldiers: int
    in_combat_now: int
    cohort_overflow: int      # own cohorts that did not fit the 48 rows (the smallest are left out)


@dataclass(frozen=True, slots=True)
class ObservationE1:
    version: str
    global_features: tuple[float, ...]
    node_features: tuple[tuple[float, ...], ...]
    node_ints: tuple[tuple[int, int, int, int], ...]
    edge_features: tuple[tuple[float, ...], ...]
    edge_slots: tuple[tuple[int, int], ...]
    foreign_features: tuple[tuple[float, ...], ...]
    foreign_nest_slots: tuple[int, ...]
    cohort_features: tuple[tuple[float, ...], ...]
    cohort_slots: tuple[tuple[int, int], ...]
    sighting_features: tuple[tuple[float, ...], ...]
    sighting_ints: tuple[tuple[int, int, int], ...]
    event_features: tuple[tuple[float, ...], ...]
    event_ints: tuple[tuple[int, int, int], ...]
    facts: ObservationFacts


def observation_bytes(obs: ObservationE1) -> bytes:
    """The canonical byte form (big-endian ``float32`` / ``int32`` blocks with their lengths); equal observations have equal bytes."""
    out = bytearray(obs.version.encode("ascii") + b"\x00")

    def floats(block: Sequence[float]) -> None:
        out.extend(struct.pack(">I", len(block)))
        out.extend(struct.pack(f">{len(block)}f", *block))

    def ints(block: Sequence[int]) -> None:
        out.extend(struct.pack(">I", len(block)))
        out.extend(struct.pack(f">{len(block)}i", *block))

    floats(obs.global_features)
    for rows, width in ((obs.node_features, NODE_DIM), (obs.edge_features, EDGE_DIM), (obs.foreign_features, FOREIGN_DIM),
                        (obs.cohort_features, COHORT_DIM), (obs.sighting_features, SIGHTING_DIM), (obs.event_features, EVENT_DIM)):
        out.extend(struct.pack(">II", len(rows), width))
        for row in rows:
            out.extend(struct.pack(f">{width}f", *row))
    for rows in (obs.node_ints, obs.edge_slots, obs.cohort_slots, obs.sighting_ints, obs.event_ints):
        out.extend(struct.pack(">I", len(rows)))
        for row in rows:
            ints(row)
    ints(obs.foreign_nest_slots)
    f = obs.facts
    ints((f.tick, f.decision_index, f.ticks_since_last_decision, f.food, f.burn_tenths, f.workers, f.soldiers, f.idle_workers_home,
          f.idle_soldiers_home, f.scouts_out, f.brood_workers, f.brood_soldiers, f.in_combat_now, f.cohort_overflow))
    return bytes(out)


def observation_hash(obs: ObservationE1) -> str:
    return domain_hash(OBSERVATION_HASH_DOMAIN, observation_bytes(obs))


# ------------------------------------------------------------------------------------------------ rolling bookkeeping


def next_rolling(rolling: RollingE1, window: WindowStatsE1, workers: int, soldiers: int, delta: int) -> RollingE1:
    """Close the window of the last ``delta`` ticks: integer EMAs (weight 1/4) of income and burn per tick and of the population.

    Contracts 9.1: **income** is the food actually delivered into the own nest (``window.delivered``; harvested food still on the workers is not
    spendable), **burn** is the upkeep *due* (``window.upkeep_due``; it stays positive while the colony starves, unlike the upkeep paid).
    Cadence limit: these EMAs advance once per decision window (30 ticks, 10 in an alert), not once per world tick as the contract's reward
    signal does; the per-tick viability used as PPO reward is computed separately in the E3 reward task (FOR-49), outside the observation.
    """
    if delta <= 0:
        return RollingE1(rolling.income_ema_q, rolling.burn_ema_q, rolling.pop_ema_q)
    income = window.delivered * Q // delta
    burn = window.upkeep_due * Q // delta
    pop = (2 * workers + soldiers) * Q // 2
    return RollingE1((3 * rolling.income_ema_q + income) // 4, (3 * rolling.burn_ema_q + burn) // 4, (3 * rolling.pop_ema_q + pop) // 4)


RUNWAY_CAP_TICKS = 40  # ``security`` saturates at this many ticks of runway (contracts 9.1)


def net_runway_feature(food: int, rolling: RollingE1) -> float:
    """``security`` input of contracts 9.1: ``min(1, food / (burn_ema - income_ema) / 40)``; 1.0 when the income covers the burn.

    Own data only: the colony's stored food and its own two EMAs. Exact integers and one correctly rounded division.
    """
    net_q = rolling.burn_ema_q - rolling.income_ema_q
    if net_q <= 0:
        return 1.0
    return _clip01(max(0, food) * Q / (net_q * RUNWAY_CAP_TICKS))


# ------------------------------------------------------------------------------------------------ builder


def _known_distances(memory, source_slot: int) -> list[int]:
    """Shortest known path (ticks) from a slot over the edges the colony knows; -1 if there is none. Slots only, no world ids."""
    adjacency: list[list[tuple[int, int]]] = [[] for _ in memory.nodes]
    for edge in memory.edges:
        adjacency[edge.slot_a].append((edge.slot_b, edge.length))
        adjacency[edge.slot_b].append((edge.slot_a, edge.length))
    dist = [-1] * len(memory.nodes)
    dist[source_slot] = 0
    queue = [(0, source_slot)]
    while queue:
        d, slot = heapq.heappop(queue)
        if d != dist[slot]:
            continue
        for other, length in adjacency[slot]:
            candidate = d + length
            if dist[other] < 0 or candidate < dist[other]:
                dist[other] = candidate
                heapq.heappush(queue, (candidate, other))
    return dist


def build_observation(colony: ColonyStateE1, cohorts: Sequence[CohortE1], tick: int, config: ConfigE1, *, decision_index: int,
                      ticks_since_last_decision: int, window: WindowStatsE1 | None = None, rolling: RollingE1 | None = None) -> ObservationE1:
    """Observation of one living colony at ``tick`` from its own state and its own cohorts. ``window``/``rolling`` default to the colony's own."""
    if colony.status != "alive" or colony.queen_hp <= 0:
        raise ObservationError(f"colony {colony.colony_id} has no living queen: its episode is over and it takes no decisions")
    memory, economy = colony.memory, config.economy
    window = colony.window if window is None else window
    rolling = colony.rolling if rolling is None else rolling
    for cohort in cohorts:
        if cohort.colony_id != colony.colony_id:
            raise ObservationError("a cohort of another colony was passed to the observation builder")
    n_nodes = len(memory.nodes)
    if n_nodes > MAX_NODES or len(memory.edges) > MAX_EDGES or len(memory.foreign) > MAX_FOREIGN:
        raise ObservationError(f"the memory ({n_nodes} nodes, {len(memory.edges)} edges) does not fit ObservationE1/v2 (128 / 320)")
    slot_of = {known.node_id: slot for slot, known in enumerate(memory.nodes)}
    home_id = memory.nodes[0].node_id
    path = _known_distances(memory, 0)
    horizon, k = config.horizon, config.decision_k

    # ---- own units
    workers = sum(c.count for c in cohorts if c.caste == "worker")
    soldiers = sum(c.count for c in cohorts if c.caste == "soldier")

    def idle_home(c: CohortE1) -> bool:
        return c.loc.kind == "node" and c.loc.node_id == home_id and c.mission.kind == "home" and c.load == 0

    idle_workers = sum(c.count for c in cohorts if c.caste == "worker" and idle_home(c))
    idle_soldiers = sum(c.count for c in cohorts if c.caste == "soldier" and idle_home(c))
    scouts = sum(c.count for c in cohorts if c.caste == "worker" and c.mission.kind == "scout")
    foragers = sum(c.count for c in cohorts if c.caste == "worker" and c.mission.kind == "forage")
    at_nest_workers = sum(c.count for c in cohorts if c.caste == "worker" and c.loc.kind == "node" and c.loc.node_id == home_id)
    at_nest_soldiers = sum(c.count for c in cohorts if c.caste == "soldier" and c.loc.kind == "node" and c.loc.node_id == home_id)
    here_workers = [0] * n_nodes
    here_soldiers = [0] * n_nodes
    foraging_to = [0] * n_nodes
    for c in cohorts:
        if c.loc.kind == "node" and c.loc.node_id in slot_of:
            (here_workers if c.caste == "worker" else here_soldiers)[slot_of[c.loc.node_id]] += c.count
        if c.caste == "worker" and c.mission.kind == "forage" and c.mission.target_node in slot_of:
            foraging_to[slot_of[c.mission.target_node]] += c.count
    brood_w = sum(b.caste == "worker" for b in colony.brood)
    brood_s = sum(b.caste == "soldier" for b in colony.brood)
    burn_tenths = economy.worker_upkeep_tenths * workers + economy.soldier_upkeep_tenths * soldiers
    alert = any(item.code in _COMBAT_CODES and tick - item.tick <= config.alert_interval for item in memory.events)

    # ---- age ledger
    adults = workers + soldiers
    expiring = 0
    age_sum = 0
    for ledger, lifespan in ((colony.ages.workers, economy.lifespan_worker), (colony.ages.soldiers, economy.lifespan_soldier)):
        for bucket in ledger:
            age_sum += max(0, tick - bucket.bucket_start) * bucket.count
            if lifespan and bucket.bucket_start + lifespan - tick <= 300:
                expiring += bucket.count
    lifespan_ref = max(economy.lifespan_worker, economy.lifespan_soldier)

    # ---- knowledge counters
    visited = [known.visited for known in memory.nodes]
    known_patches = sum(1 for known in memory.nodes if known.visited and known.kind == "patch")
    frontier = sum(1 for known in memory.nodes if not known.visited)
    recent_hostile = (any(item.code in _THREAT_CODES and tick - item.tick <= _RECENT_TICKS for item in memory.events)
                      or any(s.aggression_observed and tick - s.tick <= _RECENT_TICKS for s in memory.sightings))
    next_hatch = min((b.hatch_tick - tick for b in colony.brood), default=0)
    squad_value = {"none": 0.0, "guard": 1 / 3, "attack": 2 / 3, "withdraw": 1.0}

    runway = net_runway_feature(colony.food, rolling)
    global_features = [
        _ratio(tick, horizon),
        _ratio(decision_index, max(1, horizon // k)),
        _ratio(ticks_since_last_decision, k),
        1.0,
        _ratio(colony.queen_hp, economy.queen_hp),
        _ratio(colony.food, economy.storage_cap),
        1.0 if colony.food >= economy.storage_cap else 0.0,
        _clip01(burn_tenths / 10 / 40),
        _clip01(rolling.income_ema_q / Q / 40),
        runway,
        _ratio(colony.starvation_debt, 5),
        _lognorm(workers, 128),
        _lognorm(soldiers, 64),
        _ratio(at_nest_workers, workers),
        _ratio(at_nest_soldiers, soldiers),
        _lognorm(brood_w, 32),
        _lognorm(brood_s, 32),
        _ratio(colony.orders.egg_level, 3),
        _ratio(colony.orders.soldier_share_q, 4),
        _ratio(next_hatch, 30),
        _ratio(idle_workers, workers),
        _lognorm(scouts, 8),
        _ratio(foragers, workers),
        squad_value[colony.orders.squads[0].mission],
        squad_value[colony.orders.squads[1].mission],
        squad_value[colony.orders.squads[2].mission],
        _ratio(n_nodes, MAX_NODES),
        _ratio(known_patches, 40),
        _ratio(len(memory.foreign), MAX_FOREIGN),
        _ratio(frontier, 32),
        _lognorm(window.harvested, 256),
        _lognorm(window.upkeep_paid, 256),
        _lognorm(window.egg_spend, 256),
        _lognorm(window.losses_workers, 64),
        _lognorm(window.losses_soldiers, 64),
        _lognorm(window.kills_observed, 64),
        1.0 if recent_hostile else 0.0,
        1.0 if alert else 0.0,
        _ratio(expiring, adults),
        _ratio(age_sum // adults if adults else 0, lifespan_ref),
    ]
    assert len(global_features) == GLOBAL_DIM

    # ---- nodes
    sightings_by_slot: dict[int, list] = {}
    for s in memory.sightings:
        sightings_by_slot.setdefault(s.node_slot, []).append(s)
    pile_at = {pile.node_slot: pile.stock for pile in memory.piles_seen}
    degree = [0] * n_nodes
    for edge in memory.edges:
        degree[edge.slot_a] += 1
        degree[edge.slot_b] += 1
    node_features: list[tuple[float, ...]] = []
    node_ints: list[tuple[int, int, int, int]] = []
    for slot, known in enumerate(memory.nodes):
        if not known.visited:
            kind_code = KIND_UNKNOWN
        elif known.kind == "patch":
            kind_code = KIND_PATCH
        elif known.kind == "nest":
            kind_code = KIND_OWN_NEST if slot == 0 else KIND_FOREIGN_NEST
        else:
            kind_code = KIND_JUNCTION
        age = tick - known.seen_tick if known.visited and known.seen_tick >= 0 else -1
        local = sightings_by_slot.get(slot, [])
        youngest = max(local, key=lambda s: s.tick) if local else None
        stock_known = known.kind == "patch" and known.last_stock_tick >= 0
        queen_status = {0: 0.0, 1: 0.5, 2: 1.0}[known.queen_seen] if known.kind == "nest" else 0.0
        node_features.append(tuple(_f32(x) for x in (
            1.0 if kind_code == KIND_JUNCTION else 0.0, 1.0 if kind_code == KIND_PATCH else 0.0,
            1.0 if kind_code == KIND_OWN_NEST else 0.0, 1.0 if kind_code == KIND_FOREIGN_NEST else 0.0,
            1.0 if known.visited else 0.0, 1.0 if known.visited and 0 <= age <= 1 else 0.0,
            _ratio(age, 1500) if age >= 0 else 0.0, 0.0 if known.visited else 1.0,
            _ratio(path[slot], 60) if path[slot] > 0 else 0.0,
            _ratio(known.last_stock, known.patch_capacity) if stock_known else 0.0,
            _ratio(known.patch_capacity, 200),
            max(-1.0, min(1.0, known.last_delta_q / 12)) if stock_known else 0.0,
            _lognorm(here_workers[slot], 32), _lognorm(here_soldiers[slot], 16), _lognorm(foraging_to[slot], 32),
            _ratio(youngest.workers_bucket, 7) if youngest else 0.0, _ratio(youngest.soldiers_bucket, 7) if youngest else 0.0,
            _ratio(len({s.colony_slot for s in local}), 3),
            _ratio(tick - youngest.tick, 1500) if youngest else 0.0,
            _ratio(known.danger_q, 255),
            _clip01(known.harvest_ema_q / Q / 12),
            _ratio(pile_at.get(slot, 0), economy.storage_cap),
            _ratio(degree[slot], 6),
            queen_status)))
        node_ints.append((kind_code, 1 if known.visited else 0, age, path[slot]))

    # ---- edges
    edge_features, edge_slots = [], []
    for edge in memory.edges:
        danger = max(memory.nodes[edge.slot_a].danger_q, memory.nodes[edge.slot_b].danger_q)
        edge_features.append(tuple(_f32(x) for x in (
            _ratio(edge.length, 6), _ratio(edge.lane_cap, 12), 1.0 if edge.traversed else 0.0, _clip01(edge.traffic_ema_q / Q),
            _ratio(tick - edge.discovered_tick, 1500), _ratio(danger, 255))))
        edge_slots.append((edge.slot_a, edge.slot_b))

    # ---- foreign colonies (slot = order of first contact)
    foreign_features, foreign_nest_slots = [], []
    for record in memory.foreign:
        nest_known = record.nest_node_slot >= 0
        queen = {0: 0.0, 1: 0.5, 2: 1.0}[memory.nodes[record.nest_node_slot].queen_seen] if nest_known else 0.0
        foreign_features.append(tuple(_f32(x) for x in (
            1.0, _ratio(tick - record.first_contact_tick, horizon), _ratio(tick - record.last_seen_tick, 1500),
            _ratio(record.last_seen_workers_bucket, 7), _ratio(record.last_seen_soldiers_bucket, 7),
            1.0 if nest_known else 0.0, queen, _lognorm(record.they_attacked_q, 16), _lognorm(record.i_attacked_q, 16),
            _lognorm(record.kills_mine_q, 64), _lognorm(record.kills_theirs_q, 64), 0.0)))
        foreign_nest_slots.append(record.nest_node_slot)

    # ---- own cohorts (the 48 largest if there are more; ties keep the lower id; rows in id order)
    ordered = sorted(cohorts, key=lambda c: c.cohort_id)
    chosen = ordered if len(ordered) <= MAX_COHORTS else sorted(sorted(ordered, key=lambda c: (-c.count, c.cohort_id))[:MAX_COHORTS],
                                                               key=lambda c: c.cohort_id)
    cohort_features, cohort_slots = [], []
    for c in chosen:
        unit_hp = economy.worker_hp if c.caste == "worker" else economy.soldier_hp
        moving = c.loc.kind == "edge"
        if moving:
            near = c.loc.edge_to if 2 * (tick - c.loc.depart_tick) >= c.loc.arrive_tick - c.loc.depart_tick else c.loc.edge_from
            node_slot = slot_of.get(near, slot_of.get(c.loc.edge_from, -1))
            eta = max(0, c.loc.arrive_tick - tick)
        else:
            node_slot, eta = slot_of.get(c.loc.node_id, -1), 0
        if node_slot < 0:
            raise ObservationError("an own cohort stands outside the colony's memory")
        mission = tuple(1.0 if c.mission.kind == name else 0.0 for name in MISSION_ORDER)
        cohort_features.append(tuple(_f32(x) for x in (
            1.0 if c.caste == "worker" else 0.0, 1.0 if c.caste == "soldier" else 0.0, _lognorm(c.count, 64),
            _ratio(c.hp_total, c.count * unit_hp), 1.0 if moving else 0.0, _ratio(eta, 60), *mission,
            _ratio(c.load, c.count * economy.carry_load) if c.caste == "worker" else 0.0, PHASE_VALUE[c.mission.phase] / 3)))
        cohort_slots.append((node_slot, slot_of.get(c.mission.target_node, -1)))

    # ---- sightings and local memory events
    sighting_features, sighting_ints = [], []
    for s in memory.sightings:
        age = tick - s.tick
        node = memory.nodes[s.node_slot]
        sighting_features.append(tuple(_f32(x) for x in (
            _ratio(age, 1500), 1.0 if age <= k else 0.0, _ratio(s.workers_bucket, 7), _ratio(s.soldiers_bucket, 7),
            1.0 if node.visited and node.kind == "nest" else 0.0, 1.0 if s.aggression_observed else 0.0)))
        sighting_ints.append((s.node_slot, s.colony_slot, age))
    event_features, event_ints = [], []
    for item in memory.events:
        age = tick - item.tick
        event_features.append(tuple(_f32(x) for x in (
            _ratio(item.code, 12), 1.0 if item.node_slot >= 0 else 0.0, 1.0 if item.colony_slot >= 0 else 0.0, _lognorm(item.magnitude, 64),
            _ratio(age, 1500), 1.0 if age <= k else 0.0, 1.0 if item.code in _THREAT_CODES else 0.0,
            1.0 if item.code == MEM_OWN_QUEEN_DEAD else 0.0)))
        event_ints.append((item.code, item.node_slot, item.colony_slot))

    facts = ObservationFacts(
        tick=tick, decision_index=decision_index, ticks_since_last_decision=ticks_since_last_decision, food=colony.food,
        burn_tenths=burn_tenths, workers=workers, soldiers=soldiers, idle_workers_home=idle_workers, idle_soldiers_home=idle_soldiers,
        scouts_out=scouts, brood_workers=brood_w, brood_soldiers=brood_s, in_combat_now=1 if alert else 0,
        cohort_overflow=len(ordered) - len(chosen))
    return ObservationE1(
        OBSERVATION_VERSION, tuple(_f32(x) for x in global_features), tuple(node_features), tuple(node_ints), tuple(edge_features),
        tuple(edge_slots), tuple(foreign_features), tuple(foreign_nest_slots), tuple(cohort_features), tuple(cohort_slots),
        tuple(sighting_features), tuple(sighting_ints), tuple(event_features), tuple(event_ints), facts)
