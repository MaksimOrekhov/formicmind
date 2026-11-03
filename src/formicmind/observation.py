"""ObservationV2: 223 float32 little-endian values (FOR-2 v0.2 section 3 + errata v0.2.1).

``build_observation`` is a pure read of one frozen ``PolicySnapshot``; it never
touches ``IntelMemory`` or the state. Every per-node array is in the side's own
canonical frame (node k of side B is world node ``MIRROR[k]``), so mirrored
situations give identical vectors.

Normalizers are constants of this observation version. ``check_observation_config``
refuses configs they were not designed for (FOR-2: a change of capacity/regen/queen
HP means a new observation version and a new golden vector).

Values are computed in float64 and rounded once to float32 (stored as the exact
float32 value in a Python float).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum

from .config import SimConfig
from .model import NOWHERE
from .rules import CASTES, NODE_COUNT, Q_TOTAL, SOLDIER, WORKER, local_to_world
from .snapshot import PolicySnapshot, SideView

OBS_V2_LENGTH = 223

# (name, length) in flatten order; offsets are cumulative and half-open.
OBS_V2_BLOCKS: tuple[tuple[str, int], ...] = (
    ("time", 2),
    ("public_rules", 10),
    ("own_global", 18),
    ("current_orders", 17),
    ("public_edges", 21),
    ("public_food_capacity", 7),
    ("public_food_regen", 7),
    ("own_node_count", 14),
    ("own_node_hp_sum", 14),
    ("nearest_arrival_eta", 14),
    ("nearest_arrival_count", 14),
    ("nearest_arrival_hp", 14),
    ("transit_total_count", 2),
    ("transit_total_hp", 2),
    ("last_food_ratio", 7),
    ("food_known", 7),
    ("food_current", 7),
    ("food_age_norm", 7),
    ("enemy_count_last", 14),
    ("enemy_known", 7),
    ("enemy_current", 7),
    ("enemy_age_norm", 7),
    ("enemy_queen_hp_last", 1),
    ("enemy_queen_known", 1),
    ("enemy_queen_current", 1),
    ("enemy_queen_age_norm", 1),
)


def _offsets() -> dict[str, tuple[int, int]]:
    table, offset = {}, 0
    for name, length in OBS_V2_BLOCKS:
        table[name] = (offset, length)
        offset += length
    return table


OBS_V2_OFFSETS: dict[str, tuple[int, int]] = _offsets()
assert sum(length for _, length in OBS_V2_BLOCKS) == OBS_V2_LENGTH


class OwnGlobalIndex(IntEnum):
    """Именованные позиции блока own_global без изменения ObservationV2."""

    FOOD = 0
    WORKERS = 1
    SOLDIERS = 2
    WORKER_HP_SUM = 3
    SOLDIER_HP_SUM = 4
    QUEEN_HP = 5
    WORKER_BROOD = 6
    SOLDIER_BROOD = 7
    WORKER_BROOD_ETA = 8
    SOLDIER_BROOD_ETA = 9
    LAY_COOLDOWN = 10
    UPKEEP_REMAINDER = 11
    STARVATION_DEBT = 12
    HARVEST = 13
    UPKEEP_PAID = 14
    EGG_SPEND = 15
    STARVATION_DELTA = 16
    WINDOW_TICKS = 17


@dataclass(frozen=True, slots=True)
class OwnGlobalObservation:
    """Типизированный вид own_global; нормализованные значения float32 не меняются."""

    food: float
    workers: float
    soldiers: float
    worker_hp_sum: float
    soldier_hp_sum: float
    queen_hp: float
    worker_brood: float
    soldier_brood: float
    worker_brood_eta: float
    soldier_brood_eta: float
    lay_cooldown: float
    upkeep_remainder: float
    starvation_debt: float
    harvest: float
    upkeep_paid: float
    egg_spend: float
    starvation_delta: float
    window_ticks: float

# Normalizer constants (FOR-2 v0.2 section 3.3).
_EDGE_NORM = 4
_CAPACITY_NORM = 120
_REGEN_NORM = 12
_QUEEN_HP_NORM = 240


class ObservationConfigError(ValueError):
    pass


def check_observation_config(config: SimConfig) -> None:
    bal = config.balance
    if bal.queen_hp != _QUEEN_HP_NORM:
        raise ObservationConfigError("queen_hp differs from the ObservationV2 normalizer")
    if any(r.capacity > _CAPACITY_NORM or r.regen > _REGEN_NORM for r in bal.resources):
        raise ObservationConfigError("capacity/regen exceed the ObservationV2 normalizers")
    if any(e.length > _EDGE_NORM for e in config.topology.edges):
        raise ObservationConfigError("edge length exceeds the ObservationV2 normalizer")


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _u(x: int | float, scale: int | float) -> float:
    """U(x; S) = x / (x + S) for x >= 0."""
    return x / (x + scale)


def _sgn_u(x: int, scale: int) -> float:
    return -_u(-x, scale) if x < 0 else _u(x, scale)


@dataclass(frozen=True, slots=True)
class ObservationV2:
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if type(self.values) is not tuple or len(self.values) != OBS_V2_LENGTH:
            raise ValueError(f"ObservationV2 needs a tuple of {OBS_V2_LENGTH} values")
        for v in self.values:
            if type(v) is not float or v != v or v in (float("inf"), float("-inf")) or f32(v) != v:
                raise ValueError("ObservationV2 values must be finite float32 numbers")

    def block(self, name: str) -> tuple[float, ...]:
        offset, length = OBS_V2_OFFSETS[name]
        return self.values[offset:offset + length]

    @property
    def own_global(self) -> OwnGlobalObservation:
        values = self.block("own_global")
        return OwnGlobalObservation(*(values[index] for index in OwnGlobalIndex))

    def enemy_counts(self, node: int) -> tuple[float, float]:
        """Вернуть нормализованные счётчики рабочих и солдат в локальном узле."""
        if type(node) is not int or not 0 <= node < 7:
            raise IndexError("enemy node must be in 0..6")
        counts = self.block("enemy_count_last")
        return counts[2 * node], counts[2 * node + 1]

    @property
    def enemy_queen_known(self) -> bool:
        return self.block("enemy_queen_known")[0] > 0.0

    @property
    def enemy_queen_current(self) -> bool:
        return self.block("enemy_queen_current")[0] > 0.0


def flatten_v2(obs: ObservationV2) -> bytes:
    return struct.pack(f"<{OBS_V2_LENGTH}f", *obs.values)


def observation_from_bytes(data: bytes) -> ObservationV2:
    if len(data) != 4 * OBS_V2_LENGTH:
        raise ValueError(f"expected {4 * OBS_V2_LENGTH} bytes")
    return ObservationV2(tuple(struct.unpack(f"<{OBS_V2_LENGTH}f", data)))


def _age_norm(seen_tick: int, tick: int, t_max: int) -> float:
    return 1.0 if seen_tick == NOWHERE else min(tick - seen_tick, t_max) / t_max


def build_observation(snapshot: PolicySnapshot, side: int) -> ObservationV2:
    config = snapshot.config
    bal = config.balance
    view: SideView = snapshot.view(side)
    tick, t_max, k_pol = snapshot.tick, config.t_max, config.policy_interval_k
    world = [local_to_world(side, k) for k in range(NODE_COUNT)]  # local -> world
    local = {w: k for k, w in enumerate(world)}                    # world -> local

    # own units, grouped
    count = [[0, 0] for _ in range(NODE_COUNT)]  # by local node, caste
    hp_sum = [[0, 0] for _ in range(NODE_COUNT)]
    transit_count, transit_hp = [0, 0], [0, 0]
    arrival: dict[tuple[int, int], list[int]] = {}  # (local dest, caste) -> [eta, count, hp]
    total_count, total_hp = [0, 0], [0, 0]
    for u in view.units:
        total_count[u.caste] += 1
        total_hp[u.caste] += u.hp
        if u.node != NOWHERE:
            k = local[u.node]
            count[k][u.caste] += 1
            hp_sum[k][u.caste] += u.hp
        else:
            transit_count[u.caste] += 1
            transit_hp[u.caste] += u.hp
            key = (local[u.to_node], u.caste)
            eta = u.arrival_tick - tick
            best = arrival.get(key)
            if best is None or eta < best[0]:
                arrival[key] = [eta, 1, u.hp]
            elif eta == best[0]:
                best[1] += 1
                best[2] += u.hp

    brood_count = [0, 0]
    brood_eta = [None, None]
    for b in view.brood:
        brood_count[b.caste] += 1
        eta = max(b.hatch_tick - tick, 0)
        if brood_eta[b.caste] is None or eta < brood_eta[b.caste]:
            brood_eta[b.caste] = eta

    w = view.window
    out: list[float] = []

    # time
    out += [tick / max(t_max - 1, 1), (t_max - tick) / t_max]
    # public_rules
    out += [
        k_pol / t_max, _u(t_max, 3000),
        _u(bal.worker_egg_cost, 100), _u(bal.soldier_egg_cost, 100),
        _u(bal.eggs_per_lay, 8), _u(bal.lay_interval_ticks, 30),
        _u(bal.worker_hatch_delay, 60), _u(bal.soldier_hatch_delay, 60),
        bal.worker_upkeep_tenths / 10, bal.soldier_upkeep_tenths / 10,
    ]
    # own_global
    out += [
        _u(view.food, 250),
        _u(total_count[WORKER], 64), _u(total_count[SOLDIER], 64),
        _u(total_hp[WORKER], 1200), _u(total_hp[SOLDIER], 1200),
        view.queen_hp / _QUEEN_HP_NORM,
        _u(brood_count[WORKER], 32), _u(brood_count[SOLDIER], 32),
        (brood_eta[WORKER] or 0) / t_max, (brood_eta[SOLDIER] or 0) / t_max,
        min(max(view.next_lay_tick - tick, 0) / max(bal.lay_interval_ticks, 1), 1),
        view.upkeep_remainder10 / 10,
        _u(view.starvation_debt, 10),
        _u(w.harvest, 300), _u(w.upkeep_paid, 300), _u(w.egg_spend, 300),
        _sgn_u(w.starvation_debt_delta, 300),
        w.window_ticks / k_pol,
    ]
    # current_orders
    out += [1.0 if view.lay_order == i else 0.0 for i in range(3)]
    out += [q / Q_TOTAL for q in view.forage_q]
    out += [1.0 if view.army_choice == i else 0.0 for i in range(7)]
    out += [1.0 if view.avoid_center else 0.0]
    # public_edges: pairs (i, j), i < j, lexicographic, local frame
    length = {(e.a, e.b): e.length for e in config.topology.edges}
    for i in range(NODE_COUNT):
        for j in range(i + 1, NODE_COUNT):
            a, b = sorted((world[i], world[j]))
            out.append(length.get((a, b), 0) / _EDGE_NORM)
    # public_food_capacity / regen
    out += [bal.resources[world[k]].capacity / _CAPACITY_NORM for k in range(NODE_COUNT)]
    out += [bal.resources[world[k]].regen / _REGEN_NORM for k in range(NODE_COUNT)]
    # own_node_count / own_node_hp_sum
    out += [_u(count[k][c], 64) for k in range(NODE_COUNT) for c in CASTES]
    out += [_u(hp_sum[k][c], 1200) for k in range(NODE_COUNT) for c in CASTES]
    # nearest_arrival (keyed by next-edge destination)
    def nearest(slot: int, scale: int | None) -> list[float]:
        vals = []
        for k in range(NODE_COUNT):
            for c in CASTES:
                rec = arrival.get((k, c))
                if rec is None:
                    vals.append(0.0)
                elif scale is None:
                    vals.append(rec[0] / _EDGE_NORM)
                else:
                    vals.append(_u(rec[slot], scale))
        return vals
    out += nearest(0, None) + nearest(1, 64) + nearest(2, 1200)
    # transit totals
    out += [_u(transit_count[c], 64) for c in CASTES]
    out += [_u(transit_hp[c], 1200) for c in CASTES]
    # intel (private memory, own frame)
    intel = [view.intel_nodes[world[k]] for k in range(NODE_COUNT)]
    out += [
        (n.food_last / bal.resources[world[k]].capacity if bal.resources[world[k]].capacity else 0.0)
        for k, n in enumerate(intel)
    ]
    out += [1.0 if n.seen_tick != NOWHERE else 0.0 for n in intel]          # food_known
    out += [1.0 if n.seen_tick == tick else 0.0 for n in intel]             # food_current
    out += [_age_norm(n.seen_tick, tick, t_max) for n in intel]             # food_age_norm
    out += [_u(x, 64) for n in intel for x in (n.enemy_workers_last, n.enemy_soldiers_last)]
    out += [1.0 if n.seen_tick != NOWHERE else 0.0 for n in intel]          # enemy_known
    out += [1.0 if n.seen_tick == tick else 0.0 for n in intel]             # enemy_current
    out += [_age_norm(n.seen_tick, tick, t_max) for n in intel]             # enemy_age_norm
    # enemy queen
    seen = view.enemy_queen_seen_tick
    out += [
        view.enemy_queen_hp_last / _QUEEN_HP_NORM,
        1.0 if seen != NOWHERE else 0.0,
        1.0 if seen == tick else 0.0,
        _age_norm(seen, tick, t_max),
    ]

    return ObservationV2(tuple(f32(float(x)) for x in out))
