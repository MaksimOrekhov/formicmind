"""Structural validation of configs and states (raises ``InvariantError``)."""

from __future__ import annotations

from .config import SimConfig
from .model import NOWHERE, SimState
from .rules import (
    ARMY_GARRISON_HOME,
    ARMY_RAID_ENEMY_HOME,
    CASTES,
    FORAGE_CATEGORIES,
    FORAGE_SUPPORT_MAX,
    LAY_SOLDIER,
    MIRROR,
    NODE_COUNT,
    Q_TOTAL,
    SIDES,
    SOLDIER,
    WORKER,
)
from .versions import RULES_VERSION


class InvariantError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvariantError(message)


def _is_int(value: object) -> bool:
    return type(value) is int


def validate_config(config: SimConfig) -> None:
    _require(config.rules_version == RULES_VERSION, f"unknown rules_version {config.rules_version!r}")
    topo, bal = config.topology, config.balance
    _require(len(topo.nodes) == NODE_COUNT, "topology must have 7 nodes")
    homes = {}
    for i, node in enumerate(topo.nodes):
        _require(node.node_id == i, "nodes must be ordered by node_id 0..6")
        if node.home_side != -1:
            homes[node.home_side] = i
    _require(sorted(homes) == list(SIDES), "each side must own exactly one home node")
    _require(homes[0] == MIRROR[homes[1]], "homes must be mirror images")
    edges: dict[tuple[int, int], int] = {}
    for edge in topo.edges:
        _require(0 <= edge.a < edge.b < NODE_COUNT, f"bad edge ({edge.a}, {edge.b})")
        _require(edge.length >= 1, "edge length must be positive")
        _require((edge.a, edge.b) not in edges, "duplicate edge")
        edges[(edge.a, edge.b)] = edge.length
    _require(
        list(edges) == sorted(edges), "edges must be sorted by (a, b)"
    )
    for (a, b), length in edges.items():
        ma, mb = sorted((MIRROR[a], MIRROR[b]))
        _require(edges.get((ma, mb)) == length, f"edge ({a}, {b}) has no mirror image")
    _require(len(bal.resources) == NODE_COUNT, "one resource record per node")
    for i, res in enumerate(bal.resources):
        _require(res.node_id == i, "resources must be ordered by node_id")
        _require(
            0 <= res.start_food <= res.capacity and res.regen >= 0 and res.max_harvest >= 0,
            f"bad resource record for node {i}",
        )
        mirrored = bal.resources[MIRROR[i]]
        _require(
            (res.start_food, res.capacity, res.regen, res.max_harvest)
            == (mirrored.start_food, mirrored.capacity, mirrored.regen, mirrored.max_harvest),
            f"resource of node {i} is not mirror-symmetric",
        )
    positives = (
        bal.worker_hp, bal.soldier_hp, bal.queen_hp, bal.worker_egg_cost, bal.soldier_egg_cost,
        bal.eggs_per_lay, bal.lay_interval_ticks, bal.worker_hatch_delay, bal.soldier_hatch_delay,
        bal.starvation_unit_cost, bal.home_defense_den, config.policy_interval_k, config.t_max,
    )
    _require(all(_is_int(v) and v >= 1 for v in positives), "balance/time constants must be positive ints")
    nonnegatives = (
        bal.start_food, bal.start_workers, bal.start_soldiers, bal.worker_upkeep_tenths,
        bal.soldier_upkeep_tenths, bal.soldier_damage, bal.home_defense_num,
    )
    _require(all(_is_int(v) and v >= 0 for v in nonnegatives), "balance constants must be non-negative ints")


def validate_state(state: SimState) -> None:
    config = state.config
    bal = config.balance
    _require(_is_int(state.tick) and state.tick >= 0, "tick must be a non-negative int")
    _require(len(state.node_food) == NODE_COUNT, "node_food must have 7 entries")
    for i, food in enumerate(state.node_food):
        _require(_is_int(food) and 0 <= food <= bal.resources[i].capacity, f"node {i} food out of range")
    _require(len(state.colonies) == len(SIDES), "exactly two colonies")

    seen_ids: set[int] = set()
    edge_set = {(e.a, e.b) for e in config.topology.edges}
    max_hp = {WORKER: bal.worker_hp, SOLDIER: bal.soldier_hp}
    hatch_delay = {WORKER: bal.worker_hatch_delay, SOLDIER: bal.soldier_hatch_delay}

    for index, colony in enumerate(state.colonies):
        tag = f"colony {index}"
        _require(colony.side == index, f"{tag}: side must equal its index")
        for name in ("food", "next_lay_tick", "starvation_debt", "upkeep_remainder10", "queen_hp"):
            _require(_is_int(getattr(colony, name)), f"{tag}: {name} must be an int")
        _require(colony.food >= 0, f"{tag}: negative food")
        _require(0 <= colony.queen_hp <= bal.queen_hp, f"{tag}: queen_hp out of range")
        _require(colony.next_lay_tick >= 0, f"{tag}: negative next_lay_tick")
        _require(0 <= colony.upkeep_remainder10 <= 9, f"{tag}: upkeep_remainder10 must be 0..9")
        _require(colony.starvation_debt >= 0, f"{tag}: negative starvation_debt")
        _require(_is_int(colony.lay_order) and 0 <= colony.lay_order <= LAY_SOLDIER, f"{tag}: bad lay_order")
        q = colony.forage_q
        _require(
            len(q) == FORAGE_CATEGORIES and all(_is_int(v) and v >= 0 for v in q),
            f"{tag}: forage_q must be 6 non-negative ints",
        )
        _require(sum(q) == Q_TOTAL, f"{tag}: forage_q must sum to {Q_TOTAL}")
        _require(1 <= sum(1 for v in q if v > 0) <= FORAGE_SUPPORT_MAX, f"{tag}: forage_q support must be 1..3")
        _require(
            _is_int(colony.army_choice) and ARMY_GARRISON_HOME <= colony.army_choice <= ARMY_RAID_ENEMY_HOME,
            f"{tag}: bad army_choice",
        )
        _require(type(colony.avoid_center) is bool, f"{tag}: avoid_center must be a bool")
        _require(
            not (colony.army_choice == ARMY_GARRISON_HOME and colony.avoid_center),
            f"{tag}: garrison_home cannot set avoid_center",
        )

        _require(
            [u.unit_id for u in colony.units] == sorted({u.unit_id for u in colony.units}),
            f"{tag}: units must be strictly ordered by unit_id",
        )
        for u in colony.units:
            _require(
                all(_is_int(v) for v in (u.unit_id, u.caste, u.hp, u.node, u.from_node, u.to_node,
                                         u.arrival_tick, u.target_node)),
                f"{tag}: unit {u.unit_id} has non-int fields",
            )
            _require(u.caste in CASTES, f"{tag}: unit {u.unit_id} has bad caste")
            _require(1 <= u.hp <= max_hp[u.caste], f"{tag}: unit {u.unit_id} hp out of range")
            _require(0 <= u.target_node < NODE_COUNT, f"{tag}: unit {u.unit_id} bad target")
            if u.node != NOWHERE:
                _require(0 <= u.node < NODE_COUNT, f"{tag}: unit {u.unit_id} bad node")
                _require(
                    (u.from_node, u.to_node, u.arrival_tick) == (NOWHERE, NOWHERE, NOWHERE),
                    f"{tag}: stationed unit {u.unit_id} has transit fields",
                )
            else:
                _require(
                    (min(u.from_node, u.to_node), max(u.from_node, u.to_node)) in edge_set,
                    f"{tag}: unit {u.unit_id} is in transit along a non-edge",
                )
                _require(u.arrival_tick >= state.tick, f"{tag}: unit {u.unit_id} arrival in the past")
            seen_ids.add(u.unit_id)

        _require(
            [b.unit_id for b in colony.brood] == sorted({b.unit_id for b in colony.brood}),
            f"{tag}: brood must be strictly ordered by unit_id",
        )
        for b in colony.brood:
            _require(
                all(_is_int(v) for v in (b.unit_id, b.caste, b.laid_tick, b.hatch_tick)),
                f"{tag}: brood {b.unit_id} has non-int fields",
            )
            _require(b.caste in CASTES, f"{tag}: brood {b.unit_id} has bad caste")
            _require(
                0 <= b.laid_tick <= state.tick and b.hatch_tick == b.laid_tick + hatch_delay[b.caste],
                f"{tag}: brood {b.unit_id} has inconsistent ticks",
            )
            _require(b.hatch_tick >= state.tick, f"{tag}: brood {b.unit_id} missed its hatch tick")
            seen_ids.add(b.unit_id)

        intel = colony.intel
        _require(len(intel.nodes) == NODE_COUNT, f"{tag}: intel must cover 7 nodes")
        for i, entry in enumerate(intel.nodes):
            values = (entry.seen_tick, entry.food_last, entry.enemy_workers_last, entry.enemy_soldiers_last)
            _require(all(_is_int(v) for v in values), f"{tag}: intel node {i} has non-int fields")
            _require(
                entry.food_last >= 0 and entry.enemy_workers_last >= 0 and entry.enemy_soldiers_last >= 0,
                f"{tag}: intel node {i} has negative values",
            )
            _require(NOWHERE <= entry.seen_tick <= state.tick, f"{tag}: intel node {i} bad seen_tick")
            if entry.seen_tick == NOWHERE:
                _require(values[1:] == (0, 0, 0), f"{tag}: unseen intel node {i} must hold zeros")
        _require(
            _is_int(intel.enemy_queen_hp_last) and _is_int(intel.enemy_queen_seen_tick),
            f"{tag}: intel queen fields must be ints",
        )
        _require(
            0 <= intel.enemy_queen_hp_last <= bal.queen_hp
            and NOWHERE <= intel.enemy_queen_seen_tick <= state.tick,
            f"{tag}: intel queen fields out of range",
        )
        if intel.enemy_queen_seen_tick == NOWHERE:
            _require(intel.enemy_queen_hp_last == 0, f"{tag}: unseen enemy queen must hold 0")

    total = sum(len(c.units) + len(c.brood) for c in state.colonies)
    _require(len(seen_ids) == total, "unit ids must be unique across both colonies and brood")
    _require(_is_int(state.next_unit_id), "next_unit_id must be an int")
    _require(all(1 <= i < state.next_unit_id for i in seen_ids), "unit id outside [1, next_unit_id)")
