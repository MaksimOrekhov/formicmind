"""Prototype rule constants and the default config (FOR-1 v0.2.1 candidates).

Node ids follow the FOR-2 canonical order for side A: ``[A, N1, N2, C, N4, N5, B]``
= ``0..6``. Side B sees the mirrored order ``[B, N4, N5, C, N1, N2, A]``.
"""

from __future__ import annotations

from .config import (
    CandidateBalance,
    EdgeSpec,
    NodeResource,
    NodeSpec,
    SimConfig,
    Topology,
)
from .versions import BALANCE_VERSION, RULES_VERSION

SIDE_A = 0
SIDE_B = 1
SIDES = (SIDE_A, SIDE_B)

WORKER = 0
SOLDIER = 1
CASTES = (WORKER, SOLDIER)

LAY_NONE = 0
LAY_WORKER = 1
LAY_SOLDIER = 2

# army_choice: 0 garrison_home, 1..5 defend(local node 1..5), 6 raid(enemy_home)
ARMY_GARRISON_HOME = 0
ARMY_RAID_ENEMY_HOME = 6

NODE_COUNT = 7
NODE_NAMES = ("A", "N1", "N2", "C", "N4", "N5", "B")
NODE_A, NODE_N1, NODE_N2, NODE_C, NODE_N4, NODE_N5, NODE_B = range(NODE_COUNT)

# Side mirror: node n of side A corresponds to MIRROR[n] of side B (an involution).
MIRROR = (NODE_B, NODE_N4, NODE_N5, NODE_C, NODE_N1, NODE_N2, NODE_A)

FORAGE_CATEGORIES = 6  # home, N1, N2, C, N4, N5 (local frame)
FORAGE_SUPPORT_MAX = 3
Q_TOTAL = 65535  # FOR-2: forage shares are integers summing to Q


def mirror_node(node: int) -> int:
    return MIRROR[node]


def local_to_world(side: int, local_node: int) -> int:
    """Map a node index of the side's own canonical frame to a world node id."""
    return local_node if side == SIDE_A else MIRROR[local_node]


def default_topology() -> Topology:
    nodes = tuple(
        NodeSpec(
            node_id=i,
            name=NODE_NAMES[i],
            home_side=SIDE_A if i == NODE_A else SIDE_B if i == NODE_B else -1,
        )
        for i in range(NODE_COUNT)
    )
    lengths = {
        (NODE_A, NODE_N1): 2,
        (NODE_N4, NODE_B): 2,
        (NODE_A, NODE_N2): 3,
        (NODE_N5, NODE_B): 3,
        (NODE_N1, NODE_C): 2,
        (NODE_C, NODE_N4): 2,
        (NODE_N2, NODE_C): 3,
        (NODE_C, NODE_N5): 3,
        (NODE_N2, NODE_N5): 4,
    }
    edges = tuple(
        EdgeSpec(a=min(a, b), b=max(a, b), length=length)
        for (a, b), length in sorted(lengths.items(), key=lambda kv: tuple(sorted(kv[0])))
    )
    return Topology(nodes=nodes, edges=edges)


def default_balance() -> CandidateBalance:
    # (start_food, capacity, regen, max_harvest) per node; FOR-1 v0.2.1 section 3.
    table = {
        NODE_A: (0, 0, 0, 0),
        NODE_N1: (60, 80, 8, 8),
        NODE_N2: (40, 60, 8, 8),
        NODE_C: (80, 120, 12, 12),
        NODE_N4: (60, 80, 8, 8),
        NODE_N5: (40, 60, 8, 8),
        NODE_B: (0, 0, 0, 0),
    }
    resources = tuple(
        NodeResource(node_id=n, start_food=s, capacity=c, regen=r, max_harvest=h)
        for n, (s, c, r, h) in sorted(table.items())
    )
    return CandidateBalance(
        resources=resources,
        start_food=60,
        start_workers=12,
        start_soldiers=4,
        worker_hp=20,
        soldier_hp=30,
        queen_hp=240,
        worker_egg_cost=10,
        soldier_egg_cost=20,
        eggs_per_lay=1,
        lay_interval_ticks=10,
        worker_hatch_delay=20,
        soldier_hatch_delay=30,
        worker_upkeep_tenths=4,
        soldier_upkeep_tenths=8,
        starvation_unit_cost=5,
        soldier_damage=2,
        home_defense_num=5,
        home_defense_den=4,
    )


def default_config() -> SimConfig:
    return SimConfig(
        rules_version=RULES_VERSION,
        balance_version=BALANCE_VERSION,
        topology=default_topology(),
        balance=default_balance(),
        policy_interval_k=30,
        t_max=3000,
    )
