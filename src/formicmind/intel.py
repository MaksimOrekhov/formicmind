"""Private intel updates (FOR-2 errata v0.2.1, P0-1 and P1-4).

``update_local_intel`` is the only writer of ``IntelMemory``. It records what a side
is allowed to sense *now* and changes nothing else in the state:

* the side's own home node, always (permanent sensor, even with zero adults);
* every node where the side has at least one stationed adult.

For each such node it stores the exact food stock and the enemy worker/soldier
counts stationed there (enemy units on edges are never visible). If the enemy home
is among the observed nodes, the enemy queen HP is stored too. Nothing else is read:
no unobserved node, no enemy orders, brood, food or RNG.

Called by the tick engine (a later PR) at the start of every tick before the policy
snapshot (``PRE_POLICY``) and after Arrival, before Combat
(``POST_ARRIVAL_PRE_COMBAT``). Both phases apply the same sensor rule; the phase
only documents the call site. A region whose last observer dies in Combat keeps its
pre-combat record and ages, because no later call observes it.
"""

from __future__ import annotations

import enum

from .model import NOWHERE, SimState
from .rules import SIDES, WORKER


class IntelPhase(enum.Enum):
    PRE_POLICY = "PRE_POLICY"
    POST_ARRIVAL_PRE_COMBAT = "POST_ARRIVAL_PRE_COMBAT"


def home_node(state: SimState, side: int) -> int:
    return next(n.node_id for n in state.config.topology.nodes if n.home_side == side)


def observed_nodes(state: SimState, side: int) -> list[int]:
    """Nodes the side senses at this moment, ascending."""
    nodes = {home_node(state, side)}
    nodes.update(u.node for u in state.colonies[side].units if u.node != NOWHERE)
    return sorted(nodes)


def update_local_intel(state: SimState, side: int, phase: IntelPhase) -> None:
    if not isinstance(phase, IntelPhase):
        raise TypeError("phase must be an IntelPhase")
    if side not in SIDES:
        raise ValueError(f"bad side {side!r}")
    colony = state.colonies[side]
    enemy = state.colonies[1 - side]
    observed = observed_nodes(state, side)

    workers: dict[int, int] = {}
    soldiers: dict[int, int] = {}
    for unit in enemy.units:
        if unit.node == NOWHERE or unit.node not in observed:
            continue
        bucket = workers if unit.caste == WORKER else soldiers
        bucket[unit.node] = bucket.get(unit.node, 0) + 1

    for node in observed:
        entry = colony.intel.nodes[node]
        if entry.seen_tick > state.tick:
            raise ValueError(f"intel of node {node} is newer than the state tick")
        entry.seen_tick = state.tick
        entry.food_last = state.node_food[node]
        entry.enemy_workers_last = workers.get(node, 0)
        entry.enemy_soldiers_last = soldiers.get(node, 0)

    if home_node(state, 1 - side) in observed:
        colony.intel.enemy_queen_hp_last = enemy.queen_hp
        colony.intel.enemy_queen_seen_tick = state.tick
