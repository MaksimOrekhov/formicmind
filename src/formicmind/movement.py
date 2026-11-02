"""Arrival and dispatch (FOR-1 v0.2.1 errata A/E; audit D1a/D1b).

* ``arrive_units`` (phase Arrival): every unit whose ``arrival_tick == tick`` becomes
  stationed at the edge's far node. A unit still in transit with an earlier arrival tick is
  a corrupted state and raises.
* ``dispatch_available`` (Policy start and AutomaticDepart): every *stationed* unit whose
  ``target_node`` differs from its node and that is not blocked leaves along the first
  edge of its canonical route; departure at tick ``t`` over an edge of length ``d`` arrives
  at ``t + d`` (no extra transfer tick). A unit that arrived this tick at an intermediate
  node therefore continues in the same tick's AutomaticDepart (D1a); nothing depends on
  whether a route was "started".
* **Blocked**: a unit standing in a node that holds at least one stationed enemy soldier
  does not leave (own soldiers do not help). Availability is evaluated on the state before
  the dispatch step for both sides at once, so the result does not depend on side order.
  Combat does not exist yet, so enemy soldiers never die: two armies that meet stay
  blocked until a fixture changes the state (see ``docs/tick_movement.md``).
* Routing is recomputed from the unit's current node and ``target_node`` every time; with
  ``avoid_center`` the soldier's *current* node may still be C (D1b).
"""

from __future__ import annotations

from .events import ArriveEvent, DepartEvent, Phase
from .model import NOWHERE, SimState
from .routing import edge_length, next_hop
from .rules import SIDES, SOLDIER


def arrive_units(state: SimState, phase: int = Phase.ARRIVAL) -> list[ArriveEvent]:
    tick = state.tick
    arrived: list[ArriveEvent] = []
    for side in SIDES:
        for unit in state.colonies[side].units:
            if unit.node != NOWHERE:
                continue
            if unit.arrival_tick < tick:
                raise ValueError(f"unit {unit.unit_id} missed its arrival tick {unit.arrival_tick}")
            if unit.arrival_tick == tick:
                arrived.append(ArriveEvent(
                    tick=tick, phase=int(phase), side=side, unit_id=unit.unit_id, caste=unit.caste,
                    node=unit.to_node, from_node=unit.from_node, target_node=unit.target_node))
    arrived.sort(key=lambda e: e.unit_id)
    index = {u.unit_id: u for c in state.colonies for u in c.units}
    for event in arrived:
        unit = index[event.unit_id]
        unit.node, unit.from_node, unit.to_node, unit.arrival_tick = event.node, NOWHERE, NOWHERE, NOWHERE
    return arrived


def enemy_soldier_nodes(state: SimState, side: int) -> frozenset[int]:
    """Nodes holding at least one stationed soldier of the *other* side."""
    return frozenset(
        u.node for u in state.colonies[1 - side].units if u.caste == SOLDIER and u.node != NOWHERE
    )


def dispatch_available(state: SimState, phase: int) -> list[DepartEvent]:
    tick = state.tick
    topology = state.config.topology
    departures: list[DepartEvent] = []
    for side in SIDES:
        colony = state.colonies[side]
        blocked = enemy_soldier_nodes(state, side)
        for unit in colony.units:
            if unit.node == NOWHERE or unit.node == unit.target_node or unit.node in blocked:
                continue
            avoid = colony.avoid_center and unit.caste == SOLDIER
            hop = next_hop(topology, unit.node, unit.target_node, avoid_center=avoid)
            departures.append(DepartEvent(
                tick=tick, phase=int(phase), side=side, unit_id=unit.unit_id, caste=unit.caste,
                from_node=unit.node, to_node=hop,
                arrival_tick=tick + edge_length(topology, unit.node, hop),
                target_node=unit.target_node))
    departures.sort(key=lambda e: e.unit_id)
    index = {u.unit_id: u for c in state.colonies for u in c.units}
    for event in departures:
        unit = index[event.unit_id]
        unit.node, unit.from_node, unit.to_node, unit.arrival_tick = (
            NOWHERE, event.from_node, event.to_node, event.arrival_tick)
    return departures
