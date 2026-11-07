"""Colony memory: the sensor phase and routing over what a colony knows (contracts section 4.3).

The sensor phase is the *only* writer of ``ColonyMemoryE1`` after tick 0 (plus the combat/queen bookkeeping at the bottom of this
module, which only records what a colony physically took part in). It runs after the arrivals of a tick: every node where a cohort of
the colony stands (and the own nest while the queen lives) becomes visited and visible, shows its incident edges, and reveals the
cohorts of other colonies standing there. Visibility is local; knowledge only ages.
"""

from __future__ import annotations

import heapq

from .events import EventE1, event
from .model import (
    ColonyMemoryE1, ForeignColonyRecord, KnownEdge, KnownNode, KnownPile, MapE1, MemoryEvent, Sighting, WorldStateE1,
)

# Codes of ``MemoryEvent`` (contracts 7.7; the ones the E1-c engine can write).
MEM_FIRST_CONTACT = 1
MEM_ATTACKED = 2
MEM_WE_KILLED = 3
MEM_WE_LOST = 4
MEM_QUEEN_HIT = 5
MEM_OWN_QUEEN_DEAD = 6
MEM_SCOUT_ARRIVED = 7

SIGHTING_LIMIT = 32
MEMORY_EVENT_LIMIT = 16

_INCIDENT_CACHE: dict[str, list[list[tuple[int, object]]]] = {}


def bucket(count: int) -> int:
    """``0`` for nobody, otherwise ``min(7, 1 + floor(log2 n))``: the approximate strength a colony can perceive."""
    return min(7, count.bit_length())


def _incident(world_map: MapE1) -> list[list[tuple[int, object]]]:
    cached = _INCIDENT_CACHE.get(world_map.map_hash)
    if cached is None:
        cached = [[] for _ in world_map.nodes]
        for edge in world_map.edges:
            cached[edge.a].append((edge.b, edge))
            cached[edge.b].append((edge.a, edge))
        for row in cached:
            row.sort(key=lambda item: item[0])
        if len(_INCIDENT_CACHE) > 8:
            _INCIDENT_CACHE.clear()
        _INCIDENT_CACHE[world_map.map_hash] = cached
    return cached


def push_memory_event(memory: ColonyMemoryE1, code: int, node_slot: int, colony_slot: int, magnitude: int, tick: int) -> None:
    memory.events.append(MemoryEvent(code, node_slot, colony_slot, magnitude, tick))
    if len(memory.events) > MEMORY_EVENT_LIMIT:
        del memory.events[0]


def slot_of_node(memory: ColonyMemoryE1) -> dict[int, int]:
    return {known.node_id: slot for slot, known in enumerate(memory.nodes)}


# ------------------------------------------------------------------------------------------------ sensor phase


class _Observer:
    """Per colony, per tick working view of the memory (slot index, edge set, foreign index)."""

    def __init__(self, state: WorldStateE1, world_map: MapE1, colony_id: int, events: list[EventE1]) -> None:
        self.state, self.world_map, self.colony_id, self.events = state, world_map, colony_id, events
        self.tick = state.tick
        self.memory = state.colonies[colony_id].memory
        self.slot_of = slot_of_node(self.memory)
        self.edge_keys = {(min(self.memory.nodes[e.slot_a].node_id, self.memory.nodes[e.slot_b].node_id),
                           max(self.memory.nodes[e.slot_a].node_id, self.memory.nodes[e.slot_b].node_id))
                          for e in self.memory.edges}
        self.foreign_slot = {record.colony_id: slot for slot, record in enumerate(self.memory.foreign)}
        self.new_edges = False

    def node_slot(self, node_id: int) -> int:
        slot = self.slot_of.get(node_id)
        if slot is None:
            slot = len(self.memory.nodes)
            self.memory.nodes.append(KnownNode(node_id, "unknown", False, self.tick, -1, 0, 0, -1, 0, -1, 0, -1, 0, 0))
            self.slot_of[node_id] = slot
        return slot

    def ensure_foreign(self, other: int, node_id: int) -> int:
        slot = self.foreign_slot.get(other)
        if slot is None:
            slot = len(self.memory.foreign)
            self.memory.foreign.append(ForeignColonyRecord(other, self.tick, self.tick, 0, 0, -1, 0, 0, 0, 0))
            self.foreign_slot[other] = slot
            self.events.append(event(self.tick, "contact", self.colony_id, other, node_id, slot))
            push_memory_event(self.memory, MEM_FIRST_CONTACT, self.slot_of.get(node_id, -1), slot, 0, self.tick)
        return slot

    def observe(self, node_id: int, presence: dict[int, list[int]], stock: dict[int, int], piles: dict[int, int]) -> None:
        memory, tick, truth = self.memory, self.tick, self.world_map.nodes[node_id]
        slot = self.node_slot(node_id)
        known = memory.nodes[slot]
        if not known.visited:
            known.visited = True
            known.kind = truth.kind
            if truth.patch:
                known.patch_capacity = truth.patch[0].capacity
            self.events.append(event(tick, "discover", self.colony_id, -1, node_id, slot))
        known.seen_tick = tick
        known.danger_q = known.danger_q * 31 // 32

        for neighbor, edge in _incident(self.world_map)[node_id]:
            other_slot = self.node_slot(neighbor)
            key = (edge.a, edge.b)
            if key not in self.edge_keys:
                self.edge_keys.add(key)
                a, b = sorted((slot, other_slot))
                memory.edges.append(KnownEdge(a, b, edge.length, edge.lane_cap, tick, False, 0))
                self.new_edges = True

        if truth.patch:
            amount = stock[node_id]
            known.last_delta_q = amount - known.last_stock if known.last_stock_tick >= 0 else 0
            known.last_stock = amount
            known.last_stock_tick = tick
        self._observe_pile(slot, piles.get(node_id, 0))

        if truth.kind == "nest":
            owner = truth.nest_of
            queen_alive = self.state.colonies[owner].queen_hp > 0
            known.queen_seen = 1 if queen_alive else 2
            known.queen_seen_tick = tick
            if owner != self.colony_id:
                owner_slot = self.ensure_foreign(owner, node_id)
                known.nest_owner_slot = owner_slot
                record = memory.foreign[owner_slot]
                if record.nest_node_slot != slot:
                    record.nest_node_slot = slot
                    self.events.append(event(tick, "nest_found", self.colony_id, owner, node_id, slot, known.queen_seen))
        for other in sorted(presence):
            if other == self.colony_id:
                continue
            workers, soldiers, attacking = presence[other]
            other_slot = self.ensure_foreign(other, node_id)
            record = memory.foreign[other_slot]
            record.last_seen_tick = tick
            record.last_seen_workers_bucket = bucket(workers)
            record.last_seen_soldiers_bucket = bucket(soldiers)
            sightings = memory.sightings
            sightings[:] = [s for s in sightings if not (s.node_slot == slot and s.colony_slot == other_slot)]
            sightings.append(Sighting(slot, other_slot, tick, record.last_seen_workers_bucket,
                                      record.last_seen_soldiers_bucket, bool(attacking)))
            if len(sightings) > SIGHTING_LIMIT:
                del sightings[0]

    def _observe_pile(self, slot: int, amount: int) -> None:
        piles = self.memory.piles_seen
        for index, pile in enumerate(piles):
            if pile.node_slot == slot:
                if amount > 0:
                    pile.stock, pile.seen_tick = amount, self.tick
                else:
                    del piles[index]
                return
        if amount > 0:
            piles.append(KnownPile(slot, amount, self.tick))
            piles.sort(key=lambda p: p.node_slot)

    def finish(self) -> None:
        if self.new_edges:
            self.memory.edges.sort(key=lambda e: (e.slot_a, e.slot_b))


def sense(state: WorldStateE1, world_map: MapE1, events: list[EventE1]) -> None:
    """Sensor phase (after Arrival): update the private memory of every colony from what stands in the nodes it occupies."""
    presence: dict[int, dict[int, list[int]]] = {}
    for cohort in state.cohorts:
        if cohort.loc.kind != "node":
            continue
        row = presence.setdefault(cohort.loc.node_id, {}).setdefault(cohort.colony_id, [0, 0, 0])
        row[0 if cohort.caste == "worker" else 1] += cohort.count
        if cohort.mission.kind == "attack":
            row[2] = 1
    stock = {patch.node_id: patch.stock for patch in state.patches}
    piles = {pile.node_id: pile.stock for pile in state.piles}
    for colony in state.colonies:
        if colony.status == "extinct":
            continue
        colony_id = colony.colony_id
        nodes = {node for node, row in presence.items() if colony_id in row}
        if colony.queen_hp > 0:
            nodes.add(world_map.nests[colony_id])  # the motionless queen watches her own nest
        if not nodes:
            continue
        observer = _Observer(state, world_map, colony_id, events)
        for node_id in sorted(nodes):
            observer.observe(node_id, presence.get(node_id, {}), stock, piles)
        observer.finish()


# ------------------------------------------------------------------------------------------------ routing over known edges


def known_adjacency(memory: ColonyMemoryE1) -> dict[int, list[tuple[int, int, int]]]:
    """``node_id -> [(neighbor_node_id, length, neighbor_slot)]`` over the edges the colony knows."""
    adjacency: dict[int, list[tuple[int, int, int]]] = {}
    for edge in memory.edges:
        a, b = memory.nodes[edge.slot_a].node_id, memory.nodes[edge.slot_b].node_id
        adjacency.setdefault(a, []).append((b, edge.length, edge.slot_b))
        adjacency.setdefault(b, []).append((a, edge.length, edge.slot_a))
    return adjacency


def known_distances(memory: ColonyMemoryE1, target: int,
                    adjacency: dict[int, list[tuple[int, int, int]]] | None = None) -> dict[int, int]:
    adjacency = known_adjacency(memory) if adjacency is None else adjacency
    dist = {target: 0}
    queue = [(0, target)]
    while queue:
        d, node = heapq.heappop(queue)
        if d != dist.get(node):
            continue
        for neighbor, length, _ in adjacency.get(node, ()):
            candidate = d + length
            if candidate < dist.get(neighbor, 1 << 60):
                dist[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return dist


def known_next_hop(memory: ColonyMemoryE1, source: int, target: int) -> int:
    """First node of a shortest path over *known* edges (tie: smaller memory slot); ``-1`` if the colony knows no way."""
    if source == target:
        return source
    adjacency = known_adjacency(memory)
    dist = known_distances(memory, target, adjacency)
    if source not in dist:
        return -1
    options = [(length + dist[neighbor], slot, neighbor) for neighbor, length, slot in adjacency[source] if neighbor in dist]
    return min(options)[2] if options else -1


def known_path_ticks(memory: ColonyMemoryE1, source: int, target: int) -> int:
    """Length in ticks of the shortest known path, ``-1`` if there is none."""
    return known_distances(memory, target).get(source, -1)
