"""Scripted war-mechanics roles (``scout``, ``turtle``, ``raider``, ``retreat``) for E1-c diagnostics.

These are *checks of the mechanics*, not teachers of a future policy. A role decides only from the colony's own memory and its own
cohorts and acts only through the validated order API of ``orders.py``: it cannot name a node or a colony it has not discovered,
and an order it is not allowed to give raises ``IllegalOrderError`` (caught here: the role simply does nothing this decision).
There is no war timer: a raider attacks when its own army is large enough for what it last saw of a neighbour whose nest it has
found by scouting, and not otherwise.
"""

from __future__ import annotations

from .config import ConfigE1
from .events import EventE1
from .memory import known_distances
from .model import MapE1, WorldStateE1
from .orders import IllegalOrderError, order_attack, order_guard, order_scout, order_withdraw

ROLE_NAMES = ("scout", "turtle", "raider", "retreat")

MAX_SCOUTS_OUT = {"scout": 2, "turtle": 1, "raider": 2, "retreat": 2}
SCOUT_RANGE = 30      # farthest known-path distance (ticks from the nest) a scout is sent to
RAID_ARMY = 8         # soldiers a raider/retreat colony tries to have (counting brood) before it stops buying soldiers
TURTLE_ARMY = 6
MIN_FORCE = 6         # fewest soldiers a raid is sent with
SCOUT_MIN_WORKERS = 12  # a trip takes a worker away from foraging; a thin colony cannot afford one
SCOUT_FOOD_FLOOR = 30
ARMY_MIN_WORKERS = 14   # an army is bought only by a colony whose workers already feed it
RAID_FOOD_FLOOR = 150  # warehouse stock a raid leaves with
ARMY_FOOD_FLOOR = 90  # an army is only paid for from a comfortable warehouse: soldiers starve first, so a poor colony cannot keep one


def _units(state: WorldStateE1, colony_id: int, caste: str, mission_kind: str | None = None) -> int:
    return sum(c.count for c in state.cohorts if c.colony_id == colony_id and c.caste == caste
               and (mission_kind is None or c.mission.kind == mission_kind))


def soldier_share_for(role: str, state: WorldStateE1, colony_id: int) -> int:
    """Soldier share (quarters of eggs) a role asks for; the economy fixtures leave it at 0."""
    colony = state.colonies[colony_id]
    soldiers = _units(state, colony_id, "soldier") + sum(b.caste == "soldier" for b in colony.brood)
    if colony.food < ARMY_FOOD_FLOOR or _units(state, colony_id, "worker") < ARMY_MIN_WORKERS:
        return 0
    if role == "turtle":
        return 1 if soldiers < TURTLE_ARMY else 0
    if role in ("raider", "retreat"):
        return 2 if soldiers < RAID_ARMY else 0
    return 0


def explore(state: WorldStateE1, config: ConfigE1, world_map: MapE1, colony_id: int, sink: list[EventE1], max_out: int) -> None:
    """Keep up to ``max_out`` workers walking to the nearest discovered-but-unvisited node over known edges."""
    colony = state.colonies[colony_id]
    memory = colony.memory
    if colony.food < SCOUT_FOOD_FLOOR or _units(state, colony_id, "worker") < SCOUT_MIN_WORKERS:
        return
    scouts = [c for c in state.cohorts if c.colony_id == colony_id and c.mission.kind == "scout"]
    if sum(c.count for c in scouts) >= max_out:
        return
    taken = {c.mission.target_node for c in scouts if c.mission.phase == "go"}
    distance = known_distances(memory, world_map.nests[colony_id])
    # the farthest reachable unknown node first: the walk there reveals every node and edge on the way
    frontier = sorted((-distance[known.node_id], slot) for slot, known in enumerate(memory.nodes)
                      if not known.visited and known.node_id in distance and known.node_id not in taken
                      and distance[known.node_id] <= SCOUT_RANGE)
    for _, slot in frontier[:1]:
        try:
            order_scout(state, config, world_map, colony_id, slot, 1, sink)
        except IllegalOrderError:
            pass


def _enemy_bounds(bucket_value: int) -> tuple[int, int]:
    """(lower, upper) number of soldiers consistent with a strength bucket: ``2^(b-1) <= n < 2^b``; bucket 7 means 64 or more."""
    if bucket_value == 0:
        return 0, 0
    return 1 << (bucket_value - 1), (127 if bucket_value >= 7 else (1 << bucket_value) - 1)


def _raid_target(state: WorldStateE1, colony_id: int, world_map: MapE1) -> tuple[int, int, int] | None:
    """Pick the known neighbour nest with the weakest last-seen army: ``(node_slot, colony_slot, enemy_upper_bound)``."""
    memory = state.colonies[colony_id].memory
    distance = known_distances(memory, world_map.nests[colony_id])
    options = []
    for colony_slot, record in enumerate(memory.foreign):
        if record.nest_node_slot < 0 or memory.nodes[record.nest_node_slot].queen_seen == 2:
            continue  # nest unknown, or the queen is known to be dead
        node = memory.nodes[record.nest_node_slot]
        if node.node_id not in distance:
            continue  # only the colony's own memory decides: whether the neighbour still exists is not observed
        lower, upper = _enemy_bounds(record.last_seen_soldiers_bucket)
        options.append((upper, distance[node.node_id], colony_slot, record.nest_node_slot))
    if not options:
        return None
    upper, _, colony_slot, node_slot = min(options)
    return node_slot, colony_slot, upper


def raider(state: WorldStateE1, config: ConfigE1, world_map: MapE1, colony_id: int, sink: list[EventE1], careful: bool) -> None:
    role = "retreat" if careful else "raider"
    """Explore until a neighbour's nest is found; attack with the whole idle army once it is big enough for that neighbour."""
    if any(c.colony_id == colony_id and c.mission.kind == "attack" for c in state.cohorts):
        return  # a raid is on; ``retreat`` may call it off in ``react``
    target = _raid_target(state, colony_id, world_map)
    if target is None:
        explore(state, config, world_map, colony_id, sink, MAX_SCOUTS_OUT[role])
        return
    node_slot, colony_slot, enemy_upper = target
    idle_soldiers = sum(c.count for c in state.cohorts if c.colony_id == colony_id and c.caste == "soldier"
                        and c.mission.kind == "home" and c.loc.kind == "node" and c.loc.node_id == world_map.nests[colony_id])
    needed = max(MIN_FORCE, (2 * enemy_upper + 2) if careful else (enemy_upper + 2))
    # soldiers starve first: a raid leaves only when the warehouse can keep the army fed for the whole trip
    if idle_soldiers >= needed and state.colonies[colony_id].food >= RAID_FOOD_FLOOR:
        try:
            order_attack(state, config, world_map, colony_id, node_slot, colony_slot, 4, 0, sink)
        except IllegalOrderError:
            pass


def turtle(state: WorldStateE1, config: ConfigE1, world_map: MapE1, colony_id: int, sink: list[EventE1]) -> None:
    """Hold the nest with the army and put a guard on the visited patch where a neighbour was seen most recently."""
    explore(state, config, world_map, colony_id, sink, MAX_SCOUTS_OUT["turtle"])
    if any(c.colony_id == colony_id and c.mission.kind == "guard" for c in state.cohorts):
        return
    memory = state.colonies[colony_id].memory
    seen = [(-s.tick, s.node_slot) for s in memory.sightings if memory.nodes[s.node_slot].kind == "patch"]
    if seen and _units(state, colony_id, "soldier") >= 4:
        try:
            order_guard(state, config, world_map, colony_id, min(seen)[1], 2, 1, sink)
        except IllegalOrderError:
            pass


def react(state: WorldStateE1, config: ConfigE1, world_map: MapE1, colony_id: int, role: str, sink: list[EventE1]) -> None:
    """Extra decision during a fight: ``retreat`` calls the raid off when its soldiers there no longer outnumber what it sees."""
    if role != "retreat":
        return
    memory = state.colonies[colony_id].memory
    for cohort in sorted(state.cohorts, key=lambda c: c.cohort_id):
        if cohort.colony_id != colony_id or cohort.mission.kind != "attack" or cohort.loc.kind != "node":
            continue
        node_id = cohort.loc.node_id
        own = sum(c.count for c in state.cohorts if c.colony_id == colony_id and c.caste == "soldier"
                  and c.mission.kind == "attack" and c.loc.kind == "node" and c.loc.node_id == node_id)
        slots = {known.node_id: slot for slot, known in enumerate(memory.nodes)}
        node_slot = slots.get(node_id, -1)
        enemy_lower = 0
        for sighting in memory.sightings:
            if sighting.node_slot == node_slot and memory.foreign[sighting.colony_slot].colony_id == cohort.mission.target_colony:
                enemy_lower = _enemy_bounds(sighting.soldiers_bucket)[0]
        if own <= enemy_lower:
            try:
                order_withdraw(state, config, world_map, colony_id, node_slot, sink)
            except IllegalOrderError:
                pass
            return


def decide(state: WorldStateE1, config: ConfigE1, world_map: MapE1, colony_id: int, role: str, sink: list[EventE1]) -> None:
    """Planned decision of a role for a living colony."""
    if role == "scout":
        explore(state, config, world_map, colony_id, sink, MAX_SCOUTS_OUT["scout"])
    elif role == "turtle":
        turtle(state, config, world_map, colony_id, sink)
    elif role in ("raider", "retreat"):
        react(state, config, world_map, colony_id, role, sink)
        raider(state, config, world_map, colony_id, sink, careful=role == "retreat")
