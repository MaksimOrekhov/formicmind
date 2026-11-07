"""Validated orders for scouting and war (contracts section 5 and 8.1; RFC 7.3).

Every function checks the *whole* order against the colony's own memory and its own cohorts first and only then changes the world,
so an illegal order raises ``IllegalOrderError`` and leaves the state (and its hash) untouched: a technical error, never a defeat.
Targets are memory **slots**, not world ids: a colony cannot aim at a node it has not discovered, and cannot attack a nest it has
neither visited nor seen a fresh sighting of. Cohorts walk only over edges the colony knows (``memory.known_next_hop``).
"""

from __future__ import annotations

from .cohorts import split_cohort
from .config import ConfigE1
from .events import MISSION_CODES, EventE1, event
from .memory import known_path_ticks
from .model import CohortE1, ColonyStateE1, MapE1, Mission, SquadOrder, WorldStateE1

MAX_SCOUTS_PER_ORDER = 4
SCOUT_REVISIT_AGE = 90       # a visited node may be scouted again once its information is this old (contracts 8.1)
ATTACK_SIGHTING_AGE = 150    # a sighting is a legal attack target while it is at most this old (contracts 8.1)


class IllegalOrderError(ValueError):
    """The order cannot be executed in the current state; nothing was changed."""


def _fail(message: str) -> None:
    raise IllegalOrderError(message)


def _colony(state: WorldStateE1, colony_id: int) -> ColonyStateE1:
    if type(colony_id) is not int or not 0 <= colony_id < len(state.colonies):
        _fail(f"unknown colony {colony_id!r}")
    colony = state.colonies[colony_id]
    if colony.status != "alive":
        _fail(f"colony {colony_id} is {colony.status} and takes no orders")
    return colony


def _idle_at_home(state: WorldStateE1, world_map: MapE1, colony_id: int, caste: str) -> list[CohortE1]:
    home = world_map.nests[colony_id]
    return sorted((c for c in state.cohorts if c.colony_id == colony_id and c.caste == caste and c.loc.kind == "node"
                   and c.loc.node_id == home and c.mission.kind == "home" and c.load == 0), key=lambda c: c.cohort_id)


def _take(state: WorldStateE1, pool: list[CohortE1], units: int) -> list[CohortE1]:
    taken = []
    for cohort in pool:
        if units <= 0:
            break
        part = split_cohort(state, cohort, min(units, cohort.count))
        taken.append(part)
        units -= part.count
    return taken


def _known_slot(colony: ColonyStateE1, slot: int, label: str) -> int:
    if type(slot) is not int or not 0 <= slot < len(colony.memory.nodes):
        _fail(f"{label}: slot {slot!r} is not in the colony's memory")
    return slot


def _need_path(colony: ColonyStateE1, world_map: MapE1, node_id: int, label: str) -> None:
    if known_path_ticks(colony.memory, world_map.nests[colony.colony_id], node_id) < 0:
        _fail(f"{label}: no path over known edges")


def _emit(sink: list[EventE1] | None, tick: int, colony: int, other: int, node: int, mission: str, units: int) -> None:
    if sink is not None:
        sink.append(event(tick, "dispatch", colony, other, node, MISSION_CODES[mission], units))


def _record_squad(colony: ColonyStateE1, squad: int, mission: str, size_q: int, node_slot: int, colony_slot: int) -> None:
    colony.orders.squads[squad] = SquadOrder(mission, size_q, node_slot, colony_slot)  # type: ignore[arg-type]


def _check_squad(squad: int) -> None:
    if type(squad) is not int or not 0 <= squad < 3:
        _fail(f"squad index {squad!r} must be 0, 1 or 2")


def order_scout(state: WorldStateE1, config: ConfigE1, world_map: MapE1, colony_id: int, target_slot: int, workers: int = 1,
                sink: list[EventE1] | None = None) -> list[int]:
    """Send ``workers`` idle workers to look at a memory slot. Returns the ids of the dispatched cohorts."""
    colony = _colony(state, colony_id)
    slot = _known_slot(colony, target_slot, "scout")
    if slot == 0:
        _fail("scout: the own nest is not a target")
    if type(workers) is not int or not 1 <= workers <= MAX_SCOUTS_PER_ORDER:
        _fail(f"scout: 1..{MAX_SCOUTS_PER_ORDER} workers per order")
    known = colony.memory.nodes[slot]
    if known.visited and state.tick - known.seen_tick < SCOUT_REVISIT_AGE:
        _fail("scout: the node was seen too recently to be worth a trip")
    _need_path(colony, world_map, known.node_id, "scout")
    pool = _idle_at_home(state, world_map, colony_id, "worker")
    if sum(c.count for c in pool) < workers:
        _fail("scout: not enough idle workers at the nest")
    cohorts = _take(state, pool, workers)
    for cohort in cohorts:
        cohort.mission = Mission("scout", known.node_id, -1, "go", 0)
    _emit(sink, state.tick, colony_id, -1, known.node_id, "scout", workers)
    return [c.cohort_id for c in cohorts]


def _squad_units(state: WorldStateE1, world_map: MapE1, colony_id: int, size_q: int) -> list[CohortE1]:
    if type(size_q) is not int or not 1 <= size_q <= 4:
        _fail(f"squad size_q {size_q!r} must be 1..4")
    pool = _idle_at_home(state, world_map, colony_id, "soldier")
    available = sum(c.count for c in pool)
    total = sum(c.count for c in state.cohorts if c.colony_id == colony_id and c.caste == "soldier")
    wanted = min(available, max(1, total * size_q // 4))
    if wanted < 1:
        _fail("squad: no idle soldiers at the nest")
    return _take(state, pool, wanted)


def order_guard(state: WorldStateE1, config: ConfigE1, world_map: MapE1, colony_id: int, node_slot: int, size_q: int = 1,
                squad: int = 0, sink: list[EventE1] | None = None) -> list[int]:
    """Station soldiers on a node the colony has visited; they only answer an attack, they never start one."""
    colony = _colony(state, colony_id)
    _check_squad(squad)
    slot = _known_slot(colony, node_slot, "guard")
    known = colony.memory.nodes[slot]
    if not known.visited:
        _fail("guard: only a visited node can be guarded")
    _need_path(colony, world_map, known.node_id, "guard")
    if not _idle_at_home(state, world_map, colony_id, "soldier"):
        _fail("squad: no idle soldiers at the nest")
    cohorts = _squad_units(state, world_map, colony_id, size_q)
    for cohort in cohorts:
        cohort.mission = Mission("guard", known.node_id, -1, "go", 0)
    _record_squad(colony, squad, "guard", size_q, slot, -1)
    _emit(sink, state.tick, colony_id, -1, known.node_id, "guard", sum(c.count for c in cohorts))
    return [c.cohort_id for c in cohorts]


def attack_target_is_legal(state: WorldStateE1, colony_id: int, node_slot: int, colony_slot: int) -> bool:
    """The colony may aim at ``(node_slot, colony_slot)`` only through what it knows: the other colony's nest, or a fresh sighting."""
    memory = state.colonies[colony_id].memory
    if not (0 <= node_slot < len(memory.nodes) and 0 <= colony_slot < len(memory.foreign)):
        return False
    if memory.foreign[colony_slot].nest_node_slot == node_slot:
        return True
    return any(s.node_slot == node_slot and s.colony_slot == colony_slot and state.tick - s.tick <= ATTACK_SIGHTING_AGE
               for s in memory.sightings)


def order_attack(state: WorldStateE1, config: ConfigE1, world_map: MapE1, colony_id: int, node_slot: int, colony_slot: int,
                 size_q: int = 4, squad: int = 0, sink: list[EventE1] | None = None) -> list[int]:
    """Send soldiers to hit ``colony_slot`` at ``node_slot``. The target must come from the colony's own memory."""
    colony = _colony(state, colony_id)
    _check_squad(squad)
    if type(node_slot) is not int or type(colony_slot) is not int or not attack_target_is_legal(state, colony_id, node_slot, colony_slot):
        _fail("attack: the target is neither a known nest nor a fresh sighting of that colony")
    known = colony.memory.nodes[node_slot]
    target = colony.memory.foreign[colony_slot].colony_id
    if target == colony_id:
        _fail("attack: not a foreign colony")
    # Whether the target still exists is NOT checked: that would leak a state the colony has not observed. An attack on a colony that
    # is already gone is legal if the colony does not know it; the engine simply finds nobody to fight (combat.resolve_combat).
    _need_path(colony, world_map, known.node_id, "attack")
    if not _idle_at_home(state, world_map, colony_id, "soldier"):
        _fail("squad: no idle soldiers at the nest")
    cohorts = _squad_units(state, world_map, colony_id, size_q)
    for cohort in cohorts:
        cohort.mission = Mission("attack", known.node_id, target, "go", 0)
    _record_squad(colony, squad, "attack", size_q, node_slot, colony_slot)
    _emit(sink, state.tick, colony_id, target, known.node_id, "attack", sum(c.count for c in cohorts))
    return [c.cohort_id for c in cohorts]


def order_withdraw(state: WorldStateE1, config: ConfigE1, world_map: MapE1, colony_id: int, node_slot: int = -1,
                   sink: list[EventE1] | None = None) -> list[int]:
    """Recall guards/attackers (all of them, or those at/bound for ``node_slot``) to the nest over known edges."""
    colony = _colony(state, colony_id)
    home = world_map.nests[colony_id]
    node_id = -1
    if node_slot != -1:
        node_id = colony.memory.nodes[_known_slot(colony, node_slot, "withdraw")].node_id
    chosen = []
    for cohort in sorted(state.cohorts, key=lambda c: c.cohort_id):
        if cohort.colony_id != colony_id or cohort.mission.kind not in ("guard", "attack"):
            continue
        place = cohort.loc.node_id if cohort.loc.kind == "node" else cohort.loc.edge_to
        if node_id != -1 and cohort.mission.target_node != node_id and place != node_id:
            continue
        chosen.append(cohort)
    if not chosen:
        _fail("withdraw: no squad to recall")
    for cohort in chosen:
        if cohort.loc.kind == "node" and known_path_ticks(colony.memory, cohort.loc.node_id, home) < 0:
            _fail("withdraw: a squad knows no way home")
    for cohort in chosen:
        cohort.mission = Mission("withdraw", home, -1, "return", 0)
    for index in range(3):
        if colony.orders.squads[index].mission in ("guard", "attack"):
            _record_squad(colony, index, "withdraw", colony.orders.squads[index].size_q, -1, -1)
    _emit(sink, state.tick, colony_id, -1, home if node_id == -1 else node_id, "withdraw", sum(c.count for c in chosen))
    return [c.cohort_id for c in chosen]
