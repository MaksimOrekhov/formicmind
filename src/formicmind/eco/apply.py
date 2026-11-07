"""Turn a validated ``ActionE1`` into the colony's orders and cohort missions (contracts 5 and 8.2; RFC 7.3).

This is the only place where an action touches the world, and it works strictly through the colony's own memory: every target is a **slot**, every
route is over edges the colony knows, scouts, squads and foragers are taken from the units idle at the own nest. The calls are the validated order
API of ``orders.py`` (``order_scout`` / ``order_guard`` / ``order_attack`` / ``order_withdraw``) plus the forage dispatch below; combat physics,
logistics and demography are untouched.

Order inside one colony: the standing economic orders are recorded first, then WITHDRAW squads, scouts, GUARD / ATTACK squads (each takes its
share of the soldiers still idle, a squad with nobody left does nothing) and last the foragers, who take the workers the scouts left. A squad
order identical to a squad that is still out is not sent twice. The caller applies the colonies one after another on the snapshot they all
decided on; a colony's orders only move its own units, so the order of colonies cannot change the result.
"""

from __future__ import annotations

from .action import ActionE1, squad_take
from .config import ConfigE1
from .events import EventE1
from .memory import known_path_ticks
from .model import CohortE1, ColonyStateE1, MapE1, Mission, SquadOrder, WorldStateE1
from .orders import IllegalOrderError, order_attack, order_guard, order_scout, order_withdraw
from .cohorts import split_cohort


def _idle_home(state: WorldStateE1, world_map: MapE1, colony_id: int, caste: str) -> list[CohortE1]:
    home = world_map.nests[colony_id]
    return sorted((c for c in state.cohorts if c.colony_id == colony_id and c.caste == caste and c.loc.kind == "node"
                   and c.loc.node_id == home and c.mission.kind == "home" and c.load == 0), key=lambda c: c.cohort_id)


def _units(state: WorldStateE1, colony_id: int, caste: str, kind: str | None = None) -> int:
    return sum(c.count for c in state.cohorts if c.colony_id == colony_id and c.caste == caste and (kind is None or c.mission.kind == kind))


def _squad_out(state: WorldStateE1, colony_id: int, kind: str, node_id: int, target_colony: int) -> bool:
    return any(c.colony_id == colony_id and c.caste == "soldier" and c.mission.kind == kind and c.mission.target_node == node_id
               and (kind != "attack" or c.mission.target_colony == target_colony) for c in state.cohorts)


def _recallable(state: WorldStateE1, colony_id: int, node_id: int) -> bool:
    for c in state.cohorts:
        if c.colony_id != colony_id or c.mission.kind not in ("guard", "attack"):
            continue
        place = c.loc.node_id if c.loc.kind == "node" else c.loc.edge_to
        if node_id == -1 or c.mission.target_node == node_id or place == node_id:
            return True
    return False


def dispatch_forage(state: WorldStateE1, config: ConfigE1, world_map: MapE1, colony: ColonyStateE1) -> int:
    """Send idle workers to the colony's recorded foraging plan. Returns the number of workers sent.

    ``want = floor((workers - scouts) * forage_frac_q / 8)`` foragers in total, split by the shares (largest remainder, ties to the lower plan
    index); every target is topped up to its part from the workers idle at the nest. Foragers already out are never recalled: a target that
    left the plan simply stops being refilled when its cohort comes home. Orphans re-run their last plan (standing orders, contracts 5).
    """
    orders, memory = colony.orders, colony.memory
    if not orders.forage_targets or orders.forage_frac_q <= 0:
        return 0
    colony_id = colony.colony_id
    workers = _units(state, colony_id, "worker")
    want_total = (workers - _units(state, colony_id, "worker", "scout")) * orders.forage_frac_q // 8
    exact = [want_total * share for share in orders.forage_shares_q]
    parts = [value // 16 for value in exact]
    for index in sorted(range(len(parts)), key=lambda i: (-(exact[i] % 16), i))[:want_total - sum(parts)]:
        parts[index] += 1
    idle = _idle_home(state, world_map, colony_id, "worker")
    sent = 0
    for slot, part in zip(orders.forage_targets, parts):
        node = memory.nodes[slot]
        if not node.visited or node.kind != "patch" or known_path_ticks(memory, memory.nodes[0].node_id, node.node_id) < 0:
            continue
        active = sum(c.count for c in state.cohorts if c.colony_id == colony_id and c.caste == "worker"
                     and c.mission.kind == "forage" and c.mission.target_node == node.node_id)
        need = max(0, part - active)
        while need and idle:
            cohort = idle.pop(0)
            take = min(need, cohort.count)
            traveler = split_cohort(state, cohort, take)
            traveler.mission = Mission("forage", node.node_id, -1, "go", 0)
            if traveler is not cohort:
                idle.insert(0, cohort)
            need -= take
            sent += take
    return sent


def apply_action(state: WorldStateE1, config: ConfigE1, world_map: MapE1, colony_id: int, action: ActionE1,
                 sink: list[EventE1] | None = None) -> None:
    """Apply a masks-validated action of an alive colony. Raises ``IllegalOrderError`` if the order API refuses part of it (a bug, not a defeat)."""
    colony = state.colonies[colony_id]
    if colony.status != "alive":
        raise IllegalOrderError(f"colony {colony_id} is {colony.status} and takes no orders")
    orders, memory = colony.orders, colony.memory
    orders.egg_level, orders.soldier_share_q, orders.workers_flee = action.egg_level, action.soldier_share_q, action.workers_flee
    orders.forage_frac_q = action.forage_frac_q
    orders.forage_targets, orders.forage_shares_q = list(action.forage_targets), list(action.forage_shares_q)

    # recall first: the units come home and are idle again for the orders below only after they arrive, not within this decision
    for index, squad in enumerate(action.squads):
        if squad.mission != "withdraw":
            continue
        recorded = orders.squads[index]
        node_id = memory.nodes[recorded.target_node_slot].node_id if recorded.mission in ("guard", "attack") and recorded.target_node_slot >= 0 else -2
        if node_id != -2 and _recallable(state, colony_id, node_id):
            order_withdraw(state, config, world_map, colony_id, recorded.target_node_slot, sink)
        orders.squads[index] = SquadOrder("withdraw", recorded.size_q, -1, -1)

    # scouts (one-off): n workers split over one or two targets, the first gets the odd one
    scouts = min(action.scout_n, sum(c.count for c in _idle_home(state, world_map, colony_id, "worker")))
    if scouts and action.scout_targets:
        targets = action.scout_targets[:scouts]
        for position, slot in enumerate(targets):
            share = scouts // len(targets) + (1 if position < scouts % len(targets) else 0)
            if share:
                order_scout(state, config, world_map, colony_id, slot, share, sink)

    # squads
    for index, squad in enumerate(action.squads):
        if squad.mission not in ("guard", "attack"):
            continue
        idle = sum(c.count for c in _idle_home(state, world_map, colony_id, "soldier"))
        total = _units(state, colony_id, "soldier")
        node_id = memory.nodes[squad.node_slot].node_id
        target_colony = memory.foreign[squad.colony_slot].colony_id if squad.mission == "attack" else -1
        if _squad_out(state, colony_id, squad.mission, node_id, target_colony):
            orders.squads[index] = SquadOrder(squad.mission, squad.size_q, squad.node_slot, squad.colony_slot)  # type: ignore[arg-type]
            continue
        if idle < 1 or squad_take(total, idle, squad.size_q) < 1:
            continue  # the soldiers this squad could have taken are out already (contracts 8.2: not more than what is left)
        if squad.mission == "guard":
            order_guard(state, config, world_map, colony_id, squad.node_slot, squad.size_q, index, sink)
        else:
            order_attack(state, config, world_map, colony_id, squad.node_slot, squad.colony_slot, squad.size_q, index, sink)

    dispatch_forage(state, config, world_map, colony)
