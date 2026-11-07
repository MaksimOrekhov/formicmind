"""Deterministic E1 world ticks: cohort travel, shared harvest, food ledger, demography, exploration, contacts and local combat.

There is no observation builder or learned policy. A controller (a diagnostic fixture) may set cohort missions before the tick,
through the validated order API for scouting and war; all physical effects are resolved here from the shared world state.
Phases of a tick (contracts section 6): Policy, Hatch/Age, Arrival (+sensors), Combat, QueenTerminal, Harvest, Upkeep, Lay,
Regeneration, Depart, Terminal.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from .combat import queen_terminal, resolve_combat
from .config import ConfigE1
from .events import EventE1, event
from .graph import INF, adjacency
from .initial import Q
from .model import (
    AgeBucket, BroodRecord, ColonyStateE1, CohortE1, EdgeFlow, Location, MapE1, Mission, PileState,
    RunStateE1, WorldStateE1,
)
from .cohorts import (
    _add_pile, _book_non_age_deaths, _home_mission, _node_location, _split_load, _unit_loads, _where_rank,
    remove_cohort_members, split_cohort,
)
from .memory import MEM_SCOUT_ARRIVED, known_next_hop, push_memory_event, sense, slot_of_node
from .validate import InvariantError, validate_state


class RouteTable:
    """All-pairs shortest routes with stable node-id tie breaks."""

    def __init__(self, world_map: MapE1) -> None:
        self.adj = adjacency(len(world_map.nodes), [(edge.a, edge.b, edge.length) for edge in world_map.edges])
        for row in self.adj:
            row.sort()
        self.edges = {(edge.a, edge.b): edge for edge in world_map.edges}
        self.outbox: list[EventE1] = []  # events emitted by the order API during the controller call; drained by the tick
        self.alert: frozenset[int] = frozenset()  # colonies whose extra (combat) decision this is; empty on a planned decision
        self.dist: list[list[int]] = []
        self.first: list[list[int]] = []
        for source in range(len(self.adj)):
            distances = [INF] * len(self.adj)
            first_hop = [-1] * len(self.adj)
            predecessor = [INF] * len(self.adj)
            distances[source] = 0
            first_hop[source] = source
            predecessor[source] = source
            queue = [(0, source)]
            while queue:
                distance, node = heapq.heappop(queue)
                if distance != distances[node]:
                    continue
                for neighbor, length in self.adj[node]:
                    candidate = distance + length
                    candidate_first = neighbor if node == source else first_hop[node]
                    if (candidate, candidate_first, node) < (
                        distances[neighbor], first_hop[neighbor] if first_hop[neighbor] >= 0 else INF,
                        predecessor[neighbor],
                    ):
                        distances[neighbor] = candidate
                        first_hop[neighbor] = candidate_first
                        predecessor[neighbor] = node
                        heapq.heappush(queue, (candidate, neighbor))
            self.dist.append(distances)
            self.first.append(first_hop)

    def distance(self, source: int, target: int) -> int:
        return self.dist[source][target]

    def next_hop(self, source: int, target: int) -> int:
        return self.first[source][target]


@dataclass(slots=True)
class TickRecord:
    tick: int
    harvested: int = 0
    delivered: int = 0
    regenerated: int = 0
    upkeep_due: int = 0
    upkeep_paid: int = 0
    upkeep_missing: int = 0
    egg_spend: int = 0
    hatched_workers: int = 0
    hatched_soldiers: int = 0
    laid_workers: int = 0
    laid_soldiers: int = 0
    old_age_workers: int = 0
    old_age_soldiers: int = 0
    starvation_workers: int = 0
    starvation_soldiers: int = 0
    contention_nodes: tuple[int, ...] = ()
    depleted_nodes: tuple[int, ...] = ()
    harvested_by_zone: dict[str, int] = field(default_factory=dict)
    delivered_by_colony: dict[int, int] = field(default_factory=dict)
    upkeep_due_by_colony: dict[int, int] = field(default_factory=dict)
    events: list[EventE1] = field(default_factory=list)
    combat_nodes: tuple[int, ...] = ()
    combat_losses_workers: int = 0
    combat_losses_soldiers: int = 0


Controller = Callable[[WorldStateE1, ConfigE1, MapE1, RouteTable], None]


def _food_total(state: WorldStateE1) -> int:
    return (sum(patch.stock for patch in state.patches) + sum(pile.stock for pile in state.piles)
            + sum(colony.food for colony in state.colonies) + sum(cohort.load for cohort in state.cohorts))


def _recompute_edge_flow(state: WorldStateE1, world_map: MapE1) -> None:
    totals: dict[tuple[int, int, int], int] = defaultdict(int)
    edge_by_key = {(edge.a, edge.b): edge for edge in world_map.edges}
    for cohort in state.cohorts:
        if cohort.loc.kind != "edge":
            continue
        a, b = sorted((cohort.loc.edge_from, cohort.loc.edge_to))
        edge = edge_by_key[(a, b)]
        direction = 0 if cohort.loc.edge_from == edge.a else 1
        totals[(a, b, direction)] += cohort.count
    state.edge_flow = [EdgeFlow(a, b, direction, units) for (a, b, direction), units in sorted(totals.items())]


def _hatch_and_age(state: WorldStateE1, config: ConfigE1, world_map: MapE1, record: TickRecord) -> None:
    tick, economy = state.tick, config.economy
    for colony in state.colonies:
        if colony.status == "extinct":
            continue
        due = [brood for brood in colony.brood if brood.hatch_tick <= tick]
        colony.brood = [brood for brood in colony.brood if brood.hatch_tick > tick]
        for brood in due:
            hp = economy.worker_hp if brood.caste == "worker" else economy.soldier_hp
            state.cohorts.append(CohortE1(brood.cohort_id, colony.colony_id, brood.caste, 1, hp,
                                          _node_location(world_map.nests[colony.colony_id]), _home_mission(), 0))
            bucket_start = (tick // economy.age_bucket) * economy.age_bucket
            ledger = colony.ages.workers if brood.caste == "worker" else colony.ages.soldiers
            for bucket in ledger:
                if bucket.bucket_start == bucket_start:
                    bucket.count += 1
                    break
            else:
                ledger.append(AgeBucket(bucket_start, 1))
                ledger.sort(key=lambda bucket: bucket.bucket_start)
            if brood.caste == "worker":
                record.hatched_workers += 1
            else:
                record.hatched_soldiers += 1

    for colony in state.colonies:
        if colony.status == "extinct":
            continue
        for caste, lifespan, ledger in (
            ("worker", economy.lifespan_worker, colony.ages.workers),
            ("soldier", economy.lifespan_soldier, colony.ages.soldiers),
        ):
            if not lifespan:
                continue
            expired = [bucket for bucket in ledger if bucket.bucket_start + lifespan <= tick]
            if not expired:
                continue
            expired_count = sum(bucket.count for bucket in expired)
            keep = [bucket for bucket in ledger if bucket not in expired]
            if caste == "worker":
                colony.ages.workers = keep
                record.old_age_workers += expired_count
            else:
                colony.ages.soldiers = keep
                record.old_age_soldiers += expired_count
            removed, _ = remove_cohort_members(state, config, world_map, colony.colony_id, caste,
                                                expired_count, book_age_deaths=False)
            if removed != expired_count:
                raise InvariantError(f"colony {colony.colony_id}: age ledger exceeds adult {caste} cohorts")
    state.cohorts.sort(key=lambda cohort: cohort.cohort_id)


def _arrivals(state: WorldStateE1, config: ConfigE1, world_map: MapE1, record: TickRecord) -> None:
    tick, economy = state.tick, config.economy
    for cohort in sorted(state.cohorts, key=lambda item: item.cohort_id):
        if cohort.loc.kind == "edge":
            if cohort.loc.arrive_tick > tick:
                continue
            node_id = cohort.loc.edge_to
            cohort.loc = _node_location(node_id)
            if cohort.mission.kind == "forage":
                if cohort.mission.phase == "go" and node_id == cohort.mission.target_node:
                    if world_map.nodes[node_id].patch or any(pile.node_id == node_id and pile.stock for pile in state.piles):
                        cohort.mission.phase = "work"
                        cohort.mission.dwell_left = economy.harvest_patience
                    else:
                        cohort.mission.phase = "return"
            elif cohort.mission.phase == "go" and node_id == cohort.mission.target_node:
                _reached_target(cohort, economy)
        if (cohort.loc.kind == "node" and cohort.loc.node_id == world_map.nests[cohort.colony_id]
                and cohort.mission.phase == "return"):
            colony = state.colonies[cohort.colony_id]
            accepted = min(cohort.load, economy.storage_cap - colony.food)
            colony.food += accepted
            cohort.load -= accepted
            colony.window.delivered += accepted
            record.delivered += accepted
            record.delivered_by_colony[cohort.colony_id] = record.delivered_by_colony.get(cohort.colony_id, 0) + accepted
            if cohort.load == 0:
                cohort.mission = _home_mission()


def _reached_target(cohort: CohortE1, economy) -> None:
    """A scout, guard or attack cohort stands on its target node: it starts to look around, hold or fight."""
    mission = cohort.mission
    mission.phase = "work"
    mission.dwell_left = {"scout": economy.scout_dwell, "attack": economy.attack_dwell}.get(mission.kind, 0)


def _mission_dwell(state: WorldStateE1, config: ConfigE1, world_map: MapE1) -> None:
    """Scouts look around for ``scout_dwell`` ticks and turn home; attackers keep the node while their target is there
    (``attack_dwell`` ticks after it left or died) and then hold it as guards."""
    economy = config.economy
    present = {(c.loc.node_id, c.colony_id) for c in state.cohorts if c.loc.kind == "node"}
    for cohort in state.cohorts:
        mission = cohort.mission
        if cohort.loc.kind != "node" or mission.phase != "work":
            continue
        if mission.kind == "scout":
            if mission.dwell_left > 0:
                mission.dwell_left -= 1
                continue
            memory = state.colonies[cohort.colony_id].memory
            push_memory_event(memory, MEM_SCOUT_ARRIVED, slot_of_node(memory).get(cohort.loc.node_id, -1), -1, 0, state.tick)
            cohort.mission = Mission("scout", world_map.nests[cohort.colony_id], -1, "return", 0)
        elif mission.kind == "attack":
            target = mission.target_colony
            there = (cohort.loc.node_id, target) in present or (
                world_map.nests[target] == cohort.loc.node_id and state.colonies[target].queen_hp > 0
                and state.colonies[target].status != "extinct")
            if there:
                mission.dwell_left = economy.attack_dwell
            elif mission.dwell_left > 0:
                mission.dwell_left -= 1
            else:
                cohort.mission = Mission("guard", cohort.loc.node_id, -1, "work", 0)


def _harvest(state: WorldStateE1, config: ConfigE1, world_map: MapE1, record: TickRecord) -> None:
    patches = {patch.node_id: patch for patch in state.patches}
    specs = {node.node_id: node.patch[0] for node in world_map.nodes if node.patch}
    piles = {pile.node_id: pile for pile in state.piles}
    source_nodes = sorted(set(patches) | set(piles))
    demand_by_node: dict[int, dict[int, list[tuple[CohortE1, int]]]] = {
        node_id: defaultdict(list) for node_id in source_nodes
    }
    for cohort in state.cohorts:
        if cohort.caste != "worker" or cohort.mission.kind != "forage" or cohort.mission.phase != "work" or cohort.loc.kind != "node":
            continue
        node_id = cohort.loc.node_id
        if cohort.mission.target_node != node_id or node_id not in demand_by_node:
            continue
        free = cohort.count * config.economy.carry_load - cohort.load
        demand = min(free, cohort.count * config.economy.harvest_rate)
        if demand > 0:
            demand_by_node[node_id][cohort.colony_id].append((cohort, demand))

    contended: list[int] = []
    depleted: list[int] = []
    for node_id in source_nodes:
        groups = demand_by_node[node_id]
        colony_demands = {colony_id: sum(demand for _, demand in claims)
                          for colony_id, claims in groups.items()}
        total_demand = sum(colony_demands.values())
        if total_demand <= 0:
            continue
        if node_id in patches:
            spec, patch = specs[node_id], patches[node_id]
            supply = min(spec.site_cap, patch.stock)
            if total_demand > supply and len(colony_demands) > 1:
                contended.append(node_id)
            served = min(total_demand, supply)
            before = patch.stock
            colony_allocations = {colony_id: (demand if total_demand <= supply
                                               else served * demand // total_demand)
                                  for colony_id, demand in colony_demands.items()}
            patch.stock -= sum(colony_allocations.values())
            if before > 0 and patch.stock == 0:
                depleted.append(node_id)
        else:
            pile = piles[node_id]
            supply = pile.stock
            served = min(total_demand, supply)
            colony_allocations = {colony_id: (demand if total_demand <= supply
                                               else served * demand // total_demand)
                                  for colony_id, demand in colony_demands.items()}
            pile.stock -= sum(colony_allocations.values())
        for colony_id in sorted(groups):
            claims = sorted(groups[colony_id], key=lambda pair: pair[0].cohort_id)
            grant = colony_allocations[colony_id]
            allocations = [grant * demand // colony_demands[colony_id] for _, demand in claims]
            remainder = grant - sum(allocations)
            for index, (_, demand) in enumerate(claims):
                extra = min(remainder, demand - allocations[index])
                allocations[index] += extra
                remainder -= extra
                if remainder == 0:
                    break
            for (cohort, _), amount in zip(claims, allocations):
                if amount:
                    cohort.load += amount
                    state.colonies[colony_id].window.harvested += amount
                    record.harvested += amount
                    zone = specs[node_id].zone if node_id in specs else "pile"
                    record.harvested_by_zone[zone] = record.harvested_by_zone.get(zone, 0) + amount

    state.piles = [pile for pile in state.piles if pile.stock > 0]
    record.contention_nodes = tuple(contended)
    record.depleted_nodes = tuple(depleted)

    for cohort in state.cohorts:
        if cohort.mission.kind != "forage" or cohort.mission.phase != "work":
            continue
        if cohort.load >= cohort.count * config.economy.carry_load:
            cohort.mission.phase = "return"
            continue
        if cohort.loc.kind == "node":
            node_id = cohort.loc.node_id
            has_stock = (node_id in patches and patches[node_id].stock > 0) or (
                node_id in piles and piles[node_id].stock > 0
            )
            if has_stock:
                cohort.mission.dwell_left = config.economy.harvest_patience
            elif cohort.mission.dwell_left > 0:
                cohort.mission.dwell_left -= 1
            else:
                cohort.mission.phase = "return"


def _upkeep(state: WorldStateE1, config: ConfigE1, world_map: MapE1, record: TickRecord) -> None:
    economy = config.economy
    for colony in state.colonies:
        if colony.status == "extinct":
            continue
        workers = sum(c.count for c in state.cohorts if c.colony_id == colony.colony_id and c.caste == "worker")
        soldiers = sum(c.count for c in state.cohorts if c.colony_id == colony.colony_id and c.caste == "soldier")
        due, colony.upkeep_remainder10 = divmod(
            colony.upkeep_remainder10 + economy.worker_upkeep_tenths * workers
            + economy.soldier_upkeep_tenths * soldiers, 10
        )
        paid = min(due, colony.food)
        missing = due - paid
        colony.food -= paid
        colony.window.upkeep_due += due
        colony.window.upkeep_paid += paid
        record.upkeep_due += due
        record.upkeep_due_by_colony[colony.colony_id] = due
        record.upkeep_paid += paid
        record.upkeep_missing += missing
        if missing:
            colony.starvation_debt += missing
            while colony.starvation_debt >= economy.starvation_unit_cost and workers + soldiers:
                caste = "soldier" if soldiers else "worker"
                removed, _ = remove_cohort_members(state, config, world_map, colony.colony_id, caste, 1,
                                                   book_age_deaths=True)
                if not removed:
                    raise InvariantError(f"colony {colony.colony_id}: starvation could not find a {caste}")
                if caste == "soldier":
                    soldiers -= 1
                    colony.window.losses_soldiers += 1
                    record.starvation_soldiers += 1
                else:
                    workers -= 1
                    colony.window.losses_workers += 1
                    record.starvation_workers += 1
                colony.starvation_debt -= economy.starvation_unit_cost
        elif due > 0 and colony.starvation_debt:
            colony.starvation_debt = 0


def _lay(state: WorldStateE1, config: ConfigE1, record: TickRecord) -> None:
    tick, economy = state.tick, config.economy
    for colony in state.colonies:
        if colony.status != "alive" or colony.queen_hp <= 0 or tick < colony.next_lay_tick:
            continue
        colony.lay_acc_q = min(3, colony.lay_acc_q + colony.orders.egg_level)
        if colony.lay_acc_q < 3:
            colony.next_lay_tick = tick + economy.lay_interval_ticks
            continue
        soldiers = sum(c.count for c in state.cohorts if c.colony_id == colony.colony_id and c.caste == "soldier")
        future_soldiers = soldiers + sum(b.caste == "soldier" for b in colony.brood)
        colony.soldier_acc_q = min(7, colony.soldier_acc_q + colony.orders.soldier_share_q)
        soldier_egg = colony.soldier_acc_q >= 4
        if future_soldiers < 2:
            soldier_egg = True
        caste = "soldier" if soldier_egg else "worker"
        cost = (economy.soldier_egg_cost if caste == "soldier" else economy.worker_egg_cost) * economy.eggs_per_lay
        if colony.food < cost:
            # A failed lay attempt does not consume its timer or accumulator.
            continue
        colony.food -= cost
        colony.window.egg_spend += cost
        record.egg_spend += cost
        if caste == "soldier":
            colony.soldier_acc_q = max(0, colony.soldier_acc_q - 4)
            record.laid_soldiers += economy.eggs_per_lay
        else:
            record.laid_workers += economy.eggs_per_lay
        for _ in range(economy.eggs_per_lay):
            cohort_id = state.next_cohort_id
            state.next_cohort_id += 1
            delay = economy.soldier_hatch_delay if caste == "soldier" else economy.worker_hatch_delay
            colony.brood.append(BroodRecord(cohort_id, caste, tick, tick + delay))
        colony.brood.sort(key=lambda brood: brood.cohort_id)
        colony.lay_acc_q -= 3
        colony.next_lay_tick = tick + economy.lay_interval_ticks
def _regenerate(state: WorldStateE1, world_map: MapE1, record: TickRecord) -> None:
    specs = {node.node_id: node.patch[0] for node in world_map.nodes if node.patch}
    for patch in state.patches:
        before = patch.stock
        patch.stock = min(specs[patch.node_id].capacity, before + specs[patch.node_id].regen)
        record.regenerated += patch.stock - before


def _arrived_home(cohort: CohortE1, colony_id: int, world_map: MapE1) -> bool:
    return cohort.loc.kind == "node" and cohort.loc.node_id == world_map.nests[colony_id]


def _departures(state: WorldStateE1, config: ConfigE1, world_map: MapE1, routes: RouteTable) -> None:
    economy, tick = config.economy, state.tick
    occupancy: dict[tuple[int, int, int], int] = defaultdict(int)
    edge_by_key = {(edge.a, edge.b): edge for edge in world_map.edges}
    for cohort in state.cohorts:
        if cohort.loc.kind == "edge":
            a, b = sorted((cohort.loc.edge_from, cohort.loc.edge_to))
            edge = edge_by_key[(a, b)]
            direction = 0 if cohort.loc.edge_from == edge.a else 1
            occupancy[(a, b, direction)] += cohort.count

    for cohort in sorted(state.cohorts, key=lambda item: item.cohort_id):
        if cohort.loc.kind != "node":
            continue
        mission = cohort.mission
        home = world_map.nests[cohort.colony_id]
        if mission.kind == "forage":
            if cohort.caste != "worker" or mission.phase not in ("go", "return"):
                continue
            target = mission.target_node if mission.phase == "go" else home
            known = False
        elif mission.kind in ("scout", "guard", "attack", "withdraw"):
            if mission.phase not in ("go", "return"):
                continue  # work: hold the node
            target = mission.target_node if mission.phase == "go" else home
            known = True
        else:
            continue
        if cohort.loc.node_id == target:
            if mission.phase == "go":
                if mission.kind == "forage":
                    mission.phase = "work"
                    mission.dwell_left = economy.harvest_patience
                else:
                    _reached_target(cohort, economy)
            elif cohort.load == 0:
                cohort.mission = _home_mission()
            continue
        if known:
            memory = state.colonies[cohort.colony_id].memory
            next_node = known_next_hop(memory, cohort.loc.node_id, target)
            if next_node < 0 and mission.phase == "go":  # the way is not known any more: go home
                mission.phase, mission.target_node = "return", home
                next_node = known_next_hop(memory, cohort.loc.node_id, home)
            if next_node < 0:
                raise InvariantError(f"cohort {cohort.cohort_id} knows no way to continue its {mission.kind} mission")
        else:
            next_node = routes.next_hop(cohort.loc.node_id, target)
            if next_node < 0:
                mission.phase = "return"
                mission.target_node = home
                continue
        a, b = sorted((cohort.loc.node_id, next_node))
        edge = edge_by_key[(a, b)]
        direction = 0 if cohort.loc.node_id == edge.a else 1
        slot = (a, b, direction)
        # RFC V-7: soft lane capacity. A cohort always departs; the units already on this directed edge
        # (including cohorts that left earlier in this same phase, in cohort_id order) lengthen its trip
        # by floor(in_flight_same_direction / lane_cap). There are no queues and nothing is split.
        delay = occupancy[slot] // edge.lane_cap
        cohort.loc = Location("edge", -1, cohort.loc.node_id, next_node, tick, tick + edge.length + delay)
        occupancy[slot] += cohort.count


def _terminal(state: WorldStateE1, config: ConfigE1, world_map: MapE1, events: list[EventE1]) -> None:
    """Contracts §9.5: economic collapse of a colony with a living queen, and extinction of an empty orphan."""
    tick = state.tick
    for colony in state.colonies:
        if colony.status == "extinct":
            continue
        adults = sum(cohort.count for cohort in state.cohorts if cohort.colony_id == colony.colony_id)
        if colony.queen_hp > 0:
            # Collapse: no adults, no worker brood, and not enough food to lay a worker egg.
            # An empty colony that can still pay for a worker egg is alive and is not terminal.
            worker_brood = any(brood.caste == "worker" for brood in colony.brood)
            reason = ("collapse" if adults == 0 and not worker_brood
                      and colony.food < config.economy.worker_egg_cost * config.economy.eggs_per_lay else "")
        else:
            # Orphan (queen dead): extinct only once no adults and no brood remain.
            reason = "no_adults_or_brood" if adults == 0 and not colony.brood else ""
        if reason:
            food = colony.food
            _add_pile(state, world_map.nests[colony.colony_id], food)
            colony.food = 0
            colony.brood = []  # leftover soldier brood dies with the colony
            colony.status = "extinct"
            outcome = state.outcomes[colony.colony_id]
            if not outcome.finished:  # the outcome of a colony is written once: an orphan keeps "queen_death"
                outcome.finished, outcome.reason, outcome.terminal_tick = True, reason, tick
            if reason == "collapse":
                events.append(event(tick, "collapse", colony.colony_id, -1, world_map.nests[colony.colony_id]))
            events.append(event(tick, "extinct", colony.colony_id, -1, world_map.nests[colony.colony_id], food))
    if all(colony.status == "extinct" for colony in state.colonies):
        state.world_outcome.finished = True
        state.world_outcome.reason = "all_colonies_extinct"
        state.world_outcome.terminal_tick = tick
    elif not any(colony.status == "alive" for colony in state.colonies):
        state.world_outcome.finished = True
        state.world_outcome.reason = "no_alive_colonies"
        state.world_outcome.terminal_tick = tick
    elif tick + 1 >= config.horizon:
        state.world_outcome.finished = True
        state.world_outcome.reason = "horizon"
        state.world_outcome.terminal_tick = tick


_COMBAT_CODES = (2, 3, 4, 5)  # MEM_ATTACKED, MEM_WE_KILLED, MEM_WE_LOST, MEM_QUEEN_HIT


def _alert_colonies(state: WorldStateE1, window: int) -> frozenset[int]:
    """Living colonies that were in a fight within the last ``window`` ticks: they get an extra decision (contracts 8.4)."""
    return frozenset(
        colony.colony_id for colony in state.colonies
        if colony.status == "alive" and any(item.code in _COMBAT_CODES and state.tick - item.tick <= window
                                            for item in colony.memory.events)
    )


def step_world_tick(state: WorldStateE1, run: RunStateE1, config: ConfigE1, world_map: MapE1,
                    controller: Controller | None = None, *, routes: RouteTable | None = None,
                    validate_input: bool = True) -> TickRecord:
    """Apply one deterministic world tick in place and return its economic/demographic counters.

    The reserved world RNG stream is intentionally unused. With the same checkpoint and the same
    stateless fixture/controller, continuation produces the same state and checkpoint hashes.
    """
    del run  # RunState is part of checkpoints; E1-b tick rules do not consume randomness.
    if state.world_outcome.finished:
        raise ValueError("cannot step a terminal world")
    if state.tick >= config.horizon:
        raise ValueError("cannot step past the configured horizon")
    # A public call checks its input by default. Long sequential runs can skip the redundant
    # input and static-memory checks after tick 0: the fixture may change orders and missions,
    # while the engine validates every dynamic invariant after applying each tick.
    if validate_input:
        validate_state(state, config, world_map)
    route_table = routes or RouteTable(world_map)
    before_food = _food_total(state)
    record = TickRecord(state.tick)
    if controller is not None:
        planned = state.tick % config.decision_k == 0
        alert = frozenset()
        if not planned and config.alert_interval and state.tick % config.alert_interval == 0:
            alert = _alert_colonies(state, config.alert_interval)
        if planned or alert:
            route_table.alert = alert
            route_table.outbox.clear()
            controller(state, config, world_map, route_table)
            record.events.extend(route_table.outbox)
            route_table.outbox.clear()

    _hatch_and_age(state, config, world_map, record)
    _arrivals(state, config, world_map, record)
    sense(state, world_map, record.events)
    queen_terminal(state, config, world_map, resolve_combat(state, config, world_map, record.events, record),
                   record.events)
    _harvest(state, config, world_map, record)
    _upkeep(state, config, world_map, record)
    _lay(state, config, record)
    _regenerate(state, world_map, record)
    _mission_dwell(state, config, world_map)
    _departures(state, config, world_map, route_table)
    _terminal(state, config, world_map, record.events)
    state.tick += 1
    _recompute_edge_flow(state, world_map)
    validate_state(state, config, world_map, validate_memory=validate_input,
                   validate_config_binding=validate_input)
    after_food = _food_total(state)
    expected_food = before_food + record.regenerated - record.upkeep_paid - record.egg_spend
    if after_food != expected_food:
        raise InvariantError(f"food ledger mismatch at tick {record.tick}: {after_food} != {expected_food}")
    return record
