"""Local combat and the queen terminal of E1-c (contracts sections 5, 6 and 9.5; L0 rules of ``docs/combat_terminal.md`` on cohorts).

* Combat happens only in a node, between units that *stand* there, and only because of an explicit ``attack`` mission whose target
  colony is present (or whose nest-queen is exposed there). Colonies that are not targeted are never touched.
* Damage is computed from the pre-combat state and applied simultaneously, so a soldier killed in this phase still strikes.
  A colony's soldiers deal ``soldier_damage`` each; at their own nest the sum is multiplied by ``home_defense_num / home_defense_den``
  (floor). Attack cohorts hit the colony they target; every other soldier of an attacked colony answers all of its attackers,
  split by attacker strength with floor division (the remainder is lost). There is no priority by colony id.
* A hit lands on exactly one category of the target: soldiers, else workers, else the queen (only in her nest and only if no
  enemy soldier or worker of hers stands there). Damage is poured by ``cohort_id``; what a category cannot absorb is wasted.
* Dead workers drop their cargo as a pile at the node and the age ledger is debited proportionally, so food and demography stay exact.
"""

from __future__ import annotations

from collections import defaultdict

from .cohorts import _add_pile, _book_non_age_deaths, apply_hp_loss
from .config import ConfigE1
from .events import CASTE_CODES, EventE1, event
from .memory import (
    MEM_ATTACKED, MEM_OWN_QUEEN_DEAD, MEM_QUEEN_HIT, MEM_WE_KILLED, MEM_WE_LOST, push_memory_event,
)
from .model import CohortE1, MapE1, Mission, WorldStateE1

DANGER_STEP = 64
DANGER_MAX = 255


def _foreign_slot(memory, colony_id: int) -> int:
    for slot, record in enumerate(memory.foreign):
        if record.colony_id == colony_id:
            return slot
    return -1


def _node_slot(memory, node_id: int) -> int:
    for slot, known in enumerate(memory.nodes):
        if known.node_id == node_id:
            return slot
    return -1


def resolve_combat(state: WorldStateE1, config: ConfigE1, world_map: MapE1, events: list[EventE1], record) -> dict[int, int]:
    """Resolve every battle of this tick; return ``{colony_id: killer colony}`` for queens that died."""
    by_node: dict[int, list[CohortE1]] = defaultdict(list)
    for cohort in state.cohorts:
        if cohort.loc.kind == "node":
            by_node[cohort.loc.node_id].append(cohort)
    killers: dict[int, int] = {}
    nests = world_map.nests
    for node in sorted(by_node):
        cohorts = by_node[node]
        present = {c.colony_id for c in cohorts}
        hostile = []
        for cohort in cohorts:
            mission = cohort.mission
            if cohort.caste != "soldier" or mission.kind != "attack" or mission.phase != "work" or mission.target_node != node:
                continue
            target = mission.target_colony
            if target == cohort.colony_id or state.colonies[target].status == "extinct":
                continue
            if target in present or (nests[target] == node and state.colonies[target].queen_hp > 0):
                hostile.append(cohort)
        if hostile:
            _battle(state, config, world_map, node, cohorts, hostile, events, record, killers)
    return killers


def _battle(state, config, world_map, node, cohorts, hostile, events, record, killers) -> None:
    econ, tick, nests = config.economy, state.tick, world_map.nests
    soldiers: dict[int, int] = defaultdict(int)
    workers: dict[int, int] = defaultdict(int)
    for cohort in cohorts:
        (soldiers if cohort.caste == "soldier" else workers)[cohort.colony_id] += cohort.count
    striking: dict[tuple[int, int], int] = defaultdict(int)   # (attacker, target) -> soldiers striking
    attacking_total: dict[int, int] = defaultdict(int)
    for cohort in hostile:
        striking[(cohort.colony_id, cohort.mission.target_colony)] += cohort.count
        attacking_total[cohort.colony_id] += cohort.count

    def scaled(colony: int, amount: int) -> int:
        return amount * econ.home_defense_num // econ.home_defense_den if nests[colony] == node else amount

    damage: dict[tuple[int, int], int] = defaultdict(int)     # (source, target) -> damage dealt this tick
    for (source, target), count in striking.items():
        damage[(source, target)] += scaled(source, count * econ.soldier_damage)
    attackers_of: dict[int, dict[int, int]] = defaultdict(dict)
    for (source, target), count in striking.items():
        attackers_of[target][source] = count
    for target in sorted(attackers_of):
        defenders = soldiers[target] - attacking_total.get(target, 0)
        if defenders <= 0:
            continue
        base, total = defenders * econ.soldier_damage, sum(attackers_of[target].values())
        for source in sorted(attackers_of[target]):
            part = scaled(target, base * attackers_of[target][source] // total)
            if part:
                damage[(target, source)] += part

    incoming: dict[int, int] = defaultdict(int)
    for (_, target), amount in damage.items():
        incoming[target] += amount
    # categories are decided from the pre-combat state (a phase that kills the last defender does not hit the queen)
    tier = {}
    for target in incoming:
        if soldiers[target] > 0:
            tier[target] = "soldier"
        elif workers[target] > 0:
            tier[target] = "worker"
        elif nests[target] == node and state.colonies[target].queen_hp > 0:
            tier[target] = "queen"
        else:
            tier[target] = ""
    applied: dict[int, int] = {}
    killed: dict[int, int] = {}
    for target in sorted(incoming):
        colony = state.colonies[target]
        if not tier[target]:
            applied[target], killed[target] = 0, 0
            continue
        if tier[target] == "queen":
            hit = min(incoming[target], colony.queen_hp)
            colony.queen_hp -= hit
            applied[target], killed[target] = hit, 0
            events.append(event(tick, "queen_damage", target, -1, node, hit, colony.queen_hp))
            continue
        remaining, dead = incoming[target], 0
        for cohort in sorted((c for c in cohorts if c.colony_id == target and c.caste == tier[target]),
                             key=lambda c: c.cohort_id):
            if remaining <= 0:
                break
            take = min(remaining, cohort.hp_total)
            dead += apply_hp_loss(state, config, cohort, take, node)
            remaining -= take
        applied[target], killed[target] = incoming[target] - remaining, dead
        if dead:
            caste = tier[target]
            _book_non_age_deaths(colony, caste, dead)
            if caste == "worker":
                colony.window.losses_workers += dead
                record.combat_losses_workers += dead
            else:
                colony.window.losses_soldiers += dead
                record.combat_losses_soldiers += dead
            events.append(event(tick, "death", target, -1, node, CASTE_CODES[caste], dead))
    state.cohorts[:] = [c for c in state.cohorts if c.count > 0]

    for (source, target), amount in sorted(damage.items()):
        share = applied[target] * amount // incoming[target] if incoming[target] else 0
        events.append(event(tick, "combat", source, target, node, amount, share))
    record.combat_nodes += (node,)
    _remember(state, node, damage, striking, incoming, killed, tier, events, killers)
    _flee(state, world_map, node, incoming)


def _remember(state, node, damage, striking, incoming, killed, tier, events, killers) -> None:
    """What the colonies that took part in the fight can know of it: who hit whom, losses, kills, a dangerous node."""
    tick = state.tick
    for (source, target), amount in sorted(damage.items()):
        if amount <= 0:
            continue
        kills = killed[target] * amount // incoming[target] if incoming[target] else 0  # losses of ``target`` caused by ``source``
        attacked = striking.get((source, target), 0) > 0
        for me, other in ((target, source), (source, target)):
            memory = state.colonies[me].memory
            other_slot, node_slot = _foreign_slot(memory, other), _node_slot(memory, node)
            if other_slot < 0:
                continue
            record = memory.foreign[other_slot]
            if node_slot >= 0:
                known = memory.nodes[node_slot]
                known.danger_q = min(DANGER_MAX, known.danger_q + DANGER_STEP)
            if me == target:
                if attacked:
                    record.they_attacked_q += 1
                    push_memory_event(memory, MEM_ATTACKED, node_slot, other_slot, 1, tick)
                if kills:
                    record.kills_mine_q += kills
                    push_memory_event(memory, MEM_WE_LOST, node_slot, other_slot, kills, tick)
                if tier[target] == "queen":
                    push_memory_event(memory, MEM_QUEEN_HIT, node_slot, other_slot, amount, tick)
            else:
                if attacked:
                    record.i_attacked_q += 1
                if kills:
                    record.kills_theirs_q += kills
                    state.colonies[me].window.kills_observed += kills
                    push_memory_event(memory, MEM_WE_KILLED, node_slot, other_slot, kills, tick)
    # a queen at zero hit points: remember who dealt the most (ties: the smaller colony id; no other order matters)
    for target, colony in enumerate(state.colonies):
        if tier.get(target) == "queen" and colony.queen_hp == 0 and target not in killers:
            sources = [(amount, -source) for (source, t), amount in damage.items() if t == target]
            killers[target] = -max(sources)[1] if sources else -1


def _flee(state, world_map, node, incoming) -> None:
    """``workers_flee``: attacked workers away from their nest turn home once, with their cargo."""
    for target in sorted(incoming):
        colony = state.colonies[target]
        home = world_map.nests[target]
        if not colony.orders.workers_flee or home == node:
            continue
        for cohort in state.cohorts:
            if cohort.colony_id == target and cohort.caste == "worker" and cohort.loc.kind == "node" and cohort.loc.node_id == node:
                cohort.mission = Mission("withdraw", home, -1, "return", 0)


def queen_terminal(state: WorldStateE1, config: ConfigE1, world_map: MapE1, killers: dict[int, int],
                   events: list[EventE1]) -> None:
    """``queen_hp == 0``: the colony becomes an orphan. It keeps living (contracts 9.5); its soldiers fall back to the nest."""
    tick = state.tick
    for colony in state.colonies:
        if colony.status != "alive" or colony.queen_hp > 0:
            continue
        colony.status = "orphan"
        outcome = state.outcomes[colony.colony_id]
        if not outcome.finished:
            outcome.finished, outcome.reason, outcome.terminal_tick = True, "queen_death", tick
        colony.orders.egg_level = 0
        colony.orders.soldier_share_q = 0
        home = world_map.nests[colony.colony_id]
        memory = colony.memory
        memory.nodes[0].queen_seen = 2
        memory.nodes[0].queen_seen_tick = tick
        push_memory_event(memory, MEM_OWN_QUEEN_DEAD, 0, -1, 0, tick)
        events.append(event(tick, "queen_dead", colony.colony_id, killers.get(colony.colony_id, -1), home))
        for cohort in state.cohorts:
            if cohort.colony_id == colony.colony_id and cohort.caste == "soldier" and cohort.mission.kind in ("attack", "guard"):
                if cohort.loc.kind == "node" and cohort.loc.node_id == home:
                    cohort.mission = Mission("home", -1, -1, "idle", 0)
                else:
                    cohort.mission = Mission("withdraw", home, -1, "return", 0)
