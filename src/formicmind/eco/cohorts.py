"""Cohort bookkeeping shared by the world tick, combat and the order API: splitting, removing members, cargo drops, age ledger."""

from __future__ import annotations

from .config import ConfigE1
from .model import AgeBucket, CohortE1, ColonyStateE1, Location, MapE1, Mission, PileState, WorldStateE1


def _node_location(node_id: int) -> Location:
    return Location("node", node_id, -1, -1, 0, 0)


def _home_mission() -> Mission:
    return Mission("home", -1, -1, "idle", 0)


def _add_pile(state: WorldStateE1, node_id: int, amount: int) -> None:
    if amount <= 0:
        return
    for pile in state.piles:
        if pile.node_id == node_id:
            pile.stock += amount
            return
    state.piles.append(PileState(node_id, amount))
    state.piles.sort(key=lambda pile: pile.node_id)


def _unit_loads(load: int, count: int) -> tuple[int, int]:
    """Return (load on the first unit, load left on survivors) for a single removed unit."""
    base, extra = divmod(load, count)
    return base + int(extra > 0), load - base - int(extra > 0)


def _where_rank(cohort: CohortE1, home: int) -> int:
    if cohort.loc.kind == "edge":
        return 2
    return 0 if cohort.loc.node_id == home else 1


def remove_cohort_members(state: WorldStateE1, config: ConfigE1, world_map: MapE1, colony_id: int,
                          caste: str, count: int, *, book_age_deaths: bool) -> tuple[int, int]:
    """Remove adults in deterministic physical order; dead cargo becomes food at the death node."""
    if count <= 0:
        return 0, 0
    home = world_map.nests[colony_id]
    victims = sorted((c for c in state.cohorts if c.colony_id == colony_id and c.caste == caste),
                     key=lambda c: (_where_rank(c, home), c.cohort_id))
    removed = dropped = 0
    unit_hp = config.economy.worker_hp if caste == "worker" else config.economy.soldier_hp
    for cohort in victims:
        if removed >= count:
            break
        take = min(cohort.count, count - removed)
        for _ in range(take):
            if cohort.count <= 1:
                drop = cohort.load
            else:
                drop, remaining_load = _unit_loads(cohort.load, cohort.count)
                cohort.load = remaining_load
            death_node = cohort.loc.node_id if cohort.loc.kind == "node" else cohort.loc.edge_from
            _add_pile(state, death_node, drop)
            dropped += drop
            cohort.count -= 1
            cohort.hp_total -= min(unit_hp, cohort.hp_total)
            removed += 1
        if cohort.count == 0:
            state.cohorts.remove(cohort)
    if book_age_deaths and removed:
        _book_non_age_deaths(state.colonies[colony_id], caste, removed)
    return removed, dropped


def _book_non_age_deaths(colony: ColonyStateE1, caste: str, count: int) -> None:
    ledger = colony.ages.workers if caste == "worker" else colony.ages.soldiers
    total = sum(bucket.count for bucket in ledger)
    if not total or not count:
        return
    count = min(count, total)
    quotients = [(bucket.count * count) // total for bucket in ledger]
    remainders = [(bucket.count * count) % total for bucket in ledger]
    left = count - sum(quotients)
    order = sorted(range(len(ledger)), key=lambda i: (-remainders[i], ledger[i].bucket_start))
    for index in order[:left]:
        quotients[index] += 1
    kept = [AgeBucket(bucket.bucket_start, bucket.count - take)
            for bucket, take in zip(ledger, quotients) if bucket.count - take > 0]
    if caste == "worker":
        colony.ages.workers = kept
    else:
        colony.ages.soldiers = kept


def _split_load(load: int, count: int, moving: int) -> int:
    base, extra = divmod(load, count)
    return base * moving + min(extra, moving)


def split_cohort(state: WorldStateE1, cohort: CohortE1, count: int) -> CohortE1:
    """Split ``count`` adults from a cohort, preserving cargo and hit points exactly."""
    if not 0 < count <= cohort.count:
        raise ValueError("split count must be between one and the cohort size")
    if count == cohort.count:
        return cohort
    unit_hp = cohort.hp_total // cohort.count
    part_load = _split_load(cohort.load, cohort.count, count)
    part = CohortE1(
        state.next_cohort_id, cohort.colony_id, cohort.caste, count, count * unit_hp,
        Location(cohort.loc.kind, cohort.loc.node_id, cohort.loc.edge_from, cohort.loc.edge_to,
                 cohort.loc.depart_tick, cohort.loc.arrive_tick),
        Mission(cohort.mission.kind, cohort.mission.target_node, cohort.mission.target_colony,
                cohort.mission.phase, cohort.mission.dwell_left),
        part_load,
    )
    state.next_cohort_id += 1
    cohort.count -= count
    cohort.hp_total -= count * unit_hp
    cohort.load -= part_load
    state.cohorts.append(part)
    state.cohorts.sort(key=lambda item: item.cohort_id)
    return part


def apply_hp_loss(state: WorldStateE1, config: ConfigE1, cohort: CohortE1, loss: int, node_id: int) -> int:
    """Remove ``loss`` hit points from a cohort (pool semantics: ``count = ceil(hp_total / unit_hp)``).

    Units that die drop their share of the cargo as a pile at ``node_id``. The cohort is *not* removed from the state when it
    reaches zero; the caller does that. Returns the number of units killed.
    """
    unit_hp = config.economy.worker_hp if cohort.caste == "worker" else config.economy.soldier_hp
    loss = min(loss, cohort.hp_total)
    cohort.hp_total -= loss
    alive = -(-cohort.hp_total // unit_hp)
    killed = cohort.count - alive
    for _ in range(killed):
        if cohort.count <= 1:
            drop = cohort.load
            cohort.load = 0
        else:
            drop, remaining = _unit_loads(cohort.load, cohort.count)
            cohort.load = remaining
        _add_pile(state, node_id, drop)
        cohort.count -= 1
    return killed
