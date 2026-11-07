"""ActionE1/v1 and the masks that bound it (contracts section 8).

``action_masks(obs)`` is a pure function of an ``ObservationE1``: this module (like ``decoder.py``) imports neither ``WorldStateE1`` nor ``MapE1`` nor
the engine, so a mask can only use what the colony observes. Targets are local slots. A target is legal when the colony knows a path to it over the
edges it has itself discovered (``node_ints[...][3] >= 0``) and its local information has the right freshness; an enemy colony can be aimed at only
through a sighting no older than 150 ticks or a nest the colony has found, and a colony the agent has not met has no slot at all. Nothing in the
mask depends on whether the target colony is still alive: the colony cannot know that.
"""

from __future__ import annotations

from dataclasses import dataclass

from .observation import KIND_PATCH, ObservationE1

ACTION_VERSION = "ActionE1/v1"
EGG_LEVELS, SOLDIER_SHARES, FORAGE_FRACS, SCOUT_NS, FLEE_OPTIONS = 4, 5, 9, 5, 2
SQUADS = 3
SQUAD_MISSIONS = ("none", "guard", "attack", "withdraw")
SQUAD_SIZES = 5                  # size_q 0..4; 0 only for NONE / WITHDRAW
MAX_ATTACK_TARGETS = 32 + 7      # fresh sightings + known nests (contracts 8.1)
MAX_FORAGE_TARGETS, MAX_SCOUT_TARGETS = 3, 2
MAX_SCOUTS_PER_ORDER = 4
SCOUT_REVISIT_AGE = 90           # a visited node may be scouted again once the information is this old
ATTACK_SIGHTING_AGE = 150        # a sighting is an attack target while it is at most this old
SHARE_TOTAL = 16


class ActionError(ValueError):
    """The action is not legal for its masks (a technical error of the policy, never a defeat of the colony)."""


@dataclass(frozen=True, slots=True)
class SquadActionE1:
    mission: str      # one of SQUAD_MISSIONS
    size_q: int       # 0..4 quarters of all soldiers; 0 unless GUARD / ATTACK
    node_slot: int    # target node slot, -1 = none
    colony_slot: int  # target foreign colony slot (ATTACK only), -1 = none


NO_SQUAD = SquadActionE1("none", 0, -1, -1)


@dataclass(frozen=True, slots=True)
class ActionE1:
    egg_level: int
    soldier_share_q: int
    forage_frac_q: int
    scout_n: int
    workers_flee: bool
    forage_targets: tuple[int, ...] = ()
    forage_shares_q: tuple[int, ...] = ()
    scout_targets: tuple[int, ...] = ()
    squads: tuple[SquadActionE1, SquadActionE1, SquadActionE1] = (NO_SQUAD, NO_SQUAD, NO_SQUAD)
    version: str = ACTION_VERSION


def action_to_canonical(action: ActionE1) -> dict:
    """Plain JSON data of an action (slots only), for decision records and hashes."""
    return {
        "version": action.version, "egg_level": action.egg_level, "soldier_share_q": action.soldier_share_q,
        "forage_frac_q": action.forage_frac_q, "scout_n": action.scout_n, "workers_flee": action.workers_flee,
        "forage_targets": list(action.forage_targets), "forage_shares_q": list(action.forage_shares_q),
        "scout_targets": list(action.scout_targets),
        "squads": [{"mission": s.mission, "size_q": s.size_q, "node_slot": s.node_slot, "colony_slot": s.colony_slot} for s in action.squads],
    }


@dataclass(frozen=True, slots=True)
class ActionMasksE1:
    """Legal choices for one decision. All blocks are functions of the observation; the three integers are the unit counts it reports."""

    egg_level: tuple[bool, ...]
    soldier_share: tuple[bool, ...]
    forage_frac: tuple[bool, ...]
    scout_n: tuple[bool, ...]
    workers_flee: tuple[bool, ...]
    forage_target: tuple[bool, ...]          # per node slot
    scout_target: tuple[bool, ...]           # per node slot
    guard_target: tuple[bool, ...]           # per node slot
    attack_pairs: tuple[tuple[int, int], ...]  # candidate (node_slot, colony_slot), at most 39, no padding
    squad_mission: tuple[tuple[bool, ...], ...]  # 3 x 4
    squad_size: tuple[tuple[bool, ...], ...]     # 3 x 5
    soldiers_total: int
    soldiers_idle_home: int
    workers_idle_home: int


def attack_candidates(obs: ObservationE1) -> tuple[tuple[int, int], ...]:
    """Pairs ``(node_slot, colony_slot)`` the colony may attack: fresh sightings first (observation order), then nests it has found."""
    pairs: list[tuple[int, int]] = []
    nodes = obs.node_ints
    for node_slot, colony_slot, age in obs.sighting_ints:
        if age <= ATTACK_SIGHTING_AGE and nodes[node_slot][3] >= 0 and (node_slot, colony_slot) not in pairs:
            pairs.append((node_slot, colony_slot))
    for colony_slot, node_slot in enumerate(obs.foreign_nest_slots):
        if node_slot >= 0 and nodes[node_slot][3] >= 0 and (node_slot, colony_slot) not in pairs:
            pairs.append((node_slot, colony_slot))
    return tuple(pairs[:MAX_ATTACK_TARGETS])


def action_masks(obs: ObservationE1) -> ActionMasksE1:
    facts = obs.facts
    nodes = obs.node_ints
    forage = tuple(kind == KIND_PATCH and visited == 1 and path >= 0 and slot != 0 for slot, (kind, visited, _age, path) in enumerate(nodes))
    scout = tuple(slot != 0 and path >= 0 and (visited == 0 or age >= SCOUT_REVISIT_AGE) for slot, (_kind, visited, age, path) in enumerate(nodes))
    guard = tuple(visited == 1 and path >= 0 for (_kind, visited, _age, path) in nodes)
    attack = attack_candidates(obs)
    can_forage = any(forage) and facts.workers > 0
    max_scouts = min(MAX_SCOUTS_PER_ORDER, facts.idle_workers_home) if any(scout) else 0
    soldiers_free = facts.idle_soldiers_home >= 1
    mission = tuple((True, soldiers_free and any(guard), soldiers_free and bool(attack), True) for _ in range(SQUADS))
    size = tuple((True, *(soldiers_free,) * 4) for _ in range(SQUADS))
    return ActionMasksE1(
        egg_level=(True,) * EGG_LEVELS,
        soldier_share=(True,) * SOLDIER_SHARES,
        forage_frac=(True, *(can_forage,) * (FORAGE_FRACS - 1)),
        scout_n=tuple(n <= max_scouts for n in range(SCOUT_NS)),
        workers_flee=(True, True),
        forage_target=forage, scout_target=scout, guard_target=guard, attack_pairs=attack,
        squad_mission=mission, squad_size=size,
        soldiers_total=facts.soldiers, soldiers_idle_home=facts.idle_soldiers_home, workers_idle_home=facts.idle_workers_home)


def squad_take(total: int, available: int, size_q: int) -> int:
    """Soldiers a squad of ``size_q`` quarters takes from the idle ones (the rule of ``orders._squad_units``)."""
    return min(available, max(1, total * size_q // 4))


def validate_action(action: ActionE1, masks: ActionMasksE1) -> None:
    """Raise ``ActionError`` unless ``action`` is legal for ``masks`` (types, ranges, masks, shares, ``squad`` bookkeeping)."""

    def fail(message: str) -> None:
        raise ActionError(message)

    def integer(value: object) -> bool:
        return type(value) is int

    if not isinstance(action, ActionE1) or action.version != ACTION_VERSION:
        fail("not an ActionE1/v1")
    for name, value, mask in (("egg_level", action.egg_level, masks.egg_level), ("soldier_share_q", action.soldier_share_q, masks.soldier_share),
                              ("forage_frac_q", action.forage_frac_q, masks.forage_frac), ("scout_n", action.scout_n, masks.scout_n)):
        if not integer(value) or not 0 <= value < len(mask) or not mask[value]:
            fail(f"{name}={value!r} is not allowed")
    if type(action.workers_flee) is not bool:
        fail("workers_flee must be a bool")
    forage, shares = tuple(action.forage_targets), tuple(action.forage_shares_q)
    if len(forage) > MAX_FORAGE_TARGETS or len(forage) != len(shares) or len(set(forage)) != len(forage):
        fail("forage targets: at most 3 distinct slots with one share each")
    if any(not integer(s) or not 0 <= s < len(masks.forage_target) or not masks.forage_target[s] for s in forage):
        fail("a forage target is not a legal slot")
    if (sum(shares) != SHARE_TOTAL) if forage else bool(shares):
        fail("forage shares must sum to 16 (and be empty without targets)")
    if not all(integer(v) and v >= 1 for v in shares):
        fail("forage shares are positive integers")
    if (action.forage_frac_q > 0) != bool(forage):
        fail("forage_frac_q > 0 needs 1..3 forage targets, and none without it")
    scouts = tuple(action.scout_targets)
    if len(scouts) > MAX_SCOUT_TARGETS or len(set(scouts)) != len(scouts):
        fail("scout targets: at most 2 distinct slots")
    if any(not integer(s) or not 0 <= s < len(masks.scout_target) or not masks.scout_target[s] for s in scouts):
        fail("a scout target is not a legal slot")
    if action.scout_n > 0 and not 1 <= len(scouts) <= action.scout_n:
        fail("scout_n > 0 needs 1 target (2 only with 2+ scouts)")
    if action.scout_n == 0 and scouts:
        fail("scout targets without scouts")
    if len(action.squads) != SQUADS:
        fail("three squad orders are required")
    available = masks.soldiers_idle_home
    for index, squad in enumerate(action.squads):
        if not isinstance(squad, SquadActionE1) or squad.mission not in SQUAD_MISSIONS:
            fail(f"squad {index}: bad mission")
        code = SQUAD_MISSIONS.index(squad.mission)
        if not masks.squad_mission[index][code]:
            fail(f"squad {index}: mission {squad.mission} is not allowed")
        if squad.mission in ("none", "withdraw"):
            if (squad.size_q, squad.node_slot, squad.colony_slot) != (0, -1, -1):
                fail(f"squad {index}: {squad.mission} carries no size or target")
            continue
        if not integer(squad.size_q) or not 1 <= squad.size_q < SQUAD_SIZES or not masks.squad_size[index][squad.size_q]:
            fail(f"squad {index}: size_q {squad.size_q!r} is not allowed")
        if available < 1:
            fail(f"squad {index}: no idle soldiers are left for it")
        available -= squad_take(masks.soldiers_total, available, squad.size_q)
        if squad.mission == "guard":
            if (not integer(squad.node_slot) or not 0 <= squad.node_slot < len(masks.guard_target) or not masks.guard_target[squad.node_slot]
                    or squad.colony_slot != -1):
                fail(f"squad {index}: guard target is not legal")
        elif (squad.node_slot, squad.colony_slot) not in masks.attack_pairs:
            fail(f"squad {index}: attack target is neither a fresh sighting nor a known nest")
    if action.scout_n > masks.workers_idle_home:
        fail("more scouts than idle workers at the nest")

