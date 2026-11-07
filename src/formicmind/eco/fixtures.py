"""Deterministic diagnostic fixtures. They are not agents, observations or teachers.

* Economy fixtures ``idle``, ``peaceful_fair``, ``greedy_scarcity``, ``hoarder`` (E1-b) only assign foraging and laying and are
  deliberately omniscient about patch geometry; they never give a military order, so peaceful colonies stay peaceful.
* Roles ``scout``, ``turtle``, ``raider``, ``retreat`` (E1-c, ``roles.py``) run the ``peaceful_fair`` economy and add exploration or war
  *through the validated order API and the colony's own memory only*.

A ``FixtureController`` gives every colony a role: ``default`` for all, ``roles`` for chosen colony ids.
"""

from __future__ import annotations

from math import ceil

from .config import ConfigE1
from .engine import RouteTable, split_cohort
from .model import MapE1, Mission, WorldStateE1
from .roles import ROLE_NAMES, decide, react, soldier_share_for

ECONOMY_NAMES = ("idle", "peaceful_fair", "greedy_scarcity", "hoarder")
FIXTURE_NAMES = (*ECONOMY_NAMES, *ROLE_NAMES)
_ALIASES = {"fair": "peaceful_fair", "greedy": "greedy_scarcity", "guard": "turtle"}
FIXTURE_CHOICES = (*FIXTURE_NAMES, *_ALIASES)
_ZONE_ORDER = {"near": 0, "border": 1, "far": 2}


def normalize_fixture(name: str) -> str:
    canonical = _ALIASES.get(name, name)
    if canonical not in FIXTURE_NAMES:
        raise ValueError(f"fixture must be one of {FIXTURE_NAMES}")
    return canonical


def _economy_of(name: str) -> str:
    return name if name in ECONOMY_NAMES else "peaceful_fair"


class FixtureController:
    """Per-colony roles over one shared economy routine."""

    def __init__(self, default: str, roles: dict[int, str] | None = None) -> None:
        self.default = normalize_fixture(default)
        self.roles = {int(colony): normalize_fixture(role) for colony, role in (roles or {}).items()}
        self.name = self.default

    def role_of(self, colony_id: int) -> str:
        return self.roles.get(colony_id, self.default)

    def __call__(self, state: WorldStateE1, config: ConfigE1, world_map: MapE1, routes: RouteTable) -> None:
        if routes.alert:  # extra decision during a fight: only the war roles react, the economy waits for its planned decision
            for colony_id in sorted(routes.alert):
                react(state, config, world_map, colony_id, self.role_of(colony_id), routes.outbox)
            return
        # Military orders first: scouts and squads take their units from those idle at the nest before foraging claims them.
        for colony in state.colonies:
            if colony.status == "alive" and self.role_of(colony.colony_id) in ROLE_NAMES:
                decide(state, config, world_map, colony.colony_id, self.role_of(colony.colony_id), routes.outbox)
        self._economy(state, config, world_map, routes)

    def _economy(self, state: WorldStateE1, config: ConfigE1, world_map: MapE1, routes: RouteTable) -> None:
        modes = {colony.colony_id: _economy_of(self.role_of(colony.colony_id)) for colony in state.colonies}
        for colony in state.colonies:
            mode = modes[colony.colony_id]
            if mode == "idle":
                colony.orders.egg_level = 0
                colony.orders.soldier_share_q = 0
                continue
            if colony.status != "alive":
                continue
            workers = sum(c.count for c in state.cohorts if c.colony_id == colony.colony_id and c.caste == "worker")
            soldiers = sum(c.count for c in state.cohorts if c.colony_id == colony.colony_id and c.caste == "soldier")
            if mode == "hoarder":
                colony.orders.egg_level = 0
            else:
                due_tenths = (colony.upkeep_remainder10 + config.economy.worker_upkeep_tenths * workers
                              + config.economy.soldier_upkeep_tenths * soldiers)
                projected_upkeep = due_tenths // 10
                burn_tenths = (config.economy.worker_upkeep_tenths * workers
                               + config.economy.soldier_upkeep_tenths * soldiers)
                reserve = (burn_tenths * 25 + 19) // 20
                colony.orders.egg_level = (
                    3 if colony.food - projected_upkeep >= reserve + config.economy.worker_egg_cost else 0
                )
            colony.orders.soldier_share_q = soldier_share_for(self.role_of(colony.colony_id), state, colony.colony_id)

        active: dict[tuple[int, int], int] = {}
        available: dict[int, list] = {}
        for colony in state.colonies:
            home = world_map.nests[colony.colony_id]
            available[colony.colony_id] = []
            for cohort in state.cohorts:
                if cohort.colony_id != colony.colony_id or cohort.caste != "worker":
                    continue
                if cohort.mission.kind == "forage" and cohort.mission.phase in ("go", "work", "return"):
                    key = (colony.colony_id, cohort.mission.target_node)
                    active[key] = active.get(key, 0) + cohort.count
                elif (cohort.loc.kind == "node" and cohort.loc.node_id == home
                      and cohort.mission.kind == "home" and cohort.mission.phase == "idle"):
                    available[colony.colony_id].append(cohort)
            available[colony.colony_id].sort(key=lambda c: c.cohort_id)

        for colony in state.colonies:
            mode = modes[colony.colony_id]
            if mode == "idle" or colony.status == "extinct" or colony.food >= config.economy.storage_cap:
                continue  # an orphan keeps its foraging plan (contracts 5), re-sized to the workers it has left
            nest = world_map.nests[colony.colony_id]
            patch_targets = []
            for node in world_map.nodes:
                if not node.patch:
                    continue
                distance = routes.distance(nest, node.node_id)
                if distance >= 16:
                    continue  # a trip at distance 16 has zero net worker income in the liquid model
                patch = node.patch[0]
                sharers = sum(routes.distance(other_nest, node.node_id) <= config.map_params.contest_radius
                              for other_nest in world_map.nests)
                sharers = max(1, sharers)
                if mode == "peaceful_fair":
                    # Target the calculator's fair share, corrected by the actual graph distance.
                    target_workers = max(1, (patch.regen * (2 * distance + 8) + 16 * sharers - 1)
                                         // (16 * sharers))
                else:
                    # Each colony makes a peaceful labour claim up to the site's real throughput.
                    target_workers = max(1, ceil(patch.site_cap / config.economy.harvest_rate))
                patch_targets.append((distance, _ZONE_ORDER[patch.zone], node.node_id, target_workers))
            patch_targets.sort()
            idle_cohorts = available[colony.colony_id]
            for _, _, node_id, target_workers in patch_targets:
                have = active.get((colony.colony_id, node_id), 0)
                need = max(0, target_workers - have)
                while need and idle_cohorts:
                    cohort = idle_cohorts.pop(0)
                    if cohort.mission.kind == "forage" or cohort.count <= 0:
                        continue
                    sending = min(need, cohort.count)
                    traveler = split_cohort(state, cohort, sending)
                    traveler.mission = Mission("forage", node_id, -1, "go", 0)
                    if traveler is not cohort:
                        # The unsent remainder is still at home and can fill another patch quota.
                        idle_cohorts.insert(0, cohort)
                    need -= sending


PeacefulFixture = FixtureController  # the E1-b name


def fixture_controller(name: str, roles: dict[int, str] | None = None) -> FixtureController:
    return FixtureController(name, roles)
