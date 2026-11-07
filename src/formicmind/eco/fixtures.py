"""Deterministic, peaceful diagnostics for E1-b. These fixtures are not agents or observations."""

from __future__ import annotations

from math import ceil

from .config import ConfigE1
from .engine import RouteTable, split_cohort
from .model import MapE1, Mission, WorldStateE1

FIXTURE_NAMES = ("idle", "peaceful_fair", "greedy_scarcity", "hoarder")
_ALIASES = {"fair": "peaceful_fair", "greedy": "greedy_scarcity"}
FIXTURE_CHOICES = (*FIXTURE_NAMES, *_ALIASES)
_ZONE_ORDER = {"near": 0, "border": 1, "far": 2}


def normalize_fixture(name: str) -> str:
    canonical = _ALIASES.get(name, name)
    if canonical not in FIXTURE_NAMES:
        raise ValueError(f"fixture must be one of {FIXTURE_NAMES}")
    return canonical


class PeacefulFixture:
    """Full-map test policy that only assigns forage/home missions and laying orders.

    It is intentionally omniscient about map geometry and patch roles. That access is confined to
    diagnostic fixtures; E1-b does not build ObservationE1 or expose this policy to an agent.
    """

    def __init__(self, name: str) -> None:
        self.name = normalize_fixture(name)

    def __call__(self, state: WorldStateE1, config: ConfigE1, world_map: MapE1, routes: RouteTable) -> None:
        if self.name == "idle":
            for colony in state.colonies:
                colony.orders.egg_level = 0
                colony.orders.soldier_share_q = 0
            return
        for colony in state.colonies:
            if colony.status != "alive":
                continue
            workers = sum(c.count for c in state.cohorts if c.colony_id == colony.colony_id and c.caste == "worker")
            soldiers = sum(c.count for c in state.cohorts if c.colony_id == colony.colony_id and c.caste == "soldier")
            if self.name == "hoarder":
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
            colony.orders.soldier_share_q = 0

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
                      and cohort.mission.phase == "idle"):
                    available[colony.colony_id].append(cohort)
            available[colony.colony_id].sort(key=lambda c: c.cohort_id)

        for colony in state.colonies:
            if colony.status != "alive" or colony.food >= config.economy.storage_cap:
                continue
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
                if self.name == "peaceful_fair":
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


def fixture_controller(name: str) -> PeacefulFixture:
    return PeacefulFixture(name)
