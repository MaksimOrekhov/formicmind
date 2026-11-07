"""Typed schemas of Ecosystem v1: MapE1 (static) and WorldStateE1 / ColonyStateE1 / CohortE1 / ColonyMemoryE1 (dynamic).

Field names follow docs/ecosystem_v1_contracts.md sections 3-4. Deviations are additions that the contract implies but did not spell out
(each marked ``# +``): there is no ``null`` in canonical JSON, so optional values are empty tuples or ``-1`` sentinels, and the
colony memory stores the world ``node_id`` next to its slot (the list index) so that it is self-contained.

``MapE1`` and its parts are frozen (tuples); the state classes are plain mutable dataclasses for the future engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .codec import U64

Caste = Literal["worker", "soldier"]
NodeKind = Literal["junction", "patch", "nest"]
KnownKind = Literal["unknown", "junction", "patch", "nest"]
Zone = Literal["near", "border", "far"]
ColonyStatus = Literal["alive", "orphan", "extinct"]
MissionKind = Literal["home", "forage", "scout", "guard", "attack", "withdraw"]
MissionPhase = Literal["idle", "go", "work", "return"]
SquadMission = Literal["none", "guard", "attack", "withdraw"]
LocationKind = Literal["node", "edge"]


# ------------------------------------------------------------------------------------------------ map (static)


@dataclass(frozen=True, slots=True)
class PatchSpec:
    zone: Zone  # + generation intent only: it implies NO ownership and is never shown to a policy
    capacity: int
    regen: int
    site_cap: int
    start_stock: int


@dataclass(frozen=True, slots=True)
class NodeE1:
    node_id: int
    kind: NodeKind
    nest_of: int  # colony_id owning the nest, -1 for non-nests
    patch: tuple[PatchSpec, ...]  # + empty for non-patches, exactly one element for patches (no null in canonical JSON)
    x: int  # integer layout coordinates, viewer only
    y: int


@dataclass(frozen=True, slots=True)
class EdgeE1:
    a: int
    b: int
    length: int
    lane_cap: int
    chokepoint: bool


@dataclass(frozen=True, slots=True)
class GenerationInfo:
    algorithm: str
    attempts: int  # 1 = the first candidate layout passed every check


@dataclass(frozen=True, slots=True)
class MapE1:
    schema: str
    map_seed: U64
    config_hash: str
    n_colonies: int
    nodes: tuple[NodeE1, ...]
    edges: tuple[EdgeE1, ...]
    nests: tuple[int, ...]  # nests[colony_id] = node_id of the nest
    generation: GenerationInfo
    map_hash: str


# ------------------------------------------------------------------------------------------------ colony memory


@dataclass(slots=True)
class KnownNode:
    node_id: int  # + world id behind the slot (the slot is the list index); never leaves the colony's memory
    kind: KnownKind  # + "unknown" until the node has been visited
    visited: bool
    discovered_tick: int
    seen_tick: int  # -1 = never seen
    patch_capacity: int  # 0 unless a visited patch
    last_stock: int
    last_stock_tick: int  # -1 = never
    last_delta_q: int
    nest_owner_slot: int  # -1 = unknown
    queen_seen: int  # 0 unknown, 1 alive, 2 dead
    queen_seen_tick: int
    danger_q: int
    harvest_ema_q: int


@dataclass(slots=True)
class KnownEdge:
    slot_a: int
    slot_b: int
    length: int
    lane_cap: int
    discovered_tick: int
    traversed: bool
    traffic_ema_q: int


@dataclass(slots=True)
class ForeignColonyRecord:
    colony_id: int  # + world id behind the slot (the slot is the list index); never leaves the colony's memory
    first_contact_tick: int
    last_seen_tick: int
    last_seen_workers_bucket: int
    last_seen_soldiers_bucket: int
    nest_node_slot: int  # -1 = unknown
    they_attacked_q: int
    i_attacked_q: int
    kills_mine_q: int
    kills_theirs_q: int


@dataclass(slots=True)
class Sighting:
    node_slot: int
    colony_slot: int
    tick: int
    workers_bucket: int
    soldiers_bucket: int
    aggression_observed: bool


@dataclass(slots=True)
class MemoryEvent:
    code: int
    node_slot: int
    colony_slot: int
    magnitude: int
    tick: int


@dataclass(slots=True)
class KnownPile:
    node_slot: int
    stock: int
    seen_tick: int


@dataclass(slots=True)
class ColonyMemoryE1:
    nodes: list[KnownNode] = field(default_factory=list)
    edges: list[KnownEdge] = field(default_factory=list)
    foreign: list[ForeignColonyRecord] = field(default_factory=list)
    sightings: list[Sighting] = field(default_factory=list)
    events: list[MemoryEvent] = field(default_factory=list)
    piles_seen: list[KnownPile] = field(default_factory=list)


# ------------------------------------------------------------------------------------------------ colony / cohort state


@dataclass(slots=True)
class AgeBucket:
    bucket_start: int  # may be negative: start adults are born "before" tick 0
    count: int


@dataclass(slots=True)
class AgeLedger:
    workers: list[AgeBucket] = field(default_factory=list)
    soldiers: list[AgeBucket] = field(default_factory=list)


@dataclass(slots=True)
class BroodRecord:
    cohort_id: int
    caste: Caste
    laid_tick: int
    hatch_tick: int


@dataclass(slots=True)
class SquadOrder:
    mission: SquadMission
    size_q: int
    target_node_slot: int  # -1 = none
    target_colony_slot: int  # -1 = none


@dataclass(slots=True)
class ColonyOrders:
    egg_level: int
    soldier_share_q: int
    forage_frac_q: int
    scout_n: int
    workers_flee: bool
    forage_targets: list[int]
    forage_shares_q: list[int]
    scout_targets: list[int]
    squads: list[SquadOrder]


@dataclass(slots=True)
class WindowStatsE1:
    harvested: int = 0        # food loaded onto workers at patches (not yet spendable)
    delivered: int = 0        # food actually put into the own nest storage: the colony's income (contracts 9.1)
    upkeep_due: int = 0       # upkeep owed by the units alive that tick: the colony's burn (contracts 9.1), paid or not
    upkeep_paid: int = 0      # upkeep actually taken from the storage
    egg_spend: int = 0
    losses_workers: int = 0
    losses_soldiers: int = 0
    kills_observed: int = 0


@dataclass(slots=True)
class RollingE1:
    income_ema_q: int
    burn_ema_q: int
    pop_ema_q: int


@dataclass(slots=True)
class Location:
    kind: LocationKind
    node_id: int  # the node (kind == "node"); -1 on an edge
    edge_from: int  # -1 at a node
    edge_to: int
    depart_tick: int
    arrive_tick: int


@dataclass(slots=True)
class Mission:
    kind: MissionKind
    target_node: int  # -1 = none
    target_colony: int  # -1 = none
    phase: MissionPhase
    dwell_left: int


@dataclass(slots=True)
class CohortE1:
    cohort_id: int
    colony_id: int
    caste: Caste
    count: int
    hp_total: int
    loc: Location
    mission: Mission
    load: int


@dataclass(slots=True)
class ColonyStateE1:
    colony_id: int
    status: ColonyStatus
    food: int
    queen_hp: int
    next_lay_tick: int
    lay_acc_q: int
    soldier_acc_q: int
    upkeep_remainder10: int
    starvation_debt: int
    brood: list[BroodRecord]
    ages: AgeLedger
    orders: ColonyOrders
    memory: ColonyMemoryE1
    window: WindowStatsE1
    rolling: RollingE1


@dataclass(slots=True)
class ColonyOutcome:
    finished: bool
    reason: str  # "" while the colony is running
    terminal_tick: int  # -1 while running


@dataclass(slots=True)
class WorldOutcome:
    finished: bool
    reason: str
    terminal_tick: int


@dataclass(slots=True)
class PatchState:
    node_id: int
    stock: int


@dataclass(slots=True)
class PileState:
    node_id: int
    stock: int


@dataclass(slots=True)
class EdgeFlow:
    a: int
    b: int
    direction: int  # 0: a -> b, 1: b -> a
    in_flight_units: int


@dataclass(slots=True)
class WorldStateE1:
    schema: str
    config_hash: str
    map_hash: str
    tick: int
    patches: list[PatchState]
    piles: list[PileState]
    colonies: list[ColonyStateE1]
    cohorts: list[CohortE1]
    next_cohort_id: int
    edge_flow: list[EdgeFlow]
    outcomes: list[ColonyOutcome]
    world_outcome: WorldOutcome


# ------------------------------------------------------------------------------------------------ run state (RNG)


@dataclass(slots=True)
class StreamState:
    stream_id: str
    state: U64
    inc: U64


@dataclass(slots=True)
class RunStateE1:
    root_seed: U64
    streams: list[StreamState]
