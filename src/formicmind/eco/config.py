"""ConfigE1: the immutable, versioned configuration of an Ecosystem v1 world (contracts section 2).

Every number is an integer. The economy and resource numbers are the *candidate* values of docs/ecosystem_v1_economy.md
(``eco1-candidate-unvalidated``); nothing here is a validated balance. Scenario scaling is exact integer arithmetic
(per-mille, round half up), identical to the rounding of ``scripts/ecosystem_budget.py`` for the tabulated values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

from ..canonical import CanonicalError, canonical_dumps, canonical_loads, domain_hash
from .codec import SchemaError, from_canonical, to_canonical
from .versions import CONFIG_SCHEMA, ECO_BALANCE_VERSION, ECO_RULES_VERSION, HASH_DOMAIN_CONFIG

Scenario = Literal["abundance", "moderate", "severe"]
SCENARIOS: tuple[str, ...] = ("abundance", "moderate", "severe")
SCENARIO_PERMILLE = {"abundance": 1500, "moderate": 1000, "severe": 600}
ZONES: tuple[str, ...] = ("near", "border", "far")
SIZES: tuple[int, ...] = (6, 7, 8)


class ConfigError(CanonicalError):
    """A configuration value is outside its allowed range."""


@dataclass(frozen=True, slots=True)
class EconomyParams:
    worker_upkeep_tenths: int = 4
    soldier_upkeep_tenths: int = 8
    worker_egg_cost: int = 10
    soldier_egg_cost: int = 20
    worker_hatch_delay: int = 20
    soldier_hatch_delay: int = 30
    lay_interval_ticks: int = 20
    eggs_per_lay: int = 1
    start_workers: int = 10
    start_soldiers: int = 2
    start_food: int = 80
    storage_cap: int = 600
    worker_hp: int = 20
    soldier_hp: int = 30
    queen_hp: int = 240
    soldier_damage: int = 2
    home_defense_num: int = 5
    home_defense_den: int = 4
    starvation_unit_cost: int = 5
    carry_load: int = 16
    harvest_rate: int = 2
    lane_cap: int = 8
    narrow_lane_cap: int = 4
    harvest_patience: int = 8
    scout_dwell: int = 2
    attack_dwell: int = 3
    initial_known_radius: int = 5
    lifespan_worker: int = 2000
    lifespan_soldier: int = 2000
    start_age_spread_permille: int = 500
    age_bucket: int = 100


@dataclass(frozen=True, slots=True)
class PatchBase:
    """Base (``moderate``, sigma = 1.0) parameters of one food patch."""

    capacity: int
    regen: int
    site_cap: int


@dataclass(frozen=True, slots=True)
class ResourceParams:
    near: PatchBase = PatchBase(100, 5, 10)  # placed close to one nest
    border: PatchBase = PatchBase(120, 6, 12)  # between two neighbouring nests
    far: PatchBase = PatchBase(160, 8, 16)  # at the junction of three neighbours


@dataclass(frozen=True, slots=True)
class MapParams:
    n_nodes: int = 96
    near_per_nest: int = 2
    n_border: int = 8
    n_far: int = 8
    world_side: int = 894  # integer units; the layout is only for the viewer, rules use path lengths in ticks
    units_per_tick: int = 24
    min_edge_ticks: int = 2
    max_edge_ticks: int = 6
    junction_min_sep: int = 40  # units
    surroundings_per_nest: int = 6  # plain junctions placed 2.2-4.8 ticks around every nest
    edge_k: int = 3  # extra edges join each node to its k nearest neighbours
    max_degree: int = 5
    max_attempts: int = 40
    # structural limits checked on every generated map
    min_nest_path: int = 14  # ticks between any two nests
    max_nest_nn_path: int = 40  # ticks from a nest to its nearest neighbour nest
    near_min_path: int = 3
    near_max_path: int = 6
    economic_radius: int = 10  # ticks: supply "within reach" (economy section 2)
    contest_radius: int = 14  # ticks: a patch within this path of two or more nests is contested (profitable: break-even is 16)
    choke_radius: int = 8
    # fairness thresholds (RFC 5.2); coefficients of variation in per-mille
    max_near_cv_permille: int = 100
    max_radius_cv_permille: int = 200
    max_choke_near: int = 2
    max_narrow_share_permille: int = 150


@dataclass(frozen=True, slots=True)
class OptionsE1:
    stop_when_alone: bool = False
    loot_on_queen_death: bool = False
    workers_flee_default: bool = False


@dataclass(frozen=True, slots=True)
class ConfigE1:
    schema: str
    eco_rules_version: str
    eco_balance_version: str
    n_colonies: int
    scenario: Scenario
    horizon: int
    decision_k: int
    alert_interval: int
    economy: EconomyParams
    resources: ResourceParams
    map_params: MapParams
    options: OptionsE1

    def patch_count(self) -> int:
        m = self.map_params
        return m.near_per_nest * self.n_colonies + m.n_border + m.n_far


def default_config(n_colonies: int = 8, scenario: str = "moderate", n_nodes: int | None = None) -> ConfigE1:
    """The documented default world for ``n_colonies`` colonies (8 -> 96 nodes, 32 patches, 6000 ticks)."""
    if n_colonies not in SIZES:
        raise ConfigError(f"n_colonies must be one of {SIZES}, got {n_colonies!r}")
    if scenario not in SCENARIOS:
        raise ConfigError(f"scenario must be one of {SCENARIOS}, got {scenario!r}")
    nodes = 8 * n_colonies + 32 if n_nodes is None else n_nodes
    side = math.isqrt(n_colonies * 100_000)
    params = replace(MapParams(), n_nodes=nodes, n_border=n_colonies, n_far=n_colonies, world_side=side)
    cfg = ConfigE1(CONFIG_SCHEMA, ECO_RULES_VERSION, ECO_BALANCE_VERSION, n_colonies, scenario, 6000, 30, 10,
                   EconomyParams(), ResourceParams(), params, OptionsE1())
    validate_config(cfg)
    return cfg


def scaled(value: int, scenario: str) -> int:
    """``value * sigma`` with sigma in per-mille, round half up, at least 1 for positive inputs."""
    permille = SCENARIO_PERMILLE[scenario]
    out = (value * permille + 500) // 1000
    return max(1, out) if value > 0 else 0


def patch_parameters(config: ConfigE1, zone: str) -> tuple[int, int, int]:
    """``(capacity, regen, site_cap)`` of a patch in ``zone`` for the config's scenario."""
    base = getattr(config.resources, zone)
    return (scaled(base.capacity, config.scenario), scaled(base.regen, config.scenario),
            scaled(base.site_cap, config.scenario))


def _check(cond: bool, message: str) -> None:
    if not cond:
        raise ConfigError(message)


def validate_config(config: ConfigE1) -> None:
    _check(config.schema == CONFIG_SCHEMA, f"unknown config schema {config.schema!r}")
    _check(config.eco_rules_version == ECO_RULES_VERSION, "unsupported eco_rules_version")
    _check(config.eco_balance_version == ECO_BALANCE_VERSION, "unsupported eco_balance_version")
    _check(config.n_colonies in SIZES, "n_colonies must be 6, 7 or 8")
    _check(config.scenario in SCENARIOS, "unknown scenario")
    _check(config.horizon >= 1 and config.decision_k >= 1 and config.alert_interval >= 0, "bad time parameters")
    _check(config.alert_interval == 0 or config.alert_interval <= config.decision_k, "alert_interval must not exceed decision_k")
    m, e = config.map_params, config.economy
    _check(80 <= m.n_nodes <= 120, "n_nodes must be in 80..120")
    patches = config.patch_count()
    _check(24 <= patches <= 40, "the patch count must be in 24..40")
    _check(1 <= m.near_per_nest <= 3 and m.n_border >= 0 and m.n_far >= 0, "bad patch counts")
    _check(m.n_nodes >= config.n_colonies + patches + 8, "too few nodes for the nests and patches")
    _check(m.units_per_tick >= 1 and 1 <= m.min_edge_ticks <= m.max_edge_ticks, "bad edge length parameters")
    _check(2 <= m.max_degree <= 8 and 1 <= m.edge_k <= 8 and m.max_attempts >= 1, "bad graph parameters")
    _check(m.world_side >= 200 and m.junction_min_sep >= 1 and 0 <= m.surroundings_per_nest <= 12, "bad world size")
    _check(1 <= m.near_min_path <= m.near_max_path and m.min_nest_path <= m.max_nest_nn_path, "bad path limits")
    _check(0 <= m.max_near_cv_permille <= 1000 and 0 <= m.max_radius_cv_permille <= 1000, "bad CV thresholds")
    _check(0 <= m.max_narrow_share_permille <= 1000 and m.max_choke_near >= 0, "bad chokepoint thresholds")
    for name, value in vars_of(e).items():
        _check(type(value) is int and value >= 0, f"economy.{name} must be a non-negative integer")
    _check(e.worker_hp >= 1 and e.soldier_hp >= 1 and e.queen_hp >= 1 and e.home_defense_den >= 1, "bad HP parameters")
    _check(e.lane_cap >= e.narrow_lane_cap >= 1, "narrow_lane_cap must not exceed lane_cap")
    _check(e.carry_load >= 1 and e.harvest_rate >= 1 and e.lay_interval_ticks >= 1 and e.age_bucket >= 1, "bad economy parameters")
    _check(e.start_workers >= 1 and e.start_food <= e.storage_cap, "bad start state")
    _check(0 <= e.start_age_spread_permille <= 1000, "bad start_age_spread_permille")
    for zone in ZONES:
        base = getattr(config.resources, zone)
        _check(base.capacity >= 1 and base.regen >= 1 and base.site_cap >= base.regen, f"bad resource parameters for {zone}")
        cap, regen, site = patch_parameters(config, zone)
        _check(cap >= regen >= 1 and site >= regen, f"scenario scaling breaks the invariants of {zone}")


def vars_of(obj: object) -> dict[str, int]:
    return {name: getattr(obj, name) for name in obj.__slots__}  # type: ignore[attr-defined]


def config_to_canonical(config: ConfigE1) -> dict:
    return to_canonical(config)


def config_from_canonical(data: object) -> ConfigE1:
    try:
        config = from_canonical(ConfigE1, data)
    except SchemaError:
        raise
    validate_config(config)
    return config


def config_to_bytes(config: ConfigE1) -> bytes:
    return canonical_dumps(config_to_canonical(config))


def config_from_bytes(data: bytes) -> ConfigE1:
    return config_from_canonical(canonical_loads(data))


def config_hash(config: ConfigE1) -> str:
    return domain_hash(HASH_DOMAIN_CONFIG, config_to_bytes(config))
