"""Fixed, deterministic L0 strategic baselines (FOR-19).

These are hand-written design profiles, not tuned or validated strong agents. They
only inspect ObservationV2 and ActionMaskV2 and return the common 17-logit contract;
the shared policy decoder remains the only action-selection/validation path.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from .actions import ActionMaskV2
from .decoder import PolicyLogitsV2
from .observation import ObservationV2
from .rng import Pcg32


@dataclass(frozen=True, slots=True)
class SharedBaselineConfig:
    food_stale_freshness_slope: float = 3.0
    unknown_food_capacity_prior: float = 0.20
    food_regeneration_weight: float = 1.5
    public_route_cost_per_edge_length: float = 0.08
    unreachable_route_penalty: float = 8.0
    food_score_floor: float = 0.01
    enemy_stale_freshness_slope: float = 3.0
    enemy_worker_threat_weight: float = 0.20
    forage_support_limit: int = 3
    forage_logit_positive_base: float = 1.0
    forage_logit_delta_floor: float = -0.75
    forage_logit_delta_ceiling: float = 1.0
    forage_logit_rank_penalty: float = 0.01
    forage_logit_unselected: float = -1.0
    selected_categorical_logit: float = 2.0
    unselected_categorical_logit: float = -2.0


@dataclass(frozen=True, slots=True)
class RandomBaselineConfig:
    forage_support_max: int
    base_random_logits: int
    forage_support_selection: str
    forage_target_selection: str
    avoid_center_threshold: float
    selection: str


@dataclass(frozen=True, slots=True)
class TurtleBaselineConfig:
    """Пороги starvation_debt заданы в нормализованных единицах ObservationV2."""

    defense_threat: float
    worker_soldier_ratio: float
    lay_none_food_below: int
    lay_none_starvation_debt: float
    lay_none_starvation_delta: float
    lay_soldier_food_at_least: int
    soldier_lay_requires_known_threat: bool
    forage_support_normal: int
    forage_support_when_food_below: int
    avoid_center_when_allowed: bool


@dataclass(frozen=True, slots=True)
class RushBaselineConfig:
    """Порог lay_none_starvation_debt задан для сигнала debt/(debt+10)."""

    minimum_soldiers_to_raid: int
    soldier_worker_ratio: float
    defense_threat: float
    lay_none_food_below: int
    lay_none_starvation_debt: float
    forage_support: int
    avoid_center_when_allowed: bool


@dataclass(frozen=True, slots=True)
class BalancedBaselineConfig:
    """Пороги starvation_debt заданы в нормализованных единицах ObservationV2."""

    soldier_worker_ratio: float
    minimum_soldiers_to_raid: int
    defense_threat: float
    defense_local_nodes: str
    hungry_food_below: int
    lay_none_food_below: int
    lay_none_starvation_debt: float
    soldier_lay_food_at_least: int
    hungry_starvation_debt: float
    soldier_lay_target: str
    hungry_starvation_delta: float
    forage_support_normal: int
    forage_support_when_hungry: int
    raid_requires_current_known_enemy_queen: bool
    avoid_center_when_defending: bool
    avoid_center_score_when_idle: float


@dataclass(frozen=True, slots=True)
class RaiderBaselineConfig:
    """Порог hungry_starvation_debt задан для сигнала debt/(debt+10)."""

    minimum_soldiers_to_raid: int
    soldier_worker_ratio_to_raid: float
    defense_target_requires_current_enemy_soldier: bool
    defense_local_nodes: str
    defense_soldier_threshold: float
    raid_requires_current_known_enemy_queen: bool
    visible_intercept_priority: str
    outermost_reachable_position_fallback: bool
    hungry_food_below: int
    hungry_starvation_debt: float
    hungry_starvation_delta: float
    lay_none_food_below: int
    soldier_lay_food_at_least: int
    forage_support: int
    avoid_center_when_allowed: bool


ProfileConfig = (RandomBaselineConfig | TurtleBaselineConfig | RushBaselineConfig
                 | BalancedBaselineConfig | RaiderBaselineConfig)


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    shared: SharedBaselineConfig
    profile: ProfileConfig


@dataclass(frozen=True, slots=True)
class BaselineSpec:
    name: str
    version: str
    config: BaselineConfig
    policy_factory: Callable[[BaselineConfig, bool], "BaselinePolicy"]

    @property
    def full_id(self) -> str:
        return f"{self.name}:{self.version}"

    @property
    def policy_id(self) -> str:
        """The RNG grammar uses a colon-free slug that still includes the version."""
        return f"{self.name}-{self.version}"

    @property
    def parameters(self) -> tuple[tuple[str, int | float | str | bool], ...]:
        """Совместимое представление профильных параметров в старом формате tuple."""
        return tuple(asdict(self.config.profile).items())

    @property
    def factory(self) -> Callable[[], "BaselinePolicy"]:
        """Совместимая безаргументная фабрика для прежних клиентов BaselineSpec."""
        return lambda: self.build()

    def parameter_object(self, config: BaselineConfig | None = None) -> dict[str, Any]:
        """Сериализовать тот же неизменяемый объект конфигурации, что получает политика."""
        actual = self.config if config is None else config
        return {"shared": asdict(actual.shared), "profile": asdict(actual.profile)}

    def digest(self, config: BaselineConfig | None = None) -> str:
        payload = json.dumps(
            {"id": self.full_id, "parameters": self.parameter_object(config)},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        return hashlib.sha256(payload.encode("ascii")).hexdigest()

    def build(self, config: BaselineConfig | None = None) -> "BaselinePolicy":
        actual = self.config if config is None else config
        return self.policy_factory(actual, self.version == "v2")


class BaselinePolicy:
    """Stateless policy helper base class."""

    def __call__(
        self, observation: ObservationV2, mask: ActionMaskV2, policy_rng: Pcg32
    ) -> PolicyLogitsV2:
        raise NotImplementedError


def _decode_u(value: float, scale: float) -> float:
    """Invert the observation contract's x/(x+scale) normalizer."""
    return 0.0 if value <= 0.0 else scale * value / max(1e-9, 1.0 - value)


ACTION_LOGIT_COUNT = 17
LAY_LOGIT_START = 0
FORAGE_LOGIT_START = 3
ARMY_LOGIT_START = 9
AVOID_CENTER_LOGIT_INDEX = 16


def _logits(lay: int, forage: list[float], army: int, avoid: float,
            shared: SharedBaselineConfig) -> PolicyLogitsV2:
    values = [shared.unselected_categorical_logit] * ACTION_LOGIT_COUNT
    values[LAY_LOGIT_START + lay] = shared.selected_categorical_logit
    values[FORAGE_LOGIT_START:ARMY_LOGIT_START] = forage
    values[ARMY_LOGIT_START + army] = shared.selected_categorical_logit
    values[AVOID_CENTER_LOGIT_INDEX] = avoid
    return PolicyLogitsV2(values)


def _food_scores(observation: ObservationV2, config: SharedBaselineConfig) -> list[float]:
    """Estimate forage value from permitted intel and public resource properties."""
    ratios = observation.block("last_food_ratio")
    known = observation.block("food_known")
    current = observation.block("food_current")
    age = observation.block("food_age_norm")
    capacity = observation.block("public_food_capacity")
    regen = observation.block("public_food_regen")
    edges = observation.block("public_edges")

    # Shortest public route distance from local home (0). Edges and topology are public.
    distances = [float("inf")] * 7
    distances[0] = 0.0
    edge_index = 0
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(7)]
    for i in range(7):
        for j in range(i + 1, 7):
            length = edges[edge_index]
            edge_index += 1
            if length > 0.0:
                adjacency[i].append((j, length))
                adjacency[j].append((i, length))
    pending = set(range(7))
    while pending:
        node = min(pending, key=lambda item: distances[item])
        pending.remove(node)
        if distances[node] == float("inf"):
            break
        for neighbour, length in adjacency[node]:
            distances[neighbour] = min(distances[neighbour], distances[node] + length)

    scores = []
    for node in range(6):  # ActionV2 has forage categories for local nodes 0..5.
        if known[node]:
            freshness = 1.0 if current[node] else max(0.0, 1.0 - config.food_stale_freshness_slope * age[node])
            expected_stock = ratios[node] * capacity[node] * freshness
        else:
            # Unknown stock is not treated as current intel; this is only a public-capacity prior.
            expected_stock = capacity[node] * config.unknown_food_capacity_prior
        route_cost = 1.0 + config.public_route_cost_per_edge_length * (
            distances[node] if distances[node] != float("inf") else config.unreachable_route_penalty
        )
        scores.append((expected_stock + config.food_regeneration_weight * regen[node]
                       + config.food_score_floor) / route_cost)
    return scores


def _forage_logits(scores: list[float], support: int, config: SharedBaselineConfig) -> list[float]:
    support = min(config.forage_support_limit, max(1, support), len(scores))
    ranked = sorted(range(len(scores)), key=lambda node: (-scores[node], node))
    selected = ranked[:support]
    peak = max(scores, default=0.0)
    result = [config.forage_logit_unselected] * 6
    for rank, node in enumerate(selected):
        # Keep positive support fixed while preserving a deterministic preference order.
        result[node] = (config.forage_logit_positive_base + max(
            config.forage_logit_delta_floor,
            min(config.forage_logit_delta_ceiling, (scores[node] - peak) / max(1.0, peak)),
        ) - rank * config.forage_logit_rank_penalty)
    return result


_INTEGER_DECODE_LIMITS: dict[float, int] = {
    10.0: 512,
    32.0: 2048,
    64.0: 4096,
    250.0: 32768,
    300.0: 32768,
    1200.0: 32768,
}


def _decode_u_integer(value: float, scale: float, *, maximum: int | None = None) -> int:
    """Восстановить целое, закодированное U(x; S) и один раз округлённое до float32.

    В заданных диапазонах ошибка нормализации float32 и обратного преобразования
    меньше 0,5 (границы полностью перебираются регрессионным тестом). Более крупные
    значения ограничиваются специальным значением на единицу выше диапазона; пороги
    стратегий для этих показателей лежат существенно ниже границ диапазона.
    """
    if maximum is None:
        try:
            maximum = _INTEGER_DECODE_LIMITS[float(scale)]
        except KeyError as exc:
            raise ValueError(f"no integer decode bound registered for scale {scale}") from exc
    if value <= 0.0:
        return 0
    if value >= 1.0:
        return maximum + 1
    estimate = _decode_u(value, scale)
    if estimate >= maximum + 0.5:
        return maximum + 1
    return max(0, math.floor(estimate + 0.5))


def _known_threats(observation: ObservationV2, config: SharedBaselineConfig, *,
                   integer_metrics: bool = False) -> list[float]:
    known = observation.block("enemy_known")
    current = observation.block("enemy_current")
    ages = observation.block("enemy_age_norm")
    threats = []
    for node in range(7):
        if not known[node]:
            threats.append(0.0)
            continue
        freshness = 1.0 if current[node] else max(0.0, 1.0 - config.enemy_stale_freshness_slope * ages[node])
        workers_value, soldiers_value = observation.enemy_counts(node)
        if integer_metrics:
            workers = _decode_u_integer(workers_value, 64.0)
            soldiers = _decode_u_integer(soldiers_value, 64.0)
        else:
            workers = _decode_u(workers_value, 64.0)
            soldiers = _decode_u(soldiers_value, 64.0)
        threats.append(freshness * (soldiers + config.enemy_worker_threat_weight * workers))
    return threats


def _known_soldier_threats(observation: ObservationV2, config: SharedBaselineConfig, *,
                           integer_metrics: bool = False) -> list[float]:
    """Estimate enemy soldiers only, with the same visibility freshness rules."""
    known = observation.block("enemy_known")
    current = observation.block("enemy_current")
    ages = observation.block("enemy_age_norm")
    threats = []
    for node in range(7):
        if not known[node]:
            threats.append(0.0)
            continue
        freshness = 1.0 if current[node] else max(0.0, 1.0 - config.enemy_stale_freshness_slope * ages[node])
        _, soldiers_value = observation.enemy_counts(node)
        soldiers = (_decode_u_integer(soldiers_value, 64.0) if integer_metrics
                    else _decode_u(soldiers_value, 64.0))
        threats.append(freshness * soldiers)
    return threats


def _defensive_target(threats: list[float], mask: ActionMaskV2, threshold: float) -> int | None:
    if threats[0] >= threshold and mask.army_choice[0]:
        return 0
    candidates = [node for node in range(1, 6) if mask.army_choice[node] and threats[node] >= threshold]
    return max(candidates, key=lambda node: (threats[node], -node)) if candidates else None


def _own_counts(observation: ObservationV2, *, integer_metrics: bool = False) -> tuple[float, float, float]:
    own = observation.own_global
    if integer_metrics:
        return (
            _decode_u_integer(own.workers, 64.0),
            _decode_u_integer(own.soldiers, 64.0),
            _decode_u_integer(own.food, 250.0),
        )
    return _decode_u(own.workers, 64.0), _decode_u(own.soldiers, 64.0), _decode_u(own.food, 250.0)


def _starvation_debt_signal(observation: ObservationV2) -> float:
    """Вернуть нормализованный сигнал U(debt; 10), с которым заданы пороги профилей."""
    return observation.own_global.starvation_debt


class RandomPolicy(BaselinePolicy):
    """Uniform-ish legal logits generated exclusively from the policy's PCG32 stream."""

    def __init__(self, config: BaselineConfig, integer_metrics: bool = False):
        del integer_metrics
        if not isinstance(config.profile, RandomBaselineConfig):
            raise TypeError("RandomPolicy requires RandomBaselineConfig")
        self.baseline_config = config
        self.shared, self.config = config.shared, config.profile

    def __call__(self, observation: ObservationV2, mask: ActionMaskV2, policy_rng: Pcg32) -> PolicyLogitsV2:
        del observation, mask
        values = [policy_rng.next_u32() / 2**32 for _ in range(self.config.base_random_logits)]
        if self.config.forage_support_selection != f"uniform_1_to_{self.config.forage_support_max}":
            raise ValueError("unsupported random forage support selection")
        if self.config.selection != "pcg32_uniform_legal_logits":
            raise ValueError("unsupported random policy selection")
        if self.config.forage_target_selection != "pcg32_fisher_yates":
            raise ValueError("unsupported random forage target selection")
        selected_count = 1 + policy_rng.below(self.config.forage_support_max)
        targets = list(range(6))
        for index in range(len(targets) - 1, 0, -1):
            other = policy_rng.below(index + 1)
            targets[index], targets[other] = targets[other], targets[index]
        values[FORAGE_LOGIT_START:ARMY_LOGIT_START] = [
            self.shared.forage_logit_unselected - value
            for value in values[FORAGE_LOGIT_START:ARMY_LOGIT_START]
        ]
        for target in targets[:selected_count]:
            values[FORAGE_LOGIT_START + target] = (
                self.shared.forage_logit_positive_base + policy_rng.next_u32() / 2**32
            )
        values[AVOID_CENTER_LOGIT_INDEX] = (
            1.0 if values[AVOID_CENTER_LOGIT_INDEX] >= self.config.avoid_center_threshold else -1.0
        )
        # The decoder masks the categorical choices and forbidden center routes.
        return PolicyLogitsV2(values)


class TurtlePolicy(BaselinePolicy):
    def __init__(self, config: BaselineConfig, integer_metrics: bool = False):
        if not isinstance(config.profile, TurtleBaselineConfig):
            raise TypeError("TurtlePolicy requires TurtleBaselineConfig")
        self.baseline_config = config
        self.shared, self.config, self.integer_metrics = config.shared, config.profile, integer_metrics

    def __call__(self, observation: ObservationV2, mask: ActionMaskV2, policy_rng: Pcg32) -> PolicyLogitsV2:
        del policy_rng
        c = self.config
        workers, soldiers, food = _own_counts(observation, integer_metrics=self.integer_metrics)
        own = observation.own_global
        debt_signal = _starvation_debt_signal(observation)
        hungry = food < c.lay_none_food_below or debt_signal > c.lay_none_starvation_debt \
            or own.starvation_delta > c.lay_none_starvation_delta
        threat = _known_threats(observation, self.shared, integer_metrics=self.integer_metrics)
        target = _defensive_target(threat, mask, c.defense_threat)
        if target is None:
            army = 0
        else:
            army = target
        if target == 0:
            lay = 2 if food >= c.lay_soldier_food_at_least else 0
        elif hungry:
            lay = 0
        elif (soldiers < max(2.0, workers * c.worker_soldier_ratio)
              and (not c.soldier_lay_requires_known_threat or max(threat, default=0.0) >= c.defense_threat)):
            lay = 2
        else:
            lay = 1
        scores = _food_scores(observation, self.shared)
        support = c.forage_support_normal if food >= c.lay_soldier_food_at_least else c.forage_support_when_food_below
        avoid = 1.0 if c.avoid_center_when_allowed else -1.0
        return _logits(lay, _forage_logits(scores, support, self.shared), army, avoid, self.shared)


class RushPolicy(BaselinePolicy):
    def __init__(self, config: BaselineConfig, integer_metrics: bool = False):
        if not isinstance(config.profile, RushBaselineConfig):
            raise TypeError("RushPolicy requires RushBaselineConfig")
        self.baseline_config = config
        self.shared, self.config, self.integer_metrics = config.shared, config.profile, integer_metrics

    def __call__(self, observation: ObservationV2, mask: ActionMaskV2, policy_rng: Pcg32) -> PolicyLogitsV2:
        del policy_rng
        c = self.config
        workers, soldiers, food = _own_counts(observation, integer_metrics=self.integer_metrics)
        own = observation.own_global
        debt_signal = _starvation_debt_signal(observation)
        target = _defensive_target(
            _known_threats(observation, self.shared, integer_metrics=self.integer_metrics), mask, c.defense_threat
        )
        prepared = soldiers >= c.minimum_soldiers_to_raid
        if target is not None:
            army = target
        elif prepared and mask.army_choice[6]:
            army = 6
        else:
            army = 0
        hungry = food < c.lay_none_food_below or debt_signal > c.lay_none_starvation_debt
        if hungry:
            lay = 0
        elif soldiers < c.minimum_soldiers_to_raid or soldiers < workers * c.soldier_worker_ratio:
            lay = 2
        else:
            lay = 1
        avoid = self.shared.selected_categorical_logit if c.avoid_center_when_allowed else -1.0
        return _logits(lay, _forage_logits(_food_scores(observation, self.shared), c.forage_support, self.shared),
                       army, avoid, self.shared)


class BalancedPolicy(BaselinePolicy):
    def __init__(self, config: BaselineConfig, integer_metrics: bool = False):
        if not isinstance(config.profile, BalancedBaselineConfig):
            raise TypeError("BalancedPolicy requires BalancedBaselineConfig")
        self.baseline_config = config
        self.shared, self.config, self.integer_metrics = config.shared, config.profile, integer_metrics

    def __call__(self, observation: ObservationV2, mask: ActionMaskV2, policy_rng: Pcg32) -> PolicyLogitsV2:
        del policy_rng
        c = self.config
        workers, soldiers, food = _own_counts(observation, integer_metrics=self.integer_metrics)
        own = observation.own_global
        debt_signal = _starvation_debt_signal(observation)
        threat = _known_threats(observation, self.shared, integer_metrics=self.integer_metrics)
        # Keep the balanced profile home/near-home focused. Enemy workers at the
        # center or on the far side must not make its raid readiness unreachable.
        if c.defense_local_nodes != "home_and_1_to_2":
            raise ValueError("unsupported balanced defense node profile")
        near_home_threat = [*threat[:3], 0.0, 0.0, 0.0, 0.0]
        target = _defensive_target(near_home_threat, mask, c.defense_threat)
        enough_soldiers = soldiers >= max(c.minimum_soldiers_to_raid, workers * c.soldier_worker_ratio)
        if c.raid_requires_current_known_enemy_queen:
            enough_soldiers = enough_soldiers and observation.enemy_queen_current
        if target is not None:
            army = target
        elif enough_soldiers and mask.army_choice[6]:
            army = 6
        else:
            army = 0
        hungry = food < c.hungry_food_below or debt_signal > c.hungry_starvation_debt \
            or own.starvation_delta > c.hungry_starvation_delta
        if hungry and (debt_signal > c.lay_none_starvation_debt or food < c.lay_none_food_below):
            lay = 0
        elif c.soldier_lay_target == "raid_readiness" and not enough_soldiers \
                and (food >= c.soldier_lay_food_at_least or target is not None):
            lay = 2
        else:
            lay = 1
        support = c.forage_support_when_hungry if hungry else c.forage_support_normal
        avoid = (1.0 if target and c.avoid_center_when_defending
                 else c.avoid_center_score_when_idle)
        return _logits(lay, _forage_logits(_food_scores(observation, self.shared), support, self.shared),
                       army, avoid, self.shared)


class RaiderPolicy(BaselinePolicy):
    def __init__(self, config: BaselineConfig, integer_metrics: bool = False):
        if not isinstance(config.profile, RaiderBaselineConfig):
            raise TypeError("RaiderPolicy requires RaiderBaselineConfig")
        self.baseline_config = config
        self.shared, self.config, self.integer_metrics = config.shared, config.profile, integer_metrics

    def __call__(self, observation: ObservationV2, mask: ActionMaskV2, policy_rng: Pcg32) -> PolicyLogitsV2:
        del policy_rng
        c = self.config
        workers, soldiers, food = _own_counts(observation, integer_metrics=self.integer_metrics)
        own = observation.own_global
        debt_signal = _starvation_debt_signal(observation)
        threats = _known_threats(observation, self.shared, integer_metrics=self.integer_metrics)
        soldier_threats = _known_soldier_threats(
            observation, self.shared, integer_metrics=self.integer_metrics
        )
        # Forward intercepts are for current enemy soldiers, never worker sightings.
        # Once ready, only a soldier threat at home or the first local approach delays
        # the raid; farther groups do not create an endless precondition loop.
        current = observation.block("enemy_current")
        visible_targets = [node for node in range(1, 6) if current[node] and soldier_threats[node] > 0.0
                           and mask.army_choice[node]]
        prepared_to_raid = soldiers >= max(c.minimum_soldiers_to_raid, workers * c.soldier_worker_ratio_to_raid)
        if c.raid_requires_current_known_enemy_queen:
            prepared_to_raid = prepared_to_raid and observation.enemy_queen_current
        if c.defense_local_nodes != "home_and_1":
            raise ValueError("unsupported raider defense node profile")
        near_home_soldiers = [
            soldier_threats[node] if (current[node] or not c.defense_target_requires_current_enemy_soldier) else 0.0
            for node in range(2)
        ] + [0.0] * 5
        defense_target = _defensive_target(near_home_soldiers, mask, c.defense_soldier_threshold)
        if defense_target is not None:
            army = defense_target
        elif prepared_to_raid and mask.army_choice[6]:
            army = 6
        elif visible_targets:
            if c.visible_intercept_priority == "farthest_current_soldier_before_ready":
                army = max(visible_targets, key=lambda node: (node, threats[node])) if not self.integer_metrics \
                    else max(visible_targets)
            else:
                raise ValueError("unsupported raider visible intercept priority")
        else:
            # Advance/position on the outermost reachable visible route; ActionV2 has
            # no explicit target waypoint, so this is the closest legal pressure order.
            army = (max((node for node in range(1, 6) if mask.army_choice[node]), default=0)
                    if c.outermost_reachable_position_fallback else 0)
        hungry = food < c.hungry_food_below or debt_signal > c.hungry_starvation_debt \
            or own.starvation_delta > c.hungry_starvation_delta
        if hungry and food < c.lay_none_food_below:
            lay = 0
        elif (soldiers < max(c.minimum_soldiers_to_raid, workers * c.soldier_worker_ratio_to_raid)
              and food >= c.soldier_lay_food_at_least):
            lay = 2
        else:
            lay = 1
        avoid = 1.0 if c.avoid_center_when_allowed else -1.0
        return _logits(lay, _forage_logits(_food_scores(observation, self.shared), c.forage_support, self.shared),
                       army, avoid, self.shared)


_SHARED = SharedBaselineConfig()

_PROFILE_CONFIGS: dict[str, ProfileConfig] = {
    "random": RandomBaselineConfig(3, 17, "uniform_1_to_3", "pcg32_fisher_yates", 0.5,
                                   "pcg32_uniform_legal_logits"),
    "turtle": TurtleBaselineConfig(0.5, 0.18, 55, 0.08, 0.08, 80, True, 1, 2, True),
    "rush": RushBaselineConfig(4, 0.42, 1.0, 35, 0.12, 1, False),
    "balanced": BalancedBaselineConfig(
        0.28, 6, 0.35, "home_and_1_to_2", 65, 30, 0.15, 45, 0.08, "raid_readiness", 0.08,
        2, 3, False, True, -0.25,
    ),
    "raider": RaiderBaselineConfig(
        5, 0.40, True, "home_and_1", 1.0, False, "farthest_current_soldier_before_ready", True,
        45, 0.10, 0.10, 25, 25, 2, True,
    ),
}
_POLICY_FACTORIES: dict[str, Callable[[BaselineConfig, bool], BaselinePolicy]] = {
    "random": RandomPolicy,
    "turtle": TurtlePolicy,
    "rush": RushPolicy,
    "balanced": BalancedPolicy,
    "raider": RaiderPolicy,
}
_BASELINE_NAMES = tuple(_PROFILE_CONFIGS)
_SPECS: dict[str, BaselineSpec] = {
    f"{name}:{version}": BaselineSpec(name, version, BaselineConfig(_SHARED, profile), _POLICY_FACTORIES[name])
    for name, profile in _PROFILE_CONFIGS.items()
    for version in ("v1", "v2")
}


def baseline_names() -> tuple[str, ...]:
    """Старые краткие имена, которые по-прежнему выбирают Baselines v1."""
    return _BASELINE_NAMES


def baseline_selectors() -> tuple[str, ...]:
    """Селекторы CLI и manifest, включая явные версии стратегий."""
    return (*_BASELINE_NAMES, *(f"{name}:v{version}" for name in _BASELINE_NAMES for version in (1, 2)))


def baseline_spec(name: str, version: str | None = None) -> BaselineSpec:
    if ":" in name:
        if version is not None:
            raise ValueError("version must be omitted when the baseline selector includes :version")
        name, version = name.rsplit(":", 1)
    version = "v1" if version is None else version
    if version in ("1", "2"):
        version = f"v{version}"
    key = f"{name}:{version}"
    try:
        return _SPECS[key]
    except KeyError:
        supported = ", ".join(baseline_selectors())
        raise ValueError(f"unknown baseline {key!r}; expected one of {supported}") from None


def make_baseline(name: str, version: str | None = None,
                  config: BaselineConfig | None = None) -> BaselinePolicy:
    return baseline_spec(name, version).build(config)


__all__ = [
    "BalancedBaselineConfig", "BaselineConfig", "BaselinePolicy", "BaselineSpec", "RandomBaselineConfig",
    "RaiderBaselineConfig", "RushBaselineConfig", "SharedBaselineConfig", "TurtleBaselineConfig",
    "baseline_names", "baseline_selectors", "baseline_spec", "make_baseline",
]
