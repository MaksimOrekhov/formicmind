"""Fixed, deterministic L0 strategic baselines (FOR-19).

These are hand-written design profiles, not tuned or validated strong agents. They
only inspect ObservationV2 and ActionMaskV2 and return the common 17-logit contract;
the shared policy decoder remains the only action-selection/validation path.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .actions import ActionMaskV2
from .decoder import PolicyLogitsV2
from .observation import ObservationV2
from .rng import Pcg32


@dataclass(frozen=True, slots=True)
class BaselineSpec:
    name: str
    version: str
    parameters: tuple[tuple[str, int | float | str | bool], ...]
    factory: Callable[[], "BaselinePolicy"]

    @property
    def full_id(self) -> str:
        return f"{self.name}:{self.version}"

    @property
    def policy_id(self) -> str:
        """The RNG grammar uses a colon-free slug that still includes the version."""
        return f"{self.name}-{self.version}"

    def parameter_object(self) -> dict[str, Any]:
        return {"shared": dict(_SHARED_PARAMETERS), "profile": dict(self.parameters)}

    def digest(self) -> str:
        payload = json.dumps(
            {"id": self.full_id, "parameters": self.parameter_object()},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        return hashlib.sha256(payload.encode("ascii")).hexdigest()


class BaselinePolicy:
    """Stateless policy helper base class."""

    def __call__(
        self, observation: ObservationV2, mask: ActionMaskV2, policy_rng: Pcg32
    ) -> PolicyLogitsV2:
        raise NotImplementedError


def _decode_u(value: float, scale: float) -> float:
    """Invert the observation contract's x/(x+scale) normalizer."""
    return 0.0 if value <= 0.0 else scale * value / max(1e-9, 1.0 - value)


_SHARED_PARAMETERS: dict[str, int | float | str | bool] = {
    "food_stale_freshness_slope": 3.0,
    "unknown_food_capacity_prior": 0.20,
    "food_regeneration_weight": 1.5,
    "public_route_cost_per_edge_length": 0.08,
    "unreachable_route_penalty": 8.0,
    "food_score_floor": 0.01,
    "enemy_stale_freshness_slope": 3.0,
    "enemy_worker_threat_weight": 0.20,
    "forage_support_limit": 3,
    "forage_logit_positive_base": 1.0,
    "forage_logit_delta_floor": -0.75,
    "forage_logit_delta_ceiling": 1.0,
    "forage_logit_rank_penalty": 0.01,
    "forage_logit_unselected": -1.0,
    "selected_categorical_logit": 2.0,
    "unselected_categorical_logit": -2.0,
}


def _logits(lay: int, forage: list[float], army: int, avoid: float) -> PolicyLogitsV2:
    values = [-2.0] * 17
    values[lay] = 2.0
    values[3:9] = forage
    values[9 + army] = 2.0
    values[16] = avoid
    return PolicyLogitsV2(values)


def _food_scores(observation: ObservationV2) -> list[float]:
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
            freshness = 1.0 if current[node] else max(0.0, 1.0 - 3.0 * age[node])
            expected_stock = ratios[node] * capacity[node] * freshness
        else:
            # Unknown stock is not treated as current intel; this is only a public-capacity prior.
            expected_stock = capacity[node] * 0.20
        route_cost = 1.0 + 0.08 * (distances[node] if distances[node] != float("inf") else 8.0)
        scores.append((expected_stock + 1.5 * regen[node] + 0.01) / route_cost)
    return scores


def _forage_logits(scores: list[float], support: int) -> list[float]:
    support = min(3, max(1, support), len(scores))
    ranked = sorted(range(len(scores)), key=lambda node: (-scores[node], node))
    selected = ranked[:support]
    peak = max(scores, default=0.0)
    result = [-1.0] * 6
    for rank, node in enumerate(selected):
        # Keep positive support fixed while preserving a deterministic preference order.
        result[node] = 1.0 + max(-0.75, min(1.0, (scores[node] - peak) / max(1.0, peak))) - rank * 0.01
    return result


def _own_counts(observation: ObservationV2) -> tuple[float, float, float]:
    workers, soldiers = observation.block("own_global")[1:3]
    return _decode_u(workers, 64.0), _decode_u(soldiers, 64.0), _decode_u(observation.block("own_global")[0], 250.0)


def _known_threats(observation: ObservationV2) -> list[float]:
    counts = observation.block("enemy_count_last")
    known = observation.block("enemy_known")
    current = observation.block("enemy_current")
    ages = observation.block("enemy_age_norm")
    threats = []
    for node in range(7):
        if not known[node]:
            threats.append(0.0)
            continue
        freshness = 1.0 if current[node] else max(0.0, 1.0 - 3.0 * ages[node])
        workers = _decode_u(counts[2 * node], 64.0)
        soldiers = _decode_u(counts[2 * node + 1], 64.0)
        threats.append(freshness * (soldiers + 0.2 * workers))
    return threats


def _known_soldier_threats(observation: ObservationV2) -> list[float]:
    """Estimate enemy soldiers only, with the same visibility freshness rules."""
    counts = observation.block("enemy_count_last")
    known = observation.block("enemy_known")
    current = observation.block("enemy_current")
    ages = observation.block("enemy_age_norm")
    threats = []
    for node in range(7):
        if not known[node]:
            threats.append(0.0)
            continue
        freshness = 1.0 if current[node] else max(0.0, 1.0 - 3.0 * ages[node])
        soldiers = _decode_u(counts[2 * node + 1], 64.0)
        threats.append(freshness * soldiers)
    return threats


def _defensive_target(threats: list[float], mask: ActionMaskV2, threshold: float) -> int | None:
    if threats[0] >= threshold and mask.army_choice[0]:
        return 0
    candidates = [node for node in range(1, 6) if mask.army_choice[node] and threats[node] >= threshold]
    return max(candidates, key=lambda node: (threats[node], -node)) if candidates else None


class RandomPolicy(BaselinePolicy):
    """Uniform-ish legal logits generated exclusively from the policy's PCG32 stream."""

    def __call__(self, observation: ObservationV2, mask: ActionMaskV2, policy_rng: Pcg32) -> PolicyLogitsV2:
        values = [policy_rng.next_u32() / 2**32 for _ in range(17)]
        selected_count = 1 + policy_rng.below(3)
        targets = list(range(6))
        for index in range(len(targets) - 1, 0, -1):
            other = policy_rng.below(index + 1)
            targets[index], targets[other] = targets[other], targets[index]
        values[3:9] = [-1.0 - value for value in values[3:9]]
        for target in targets[:selected_count]:
            values[3 + target] = 1.0 + policy_rng.next_u32() / 2**32
        values[16] = 1.0 if values[16] >= 0.5 else -1.0
        # The decoder masks the categorical choices and forbidden center routes.
        return PolicyLogitsV2(values)


class TurtlePolicy(BaselinePolicy):
    def __call__(self, observation: ObservationV2, mask: ActionMaskV2, policy_rng: Pcg32) -> PolicyLogitsV2:
        del policy_rng
        workers, soldiers, food = _own_counts(observation)
        own = observation.block("own_global")
        hungry = food < 55.0 or own[12] > 0.08 or own[16] > 0.08
        threat = _known_threats(observation)
        target = _defensive_target(threat, mask, 0.5)
        if target is None:
            army = 0
        else:
            army = target
        if target == 0:
            lay = 2 if food >= 80.0 else 0
        elif hungry:
            lay = 0
        elif soldiers < max(2.0, workers * 0.18) and max(threat, default=0.0) >= 0.5:
            lay = 2
        else:
            lay = 1
        scores = _food_scores(observation)
        return _logits(lay, _forage_logits(scores, 1 if food >= 80.0 else 2), army, 1.0)


class RushPolicy(BaselinePolicy):
    def __call__(self, observation: ObservationV2, mask: ActionMaskV2, policy_rng: Pcg32) -> PolicyLogitsV2:
        del policy_rng
        workers, soldiers, food = _own_counts(observation)
        own = observation.block("own_global")
        target = _defensive_target(_known_threats(observation), mask, 1.0)
        prepared = soldiers >= 4.0
        if target is not None:
            army = target
        elif prepared and mask.army_choice[6]:
            army = 6
        else:
            army = 0
        hungry = food < 35.0 or own[12] > 0.12
        if hungry:
            lay = 0
        elif soldiers < 4.0 or soldiers < workers * 0.42:
            lay = 2
        else:
            lay = 1
        return _logits(lay, _forage_logits(_food_scores(observation), 1), army, -1.0)


class BalancedPolicy(BaselinePolicy):
    def __call__(self, observation: ObservationV2, mask: ActionMaskV2, policy_rng: Pcg32) -> PolicyLogitsV2:
        del policy_rng
        workers, soldiers, food = _own_counts(observation)
        own = observation.block("own_global")
        threat = _known_threats(observation)
        # Keep the balanced profile home/near-home focused. Enemy workers at the
        # center or on the far side must not make its raid readiness unreachable.
        near_home_threat = [*threat[:3], 0.0, 0.0, 0.0, 0.0]
        target = _defensive_target(near_home_threat, mask, 0.35)
        enough_soldiers = soldiers >= max(6.0, workers * 0.28)
        if target is not None:
            army = target
        elif enough_soldiers and mask.army_choice[6]:
            army = 6
        else:
            army = 0
        hungry = food < 65.0 or own[12] > 0.08 or own[16] > 0.08
        if hungry and (own[12] > 0.15 or food < 30.0):
            lay = 0
        elif soldiers < max(6.0, workers * 0.28) and (food >= 45.0 or target is not None):
            lay = 2
        else:
            lay = 1
        support = 3 if hungry else 2
        return _logits(lay, _forage_logits(_food_scores(observation), support), army, 1.0 if target else -0.25)


class RaiderPolicy(BaselinePolicy):
    def __call__(self, observation: ObservationV2, mask: ActionMaskV2, policy_rng: Pcg32) -> PolicyLogitsV2:
        del policy_rng
        workers, soldiers, food = _own_counts(observation)
        own = observation.block("own_global")
        threats = _known_threats(observation)
        soldier_threats = _known_soldier_threats(observation)
        # Forward intercepts are for current enemy soldiers, never worker sightings.
        # Once ready, only a soldier threat at home or the first local approach delays
        # the raid; farther groups do not create an endless precondition loop.
        current = observation.block("enemy_current")
        visible_targets = [node for node in range(1, 6) if current[node] and soldier_threats[node] > 0.0
                           and mask.army_choice[node]]
        prepared_to_raid = soldiers >= max(5.0, workers * 0.40)
        near_home_soldiers = [
            soldier_threats[node] if current[node] else 0.0 for node in range(2)
        ] + [0.0] * 5
        defense_target = _defensive_target(near_home_soldiers, mask, 1.0)
        if defense_target is not None:
            army = defense_target
        elif prepared_to_raid and mask.army_choice[6]:
            army = 6
        elif visible_targets:
            army = max(visible_targets, key=lambda node: (node, threats[node]))
        else:
            # Advance/position on the outermost reachable visible route; ActionV2 has
            # no explicit target waypoint, so this is the closest legal pressure order.
            army = max((node for node in range(1, 6) if mask.army_choice[node]), default=0)
        hungry = food < 45.0 or own[12] > 0.10 or own[16] > 0.10
        if hungry and food < 25.0:
            lay = 0
        elif soldiers < max(5.0, workers * 0.40):
            lay = 2
        else:
            lay = 1
        return _logits(lay, _forage_logits(_food_scores(observation), 2), army, 1.0)


_SPECS: dict[str, BaselineSpec] = {
    "random": BaselineSpec("random", "v1", (("forage_support_max", 3),
        ("base_random_logits", 17), ("forage_support_selection", "uniform_1_to_3"),
        ("forage_target_selection", "pcg32_fisher_yates"), ("avoid_center_threshold", 0.5),
        ("selection", "pcg32_uniform_legal_logits")), RandomPolicy),
    "turtle": BaselineSpec("turtle", "v1", (("defense_threat", 0.5),
        ("worker_soldier_ratio", 0.18), ("lay_none_food_below", 55),
        ("lay_none_starvation_debt", 0.08), ("lay_none_starvation_delta", 0.08),
        ("lay_soldier_food_at_least", 80),
        ("soldier_lay_requires_known_threat", True), ("forage_support_normal", 1),
        ("forage_support_when_food_below", 2), ("avoid_center_when_allowed", True)), TurtlePolicy),
    "rush": BaselineSpec("rush", "v1", (("minimum_soldiers_to_raid", 4),
        ("soldier_worker_ratio", 0.42), ("defense_threat", 1.0), ("lay_none_food_below", 35),
        ("lay_none_starvation_debt", 0.12), ("forage_support", 1),
        ("avoid_center_when_allowed", False)), RushPolicy),
    "balanced": BaselineSpec("balanced", "v1", (("soldier_worker_ratio", 0.28),
        ("minimum_soldiers_to_raid", 6), ("defense_threat", 0.35),
        ("defense_local_nodes", "home_and_1_to_2"), ("hungry_food_below", 65),
        ("lay_none_food_below", 30), ("lay_none_starvation_debt", 0.15),
        ("soldier_lay_food_at_least", 45), ("hungry_starvation_debt", 0.08),
        ("soldier_lay_target", "raid_readiness"),
        ("hungry_starvation_delta", 0.08),
        ("forage_support_normal", 2), ("forage_support_when_hungry", 3),
        ("raid_requires_current_known_enemy_queen", False),
        ("avoid_center_when_defending", True), ("avoid_center_score_when_idle", -0.25)), BalancedPolicy),
    "raider": BaselineSpec("raider", "v1", (("minimum_soldiers_to_raid", 5),
        ("soldier_worker_ratio_to_raid", 0.40),
        ("defense_target_requires_current_enemy_soldier", True),
        ("defense_local_nodes", "home_and_1"), ("defense_soldier_threshold", 1.0),
        ("raid_requires_current_known_enemy_queen", False),
        ("visible_intercept_priority", "farthest_current_soldier_before_ready"),
        ("outermost_reachable_position_fallback", True), ("hungry_food_below", 45),
        ("hungry_starvation_debt", 0.10), ("hungry_starvation_delta", 0.10), ("lay_none_food_below", 25),
        ("soldier_lay_food_at_least", 25),
        ("forage_support", 2), ("avoid_center_when_allowed", True)), RaiderPolicy),
}


def baseline_names() -> tuple[str, ...]:
    return tuple(_SPECS)


def baseline_spec(name: str) -> BaselineSpec:
    try:
        return _SPECS[name]
    except KeyError:
        raise ValueError(f"unknown baseline {name!r}; expected one of {', '.join(_SPECS)}") from None


def make_baseline(name: str) -> BaselinePolicy:
    return baseline_spec(name).factory()


__all__ = ["BaselinePolicy", "BaselineSpec", "baseline_names", "baseline_spec", "make_baseline"]
