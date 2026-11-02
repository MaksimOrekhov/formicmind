"""ActionV2, ActionMaskV2 (30 bool) and the fail-fast validator (FOR-2 section 4).

``get_action_mask`` is a pure function of an ``ObservationV2``: legality of the
army/avoid options is derived from the public edge block of the observation
(own-side frame: home = 0, enemy home = 6, centre = 3), never from hidden state.
Masks do not depend on hidden troops, food or the enemy archetype.

Mask layout (flattened, 30 bool): ``lay[3]``, ``forage_target[6]``,
``army_choice[7]``, ``army_avoid_center[7][2]`` row-major with last index
``(false, true)``.

``forage_support_max = 3`` and ``sum(q) == 65535`` are checked by the validator, not
by the bit mask.
"""

from __future__ import annotations

from dataclasses import dataclass

from .observation import ObservationV2
from .rules import (
    ARMY_GARRISON_HOME,
    ARMY_RAID_ENEMY_HOME,
    FORAGE_CATEGORIES,
    FORAGE_SUPPORT_MAX,
    LAY_NONE,
    LAY_SOLDIER,
    NODE_COUNT,
    Q_TOTAL,
)

MASK_V2_LENGTH = 30
LAY_CHOICES = 3
ARMY_CHOICES = 7
LOCAL_HOME = 0
LOCAL_CENTER = 3
LOCAL_ENEMY_HOME = 6


class InvalidAction(ValueError):
    """The action violates the contract. Raised before any state is touched."""


class MaskError(ValueError):
    """The mask itself is malformed or leaves no legal option (a bug, not a bad action)."""


@dataclass(frozen=True, slots=True)
class ActionV2:
    lay_order: int  # 0 none, 1 worker, 2 soldier
    forage_share_q: tuple[int, ...]  # 6 shares: home, N1, N2, C, N4, N5; sum == 65535
    army_choice: int  # 0 garrison_home, 1..5 defend(local node), 6 raid(enemy_home)
    avoid_center: bool


@dataclass(frozen=True, slots=True)
class ValidatedActionV2:
    """An ``ActionV2`` that passed ``validate_action`` against a given mask.

    Python cannot make this unforgeable: anybody can construct it directly. The tick
    engine therefore never trusts the wrapper; it re-validates every action against the
    side's own observation and mask (see ``tick._phase_policy``).
    """

    action: ActionV2


@dataclass(frozen=True, slots=True)
class ActionMaskV2:
    lay: tuple[bool, ...]
    forage_target: tuple[bool, ...]
    army_choice: tuple[bool, ...]
    army_avoid_center: tuple[tuple[bool, bool], ...]

    def __post_init__(self) -> None:
        ok = (
            len(self.lay) == LAY_CHOICES
            and len(self.forage_target) == FORAGE_CATEGORIES
            and len(self.army_choice) == ARMY_CHOICES
            and len(self.army_avoid_center) == ARMY_CHOICES
            and all(len(row) == 2 for row in self.army_avoid_center)
            and all(type(b) is bool for b in self.flatten())
        )
        if not ok:
            raise MaskError("ActionMaskV2 must hold exactly 30 bool in the documented blocks")

    def flatten(self) -> tuple[bool, ...]:
        flat = [*self.lay, *self.forage_target, *self.army_choice]
        for row in self.army_avoid_center:
            flat.extend(row)
        return tuple(flat)


def flatten_mask_v2(mask: ActionMaskV2) -> tuple[bool, ...]:
    flat = mask.flatten()
    assert len(flat) == MASK_V2_LENGTH
    return flat


def mask_from_flat(flat: tuple[bool, ...]) -> ActionMaskV2:
    if len(flat) != MASK_V2_LENGTH:
        raise MaskError(f"expected {MASK_V2_LENGTH} bool")
    return ActionMaskV2(
        lay=tuple(flat[0:3]),
        forage_target=tuple(flat[3:9]),
        army_choice=tuple(flat[9:16]),
        army_avoid_center=tuple((flat[16 + 2 * i], flat[17 + 2 * i]) for i in range(ARMY_CHOICES)),
    )


def _graph_from_observation(obs: ObservationV2) -> list[list[int]]:
    """Adjacency (own-side frame) from the ``public_edges`` block (pairs i<j, lexicographic)."""
    edges = obs.block("public_edges")
    adj: list[list[int]] = [[] for _ in range(NODE_COUNT)]
    index = 0
    for i in range(NODE_COUNT):
        for j in range(i + 1, NODE_COUNT):
            if edges[index] > 0:
                adj[i].append(j)
                adj[j].append(i)
            index += 1
    return adj


def _reachable(adj: list[list[int]], src: int, dst: int, banned: frozenset[int]) -> bool:
    if src in banned or dst in banned:
        return False
    seen, stack = {src}, [src]
    while stack:
        node = stack.pop()
        if node == dst:
            return True
        for nxt in adj[node]:
            if nxt not in seen and nxt not in banned:
                seen.add(nxt)
                stack.append(nxt)
    return False


def get_action_mask(obs: ObservationV2) -> ActionMaskV2:
    adj = _graph_from_observation(obs)
    queen_alive = obs.block("own_global")[5] > 0  # own queen HP ratio

    lay = (True, queen_alive, queen_alive)
    forage = (True,) * FORAGE_CATEGORIES  # home and all five food nodes are always legal targets

    army: list[bool] = []
    avoid: list[tuple[bool, bool]] = []
    for choice in range(ARMY_CHOICES):
        target = LOCAL_HOME if choice == ARMY_GARRISON_HOME else choice  # 1..5 defend, 6 raid
        reachable = _reachable(adj, LOCAL_HOME, target, frozenset())
        army.append(reachable)
        if not reachable or choice == ARMY_GARRISON_HOME:
            avoid.append((reachable, False))
        else:
            avoid.append((True, _reachable(adj, LOCAL_HOME, target, frozenset({LOCAL_CENTER}))))
    return ActionMaskV2(lay, forage, tuple(army), tuple(avoid))


def check_action_structure(action: ActionV2) -> list[int]:
    """Mask-independent contract checks; returns the positive forage categories.

    Raises ``InvalidAction``. Used by ``validate_action`` and, as a last line of defence,
    by ``apply_joint_orders`` (a ``ValidatedActionV2`` can be constructed by hand).
    """
    if not isinstance(action, ActionV2):
        raise InvalidAction("not an ActionV2")
    if type(action.lay_order) is not int or not LAY_NONE <= action.lay_order <= LAY_SOLDIER:
        raise InvalidAction(f"invalid lay_order {action.lay_order!r}")
    if type(action.army_choice) is not int or not (
        ARMY_GARRISON_HOME <= action.army_choice <= ARMY_RAID_ENEMY_HOME
    ):
        raise InvalidAction(f"invalid army_choice {action.army_choice!r}")
    if type(action.avoid_center) is not bool:
        raise InvalidAction("avoid_center must be a bool")
    q = action.forage_share_q
    if type(q) is not tuple or len(q) != FORAGE_CATEGORIES or any(
        type(v) is not int or v < 0 for v in q
    ):
        raise InvalidAction("forage_share_q must be a tuple of 6 non-negative ints")
    if sum(q) != Q_TOTAL:
        raise InvalidAction(f"forage_share_q must sum to {Q_TOTAL}")
    positive = [i for i, v in enumerate(q) if v > 0]
    if not 1 <= len(positive) <= FORAGE_SUPPORT_MAX:
        raise InvalidAction(f"forage support must be 1..{FORAGE_SUPPORT_MAX}, got {len(positive)}")
    return positive


def validate_action(
    obs: ObservationV2, mask: ActionMaskV2, action: ActionV2
) -> ValidatedActionV2:
    """Whole-action validation. No fallback, no partial result."""
    if not isinstance(obs, ObservationV2) or not isinstance(mask, ActionMaskV2):
        raise TypeError("validate_action needs an ObservationV2 and an ActionMaskV2")
    positive = check_action_structure(action)
    if not mask.lay[action.lay_order]:
        raise InvalidAction("lay_order is masked out")
    if any(not mask.forage_target[i] for i in positive):
        raise InvalidAction("forage share on a masked-out target")
    if not mask.army_choice[action.army_choice]:
        raise InvalidAction("army_choice is masked out")
    if not mask.army_avoid_center[action.army_choice][1 if action.avoid_center else 0]:
        raise InvalidAction("avoid_center option is masked out for this army_choice")
    return ValidatedActionV2(action)
