"""The one shared decoder: ``PolicyLogitsV2`` (17 finite float32) -> ``ActionV2``.

FOR-2 errata v0.2.1, P0-2. Every controller (random, open-loop, parametric, MLP)
goes through ``decode_action_logits``; there is no other selection procedure.

Logit indices: ``[0,3)`` lay, ``[3,9)`` forage (home, N1, N2, C, N4, N5),
``[9,16)`` army (ActionV2 order), ``[16]`` avoid_center.

* lay / army: masked argmax, ties resolved by index order (none, worker, soldier;
  garrison, defend by node, raid).
* avoid_center: among the options the mask allows for the chosen army_choice,
  ``false`` scores 0 and ``true`` scores ``logits[16]``; highest wins, tie -> false.
  No allowed option means a malformed mask (``MaskError``), not a bad action.
* forage (sparse): keep allowed targets, sort by logit descending (canonical order
  breaks ties), ``m`` = number of logits strictly above 0, ``k = min(3, max(1, m))``
  reduced to the number of allowed targets, softmax in float64 (shifted by the max)
  over the first ``k``, then shares are quantized to ``Q = 65535`` by largest
  remainder via ``Decimal(str(p))`` (exact rational arithmetic afterwards; ties go
  to the lower category index). A share that quantizes to 0 stays 0; the largest
  probability always gets a positive share, so the support is 1..3.

NaN/inf (or anything that is not a finite float32) is a technical policy error
(``PolicyOutputError``) raised before decoding.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from decimal import Decimal
from fractions import Fraction

from .actions import (
    ARMY_CHOICES,
    LAY_CHOICES,
    ActionMaskV2,
    ActionV2,
    MaskError,
)
from .rules import FORAGE_CATEGORIES, FORAGE_SUPPORT_MAX, Q_TOTAL

LOGITS_V2_LENGTH = 17
_LAY = slice(0, 3)
_FORAGE = slice(3, 9)
_ARMY = slice(9, 16)
_AVOID = 16


class PolicyOutputError(ValueError):
    """Technical failure of a policy (non-finite or malformed logits), not a strategy."""


def _as_float32(value: object) -> float:
    if type(value) not in (int, float):
        raise PolicyOutputError(f"logit must be a real number, got {type(value).__name__}")
    try:
        x = float(value)
    except OverflowError as exc:
        raise PolicyOutputError("logit does not fit a float") from exc
    if not math.isfinite(x):
        raise PolicyOutputError("logit is not finite")
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError as exc:
        raise PolicyOutputError("logit does not fit a float32") from exc


class PolicyLogitsV2:
    """Exactly 17 finite float32 values (stored as exact float32 values in Python floats)."""

    __slots__ = ("values",)

    def __init__(self, values: Sequence[float]) -> None:
        if not isinstance(values, (list, tuple)) or len(values) != LOGITS_V2_LENGTH:
            raise PolicyOutputError(f"PolicyLogitsV2 needs {LOGITS_V2_LENGTH} logits")
        object.__setattr__(self, "values", tuple(_as_float32(v) for v in values))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("PolicyLogitsV2 is immutable")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PolicyLogitsV2) and self.values == other.values

    def __hash__(self) -> int:
        return hash(self.values)

    def __repr__(self) -> str:
        return f"PolicyLogitsV2({list(self.values)})"

    @classmethod
    def coerce(cls, value: object) -> "PolicyLogitsV2":
        if isinstance(value, cls):
            return value
        if isinstance(value, (list, tuple)):
            return cls(value)
        raise PolicyOutputError(f"policy returned {type(value).__name__}, not PolicyLogitsV2")


def quantize_shares(weights: Sequence[float | int | Decimal], total: int = Q_TOTAL) -> tuple[int, ...]:
    """Largest-remainder quantization of non-negative weights to integers summing to ``total``.

    Floats enter through ``Decimal(str(w))``; the arithmetic after that is exact.
    Equal remainders are resolved in favour of the lower index.
    """
    exact: list[Fraction] = []
    for w in weights:
        if isinstance(w, Decimal):
            d = w
        elif type(w) is float:
            if not math.isfinite(w):
                raise PolicyOutputError("share weight is not finite")
            d = Decimal(str(w))
        elif type(w) is int:
            d = Decimal(w)
        else:
            raise PolicyOutputError(f"bad share weight type {type(w).__name__}")
        if not d.is_finite() or d < 0:
            raise PolicyOutputError("share weights must be finite and non-negative")
        exact.append(Fraction(d))
    weight_sum = sum(exact, Fraction(0))
    if weight_sum <= 0:
        raise PolicyOutputError("share weights must have a positive sum")
    ideal = [total * w / weight_sum for w in exact]
    floors = [x.numerator // x.denominator for x in ideal]
    leftover = total - sum(floors)
    order = sorted(range(len(ideal)), key=lambda i: (-(ideal[i] - floors[i]), i))
    for i in order[:leftover]:
        floors[i] += 1
    return tuple(floors)


def encode_float_shares(weights: Sequence[float]) -> tuple[int, ...]:
    """Common encoder for strategies that produce float shares (FOR-2 section 4.1).

    Fails fast (``PolicyOutputError``) on more than 3 positive weights; use
    ``quantize_shares`` for generic quantization without the action-support contract.
    """
    if len(weights) != FORAGE_CATEGORIES:
        raise PolicyOutputError("expected 6 share weights")
    positive = sum(1 for w in weights if type(w) in (int, float, Decimal) and w > 0)
    if positive > FORAGE_SUPPORT_MAX:
        # ActionV2 allows at most 3 positive shares; silently dropping or merging weights
        # would be a hidden redistribution, so the caller must select its support first.
        raise PolicyOutputError(
            f"{positive} positive share weights exceed forage_support_max={FORAGE_SUPPORT_MAX}"
        )
    return quantize_shares(list(weights))


def decode_forage_logits(
    logits: Sequence[float], target_mask: Sequence[bool]
) -> tuple[int, ...]:
    if len(logits) != FORAGE_CATEGORIES or len(target_mask) != FORAGE_CATEGORIES:
        raise MaskError("forage decoding needs 6 logits and a 6-entry target mask")
    values = [_as_float32(v) for v in logits]
    allowed = [i for i in range(FORAGE_CATEGORIES) if target_mask[i]]
    if not allowed:
        raise MaskError("forage target mask allows no target")
    ranked = sorted(allowed, key=lambda i: (-values[i], i))
    positive = sum(1 for i in allowed if values[i] > 0)
    k = min(FORAGE_SUPPORT_MAX, max(1, positive), len(allowed))
    chosen = ranked[:k]
    peak = max(values[i] for i in chosen)
    exps = [math.exp(values[i] - peak) for i in chosen]
    norm = math.fsum(exps)
    probs = [e / norm for e in exps]
    weights: list[float | int] = [0] * FORAGE_CATEGORIES
    for i, p in zip(chosen, probs):
        weights[i] = p
    return quantize_shares(weights)


def _masked_argmax(scores: Sequence[float], allowed: Sequence[bool], what: str) -> int:
    best = -1
    for i, ok in enumerate(allowed):
        if ok and (best < 0 or scores[i] > scores[best]):
            best = i
    if best < 0:
        raise MaskError(f"{what} mask allows no option")
    return best


def decode_action_logits(logits: PolicyLogitsV2 | Sequence[float], mask: ActionMaskV2) -> ActionV2:
    policy_logits = PolicyLogitsV2.coerce(logits)
    v = policy_logits.values
    if not isinstance(mask, ActionMaskV2):
        raise TypeError("mask must be an ActionMaskV2")

    lay = _masked_argmax(v[_LAY], mask.lay, "lay")
    army = _masked_argmax(v[_ARMY], mask.army_choice, "army")
    no_avoid, avoid_ok = mask.army_avoid_center[army]
    if not (no_avoid or avoid_ok):
        raise MaskError("army_choice has no allowed avoid_center option")
    if no_avoid and avoid_ok:
        avoid = v[_AVOID] > 0.0  # false scores 0; ties go to false
    else:
        avoid = avoid_ok
    shares = decode_forage_logits(v[_FORAGE], mask.forage_target)
    return ActionV2(lay_order=lay, forage_share_q=shares, army_choice=army, avoid_center=avoid)


__all__ = [
    "ARMY_CHOICES",
    "LAY_CHOICES",
    "LOGITS_V2_LENGTH",
    "PolicyLogitsV2",
    "PolicyOutputError",
    "decode_action_logits",
    "decode_forage_logits",
    "encode_float_shares",
    "quantize_shares",
]
