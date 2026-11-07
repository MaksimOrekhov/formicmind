"""``decode(latents, masks) -> ActionE1``: the one path from policy output to an action (contracts 8.3).

A policy emits a flat vector of ``LATENT_SIZE`` finite numbers (heads laid out in ``LAYOUT``). Any finite vector decodes to a **valid** action:
masked options are never chosen, branches with a single legal option are taken whatever the logit says (a mandatory branch needs no logit and
carries no preference), and branches the previous choices make impossible are skipped (no scouts without a legal target, no squad without idle
soldiers, no forage share without a target ...). ``NaN`` / ``inf`` / non-numbers raise ``PolicyOutputError``: a technical failure, not a strategy.

Choices are greedy (arg-max, ties to the lower index); sampling for training lives in the later E3 stage and uses the same heads and masks.
Shares over the chosen forage targets are ``exp2``-weighted with an exact piecewise-linear ``exp2`` (``Fraction`` arithmetic, no ``libm``), each at
least 1/16, summing to 16, so the decoder is bit-identical on every platform. This module imports no world state, map or engine.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction

from .action import (
    EGG_LEVELS, FLEE_OPTIONS, FORAGE_FRACS, MAX_ATTACK_TARGETS, MAX_FORAGE_TARGETS, MAX_SCOUT_TARGETS, SCOUT_NS, SHARE_TOTAL, SOLDIER_SHARES,
    SQUAD_MISSIONS, SQUAD_SIZES, SQUADS, NO_SQUAD, ActionE1, ActionMasksE1, SquadActionE1, squad_take,
)
from .observation import MAX_NODES


class PolicyOutputError(ValueError):
    """The policy returned something that is not a finite numeric vector of the right length (technical error, not a defeat)."""


def _layout() -> dict[str, tuple[int, int]]:
    sizes = [("egg", EGG_LEVELS), ("soldier_share", SOLDIER_SHARES), ("forage_frac", FORAGE_FRACS), ("scout_n", SCOUT_NS),
             ("flee", FLEE_OPTIONS), ("forage_k", MAX_FORAGE_TARGETS + 1), ("forage_share", MAX_FORAGE_TARGETS),
             ("scout_k", MAX_SCOUT_TARGETS), ("forage_ptr", MAX_NODES), ("scout_ptr", MAX_NODES)]
    for squad in range(SQUADS):
        sizes += [(f"mission{squad}", len(SQUAD_MISSIONS)), (f"size{squad}", SQUAD_SIZES), (f"guard_ptr{squad}", MAX_NODES),
                  (f"attack_ptr{squad}", MAX_ATTACK_TARGETS)]
    layout, offset = {}, 0
    for name, size in sizes:
        layout[name] = (offset, size)
        offset += size
    return layout


LAYOUT = _layout()
LATENT_SIZE = sum(size for _, size in LAYOUT.values())
_FLOAT32_MAX = 3.4028234663852886e38


def check_latents(latents: Sequence[float]) -> tuple[float, ...]:
    """Validate and copy the policy output: a sequence of ``LATENT_SIZE`` finite real numbers (``bool`` is not a number here)."""
    try:
        values = tuple(latents)
    except TypeError as exc:
        raise PolicyOutputError("latents must be a sequence of numbers") from exc
    if len(values) != LATENT_SIZE:
        raise PolicyOutputError(f"expected {LATENT_SIZE} latents, got {len(values)}")
    out = []
    for index, value in enumerate(values):
        if type(value) not in (int, float):
            raise PolicyOutputError(f"latent {index} is not a real number ({type(value).__name__})")
        try:
            x = float(value)
        except OverflowError as exc:
            raise PolicyOutputError(f"latent {index} does not fit a float") from exc
        if not math.isfinite(x) or abs(x) > _FLOAT32_MAX:
            raise PolicyOutputError(f"latent {index} is not a finite float32")
        out.append(x)
    return tuple(out)


def _head(values: tuple[float, ...], name: str) -> tuple[float, ...]:
    offset, size = LAYOUT[name]
    return values[offset:offset + size]


def _argmax(logits: Sequence[float], allowed: Sequence[bool]) -> int:
    best = -1
    for index, flag in enumerate(allowed):
        if flag and (best < 0 or logits[index] > logits[best]):
            best = index
    if best < 0:
        raise PolicyOutputError("a decision branch has no legal option (malformed masks)")
    return best


def _top(logits: Sequence[float], allowed: Sequence[bool], count: int) -> list[int]:
    """The ``count`` best legal indices, highest logit first, ties to the lower index (choice without replacement)."""
    legal = [i for i, flag in enumerate(allowed) if flag]
    legal.sort(key=lambda i: (-logits[i], i))
    return legal[:count]


def _exp2_weight(x: float) -> Fraction:
    """Exact, monotone, positive ``2**x`` replacement: ``2**floor(x) * (1 + frac(x))`` over the rational value of the float."""
    value = Fraction(max(-60.0, min(60.0, x)))
    whole = math.floor(value)
    return Fraction(2) ** whole * (1 + (value - whole))


def _shares(logits: Sequence[float]) -> tuple[int, ...]:
    """Quantise softmax-like weights to sixteenths, each >= 1, total 16 (largest remainder, ties to the lower index)."""
    count = len(logits)
    if count == 1:
        return (SHARE_TOTAL,)
    weights = [_exp2_weight(x) for x in logits]
    total = sum(weights)
    spare = SHARE_TOTAL - count
    exact = [weight / total * spare for weight in weights]
    shares = [1 + int(math.floor(e)) for e in exact]
    leftover = SHARE_TOTAL - sum(shares)
    order = sorted(range(count), key=lambda i: (-(exact[i] - math.floor(exact[i])), i))
    for i in order[:leftover]:
        shares[i] += 1
    return tuple(shares)


def decode(latents: Sequence[float], masks: ActionMasksE1) -> ActionE1:
    """Finite latents + masks -> a valid ``ActionE1``. Raises ``PolicyOutputError`` for non-finite or malformed latents."""
    z = check_latents(latents)
    egg = _argmax(_head(z, "egg"), masks.egg_level)
    soldier_share = _argmax(_head(z, "soldier_share"), masks.soldier_share)
    flee = _argmax(_head(z, "flee"), masks.workers_flee) == 1

    # ---- foraging: fraction first; a positive fraction needs 1..3 legal targets, chosen by pointer logits without replacement
    frac = _argmax(_head(z, "forage_frac"), masks.forage_frac)
    forage_targets: tuple[int, ...] = ()
    forage_shares: tuple[int, ...] = ()
    if frac > 0:
        legal = [i for i, flag in enumerate(masks.forage_target) if flag]
        k_allowed = [False] + [k <= len(legal) for k in range(1, MAX_FORAGE_TARGETS + 1)]
        k = _argmax(_head(z, "forage_k"), k_allowed)
        forage_targets = tuple(_top(_head(z, "forage_ptr"), masks.forage_target, k))
        forage_shares = _shares(_head(z, "forage_share")[:k])

    # ---- scouts: n > 0 only with a legal target and idle workers; one or two targets
    n_allowed = list(masks.scout_n)
    n = _argmax(_head(z, "scout_n"), n_allowed)
    scout_targets: tuple[int, ...] = ()
    if n > 0:
        legal_scout = sum(masks.scout_target)
        t_allowed = [True, n >= 2 and legal_scout >= 2]
        t = _argmax(_head(z, "scout_k"), t_allowed) + 1
        scout_targets = tuple(_top(_head(z, "scout_ptr"), masks.scout_target, t))

    # ---- squads in order; each takes its share of the soldiers that are still idle, a squad with nobody left is NONE
    available = masks.soldiers_idle_home
    squads: list[SquadActionE1] = []
    for index in range(SQUADS):
        allowed = list(masks.squad_mission[index])
        if available < 1:
            allowed[SQUAD_MISSIONS.index("guard")] = allowed[SQUAD_MISSIONS.index("attack")] = False
        mission = SQUAD_MISSIONS[_argmax(_head(z, f"mission{index}"), allowed)]
        if mission in ("none", "withdraw"):
            squads.append(SquadActionE1(mission, 0, -1, -1) if mission == "withdraw" else NO_SQUAD)
            continue
        size_allowed = [False, *masks.squad_size[index][1:]]
        size_q = _argmax(_head(z, f"size{index}"), size_allowed)
        available -= squad_take(masks.soldiers_total, available, size_q)
        if mission == "guard":
            node = _argmax(_head(z, f"guard_ptr{index}"), masks.guard_target)
            squads.append(SquadActionE1("guard", size_q, node, -1))
        else:
            pair_allowed = [i < len(masks.attack_pairs) for i in range(MAX_ATTACK_TARGETS)]
            node, colony = masks.attack_pairs[_argmax(_head(z, f"attack_ptr{index}"), pair_allowed)]
            squads.append(SquadActionE1("attack", size_q, node, colony))
    return ActionE1(egg, soldier_share, frac, n, flee, forage_targets, forage_shares, scout_targets, tuple(squads))  # type: ignore[arg-type]
