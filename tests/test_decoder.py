"""The shared decoder: goldens D-1 (FOR-2 errata P0-2), ties, masks, quantization, fuzz."""

import math
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from formicmind.actions import (
    ActionMaskV2,
    ActionV2,
    MaskError,
    get_action_mask,
    validate_action,
)
from formicmind.decoder import (
    PolicyLogitsV2,
    PolicyOutputError,
    decode_action_logits,
    decode_forage_logits,
    encode_float_shares,
    quantize_shares,
)
from formicmind.observation import build_observation
from formicmind.rng import new_stream
from formicmind.snapshot import begin_policy_phase

from helpers import fresh_state

T, F = True, False
ALL6 = (T,) * 6
F32_MAX = 3.4028234663852886e38


def forage(logits, mask=ALL6):
    return decode_forage_logits(logits, mask)


def logits17(lay=(0, 0, 0), forage_=(0,) * 6, army=(0,) * 7, avoid=0.0):
    return PolicyLogitsV2([*lay, *forage_, *army, avoid])


def full_mask(**kw):
    base = dict(lay=(T, T, T), forage_target=ALL6, army_choice=(T,) * 7,
                army_avoid_center=((T, T),) * 7)
    base.update(kw)
    return ActionMaskV2(**base)


def g0_obs_and_mask():
    obs = build_observation(begin_policy_phase(fresh_state(0)), 0)
    return obs, get_action_mask(obs)


# --- independent reference (exact rationals; softmax via high-precision Decimal) -----------

def ref_forage(logits, mask):
    allowed = [i for i in range(6) if mask[i]]
    ranked = sorted(allowed, key=lambda i: (-logits[i], i))
    m = sum(1 for i in allowed if logits[i] > 0)
    k = min(3, max(1, m), len(allowed))
    chosen = ranked[:k]
    with localcontext() as ctx:
        ctx.prec = 80
        ctx.Emin, ctx.Emax = -10**9, 10**9
        peak = max(Decimal(logits[i]) for i in chosen)
        exps = {i: (Decimal(logits[i]) - peak).exp() for i in chosen}
        total = sum(exps.values())
        probs = {i: Fraction(exps[i] / total) for i in chosen}
    ideal = [Fraction(65535) * probs.get(i, Fraction(0)) for i in range(6)]
    q = [x.numerator // x.denominator for x in ideal]
    left = 65535 - sum(q)
    for i in sorted(range(6), key=lambda i: (-(ideal[i] - q[i]), i))[:left]:
        q[i] += 1
    return tuple(q)


# --- PolicyLogitsV2 -------------------------------------------------------------------

def test_policy_logits_hold_17_float32_values_and_are_immutable():
    logits = PolicyLogitsV2([0.1] * 17)
    assert len(logits.values) == 17
    assert logits.values[0] == 0.10000000149011612          # rounded to float32
    assert PolicyLogitsV2(logits.values) == logits
    with pytest.raises(AttributeError):
        logits.values = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    "bad",
    [
        [0.0] * 16, [0.0] * 18, "x" * 17, None,
        [0.0] * 16 + [float("nan")], [0.0] * 16 + [float("inf")], [float("-inf")] + [0.0] * 16,
        [0.0] * 16 + [1e39], [0.0] * 16 + [-1e39], [0.0] * 16 + ["1"], [0.0] * 16 + [None],
        [0.0] * 16 + [True], [0.0] * 16 + [10**400],
    ],
)
def test_policy_logits_reject_malformed_or_non_finite_output(bad):
    with pytest.raises(PolicyOutputError):
        PolicyLogitsV2(bad)  # type: ignore[arg-type]


def test_nan_and_inf_are_a_technical_error_before_decoding():
    mask = full_mask()
    for v in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(PolicyOutputError):
            decode_action_logits([0.0] * 16 + [v], mask)
        with pytest.raises(PolicyOutputError):
            decode_forage_logits([v, 0, 0, 0, 0, 0], ALL6)
    with pytest.raises(PolicyOutputError):
        decode_action_logits("not logits", mask)  # type: ignore[arg-type]


# --- lay / army / avoid_center -----------------------------------------------------------

def test_lay_argmax_with_canonical_ties_and_mask():
    mask = full_mask()
    assert decode_action_logits(logits17(lay=(1, 1, 1)), mask).lay_order == 0
    assert decode_action_logits(logits17(lay=(0, 2, 2)), mask).lay_order == 1
    assert decode_action_logits(logits17(lay=(-1, -3, -2)), mask).lay_order == 0
    assert decode_action_logits(logits17(lay=(-1, -3, -2)), full_mask(lay=(F, T, T))).lay_order == 2
    assert decode_action_logits(logits17(lay=(0, 9, 1)), full_mask(lay=(T, F, T))).lay_order == 2
    with pytest.raises(MaskError):
        decode_action_logits(logits17(), full_mask(lay=(F, F, F)))


def test_army_argmax_with_canonical_ties_and_mask():
    assert decode_action_logits(logits17(army=(0,) * 7), full_mask()).army_choice == 0
    assert decode_action_logits(logits17(army=(0, 1, 1, 0, 0, 0, 1)), full_mask()).army_choice == 1
    assert decode_action_logits(logits17(army=(0, 0, 0, 0, 0, 0, 5)), full_mask()).army_choice == 6
    masked = full_mask(army_choice=(T, T, T, T, T, T, F))
    assert decode_action_logits(logits17(army=(0, 0, 0, 0, 0, 0, 5)), masked).army_choice == 0
    assert decode_action_logits(logits17(army=(-5, -4, -3, -2, -1, -6, -7)), full_mask()).army_choice == 4
    with pytest.raises(MaskError):
        decode_action_logits(logits17(), full_mask(army_choice=(F,) * 7))


@pytest.mark.parametrize(
    "row, avoid_logit, expected",
    [
        ((T, T), 0.5, True), ((T, T), 0.0, False), ((T, T), -1.0, False),
        ((F, T), -5.0, True), ((F, T), 0.0, True),
        ((T, F), 5.0, False), ((T, F), 0.0, False),
    ],
)
def test_avoid_center_scores_false_as_zero_and_ties_go_to_false(row, avoid_logit, expected):
    mask = full_mask(army_avoid_center=((T, T),) * 6 + (row,))
    action = decode_action_logits(logits17(army=(0, 0, 0, 0, 0, 0, 1), avoid=avoid_logit), mask)
    assert action.army_choice == 6 and action.avoid_center is expected


def test_avoid_center_with_no_allowed_option_is_a_mask_error():
    mask = full_mask(army_avoid_center=((T, T),) * 6 + ((F, F),))
    with pytest.raises(MaskError):
        decode_action_logits(logits17(army=(0, 0, 0, 0, 0, 0, 1)), mask)


def test_garrison_never_decodes_to_avoid_center_with_the_real_mask():
    obs, mask = g0_obs_and_mask()
    action = decode_action_logits(logits17(army=(1, 0, 0, 0, 0, 0, 0), avoid=9.0), mask)
    assert action.army_choice == 0 and action.avoid_center is False
    action = decode_action_logits(logits17(army=(0, 0, 0, 1, 0, 0, 0), avoid=9.0), mask)  # defend(C)
    assert action.army_choice == 3 and action.avoid_center is False
    validate_action(obs, mask, action)


# --- forage decoder: goldens D-1 -------------------------------------------------------------

def test_d1_all_zero_logits_select_home_only():
    assert forage([0] * 6) == (65535, 0, 0, 0, 0, 0)


def test_d1_all_positive_equal_logits_select_first_three_canonically():
    assert forage([1] * 6) == (21845, 21845, 21845, 0, 0, 0)


def test_d1_exactly_two_positive_logits_give_support_two():
    q = forage([3, 0, 2, 0, -1, 0])
    assert [i for i, v in enumerate(q) if v] == [0, 2] and sum(q) == 65535
    assert q == ref_forage([3, 0, 2, 0, -1, 0], ALL6)
    q = forage([0.5, 2.0, -1, 0, 0, 0])
    assert [i for i, v in enumerate(q) if v] == [0, 1] and q[1] > q[0]
    assert q == ref_forage([0.5, 2.0, -1, 0, 0, 0], ALL6)
    assert forage([1, 1, 0, 0, 0, 0]) == (32768, 32767, 0, 0, 0, 0)   # exact tie: lower index wins the unit


def test_d1_extreme_gap_collapses_to_support_one_but_stays_valid():
    assert forage([1e30, 5, 1, 0, 0, 0]) == (65535, 0, 0, 0, 0, 0)
    assert forage([F32_MAX, -F32_MAX, F32_MAX, -F32_MAX, 0, F32_MAX]) == (21845, 0, 21845, 0, 0, 21845)
    assert forage([-F32_MAX, F32_MAX, -F32_MAX, -F32_MAX, -F32_MAX, -F32_MAX]) == (0, 65535, 0, 0, 0, 0)


def test_d1_quantization_underflow_keeps_the_unit_on_the_dominant_share():
    q = forage([20, 1, 1, 0, 0, 0])
    assert q == (65535, 0, 0, 0, 0, 0)
    assert q == ref_forage([20, 1, 1, 0, 0, 0], ALL6)
    q = forage([9, 8, 1, 0, 0, 0])
    assert sum(q) == 65535 and q[2] >= 0 and q == ref_forage([9, 8, 1, 0, 0, 0], ALL6)


def test_d1_ties_at_the_support_boundary_and_negative_logits():
    assert forage([1, 1, 1, 1, 0, 0]) == (21845, 21845, 21845, 0, 0, 0)
    assert forage([-5, -1, -3, -2, -4, -6]) == (0, 65535, 0, 0, 0, 0)
    assert forage([-2] * 6) == (65535, 0, 0, 0, 0, 0)


def test_forage_respects_the_target_mask():
    # a positive logit on a masked target neither counts towards k nor gets a share
    assert forage([9, 0, 0, 0, 0, 0], (F, T, T, T, T, T)) == (0, 65535, 0, 0, 0, 0)
    assert forage([5] * 6, (F, T, T, T, T, T)) == (0, 21845, 21845, 21845, 0, 0)
    # k is reduced to the number of allowed targets
    assert forage([5] * 6, (F, F, T, F, T, F)) == (0, 0, 32768, 0, 32767, 0)
    assert forage([5] * 6, (F, F, F, T, F, F)) == (0, 0, 0, 65535, 0, 0)
    with pytest.raises(MaskError):
        forage([0] * 6, (F,) * 6)


def test_forage_support_is_one_two_or_three_by_positive_count():
    for positives in range(0, 7):
        logits = [1.0] * positives + [-1.0] * (6 - positives)
        q = forage(logits)
        assert 1 <= sum(1 for v in q if v) <= 3 and sum(q) == 65535
        assert sum(1 for v in q if v) == min(3, max(1, positives))


# --- quantization / encoder -------------------------------------------------------------------

def test_encode_float_shares_golden_float_error_case():
    assert encode_float_shares([0.5, 0.5000009, 0, 0, 0, 0]) == (32767, 32768, 0, 0, 0, 0)


def test_quantize_uses_decimal_of_str_so_decimal_ties_are_exact():
    # 0.1/0.2/0.7 are exact decimal fractions here: ideals 6553.5, 13107, 45874.5 -> lower index wins
    assert quantize_shares([0.1, 0.2, 0.7]) == (6554, 13107, 45874)
    assert quantize_shares([1, 1, 1]) == (21845, 21845, 21845)
    assert quantize_shares([0, 0, 5]) == (0, 0, 65535)
    assert sum(quantize_shares([1e-300, 1, 3e-5, 1e-9])) == 65535


@pytest.mark.parametrize(
    "bad",
    [[float("nan")] + [1] * 5, [float("inf")] + [1] * 5, [-0.1, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0], [1, 1, 1], [True, 0, 0, 0, 0, 0], ["1", 0, 0, 0, 0, 0]],
)
def test_encode_float_shares_rejects_bad_input(bad):
    with pytest.raises(PolicyOutputError):
        encode_float_shares(bad)


# --- decoder is total on finite float32[17] -----------------------------------------------------

def _random_logit(rng):
    mode = rng.below(5)
    if mode == 0:
        return (rng.next_u32() / 2**32 - 0.5) * 16
    if mode == 1:
        return float(rng.below(5) - 2)
    if mode == 2:
        return (1.0, -1.0)[rng.below(2)] * (F32_MAX, 1e30, 1e-30, 0.0)[rng.below(4)]
    if mode == 3:
        return float(rng.below(3) - 1) * 0.5
    return (rng.next_u32() / 2**32 - 0.5) * 80


def _random_mask(rng):
    def subset(n):
        while True:
            bits = tuple(bool(rng.below(2)) for _ in range(n))
            if any(bits):
                return bits
    rows = tuple(((T, F), (F, T), (T, T))[rng.below(3)] for _ in range(7))
    return ActionMaskV2(subset(3), subset(6), subset(7), rows)


def test_fuzz_decoder_always_yields_a_valid_action_matching_the_reference():
    obs, _ = g0_obs_and_mask()
    rng = new_stream(31337, "test:decoder-fuzz")
    for _ in range(4000):
        raw = [_random_logit(rng) for _ in range(17)]
        mask = _random_mask(rng)
        logits = PolicyLogitsV2(raw)
        action = decode_action_logits(logits, mask)
        validate_action(obs, mask, action)           # never an InvalidAction

        v = logits.values
        argmax = lambda xs, ok: min((i for i in range(len(xs)) if ok[i]), key=lambda i: (-xs[i], i))  # noqa: E731
        assert action.lay_order == argmax(v[0:3], mask.lay)
        assert action.army_choice == argmax(v[9:16], mask.army_choice)
        row = mask.army_avoid_center[action.army_choice]
        assert action.avoid_center == (row[1] and (not row[0] or v[16] > 0))
        assert action.forage_share_q == ref_forage(v[3:9], mask.forage_target)
        assert sum(action.forage_share_q) == 65535
        assert 1 <= sum(1 for q in action.forage_share_q if q) <= 3
        assert all(mask.forage_target[i] for i, q in enumerate(action.forage_share_q) if q)
        assert isinstance(action, ActionV2) and type(action.avoid_center) is bool


def test_decoding_is_deterministic_and_side_independent():
    obs0 = build_observation(begin_policy_phase(fresh_state(0)), 0)
    obs1 = build_observation(begin_policy_phase(fresh_state(0)), 1)
    mask0, mask1 = get_action_mask(obs0), get_action_mask(obs1)
    logits = logits17(lay=(0.1, 0.9, 0.3), forage_=(0.2, 1.5, 0.7, -1, 0.7, 0.1),
                      army=(0.1, 0.2, 0.9, 0.3, 0.0, 0.1, 0.9), avoid=0.3)
    first = decode_action_logits(logits, mask0)
    assert first == decode_action_logits(logits, mask0) == decode_action_logits(logits, mask1)
    assert math.isfinite(sum(first.forage_share_q))
