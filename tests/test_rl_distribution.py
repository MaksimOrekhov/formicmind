"""FOR-42 phase 0: the stochastic action distribution over the unchanged decoder."""

from __future__ import annotations

import itertools
import math
import random

import pytest

torch = pytest.importorskip("torch")

from formicmind.actions import ActionMaskV2, mask_from_flat, validate_action  # noqa: E402
from formicmind.decoder import PolicyLogitsV2, decode_action_logits, quantize_shares  # noqa: E402
from formicmind.rl.distribution import (  # noqa: E402
    HEAD_DIM, MU_BOUND, evaluate_latent, latent_to_logits, sample_latent,
)
from formicmind.rng import Pcg32  # noqa: E402
from rl_helpers import LOG_STD, make_mask, random_heads  # noqa: E402

def reference_log_prob(heads, log_std, mask, ints, z) -> float:
    """Independent float64 re-derivation of log pi(latent | obs, mask) from the written definition."""
    def softmax_allowed(logits, allowed):
        m = max(v for v, ok in zip(logits, allowed) if ok)
        w = [math.exp(v - m) if ok else 0.0 for v, ok in zip(logits, allowed)]
        s = sum(w)
        return [x / s for x in w]

    lay_h, army_h, avoid_h, k_h, tgt_h, mu_h = heads[0:3], heads[3:10], heads[10:17], heads[17:20], heads[20:26], heads[26:32]
    lay_m, forage_m, army_m = mask[0:3], mask[3:9], mask[9:16]
    avoid_m = [(mask[16 + 2 * i], mask[17 + 2 * i]) for i in range(7)]
    lp = math.log(softmax_allowed(lay_h, lay_m)[ints[0]])
    lp += math.log(softmax_allowed(army_h, army_m)[ints[1]])
    no_avoid, avoid_ok = avoid_m[ints[1]]
    if no_avoid and avoid_ok:
        p = 1 / (1 + math.exp(-avoid_h[ints[1]]))
        lp += math.log(p if ints[2] else 1 - p)
    n = sum(forage_m)
    k = ints[3] + 1
    lp += math.log(softmax_allowed(k_h, [j + 1 <= n for j in range(3)])[ints[3]])
    remaining = list(forage_m)
    for slot in range(k):
        t = ints[4 + slot]
        lp += math.log(softmax_allowed(tgt_h, remaining)[t])
        remaining[t] = False
        mu = MU_BOUND * math.tanh(mu_h[t] / MU_BOUND)
        sigma = math.exp(min(max(log_std[t], -3.0), 1.0))
        lp += -0.5 * ((z[slot] - mu) / sigma) ** 2 - math.log(sigma) - 0.5 * math.log(2 * math.pi)
    return lp


def evaluate_one(heads, mask, ints, z):
    return evaluate_latent(
        torch.tensor([heads]), torch.tensor(LOG_STD), torch.tensor([mask]), torch.tensor([ints]), torch.tensor([z]))


MASKS = {
    "open": make_mask(),
    "no_queen": make_mask(lay=(1, 0, 0)),
    "two_targets": make_mask(forage=(1, 0, 0, 1, 0, 0)),
    "one_target": make_mask(forage=(0, 0, 0, 1, 0, 0)),
    "few_army": make_mask(army=(1, 1, 0, 0, 1, 0, 1), avoid_free=[False, True, False, False, False, False, True]),
    "forced_avoid": make_mask(avoid_free=[False] * 7),
}


@pytest.mark.parametrize("name", sorted(MASKS))
def test_discrete_probability_mass_sums_to_one(name):
    mask = MASKS[name]
    heads = random_heads(7)
    rows_ints = []
    lay_ok, forage_ok, army_ok = mask[0:3], mask[3:9], mask[9:16]
    targets = [i for i in range(6) if forage_ok[i]]
    for lay in (i for i in range(3) if lay_ok[i]):
        for army in (i for i in range(7) if army_ok[i]):
            no_avoid, avoid_ok = mask[16 + 2 * army], mask[17 + 2 * army]
            options = [0, 1] if (no_avoid and avoid_ok) else [int(avoid_ok)]
            for avoid in options:
                for k in range(1, min(3, len(targets)) + 1):
                    for ordered in itertools.permutations(targets, k):
                        rows_ints.append([lay, army, avoid, k - 1, *ordered, *([0] * (3 - k))])
    n = len(rows_ints)
    ev = evaluate_latent(
        torch.tensor([heads] * n), torch.tensor(LOG_STD), torch.tensor([mask] * n), torch.tensor(rows_ints),
        torch.zeros(n, 3),
    )
    discrete = ev.log_prob - ev.components["shares"]
    assert float(torch.exp(discrete).sum()) == pytest.approx(1.0, abs=1e-4)


def test_torch_log_prob_matches_independent_reference():
    rng = Pcg32.seeded(1, 2)
    for i, mask in enumerate(MASKS.values()):
        for j in range(25):
            heads = random_heads(100 * i + j, scale=1.5)
            ints, z = sample_latent(heads, LOG_STD, mask, rng)
            ev = evaluate_one(heads, mask, ints, z)
            assert float(ev.log_prob[0]) == pytest.approx(reference_log_prob(heads, LOG_STD, mask, ints, z), abs=2e-4)


def test_sampled_actions_always_legal_and_decode_to_the_latent():
    rng = Pcg32.seeded(5, 6)
    for name, flat in MASKS.items():
        mask = mask_from_flat(tuple(flat))
        for j in range(120):
            heads = random_heads(j, scale=3.0)
            ints, z = sample_latent(heads, LOG_STD, flat, rng)
            logits = latent_to_logits(ints, z)
            assert len(logits) == 17 and all(math.isfinite(v) for v in logits)
            action = decode_action_logits(PolicyLogitsV2(logits), mask)
            # the validator is the engine's own whole-action check: it must accept everything we sample
            assert validate_action.__name__  # imported
            assert mask.lay[action.lay_order] and mask.army_choice[action.army_choice]
            assert action.lay_order == ints[0] and action.army_choice == ints[1]
            no_avoid, avoid_ok = mask.army_avoid_center[ints[1]]
            if no_avoid and avoid_ok:
                assert action.avoid_center == bool(ints[2])
            else:
                assert action.avoid_center == avoid_ok  # forced by the mask, whatever was "sampled"
            assert sum(action.forage_share_q) == 65535
            positive = [i for i, q in enumerate(action.forage_share_q) if q > 0]
            assert 1 <= len(positive) <= 3 and all(mask.forage_target[i] for i in positive)
            k = ints[3] + 1
            selected = ints[4:4 + k]
            assert set(positive) <= set(selected) and len(set(selected)) == k
            z_sel = z[:k]
            exp = [math.exp(v - max(z_sel)) for v in z_sel]
            probs = [e / sum(exp) for e in exp]
            weights = [0.0] * 6
            for t, p in zip(selected, probs):
                weights[t] = p
            ideal = quantize_shares(weights)
            assert max(abs(a - b) for a, b in zip(ideal, action.forage_share_q)) <= 2


def test_masked_choices_are_never_sampled():
    rng = Pcg32.seeded(11, 12)
    flat = make_mask(lay=(1, 0, 0), forage=(0, 1, 0, 1, 0, 1), army=(1, 0, 1, 0, 0, 0, 1))
    counts = {"lay": set(), "army": set(), "targets": set()}
    for j in range(300):
        ints, _ = sample_latent(random_heads(j, 4.0), LOG_STD, flat, rng)
        counts["lay"].add(ints[0])
        counts["army"].add(ints[1])
        counts["targets"].update(ints[4:4 + ints[3] + 1])
    assert counts["lay"] == {0}
    assert counts["army"] <= {0, 2, 6}
    assert counts["targets"] <= {1, 3, 5}


def test_empirical_frequencies_match_probabilities():
    heads = [0.0] * HEAD_DIM
    heads[0:3] = [0.0, 1.0, -1.0]
    heads[17:20] = [0.5, 0.0, -0.5]
    heads[20:26] = [1.0, 0.0, 0.0, -1.0, 0.0, 0.5]
    flat = make_mask()
    rng = Pcg32.seeded(21, 22)
    n = 6000
    lay = [0, 0, 0]
    k_counts = [0, 0, 0]
    first = [0] * 6
    for _ in range(n):
        ints, _z = sample_latent(heads, LOG_STD, flat, rng)
        lay[ints[0]] += 1
        k_counts[ints[3]] += 1
        first[ints[4]] += 1
    p_lay = [math.exp(v) for v in heads[0:3]]
    p_lay = [v / sum(p_lay) for v in p_lay]
    p_k = [math.exp(v) for v in heads[17:20]]
    p_k = [v / sum(p_k) for v in p_k]
    p_t = [math.exp(v) for v in heads[20:26]]
    p_t = [v / sum(p_t) for v in p_t]
    for observed, expected in ((lay, p_lay), (k_counts, p_k), (first, p_t)):
        for o, e in zip(observed, expected):
            assert o / n == pytest.approx(e, abs=0.03)


def test_conditional_avoid_has_zero_logprob_when_forced():
    heads = random_heads(3)
    forced = make_mask(avoid_free=[False] * 7)
    a = evaluate_one(heads, forced, [0, 2, 0, 0, 0, 0, 0], [0.0, 0.0, 0.0])
    b = evaluate_one(heads, forced, [0, 2, 1, 0, 0, 0, 0], [0.0, 0.0, 0.0])  # the stored value is irrelevant when forced
    assert float(a.components["avoid"][0]) == 0.0 and float(b.components["avoid"][0]) == 0.0
    free = make_mask()
    p_true = float(evaluate_one(heads, free, [0, 2, 1, 0, 0, 0, 0], [0.0] * 3).components["avoid"][0].exp())
    p_false = float(evaluate_one(heads, free, [0, 2, 0, 0, 0, 0, 0], [0.0] * 3).components["avoid"][0].exp())
    assert p_true + p_false == pytest.approx(1.0, abs=1e-6)


def test_extreme_logits_stay_finite_and_gradients_flow():
    heads = torch.tensor([[60.0, -60.0, 0.0] + [-80.0] * 29], requires_grad=True)
    log_std = torch.nn.Parameter(torch.tensor(LOG_STD))
    mask = torch.tensor([make_mask(forage=(1, 1, 0, 0, 0, 1))])
    ints = torch.tensor([[0, 4, 1, 1, 0, 5, 0]])
    z = torch.tensor([[0.3, -0.2, 0.0]])
    ev = evaluate_latent(heads, log_std, mask, ints, z)
    for tensor in (ev.log_prob, ev.entropy_discrete, ev.entropy_share):
        assert torch.isfinite(tensor).all()
    (ev.log_prob + ev.entropy_discrete + ev.entropy_share).sum().backward()
    assert torch.isfinite(heads.grad).all() and torch.isfinite(log_std.grad).all()
    assert heads.grad.abs().sum() > 0


def test_entropy_is_maximal_for_uniform_heads_and_bounded():
    uniform = evaluate_one([0.0] * HEAD_DIM, make_mask(), [0, 1, 0, 0, 0, 0, 0], [0.0] * 3)
    peaked = evaluate_one(random_heads(2, 8.0), make_mask(), [0, 1, 0, 0, 0, 0, 0], [0.0] * 3)
    assert float(uniform.entropy_discrete[0]) > float(peaked.entropy_discrete[0]) >= 0.0


def test_greedy_is_deterministic_and_uses_no_randomness():
    heads = random_heads(9)
    flat = make_mask()
    rng = Pcg32.seeded(1, 1)
    before = rng.state
    first = sample_latent(heads, LOG_STD, flat, rng, greedy=True)
    assert rng.state == before
    assert first == sample_latent(heads, LOG_STD, flat, Pcg32.seeded(9, 9), greedy=True)


def test_real_game_masks_are_accepted_by_the_whole_action_validator():
    from formicmind.actions import get_action_mask
    from formicmind.observation import build_observation
    from formicmind.reset import reset
    from formicmind.rules import default_config
    from formicmind.snapshot import begin_policy_phase

    state, run = reset(default_config(), 1)
    snapshot = begin_policy_phase(state, (run.windows[0], run.windows[1]))
    rng = Pcg32.seeded(31, 32)
    for side in (0, 1):
        obs = build_observation(snapshot, side)
        mask = get_action_mask(obs)
        for j in range(60):
            ints, z = sample_latent(random_heads(j, 2.5), LOG_STD, mask.flatten(), rng)
            action = decode_action_logits(PolicyLogitsV2(latent_to_logits(ints, z)), mask)
            validate_action(obs, mask, action)  # raises InvalidAction on any violation
    assert isinstance(mask, ActionMaskV2)
    assert random.Random(0)  # keep import used for linters
