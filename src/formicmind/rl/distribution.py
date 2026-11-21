"""Stochastic action distribution over the existing ``PolicyLogitsV2`` decoder (FOR-42, phase 0).

Why this exists
---------------
``decode_action_logits`` is a deterministic function of 17 logits (masked argmax / sparse top-k softmax). PPO needs a
*sampled* decision with the exact ``log_prob`` of what was sampled. We therefore define a probability distribution over
a **latent action** and a deterministic map ``latent -> 17 logits`` whose decoding gives the executed ``ActionV2``:

    obs --net--> head parameters --(mask)--> latent ~ pi(. | obs, mask) --latent_to_logits--> 17 logits
        --decode_action_logits(logits, mask)--> ActionV2

The executed action is a deterministic function of the latent, so ``pi(latent | obs, mask)`` is a proper stochastic
policy for PPO (the environment simply sees the decoded action). ``log_prob`` is always the log-density of the sampled
latent, never of an argmax.

Latent (``ints`` int64[7] + ``z`` float32[3]):

* ``lay``   ~ Categorical over the lay options the mask allows.
* ``army``  ~ Categorical over the reachable army choices the mask allows.
* ``avoid`` ~ Bernoulli (sigmoid of ``avoid[army]``), **only** when the mask allows both options for the chosen
  army choice (conditional); otherwise it is forced and contributes log-prob 0 / entropy 0.
* ``k`` in {1,2,3}: forage support size ~ Categorical (masked to k <= number of allowed targets).
* ``t_0..t_{k-1}``: an ordered tuple of distinct allowed forage targets drawn sequentially without replacement from one
  categorical (Plackett-Luce), each step renormalised over the remaining allowed targets.
* ``z_0..z_{k-1}``: logistic-normal share logits, ``z_j ~ Normal(mu[t_j], sigma[t_j])`` (diagonal Gaussian, state-free
  learned ``sigma``); the decoder's softmax over the selected logits gives the continuous shares
  ``p = softmax(z)`` which are then quantised to Q = 65535 exactly as for every other controller.

Latent -> logits (so that the *shared* decoder reproduces the sample):

* lay / army: chosen index 1.0, the rest 0.0 (strict maximum; the decoder ignores masked-out entries anyway);
* avoid:      ``+1.0`` for true, ``-1.0`` for false (decoder: ``> 0``);
* forage:     selected targets get ``z_j - min(z) + 1.0`` (>= 1, so exactly k strictly positive logits and the decoder's
  ``k = min(3, max(1, #positive))`` is k, softmax shift-invariance gives ``softmax(z)``); every other target gets
  ``-1.0``.

Quantisation can round a tiny share to zero; the *executed* support is then smaller than the latent support. That is a
deterministic function of the latent, so the log-prob of the latent remains the correct PPO ratio.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ..rng import Pcg32

N_LAY = 3
N_ARMY = 7
N_FORAGE = 6
N_K = 3
HEAD_LAYOUT: tuple[tuple[str, int], ...] = (
    ("lay", N_LAY),
    ("army", N_ARMY),
    ("avoid", N_ARMY),
    ("kcount", N_K),
    ("target", N_FORAGE),
    ("share_mu", N_FORAGE),
)
HEAD_DIM = sum(size for _, size in HEAD_LAYOUT)  # 32
MASK_LEN = 30
INT_FIELDS = 7  # lay, army, avoid, k_index (k-1), t0, t1, t2
Z_SLOTS = 3
LOG_STD_MIN = -3.0
LOG_STD_MAX = 1.0
MU_BOUND = 6.0
NEG_INF = -1.0e9
_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)
_HALF_LOG_2PIE = 0.5 * math.log(2.0 * math.pi * math.e)

_OFFSETS: dict[str, tuple[int, int]] = {}
_pos = 0
for _name, _size in HEAD_LAYOUT:
    _OFFSETS[_name] = (_pos, _pos + _size)
    _pos += _size
del _name, _size, _pos


def split_heads(heads: Tensor) -> dict[str, Tensor]:
    if heads.shape[-1] != HEAD_DIM:
        raise ValueError(f"expected {HEAD_DIM} head outputs, got {heads.shape[-1]}")
    return {name: heads[..., a:b] for name, (a, b) in _OFFSETS.items()}


def bounded_mu(raw_mu: Tensor) -> Tensor:
    return MU_BOUND * torch.tanh(raw_mu / MU_BOUND)


def sigma_from_log_std(log_std: Tensor) -> Tensor:
    return torch.exp(torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX))


def _mask_parts(mask: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if mask.shape[-1] != MASK_LEN:
        raise ValueError(f"expected a {MASK_LEN}-entry mask")
    lay = mask[:, 0:3]
    forage = mask[:, 3:9]
    army = mask[:, 9:16]
    avoid = mask[:, 16:30].reshape(-1, N_ARMY, 2)
    return lay, forage, army, avoid


def masked_log_softmax(logits: Tensor, allowed: Tensor) -> Tensor:
    return F.log_softmax(logits.masked_fill(~allowed, NEG_INF), dim=-1)


def _entropy(log_p: Tensor, allowed: Tensor) -> Tensor:
    p = torch.exp(log_p)
    return -(p * log_p).masked_fill(~allowed, 0.0).sum(-1)


@dataclass(frozen=True, slots=True)
class LatentEvaluation:
    log_prob: Tensor
    entropy_discrete: Tensor
    entropy_share: Tensor
    components: dict[str, Tensor]


def evaluate_latent(heads: Tensor, log_std: Tensor, mask: Tensor, ints: Tensor, z: Tensor) -> LatentEvaluation:
    """Exact ``log pi(latent | obs, mask)`` plus chain-rule entropy estimates (sampled-prefix conditional entropies).

    ``heads`` (B, 32) float, ``log_std`` (6,), ``mask`` (B, 30) bool, ``ints`` (B, 7) int64, ``z`` (B, 3) float.
    """
    batch = heads.shape[0]
    h = split_heads(heads)
    m_lay, m_forage, m_army, m_avoid = _mask_parts(mask)
    rows = torch.arange(batch, device=heads.device)

    lp_lay_all = masked_log_softmax(h["lay"], m_lay)
    lp_lay = lp_lay_all.gather(1, ints[:, 0:1]).squeeze(1)
    ent_lay = _entropy(lp_lay_all, m_lay)

    lp_army_all = masked_log_softmax(h["army"], m_army)
    army = ints[:, 1]
    lp_army = lp_army_all.gather(1, army[:, None]).squeeze(1)
    ent_army = _entropy(lp_army_all, m_army)

    option_rows = m_avoid[rows, army]  # (B, 2): (false allowed, true allowed) for the chosen army choice
    free = option_rows[:, 0] & option_rows[:, 1]
    a_logit = h["avoid"].gather(1, army[:, None]).squeeze(1)
    chose_true = ints[:, 2] > 0
    lp_avoid_free = torch.where(chose_true, F.logsigmoid(a_logit), F.logsigmoid(-a_logit))
    p_true = torch.sigmoid(a_logit)
    ent_avoid_free = -(p_true * F.logsigmoid(a_logit) + (1.0 - p_true) * F.logsigmoid(-a_logit))
    zero = torch.zeros_like(a_logit)
    lp_avoid = torch.where(free, lp_avoid_free, zero)
    ent_avoid = torch.where(free, ent_avoid_free, zero)

    n_allowed = m_forage.sum(-1)
    k_allowed = torch.arange(1, N_K + 1, device=heads.device)[None, :] <= n_allowed[:, None]
    lp_k_all = masked_log_softmax(h["kcount"], k_allowed)
    k_idx = ints[:, 3]
    lp_k = lp_k_all.gather(1, k_idx[:, None]).squeeze(1)
    ent_k = _entropy(lp_k_all, k_allowed)

    mu_all = bounded_mu(h["share_mu"])
    sigma_all = sigma_from_log_std(log_std)
    remaining = m_forage.clone()
    lp_targets = torch.zeros_like(a_logit)
    ent_targets = torch.zeros_like(a_logit)
    lp_shares = torch.zeros_like(a_logit)
    ent_shares = torch.zeros_like(a_logit)
    for slot in range(Z_SLOTS):
        active = k_idx >= slot
        lp_all = masked_log_softmax(h["target"], remaining)
        chosen = ints[:, 4 + slot]
        lp_t = lp_all.gather(1, chosen[:, None]).squeeze(1)
        lp_targets = lp_targets + torch.where(active, lp_t, zero)
        ent_targets = ent_targets + torch.where(active, _entropy(lp_all, remaining), zero)
        mu = mu_all.gather(1, chosen[:, None]).squeeze(1)
        sigma = sigma_all[chosen]
        lp_z = -0.5 * ((z[:, slot] - mu) / sigma) ** 2 - torch.log(sigma) - _HALF_LOG_2PI
        lp_shares = lp_shares + torch.where(active, lp_z, zero)
        ent_shares = ent_shares + torch.where(active, torch.log(sigma) + _HALF_LOG_2PIE, zero)
        remaining = remaining.scatter(1, chosen[:, None], False)

    log_prob = lp_lay + lp_army + lp_avoid + lp_k + lp_targets + lp_shares
    entropy_discrete = ent_lay + ent_army + ent_avoid + ent_k + ent_targets
    components = {
        "lay": lp_lay, "army": lp_army, "avoid": lp_avoid, "k": lp_k, "targets": lp_targets, "shares": lp_shares,
    }
    return LatentEvaluation(log_prob, entropy_discrete, ent_shares, components)


# ----------------------------------------------------------------------------------------------------------
# Sampling (pure Python on top of the head outputs; the only randomness is the controller's own PCG32 stream)
# ----------------------------------------------------------------------------------------------------------

def _uniform(rng: Pcg32) -> float:
    return (rng.next_u32() + 0.5) / 4294967296.0


def _normal(rng: Pcg32) -> float:
    u1 = _uniform(rng)
    u2 = _uniform(rng)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _softmax_allowed(logits: Sequence[float], allowed: Sequence[bool]) -> list[float]:
    best = max(v for v, ok in zip(logits, allowed) if ok)
    weights = [math.exp(v - best) if ok else 0.0 for v, ok in zip(logits, allowed)]
    total = math.fsum(weights)
    return [w / total for w in weights]


def _draw(probs: Sequence[float], rng: Pcg32) -> int:
    target = _uniform(rng)
    acc = 0.0
    last_positive = -1
    for i, p in enumerate(probs):
        if p > 0.0:
            last_positive = i
            acc += p
            if target < acc:
                return i
    return last_positive  # floating-point slack: never a zero-probability (masked) index


def _argmax(probs: Sequence[float]) -> int:
    best = 0
    for i, p in enumerate(probs):
        if p > probs[best]:
            best = i
    return best


def sample_latent(
    heads_row: Sequence[float], log_std: Sequence[float], mask_row: Sequence[bool], rng: Pcg32, *, greedy: bool = False,
) -> tuple[list[int], list[float]]:
    """Sample (or, with ``greedy``, take the mode of) one latent action. Returns ``(ints[7], z[3])``."""
    if len(heads_row) != HEAD_DIM or len(mask_row) != MASK_LEN:
        raise ValueError("bad head or mask length")
    h = {name: list(heads_row[a:b]) for name, (a, b) in _OFFSETS.items()}
    m_lay = [bool(x) for x in mask_row[0:3]]
    m_forage = [bool(x) for x in mask_row[3:9]]
    m_army = [bool(x) for x in mask_row[9:16]]
    m_avoid = [(bool(mask_row[16 + 2 * i]), bool(mask_row[17 + 2 * i])) for i in range(N_ARMY)]

    def pick(probs: Sequence[float]) -> int:
        return _argmax(probs) if greedy else _draw(probs, rng)

    lay = pick(_softmax_allowed(h["lay"], m_lay))
    army = pick(_softmax_allowed(h["army"], m_army))
    no_avoid, avoid_ok = m_avoid[army]
    if no_avoid and avoid_ok:
        p_true = 1.0 / (1.0 + math.exp(-h["avoid"][army]))
        avoid = int(p_true > 0.5) if greedy else int(_uniform(rng) < p_true)
    else:
        avoid = int(avoid_ok)  # forced

    n_allowed = sum(m_forage)
    k_allowed = [j + 1 <= n_allowed for j in range(N_K)]
    k_idx = pick(_softmax_allowed(h["kcount"], k_allowed))
    remaining = list(m_forage)
    targets: list[int] = []
    for _ in range(k_idx + 1):
        t = pick(_softmax_allowed(h["target"], remaining))
        targets.append(t)
        remaining[t] = False
    z: list[float] = []
    for t in targets:
        mu = MU_BOUND * math.tanh(h["share_mu"][t] / MU_BOUND)
        sigma = math.exp(min(max(log_std[t], LOG_STD_MIN), LOG_STD_MAX))
        z.append(mu if greedy else mu + sigma * _normal(rng))
    ints = [lay, army, avoid, k_idx, *targets, *([0] * (Z_SLOTS - len(targets)))]
    z += [0.0] * (Z_SLOTS - len(z))
    return ints, z


def latent_to_logits(ints: Sequence[int], z: Sequence[float]) -> list[float]:
    """The 17 logits whose *shared* decode reproduces the latent (see the module docstring)."""
    v = [0.0] * 17
    v[ints[0]] = 1.0
    v[9 + ints[1]] = 1.0
    v[16] = 1.0 if ints[2] else -1.0
    k = ints[3] + 1
    for i in range(N_FORAGE):
        v[3 + i] = -1.0
    selected = z[:k]
    low = min(selected)
    for slot in range(k):
        v[3 + ints[4 + slot]] = (selected[slot] - low) + 1.0
    return v


__all__ = [
    "HEAD_DIM", "HEAD_LAYOUT", "INT_FIELDS", "LOG_STD_MAX", "LOG_STD_MIN", "MASK_LEN", "MU_BOUND", "N_ARMY", "N_FORAGE",
    "N_K", "N_LAY", "Z_SLOTS", "LatentEvaluation", "bounded_mu", "evaluate_latent",
    "latent_to_logits", "masked_log_softmax", "sample_latent", "sigma_from_log_std", "split_heads",
]
