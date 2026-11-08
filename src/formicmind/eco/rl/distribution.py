"""The stochastic policy over ``ActionE1`` (FOR-49): sampling with a private PCG32 stream, the exact ``log_prob`` / entropy of the *same* sample under
any parameters, and the latent vector through which the shared decoder turns the sample into an action.

The sample is a sequence of factorised choices that follows the decoder's own branch order:

    egg, soldier share, flee                               categorical (mask applied; a single legal option has log-prob 0)
    forage fraction                                        categorical, 0 forced when nothing can be foraged
    if fraction > 0:  k in 1..min(3, #legal targets)       categorical
                      k forage targets                     pointer choice WITHOUT replacement (Plackett-Luce), order matters
                      k share logits z_j ~ N(mu_j, sigma_j) real latents; the decoder quantises softmax-like(z) to sixteenths
    scout n                                                categorical (mask: needs a legal target and idle workers)
    if n > 0:         t in {1, 2}, t targets               categorical, pointers without replacement
    for each of 3 squads: mission                          categorical (guard / attack only while soldiers are left, as the decoder tracks)
                      size 1..4 (guard / attack)           categorical;   guard: node pointer;   attack: pair pointer

``build_latent`` writes the sample into the decoder's flat head layout so that ``decoder.decode(latent, masks)`` reproduces exactly the sampled
choices (tests prove it); the executed action is then a deterministic function of the sample, and ``log_prob`` is that of the sample, never of an
argmax. ``evaluate`` recomputes it differentiably from the stored sample (PPO never re-samples). Masked options have probability 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ...rng import Pcg32
from ..action import squad_take
from ..decoder import LATENT_SIZE, LAYOUT
from .features import EGG_AT, FLEE_AT, FRAC_AT, SCOUT_N_AT, SOLDIER_AT, Batch
from .network import NEG, SHARE_SLOTS, HeadOutputs

# layout of the integer part of a sample (-1 = branch not taken)
I_EGG, I_SOLDIER, I_FLEE, I_FRAC, I_K = 0, 1, 2, 3, 4
I_FT = 5            # 3 forage targets (local node slots)
I_SCOUT_N, I_SCOUT_T = 8, 9
I_ST = 10           # 2 scout targets
I_SQUAD = 12        # per squad: mission, size index (0..3 for size_q 1..4), guard node slot, attack pair index
SAMPLE_INTS = 24
MISSION_NONE, MISSION_GUARD, MISSION_ATTACK, MISSION_WITHDRAW = 0, 1, 2, 3
_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)
_HALF_LOG_2PIE = 0.5 * math.log(2.0 * math.pi * math.e)


@dataclass(slots=True)
class Sample:
    ints: Tensor              # (B, 24) long
    z: Tensor                 # (B, 3) float32 share logits
    mission_allowed: Tensor   # (B, 3, 4) bool: the mission options the decoder offered at each squad (depends on the soldiers left)

    @property
    def size(self) -> int:
        return int(self.ints.shape[0])


def cat_samples(samples: list[Sample]) -> Sample:
    return Sample(torch.cat([s.ints for s in samples]), torch.cat([s.z for s in samples]), torch.cat([s.mission_allowed for s in samples]))


def select_sample(sample: Sample, ids: Tensor) -> Sample:
    return Sample(sample.ints[ids], sample.z[ids], sample.mission_allowed[ids])


@dataclass(slots=True)
class Evaluation:
    log_prob: Tensor          # (B,)
    entropy_discrete: Tensor  # (B,) chain-rule entropy of the categorical / pointer choices along the sampled path
    entropy_z: Tensor         # (B,) entropy of the share logits that were actually sampled
    log_prob_z: Tensor        # (B,) the part of ``log_prob`` that belongs to the real-valued share logits


# ------------------------------------------------------------------------------------------------ log-prob (differentiable)


def _cat_term(logits: Tensor, allowed: Tensor, choice: Tensor, active: Tensor | None, zero: Tensor) -> tuple[Tensor, Tensor]:
    logp = F.log_softmax(logits.masked_fill(~allowed, NEG), dim=-1)
    chosen = logp.gather(1, choice.clamp(min=0)[:, None]).squeeze(1)
    entropy = -(logp.exp() * logp).masked_fill(~allowed, 0.0).sum(-1)
    if active is not None:
        chosen, entropy = torch.where(active, chosen, zero), torch.where(active, entropy, zero)
    return chosen, entropy


def evaluate(heads: HeadOutputs, batch: Batch, sample: Sample) -> Evaluation:
    """Exact ``log pi(sample | observation, masks)`` and entropies under the given head outputs."""
    ints, z = sample.ints, sample.z
    B = ints.shape[0]
    sm = batch["smask"]
    zero = heads.mu.new_zeros(B)
    log_prob, entropy, entropy_z, log_prob_z = zero.clone(), zero.clone(), zero.clone(), zero.clone()

    def add(term: tuple[Tensor, Tensor]) -> None:
        nonlocal log_prob, entropy
        log_prob, entropy = log_prob + term[0], entropy + term[1]

    add(_cat_term(heads.cat["egg"], sm[:, EGG_AT:EGG_AT + 4], ints[:, I_EGG], None, zero))
    add(_cat_term(heads.cat["soldier"], sm[:, SOLDIER_AT:SOLDIER_AT + 5], ints[:, I_SOLDIER], None, zero))
    add(_cat_term(heads.cat["flee"], sm[:, FLEE_AT:FLEE_AT + 2], ints[:, I_FLEE], None, zero))
    add(_cat_term(heads.cat["frac"], sm[:, FRAC_AT:FRAC_AT + 9], ints[:, I_FRAC], None, zero))

    # ---- foraging branch
    foraging = ints[:, I_FRAC] > 0
    k = ints[:, I_K]
    n_forage = heads.legal_forage.sum(-1)
    k_allowed = torch.arange(1, 4, device=ints.device)[None, :] <= n_forage[:, None]
    add(_cat_term(heads.cat["k"], k_allowed, k - 1, foraging, zero))
    remaining = heads.legal_forage.clone()
    for j in range(3):
        active = foraging & (k > j)
        chosen = ints[:, I_FT + j]
        add(_cat_term(heads.forage, remaining, chosen, active, zero))
        remaining = remaining.scatter(1, chosen.clamp(min=0)[:, None], False)
        if j < SHARE_SLOTS:
            mu, sigma = heads.mu[:, j], heads.sigma[j]
            lp_z = -0.5 * ((z[:, j] - mu) / sigma) ** 2 - torch.log(sigma) - _HALF_LOG_2PI
            log_prob = log_prob + torch.where(active, lp_z, zero)
            log_prob_z = log_prob_z + torch.where(active, lp_z, zero)
            entropy_z = entropy_z + torch.where(active, (torch.log(sigma) + _HALF_LOG_2PIE).expand(B), zero)

    # ---- scouting branch
    add(_cat_term(heads.cat["scout_n"], sm[:, SCOUT_N_AT:SCOUT_N_AT + 5], ints[:, I_SCOUT_N], None, zero))
    scouting = ints[:, I_SCOUT_N] > 0
    two_ok = (ints[:, I_SCOUT_N] >= 2) & (heads.legal_scout.sum(-1) >= 2)
    t_allowed = torch.stack([torch.ones_like(two_ok), two_ok], 1)
    add(_cat_term(heads.cat["scout_t"], t_allowed, ints[:, I_SCOUT_T] - 1, scouting, zero))
    remaining = heads.legal_scout.clone()
    for j in range(2):
        active = scouting & (ints[:, I_SCOUT_T] > j)
        chosen = ints[:, I_ST + j]
        add(_cat_term(heads.scout, remaining, chosen, active, zero))
        remaining = remaining.scatter(1, chosen.clamp(min=0)[:, None], False)

    # ---- squads
    for s in range(3):
        base = I_SQUAD + 4 * s
        mission = ints[:, base]
        add(_cat_term(heads.cat[f"mission{s}"], sample.mission_allowed[:, s], mission, None, zero))
        sized = (mission == MISSION_GUARD) | (mission == MISSION_ATTACK)
        add(_cat_term(heads.cat[f"size{s}"], torch.ones_like(heads.cat[f"size{s}"], dtype=torch.bool), ints[:, base + 1], sized, zero))
        add(_cat_term(heads.guard[s], heads.legal_guard, ints[:, base + 2], mission == MISSION_GUARD, zero))
        add(_cat_term(heads.attack[s], heads.pair_valid, ints[:, base + 3], mission == MISSION_ATTACK, zero))
    return Evaluation(log_prob, entropy, entropy_z, log_prob_z)


# ------------------------------------------------------------------------------------------------ sampling (private PCG32 per colony)


def _uniform(rng: Pcg32) -> float:
    return (rng.next_u32() + 0.5) / 4294967296.0


def _normal(rng: Pcg32) -> float:
    u1, u2 = _uniform(rng), _uniform(rng)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _softmax(logits: list[float], allowed: list[bool]) -> list[float]:
    best = max(v for v, ok in zip(logits, allowed) if ok)
    weights = [math.exp(v - best) if ok else 0.0 for v, ok in zip(logits, allowed)]
    total = math.fsum(weights)
    return [w / total for w in weights]


def _pick(probs: list[float], rng: Pcg32, greedy: bool, trace: list[float] | None = None) -> int:
    index = _pick_index(probs, rng, greedy)
    if trace is not None:
        trace.append(math.log(probs[index]))
    return index


def _pick_index(probs: list[float], rng: Pcg32, greedy: bool) -> int:
    if greedy:
        best = 0
        for i, p in enumerate(probs):
            if p > probs[best]:
                best = i
        return best
    target = _uniform(rng)
    acc, last = 0.0, -1
    for i, p in enumerate(probs):
        if p > 0.0:
            last = i
            acc += p
            if target < acc:
                return i
    return last   # floating-point slack: never a zero-probability (masked) index


def sample_rows(heads: HeadOutputs, batch: Batch, rngs: list[Pcg32], greedy: bool = False, py_logp: list[float] | None = None) -> tuple[Sample, list[list[float]]]:
    """One sample per row (row ``b`` draws from ``rngs[b]`` only) and the decoder latents that reproduce it. ``greedy`` takes the mode
    (and the mean of the share logits): the evaluation mode, never used for PPO."""
    B = batch.size
    cat = {name: t.tolist() for name, t in heads.cat.items()}
    mu, sigma = heads.mu.tolist(), heads.sigma.tolist()
    forage, scout = heads.forage.tolist(), heads.scout.tolist()
    guard, attack = heads.guard.tolist(), heads.attack.tolist()
    lf, ls, lg, pv = (heads.legal_forage.tolist(), heads.legal_scout.tolist(), heads.legal_guard.tolist(), heads.pair_valid.tolist())
    smask, squad_base, units = batch["smask"].tolist(), batch["squad_base"].tolist(), batch["units"].tolist()
    ints_out, z_out, allowed_out, latents = [], [], [], []

    for b in range(B):
        rng = rngs[b]
        trace: list[float] = []

        def pick(probs: list[float]) -> int:
            return _pick(probs, rng, greedy, trace)
        row = [-1] * SAMPLE_INTS
        z = [0.0] * SHARE_SLOTS
        sm = smask[b]
        latent = [0.0] * LATENT_SIZE

        def one_hot(name: str, index: int) -> None:
            latent[LAYOUT[name][0] + index] = 1.0

        row[I_EGG] = pick(_softmax(cat["egg"][b], sm[EGG_AT:EGG_AT + 4])); one_hot("egg", row[I_EGG])
        row[I_SOLDIER] = pick(_softmax(cat["soldier"][b], sm[SOLDIER_AT:SOLDIER_AT + 5])); one_hot("soldier_share", row[I_SOLDIER])
        row[I_FLEE] = pick(_softmax(cat["flee"][b], sm[FLEE_AT:FLEE_AT + 2])); one_hot("flee", row[I_FLEE])
        row[I_FRAC] = pick(_softmax(cat["frac"][b], sm[FRAC_AT:FRAC_AT + 9])); one_hot("forage_frac", row[I_FRAC])

        if row[I_FRAC] > 0:
            legal = lf[b]
            n_legal = sum(legal)
            k_allowed = [j + 1 <= n_legal for j in range(3)]
            row[I_K] = pick(_softmax(cat["k"][b], k_allowed)) + 1
            one_hot("forage_k", row[I_K])
            remaining = list(legal)
            for j in range(row[I_K]):
                node = pick(_softmax(forage[b], remaining))
                row[I_FT + j] = node
                remaining[node] = False
                latent[LAYOUT["forage_ptr"][0] + node] = float(row[I_K] - j)
                z[j] = mu[b][j] if greedy else mu[b][j] + sigma[j] * _normal(rng)
                trace.append(-0.5 * ((z[j] - mu[b][j]) / sigma[j]) ** 2 - math.log(sigma[j]) - _HALF_LOG_2PI)
                latent[LAYOUT["forage_share"][0] + j] = z[j]

        row[I_SCOUT_N] = pick(_softmax(cat["scout_n"][b], sm[SCOUT_N_AT:SCOUT_N_AT + 5])); one_hot("scout_n", row[I_SCOUT_N])
        if row[I_SCOUT_N] > 0:
            legal = ls[b]
            t_allowed = [True, row[I_SCOUT_N] >= 2 and sum(legal) >= 2]
            row[I_SCOUT_T] = pick(_softmax(cat["scout_t"][b], t_allowed)) + 1
            one_hot("scout_k", row[I_SCOUT_T] - 1)
            remaining = list(legal)
            for j in range(row[I_SCOUT_T]):
                node = pick(_softmax(scout[b], remaining))
                row[I_ST + j] = node
                remaining[node] = False
                latent[LAYOUT["scout_ptr"][0] + node] = float(row[I_SCOUT_T] - j)

        soldiers_total, available = units[b][0], units[b][1]
        allowed_rows = []
        for s in range(3):
            allowed = list(squad_base[b][s])
            if available < 1:
                allowed[MISSION_GUARD] = allowed[MISSION_ATTACK] = False
            allowed_rows.append(allowed)
            base = I_SQUAD + 4 * s
            mission = pick(_softmax(cat[f"mission{s}"][b], allowed))
            row[base] = mission
            one_hot(f"mission{s}", mission)
            if mission in (MISSION_NONE, MISSION_WITHDRAW):
                continue
            size_index = pick(_softmax(cat[f"size{s}"][b], [True] * 4))
            row[base + 1] = size_index
            one_hot(f"size{s}", size_index + 1)
            available -= squad_take(soldiers_total, available, size_index + 1)
            if mission == MISSION_GUARD:
                node = pick(_softmax(guard[s][b], lg[b]))
                row[base + 2] = node
                one_hot(f"guard_ptr{s}", node)
            else:
                index = pick(_softmax(attack[s][b], pv[b]))
                row[base + 3] = index
                one_hot(f"attack_ptr{s}", index)
        if py_logp is not None:
            py_logp.append(math.fsum(trace))
        ints_out.append(row)
        z_out.append(z)
        allowed_out.append(allowed_rows)
        latents.append(latent)
    sample = Sample(torch.tensor(ints_out, dtype=torch.long), torch.tensor(z_out, dtype=torch.float32),
                    torch.tensor(allowed_out, dtype=torch.bool))
    return sample, latents


__all__ = ["Sample", "Evaluation", "evaluate", "sample_rows", "cat_samples", "select_sample", "SAMPLE_INTS"]
