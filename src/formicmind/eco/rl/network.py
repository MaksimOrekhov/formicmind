"""Actor-critic for ``ObservationE1`` (FOR-49): a small graph encoder over the colony's *known* map, set encoders for the other blocks, a GRU
memory per colony for the actor and a separate one for the critic, factorised heads with pointer scores over local node slots.

Input is a ``features.Batch`` (built from an ``ObservationE1`` and its masks only). No world id, no hidden fact and no other colony enters:
every slot is local to the step. The recurrent state is passed in and out explicitly (one row per colony), so a rollout keeps one private
memory per colony and an update can unroll whole trajectories.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..observation import COHORT_DIM, EDGE_DIM, EVENT_DIM, FOREIGN_DIM, GLOBAL_DIM, SIGHTING_DIM
from .features import N_KINDS, NODE_IN, Batch

NEG = -1.0e9

# categorical heads (all sized for the *full* option set; masks are applied in the distribution)
CAT_LAYOUT: tuple[tuple[str, int], ...] = (
    ("egg", 4), ("soldier", 5), ("flee", 2), ("frac", 9), ("k", 3), ("scout_n", 5), ("scout_t", 2),
    ("mission0", 4), ("mission1", 4), ("mission2", 4), ("size0", 4), ("size1", 4), ("size2", 4),
)
CAT_DIM = sum(size for _, size in CAT_LAYOUT)
CAT_OFFSET = {}
_pos = 0
for _name, _size in CAT_LAYOUT:
    CAT_OFFSET[_name] = (_pos, _pos + _size)
    _pos += _size
del _name, _size, _pos
SHARE_SLOTS = 3
LOG_STD_MIN, LOG_STD_MAX = -3.0, 1.0
MU_BOUND = 6.0


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    node_dim: int = 64
    hidden: int = 128
    key_dim: int = 32
    message_rounds: int = 2

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class HeadOutputs:
    """Everything the distribution needs, dense where a pointer head needs a row per step."""

    cat: dict[str, Tensor]      # name -> (B, size) logits
    mu: Tensor                  # (B, 3) mean of the share logits, bounded
    sigma: Tensor               # (3,) state-free standard deviation of the share logits
    forage: Tensor              # (B, Nmax) pointer scores over node slots
    scout: Tensor               # (B, Nmax)
    guard: Tensor               # (3, B, Nmax)
    attack: Tensor              # (3, B, Pmax) scores over the attack pairs
    legal_forage: Tensor        # (B, Nmax) bool
    legal_scout: Tensor
    legal_guard: Tensor
    pair_valid: Tensor          # (B, Pmax) bool


def _mean_pool(rows: Tensor, index: Tensor, size: int) -> Tensor:
    out = rows.new_zeros((size, rows.shape[1]))
    if rows.shape[0]:
        out.index_add_(0, index, rows)
        count = rows.new_zeros(size).index_add_(0, index, rows.new_ones(rows.shape[0])).clamp(min=1.0)
        out = out / count[:, None]
    return out


def _max_pool(rows: Tensor, index: Tensor, size: int) -> Tensor:
    out = rows.new_zeros((size, rows.shape[1]))
    if rows.shape[0]:
        out = out.scatter_reduce(0, index[:, None].expand(-1, rows.shape[1]), rows, "amax", include_self=False)
    return out


def _dense(values: Tensor, batch_index: Tensor, local_index: Tensor, size: int, width: int, fill) -> Tensor:
    out = values.new_full((size, width), fill) if values.dtype != torch.bool else torch.full((size, width), bool(fill), dtype=torch.bool, device=values.device)
    if values.shape[0]:
        out[batch_index, local_index] = values
    return out


class ActorCriticE1(nn.Module):
    def __init__(self, config: NetworkConfig = NetworkConfig()) -> None:
        super().__init__()
        self.config = config
        D, H, K = config.node_dim, config.hidden, config.key_dim
        self.kind_emb = nn.Embedding(N_KINDS, 8)
        self.node_in = nn.Linear(NODE_IN + 8, D)
        self.msg = nn.ModuleList(nn.Linear(D + EDGE_DIM, D) for _ in range(config.message_rounds))
        self.upd = nn.ModuleList(nn.Linear(2 * D, D) for _ in range(config.message_rounds))
        self.norm = nn.ModuleList(nn.LayerNorm(D) for _ in range(config.message_rounds))
        self.g_in = nn.Linear(GLOBAL_DIM, D)
        self.f_in = nn.Linear(FOREIGN_DIM + D, D)
        self.c_in = nn.Linear(COHORT_DIM + 2 * D, D)
        self.s_in = nn.Linear(SIGHTING_DIM + 2 * D, D)
        self.v_in = nn.Linear(EVENT_DIM + D, D)
        self.fuse = nn.Sequential(nn.Linear(11 * D, H), nn.GELU(), nn.Linear(H, H), nn.LayerNorm(H))
        self.actor_gru = nn.GRUCell(H, H)
        self.critic_gru = nn.GRUCell(H, H)
        self.cat_head = nn.Linear(H, CAT_DIM)
        self.mu_head = nn.Linear(H, SHARE_SLOTS)
        self.log_std = nn.Parameter(torch.full((SHARE_SLOTS,), -0.5))
        self.q_forage, self.q_scout = nn.Linear(H, K), nn.Linear(H, K)
        self.q_guard, self.q_attack = nn.Linear(H, 3 * K), nn.Linear(H, 3 * K)
        self.k_forage, self.k_scout, self.k_guard = nn.Linear(D, K), nn.Linear(D, K), nn.Linear(D, K)
        self.k_attack = nn.Linear(2 * D, 3 * K)
        self.value_head = nn.Sequential(nn.Linear(H, 64), nn.GELU(), nn.Linear(64, 1))
        for head in (self.cat_head, self.mu_head):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)
        nn.init.orthogonal_(self.value_head[2].weight, gain=1.0)
        nn.init.zeros_(self.value_head[2].bias)

    # ---------------------------------------------------------------------------------------------- encoder (no memory)
    def encode(self, batch: Batch) -> tuple[Tensor, Tensor, Tensor]:
        """``x`` (B, H) per-step summary, ``node_emb`` (N, D) per node row, ``f_emb`` (F, D) per foreign-colony row."""
        B = batch.size
        n_off, f_off = batch.offsets("n"), batch.offsets("f")
        n_idx = batch.batch_index("n")
        h = F.gelu(self.node_in(torch.cat([batch["n_feat"], batch["n_legal"].to(torch.float32), self.kind_emb(batch["n_kind"])], 1)))

        e_idx = batch.batch_index("e")
        if batch["e_feat"].shape[0]:
            a = n_off[e_idx] + batch["e_slot"][:, 0]
            b = n_off[e_idx] + batch["e_slot"][:, 1]
            src, dst = torch.cat([b, a]), torch.cat([a, b])
            feat = torch.cat([batch["e_feat"], batch["e_feat"]], 0)
            degree = h.new_zeros(h.shape[0]).index_add_(0, dst, h.new_ones(dst.shape[0])).clamp(min=1.0)
        for i in range(self.config.message_rounds):
            if batch["e_feat"].shape[0]:
                message = F.gelu(self.msg[i](torch.cat([h[src], feat], 1)))
                aggregate = h.new_zeros(h.shape).index_add_(0, dst, message) / degree[:, None]
            else:
                aggregate = torch.zeros_like(h)
            h = self.norm[i](h + F.gelu(self.upd[i](torch.cat([h, aggregate], 1))))
        node_emb = h

        def node_at(slot: Tensor, batch_index: Tensor) -> Tensor:
            """Embedding of a local node slot per row; zeros where the slot is -1."""
            valid = slot >= 0
            rows = node_emb[(n_off[batch_index] + slot.clamp(min=0)).clamp(max=node_emb.shape[0] - 1)]
            return rows * valid[:, None].to(rows.dtype)

        f_idx = batch.batch_index("f")
        f_emb = F.gelu(self.f_in(torch.cat([batch["f_feat"], node_at(batch["f_nest"], f_idx)], 1)))

        def foreign_at(slot: Tensor, batch_index: Tensor) -> Tensor:
            valid = (slot >= 0) & (slot < batch.counts("f")[batch_index])
            if f_emb.shape[0] == 0:
                return node_emb.new_zeros((slot.shape[0], f_emb.shape[1]))
            rows = f_emb[(f_off[batch_index] + slot.clamp(min=0)).clamp(max=f_emb.shape[0] - 1)]
            return rows * valid[:, None].to(rows.dtype)

        c_idx = batch.batch_index("c")
        c_emb = F.gelu(self.c_in(torch.cat([batch["c_feat"], node_at(batch["c_slot"][:, 0], c_idx),
                                            node_at(batch["c_slot"][:, 1], c_idx)], 1)))
        s_idx = batch.batch_index("s")
        s_emb = F.gelu(self.s_in(torch.cat([batch["s_feat"], node_at(batch["s_int"][:, 0], s_idx),
                                            foreign_at(batch["s_int"][:, 1], s_idx)], 1)))
        v_idx = batch.batch_index("v")
        v_emb = F.gelu(self.v_in(torch.cat([batch["v_feat"], node_at(batch["v_int"][:, 1], v_idx)], 1)))

        home = node_emb[n_off]                                   # slot 0 of every step is the colony's own nest
        parts = [F.gelu(self.g_in(batch["g"])), _mean_pool(node_emb, n_idx, B), home,
                 _mean_pool(c_emb, c_idx, B), _max_pool(c_emb, c_idx, B), _mean_pool(f_emb, f_idx, B), _max_pool(f_emb, f_idx, B),
                 _mean_pool(s_emb, s_idx, B), _max_pool(s_emb, s_idx, B), _mean_pool(v_emb, v_idx, B), _max_pool(v_emb, v_idx, B)]
        return self.fuse(torch.cat(parts, 1)), node_emb, f_emb

    # ---------------------------------------------------------------------------------------------- memory and heads
    def value(self, hc: Tensor) -> Tensor:
        return self.value_head(hc).squeeze(-1)

    def heads(self, ha: Tensor, node_emb: Tensor, f_emb: Tensor, batch: Batch) -> HeadOutputs:
        B = batch.size
        K = self.config.key_dim
        cat_all = self.cat_head(ha)
        cat = {name: cat_all[:, a:b] for name, (a, b) in CAT_OFFSET.items()}
        mu = MU_BOUND * torch.tanh(self.mu_head(ha) / MU_BOUND)
        sigma = torch.exp(torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX))

        n_idx, n_local = batch.batch_index("n"), batch.local_index("n")
        n_max = max(1, int(batch.counts("n").max())) if B else 1
        scale = 1.0 / math.sqrt(K)

        def node_scores(query: Tensor, key: Tensor) -> Tensor:
            return (query[n_idx] * key).sum(-1) * scale

        forage = _dense(node_scores(self.q_forage(ha), self.k_forage(node_emb)), n_idx, n_local, B, n_max, NEG)
        scout = _dense(node_scores(self.q_scout(ha), self.k_scout(node_emb)), n_idx, n_local, B, n_max, NEG)
        qg, kg = self.q_guard(ha).view(B, 3, K), self.k_guard(node_emb)
        guard = torch.stack([_dense((qg[n_idx, s] * kg).sum(-1) * scale, n_idx, n_local, B, n_max, NEG) for s in range(3)])
        legal = batch["n_legal"]
        legal_forage = _dense(legal[:, 0], n_idx, n_local, B, n_max, False)
        legal_scout = _dense(legal[:, 1], n_idx, n_local, B, n_max, False)
        legal_guard = _dense(legal[:, 2], n_idx, n_local, B, n_max, False)

        p_idx, p_local = batch.batch_index("p"), batch.local_index("p")
        p_max = max(1, int(batch.counts("p").max())) if B else 1
        if batch["p_slot"].shape[0]:
            n_off, f_off = batch.offsets("n"), batch.offsets("f")
            node_part = node_emb[n_off[p_idx] + batch["p_slot"][:, 0]]
            colony = batch["p_slot"][:, 1]
            valid = (colony >= 0) & (colony < batch.counts("f")[p_idx])
            if f_emb.shape[0]:
                colony_part = f_emb[(f_off[p_idx] + colony.clamp(min=0)).clamp(max=f_emb.shape[0] - 1)] * valid[:, None].to(f_emb.dtype)
            else:
                colony_part = node_part.new_zeros(node_part.shape)
            keys = self.k_attack(torch.cat([node_part, colony_part], 1)).view(-1, 3, K)
            qa = self.q_attack(ha).view(B, 3, K)
            attack = torch.stack([_dense((qa[p_idx, s] * keys[:, s]).sum(-1) * scale, p_idx, p_local, B, p_max, NEG) for s in range(3)])
        else:
            attack = ha.new_full((3, B, p_max), NEG)
        pair_valid = _dense(torch.ones(p_idx.shape[0], dtype=torch.bool, device=ha.device), p_idx, p_local, B, p_max, False)
        return HeadOutputs(cat, mu, sigma, forage, scout, guard, attack, legal_forage, legal_scout, legal_guard, pair_valid)

    def step(self, batch: Batch, actor_h: Tensor, critic_h: Tensor) -> tuple[HeadOutputs, Tensor, Tensor, Tensor]:
        """One decision for each row of ``batch`` with the private memories ``actor_h`` / ``critic_h`` (B, H):
        returns ``(heads, value, new_actor_h, new_critic_h)``."""
        x, node_emb, f_emb = self.encode(batch)
        ha = self.actor_gru(x, actor_h)
        hc = self.critic_gru(x, critic_h)
        return self.heads(ha, node_emb, f_emb, batch), self.value(hc), ha, hc

    def bootstrap_value(self, batch: Batch, critic_h: Tensor) -> Tensor:
        """``Value_critic(o)`` of an observation that has no action (the final observation of a truncated trajectory)."""
        x, _, _ = self.encode(batch)
        return self.value(self.critic_gru(x, critic_h))
