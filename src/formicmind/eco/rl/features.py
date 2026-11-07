"""``ObservationE1`` + masks -> CSR-batched tensors (variable-length blocks, no world ids, no padding until a pointer head needs it).

A ``Batch`` of ``B`` decision steps stores, for every variable block (nodes, edges, foreign colonies, own cohorts, sightings, memory events,
attack pairs), the concatenated rows of all steps plus a per-step row count. Any slot is **local** to its step (slot 0 of the nodes is the
colony's own nest); global row indices are derived from the counts inside the network. ``select`` and ``cat`` are vectorised, so a rollout
stored as one big batch can be sliced into sequence minibatches without re-reading any observation.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from ..action import ActionMasksE1
from ..observation import COHORT_DIM, EDGE_DIM, EVENT_DIM, FOREIGN_DIM, GLOBAL_DIM, NODE_DIM, SIGHTING_DIM, ObservationE1

NODE_IN = NODE_DIM + 3          # the node features plus three legality flags (forage / scout / guard) derived from the masks
N_KINDS = 5                      # junction, patch, own nest, foreign nest, unknown
SCALAR_MASK = 25                 # egg 4 | soldier share 5 | forage fraction 9 | scout n 5 | flee 2
EGG_AT, SOLDIER_AT, FRAC_AT, SCOUT_N_AT, FLEE_AT = 0, 4, 9, 18, 23

# block name -> row tensors; every block has a count tensor ``<name>_cnt``
BLOCKS: dict[str, tuple[str, ...]] = {
    "n": ("n_feat", "n_kind", "n_legal"),
    "e": ("e_feat", "e_slot"),
    "f": ("f_feat", "f_nest"),
    "c": ("c_feat", "c_slot"),
    "s": ("s_feat", "s_int"),
    "v": ("v_feat", "v_int"),
    "p": ("p_slot",),
}
PER_STEP = ("g", "smask", "squad_base", "units")   # one row per step


class Batch:
    """A set of decision steps in CSR form. ``t`` maps tensor names to tensors."""

    __slots__ = ("t", "size")

    def __init__(self, t: dict[str, Tensor], size: int) -> None:
        self.t, self.size = t, size

    def __getitem__(self, name: str) -> Tensor:
        return self.t[name]

    def to(self, device) -> "Batch":
        return Batch({k: v.to(device) for k, v in self.t.items()}, self.size)

    def counts(self, block: str) -> Tensor:
        return self.t[f"{block}_cnt"]

    def batch_index(self, block: str) -> Tensor:
        """Step index of every row of a block."""
        return torch.repeat_interleave(torch.arange(self.size, device=self.t["g"].device), self.counts(block))

    def offsets(self, block: str) -> Tensor:
        """First row of every step inside the block (exclusive cumulative sum of the counts)."""
        return torch.cumsum(self.counts(block), 0) - self.counts(block)

    def local_index(self, block: str) -> Tensor:
        """Position of every row inside its own step (the local slot)."""
        counts = self.counts(block)
        total = int(counts.sum())
        return torch.arange(total, device=counts.device) - torch.repeat_interleave(self.offsets(block), counts)

    def validate(self) -> None:
        for block, names in BLOCKS.items():
            total = int(self.counts(block).sum())
            for name in names:
                if self.t[name].shape[0] != total:
                    raise ValueError(f"block {block}: {name} has {self.t[name].shape[0]} rows, counts say {total}")
        for name in PER_STEP:
            if self.t[name].shape[0] != self.size:
                raise ValueError(f"{name} has {self.t[name].shape[0]} rows for {self.size} steps")
        if self.t["g"].shape[1] != GLOBAL_DIM:
            raise ValueError("global feature width")


def _tensor(rows, width: int, dtype=torch.float32) -> Tensor:
    if not rows:
        return torch.zeros((0, width), dtype=dtype)
    return torch.tensor(rows, dtype=dtype)


def step_batch(obs: ObservationE1, masks: ActionMasksE1) -> Batch:
    """The tensors of one decision (a batch of one step). Uses only the observation and the masks derived from it."""
    n = len(obs.node_features)
    legal = [(bool(a), bool(b), bool(c)) for a, b, c in zip(masks.forage_target, masks.scout_target, masks.guard_target)]
    kinds = [row[0] for row in obs.node_ints]
    pairs = list(masks.attack_pairs)
    scalar = [*masks.egg_level, *masks.soldier_share, *masks.forage_frac, *masks.scout_n, *masks.workers_flee]
    t = {
        "g": torch.tensor([obs.global_features], dtype=torch.float32),
        "smask": torch.tensor([scalar], dtype=torch.bool),
        "squad_base": torch.tensor([masks.squad_mission], dtype=torch.bool),
        "units": torch.tensor([[masks.soldiers_total, masks.soldiers_idle_home, masks.workers_idle_home]], dtype=torch.long),
        "n_cnt": torch.tensor([n]), "n_feat": _tensor(obs.node_features, NODE_DIM), "n_kind": torch.tensor(kinds, dtype=torch.long),
        "n_legal": _tensor(legal, 3, torch.bool),
        "e_cnt": torch.tensor([len(obs.edge_features)]), "e_feat": _tensor(obs.edge_features, EDGE_DIM),
        "e_slot": _tensor(obs.edge_slots, 2, torch.long),
        "f_cnt": torch.tensor([len(obs.foreign_features)]), "f_feat": _tensor(obs.foreign_features, FOREIGN_DIM),
        "f_nest": torch.tensor(list(obs.foreign_nest_slots), dtype=torch.long),
        "c_cnt": torch.tensor([len(obs.cohort_features)]), "c_feat": _tensor(obs.cohort_features, COHORT_DIM),
        "c_slot": _tensor(obs.cohort_slots, 2, torch.long),
        "s_cnt": torch.tensor([len(obs.sighting_features)]), "s_feat": _tensor(obs.sighting_features, SIGHTING_DIM),
        "s_int": _tensor(obs.sighting_ints, 3, torch.long),
        "v_cnt": torch.tensor([len(obs.event_features)]), "v_feat": _tensor(obs.event_features, EVENT_DIM),
        "v_int": _tensor(obs.event_ints, 3, torch.long),
        "p_cnt": torch.tensor([len(pairs)]), "p_slot": _tensor(pairs, 2, torch.long),
    }
    return Batch(t, 1)


def cat(batches: Sequence[Batch]) -> Batch:
    if not batches:
        raise ValueError("no batches to concatenate")
    if len(batches) == 1:
        return batches[0]
    names = batches[0].t.keys()
    return Batch({name: torch.cat([b.t[name] for b in batches], 0) for name in names}, sum(b.size for b in batches))


def _ranges(starts: Tensor, lengths: Tensor) -> Tensor:
    """Concatenation of ``arange(start, start + length)`` for every pair, without a Python loop."""
    total = int(lengths.sum())
    if total == 0:
        return torch.zeros(0, dtype=torch.long, device=starts.device)
    shift = torch.cumsum(lengths, 0) - lengths
    return torch.repeat_interleave(starts - shift, lengths) + torch.arange(total, device=starts.device)


def select(batch: Batch, ids: Tensor) -> Batch:
    """The sub-batch of the given steps, in the given order (``ids`` may repeat or reorder)."""
    out: dict[str, Tensor] = {}
    for name in PER_STEP:
        out[name] = batch.t[name][ids]
    for block, names in BLOCKS.items():
        counts, offsets = batch.counts(block), batch.offsets(block)
        rows = _ranges(offsets[ids], counts[ids])
        out[f"{block}_cnt"] = counts[ids]
        for name in names:
            out[name] = batch.t[name][rows]
    return Batch(out, int(ids.shape[0]))
