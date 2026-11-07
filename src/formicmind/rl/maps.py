"""Mirror-valid training / held-out maps (FOR-42). Only edge *lengths and presence* change; rules and balance never do.

``map_config(map_id)`` accepts ``classic``, a named variant or ``gen:<int>`` (a seeded procedural variant). The train
and held-out sets are disjoint **by construction**: named maps are split below and procedural seeds are split by range.
``ObservationV2`` limits an edge length to 4 (checked), and every node must stay routable with and without the centre
(the same invariant the FOR-38 maps satisfy), so every ``ActionMaskV2`` army option stays well defined.
"""

from __future__ import annotations

import random

from ..config import EdgeSpec, SimConfig, Topology
from ..invariants import validate_config
from ..observation import check_observation_config
from ..routing import shortest_path
from ..rules import MIRROR, NODE_COUNT, default_config

# (edits, removed); all mirror pairs.  Nodes: A=0 N1=1 N2=2 C=3 N4=4 N5=5 B=6.
NAMED_MAP_EDITS: dict[str, tuple[dict[tuple[int, int], int], tuple[tuple[int, int], ...]]] = {
    "classic": ({}, ()),
    "cross": ({(1, 5): 4, (2, 4): 4}, ((2, 5),)),
    "dense": ({(1, 5): 4, (2, 4): 4}, ()),
    "fast_wings": ({(0, 1): 1, (4, 6): 1}, ()),
    "slow_center": ({(1, 3): 4, (3, 4): 4, (2, 3): 4, (3, 5): 4}, ()),
    "wide_ring": ({(1, 2): 2, (4, 5): 2}, ()),
    "far_flank": ({(0, 2): 4, (5, 6): 4, (2, 5): 2}, ()),
    "shortcut": ({(0, 3): 4, (3, 6): 4}, ()),
    "chain": ({(1, 4): 4}, ((2, 5),)),
}
TRAIN_NAMED_MAPS = ("classic", "cross", "fast_wings", "slow_center", "wide_ring", "shortcut")
HELDOUT_NAMED_MAPS = ("dense", "far_flank", "chain")
TRAIN_GEN_SEEDS = range(0, 1_000_000)
HELDOUT_GEN_SEEDS = range(1_000_000, 2_000_000)
_MAX_EDGE = 4


class MapError(ValueError):
    pass


def _routable(topology: Topology) -> bool:
    try:
        for source in range(NODE_COUNT):
            for destination in range(NODE_COUNT):
                shortest_path(topology, source, destination, side=0)
                if destination != 3:
                    shortest_path(topology, source, destination, side=0, avoid_center=True)
    except Exception:  # noqa: BLE001 - routing raises its own error type for unreachable pairs
        return False
    return True


def _orbit(edge: tuple[int, int]) -> frozenset[tuple[int, int]]:
    a, b = edge
    mirrored = tuple(sorted((MIRROR[a], MIRROR[b])))
    return frozenset({edge, mirrored})  # type: ignore[arg-type]


def _build(base: SimConfig, edges: dict[tuple[int, int], int]) -> SimConfig | None:
    topology = Topology(nodes=base.topology.nodes,
                        edges=tuple(EdgeSpec(a, b, length) for (a, b), length in sorted(edges.items())))
    config = SimConfig(rules_version=base.rules_version, balance_version=base.balance_version, topology=topology,
                       balance=base.balance, policy_interval_k=base.policy_interval_k, t_max=base.t_max)
    try:
        validate_config(config)
        check_observation_config(config)
    except Exception:  # noqa: BLE001
        return None
    return config if _routable(topology) else None


def _named(map_id: str) -> SimConfig:
    base = default_config()
    edges = {(e.a, e.b): e.length for e in base.topology.edges}
    edits, removed = NAMED_MAP_EDITS[map_id]
    for key in removed:
        edges.pop(key)
    edges.update(edits)
    config = _build(base, edges)
    if config is None:
        raise MapError(f"named map {map_id!r} is not a valid routable mirror topology")
    return config


def _generated(seed: int) -> SimConfig:
    base = default_config()
    rng = random.Random(f"formicmind-rl-map-{seed}")
    for _ in range(200):
        edges = {(e.a, e.b): e.length for e in base.topology.edges}
        orbits = sorted({_orbit(e) for e in edges}, key=lambda o: sorted(o))
        for orbit in orbits:
            if rng.random() < 0.55:
                length = rng.randint(1, _MAX_EDGE)
                for key in orbit:
                    edges[key] = length
        missing = sorted({_orbit((a, b)) for a in range(NODE_COUNT) for b in range(a + 1, NODE_COUNT)
                          if (a, b) not in edges}, key=lambda o: sorted(o))
        for _ in range(rng.randint(0, 2)):
            if not missing:
                break
            orbit = missing.pop(rng.randrange(len(missing)))
            length = rng.randint(1, _MAX_EDGE)
            for key in orbit:
                edges[key] = length
        if rng.random() < 0.25:
            present = sorted({_orbit(k) for k in edges}, key=lambda o: sorted(o))
            orbit = present[rng.randrange(len(present))]
            trial = {k: v for k, v in edges.items() if k not in orbit}
            if _build(base, trial) is not None:
                edges = trial
        config = _build(base, edges)
        if config is not None:
            return config
    raise MapError(f"could not generate a valid map for seed {seed}")


def map_config(map_id: str) -> SimConfig:
    if map_id in NAMED_MAP_EDITS:
        return _named(map_id)
    if map_id.startswith("gen:"):
        try:
            seed = int(map_id[4:])
        except ValueError as exc:
            raise MapError(f"bad generated map id {map_id!r}") from exc
        if seed < 0:
            raise MapError("generated map seed must be non-negative")
        return _generated(seed)
    raise MapError(f"unknown map id {map_id!r}")


def map_split(map_id: str) -> str:
    """'train' or 'heldout'; raises for anything that belongs to neither (so splits cannot silently leak)."""
    if map_id in TRAIN_NAMED_MAPS:
        return "train"
    if map_id in HELDOUT_NAMED_MAPS:
        return "heldout"
    if map_id.startswith("gen:"):
        seed = int(map_id[4:])
        if seed in TRAIN_GEN_SEEDS:
            return "train"
        if seed in HELDOUT_GEN_SEEDS:
            return "heldout"
    raise MapError(f"map {map_id!r} is in no split")


def training_map_id(rng: random.Random, generated_fraction: float = 0.5) -> str:
    if rng.random() < generated_fraction:
        return f"gen:{rng.choice(TRAIN_GEN_SEEDS)}"
    return rng.choice(TRAIN_NAMED_MAPS)


__all__ = [
    "HELDOUT_GEN_SEEDS", "HELDOUT_NAMED_MAPS", "TRAIN_GEN_SEEDS", "TRAIN_NAMED_MAPS", "MapError", "map_config",
    "map_split", "training_map_id",
]
