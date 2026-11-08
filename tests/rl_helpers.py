"""Shared helpers for the FOR-42 tests."""

from __future__ import annotations

import torch

from formicmind.rl.distribution import HEAD_DIM

LOG_STD = [-0.3, -0.5, 0.0, 0.2, -1.0, -0.3]


def make_mask(lay=(1, 1, 1), forage=(1,) * 6, army=(1,) * 7, avoid_free=None) -> list[bool]:
    """Flat 30-entry mask. garrison (0) never allows avoid=true; others allow both unless ``avoid_free`` says no."""
    avoid_free = avoid_free if avoid_free is not None else [c != 0 for c in range(7)]
    flat = [bool(x) for x in lay] + [bool(x) for x in forage] + [bool(x) for x in army]
    for c in range(7):
        if not army[c]:
            flat += [False, False]
        elif avoid_free[c]:
            flat += [True, True]
        else:
            flat += [True, False]
    return flat


def random_heads(seed: int, scale: float = 2.0) -> list[float]:
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(HEAD_DIM, generator=g) * scale).tolist()
