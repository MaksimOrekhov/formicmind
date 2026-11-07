"""Shared builders for the FOR-49 learning tests (real observations from real worlds, small models)."""

from __future__ import annotations

import functools

from test_eco_policy_helpers import AttackerPolicy, Recorder, Run
from formicmind.eco import default_config
from formicmind.eco.policy import ObservationHeuristicPolicy


@functools.lru_cache(maxsize=4)
def recorded_decisions(ticks: int = 1500, seed: int = 50012) -> tuple:
    """Every ``(ObservationE1, ActionMasksE1)`` the eight colonies of a real world were shown (two aggressors, six explorers), in decision order.
    The aggressors make contacts and attack candidates appear, the explorers scout, so all branches of the masks occur."""
    config = default_config(8, "moderate")
    policies = [Recorder(AttackerPolicy(config, "explorer") if c in (1, 3) else ObservationHeuristicPolicy(config, "explorer")) for c in range(8)]
    Run(policies=policies, seed=seed).advance(ticks)
    return tuple((obs, masks) for policy in policies for obs, _bytes, masks in policy.seen)


def peaky_model(seed: int, scale: float = 60.0):
    """A freshly initialised network whose heads are scaled up, so that sampling is confident and walks every branch of the decoder."""
    import torch

    from formicmind.eco.rl.network import ActorCriticE1

    torch.manual_seed(seed)
    model = ActorCriticE1()
    with torch.no_grad():
        for head in (model.cat_head, model.mu_head):
            head.weight.mul_(scale)
            head.bias.copy_(torch.randn_like(head.bias))
    model.eval()
    return model
