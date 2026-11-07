"""The trained network behind the ordinary ``PolicyE1`` interface (FOR-49): one instance per colony, its own memory and PCG32 stream, an
``ObservationE1`` and masks in, decoder latents out. Nothing but the observation reaches it; the same object runs in evaluation and in the replay
that the viewer shows, and ``replay-verify`` re-executes it from the model's SHA-256 named in the replay header.

A replay header names the policy as ``learned`` with the variant ``<mode>-<first 16 hex of the weights SHA-256>``; ``mode`` is ``sample`` (draw
from the stochastic policy, the way it was trained) or ``greedy`` (take the mode). ``register_model`` makes a model directory resolvable.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ...rng import Pcg32, new_stream
from ..config import ConfigE1
from ..policy import PolicySpec, register_policy_factory, stream_id_of
from .checkpoint import load_model
from .distribution import sample_rows
from .features import cat, step_batch
from .network import ActorCriticE1

MODES = ("sample", "greedy")
_REGISTRY: dict[str, tuple[ActorCriticE1, dict]] = {}


def variant_for(manifest: dict, mode: str = "sample") -> str:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    return f"{mode}-{manifest['weights_sha256'][:16]}"


def register_model(directory: str | Path) -> tuple[ActorCriticE1, dict]:
    """Load a model directory and make it resolvable by its SHA prefix in specifications."""
    model, manifest = load_model(directory)
    _REGISTRY[manifest["weights_sha256"][:16]] = (model, manifest)
    return model, manifest


class LearnedColonyPolicy:
    """``PolicyE1`` for one colony: the trained actor with a private recurrent memory and a private random stream."""

    def __init__(self, model: ActorCriticE1, rng: Pcg32, *, greedy: bool, name: str) -> None:
        self.model, self.rng, self.greedy, self.name = model, rng, greedy, name
        hidden = model.config.hidden
        self.h_actor = torch.zeros(1, hidden)
        self.h_critic = torch.zeros(1, hidden)

    def act(self, obs, masks) -> tuple[float, ...]:
        batch = cat([step_batch(obs, masks)])
        with torch.no_grad():
            heads, _value, self.h_actor, self.h_critic = self.model.step(batch, self.h_actor, self.h_critic)
            _sample, latents = sample_rows(heads, batch, [self.rng], self.greedy)
        return tuple(latents[0])

    def snapshot(self) -> dict:
        return {"rng": self.rng.to_canonical(), "h_actor": self.h_actor.flatten().tolist(), "h_critic": self.h_critic.flatten().tolist()}

    def restore(self, data: dict) -> None:
        self.rng = Pcg32.from_canonical(data["rng"])
        self.h_actor = torch.tensor([data["h_actor"]], dtype=torch.float32)
        self.h_critic = torch.tensor([data["h_critic"]], dtype=torch.float32)


def _factory(item: PolicySpec, colony: int, config: ConfigE1, policy_seed: int) -> LearnedColonyPolicy:
    mode, _, prefix = item.variant.partition("-")
    if mode not in MODES or not prefix:
        raise ValueError(f"learned policy variant must be '<sample|greedy>-<weights sha256 prefix>', got {item.variant!r}")
    if prefix not in _REGISTRY:
        raise ValueError(f"the model {prefix} is not loaded: pass its directory (--model) or call register_model()")
    model, _ = _REGISTRY[prefix]
    return LearnedColonyPolicy(model, new_stream(policy_seed, stream_id_of(item, colony)), greedy=mode == "greedy", name=f"learned:{item.variant}")


register_policy_factory("learned", _factory)
