"""Actor-critic network: 223 -> 256 -> 256 -> 256 with separate policy heads and a separate value trunk (FOR-42)."""

from __future__ import annotations

import io
import math

import torch
from torch import Tensor, nn

from ..observation import OBS_V2_LENGTH
from .distribution import HEAD_DIM, N_FORAGE

MODEL_SCHEMA = "formicmind.rl-actor-critic/v1"
HIDDEN_SIZES = (256, 256, 256)
INIT_LOG_STD = -0.3
HEAD_INIT_GAIN = 0.01


def _trunk(sizes: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = OBS_V2_LENGTH
    for width in sizes:
        linear = nn.Linear(previous, width)
        nn.init.orthogonal_(linear.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(linear.bias)
        layers += [linear, nn.Tanh()]
        previous = width
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Only ``ObservationV2`` (223 float32) enters the network; masks and sampling live outside it.

    The actor and the critic do not share parameters (no gradient interference between the policy and value losses).
    ``actor.*``, ``policy_head.*`` and ``log_std`` are everything inference needs; the critic is training-only.
    """

    def __init__(self, hidden_sizes: tuple[int, ...] = HIDDEN_SIZES) -> None:
        super().__init__()
        self.hidden_sizes = tuple(hidden_sizes)
        self.actor = _trunk(self.hidden_sizes)
        self.policy_head = nn.Linear(self.hidden_sizes[-1], HEAD_DIM)
        nn.init.orthogonal_(self.policy_head.weight, gain=HEAD_INIT_GAIN)
        nn.init.zeros_(self.policy_head.bias)
        self.log_std = nn.Parameter(torch.full((N_FORAGE,), INIT_LOG_STD))
        self.critic = _trunk(self.hidden_sizes)
        self.value_head = nn.Linear(self.hidden_sizes[-1], 1)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def policy(self, observation: Tensor) -> Tensor:
        return self.policy_head(self.actor(observation))

    def value(self, observation: Tensor) -> Tensor:
        return self.value_head(self.critic(observation)).squeeze(-1)

    def forward(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        return self.policy(observation), self.value(observation)

    def actor_state_dict(self) -> dict[str, Tensor]:
        return {
            name: tensor.detach().to("cpu").clone()
            for name, tensor in self.state_dict().items()
            if name.startswith(("actor.", "policy_head.")) or name == "log_std"
        }

    def parameter_count(self) -> dict[str, int]:
        actor = sum(t.numel() for n, t in self.named_parameters() if n.startswith(("actor.", "policy_head.")) or n == "log_std")
        total = sum(p.numel() for p in self.parameters())
        return {"actor": actor, "critic": total - actor, "total": total}


def actor_state_to_bytes(state: dict[str, Tensor]) -> bytes:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return buffer.getvalue()


def actor_state_from_bytes(data: bytes) -> dict[str, Tensor]:
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


def infer_hidden_sizes(state: dict[str, Tensor]) -> tuple[int, ...]:
    widths = []
    index = 0
    while f"actor.{2 * index}.weight" in state:
        widths.append(int(state[f"actor.{2 * index}.weight"].shape[0]))
        index += 1
    if not widths:
        raise ValueError("actor state has no trunk layers")
    return tuple(widths)


def build_actor(state: dict[str, Tensor], hidden_sizes: tuple[int, ...] | None = None) -> ActorCritic:
    """CPU inference copy: actor weights loaded, critic left at its (unused) initial values."""
    model = ActorCritic(infer_hidden_sizes(state) if hidden_sizes is None else hidden_sizes)
    missing = model.load_state_dict(state, strict=False)
    unexpected = list(missing.unexpected_keys)
    absent = [k for k in missing.missing_keys if not k.startswith(("critic.", "value_head."))]
    if unexpected or absent:
        raise ValueError(f"actor state does not match the network: unexpected={unexpected} missing={absent}")
    model.eval()
    return model


__all__ = [
    "HIDDEN_SIZES", "MODEL_SCHEMA", "ActorCritic", "actor_state_from_bytes", "actor_state_to_bytes", "build_actor",
]
