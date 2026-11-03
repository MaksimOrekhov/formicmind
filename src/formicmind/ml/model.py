"""The optional PyTorch MLP and its adapter to the existing policy boundary."""

from __future__ import annotations

import os
from collections.abc import Sequence

try:
    import torch
    from torch import nn
except ImportError as exc:  # keep PyTorch optional for the core simulator
    raise RuntimeError(
        "FormicMind ML requires the optional PyTorch dependency. Install the official PyTorch wheel, "
        "then install FormicMind with `python -m pip install -e .`."
    ) from exc

from ..actions import ActionMaskV2
from ..decoder import PolicyLogitsV2
from ..observation import OBS_V2_LENGTH, ObservationV2
from ..rng import Pcg32

MODEL_SCHEMA = "formicmind.mlp-policy/v1"
INPUT_SIZE = OBS_V2_LENGTH
HIDDEN_SIZES = (64, 64)
OUTPUT_SIZE = 17


class PolicyMLP(nn.Module):
    """Fixed 223 → 64 → 64 → 17 ReLU MLP; only ObservationV2 enters the network."""

    def __init__(self, hidden_sizes: Sequence[int] = HIDDEN_SIZES) -> None:
        super().__init__()
        if tuple(hidden_sizes) != HIDDEN_SIZES:
            raise ValueError(f"supported hidden_sizes are fixed at {HIDDEN_SIZES}")
        self.layers = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZES[0]),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1]),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZES[1], OUTPUT_SIZE),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2 or observation.shape[-1] != INPUT_SIZE:
            raise ValueError(f"MLP input must have shape (batch, {INPUT_SIZE})")
        return self.layers(observation)


class MLPPolicy:
    """Observation-only inference adapter; mask and RNG remain outside the model."""

    def __init__(self, model: PolicyMLP, device: str | torch.device) -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def __call__(
        self, observation: ObservationV2, mask: ActionMaskV2, policy_rng: Pcg32
    ) -> PolicyLogitsV2:
        del mask, policy_rng
        inputs = torch.tensor(observation.values, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            logits = self.model(inputs)[0].detach().to("cpu").tolist()
        return PolicyLogitsV2(logits)


def resolve_device(requested: str) -> torch.device:
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be one of auto, cpu, cuda")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda was requested, but this PyTorch installation has no available CUDA device")
        return torch.device("cuda")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_determinism(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    # Small batches of this MLP are quicker and repeatable without thread scheduling noise.
    torch.set_num_threads(1)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


__all__ = [
    "HIDDEN_SIZES", "INPUT_SIZE", "MODEL_SCHEMA", "OUTPUT_SIZE", "MLPPolicy", "PolicyMLP",
    "configure_determinism", "resolve_device", "synchronize",
]
