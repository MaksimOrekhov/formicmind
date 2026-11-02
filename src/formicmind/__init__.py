"""FormicMind L0 headless simulator: foundation (FOR-9) and agent boundary (FOR-10)."""

from .actions import (
    ActionMaskV2,
    ActionV2,
    InvalidAction,
    ValidatedActionV2,
    get_action_mask,
    validate_action,
)
from .canonical import canonical_dumps, canonical_loads
from .config import SimConfig
from .decoder import PolicyLogitsV2, PolicyOutputError, decode_action_logits
from .intel import IntelPhase, update_local_intel
from .model import ColonyState, SimState
from .observation import ObservationV2, build_observation, flatten_v2
from .policy import Policy, decide
from .reset import initial_state, reset
from .rules import default_config
from .runstate import RunState
from .snapshot import PolicySnapshot, WindowStats, begin_policy_phase, freeze_policy_snapshot
from .serialization import (
    checkpoint_from_bytes,
    checkpoint_hash,
    checkpoint_to_bytes,
    config_from_bytes,
    config_hash,
    config_to_bytes,
    state_from_bytes,
    state_hash,
    state_to_bytes,
)

__all__ = [
    "ActionMaskV2",
    "ActionV2",
    "ColonyState",
    "IntelPhase",
    "InvalidAction",
    "ObservationV2",
    "Policy",
    "PolicyLogitsV2",
    "PolicyOutputError",
    "PolicySnapshot",
    "RunState",
    "SimConfig",
    "SimState",
    "ValidatedActionV2",
    "WindowStats",
    "begin_policy_phase",
    "build_observation",
    "canonical_dumps",
    "canonical_loads",
    "checkpoint_from_bytes",
    "checkpoint_hash",
    "checkpoint_to_bytes",
    "config_from_bytes",
    "config_hash",
    "config_to_bytes",
    "decide",
    "decode_action_logits",
    "default_config",
    "flatten_v2",
    "freeze_policy_snapshot",
    "get_action_mask",
    "initial_state",
    "reset",
    "state_from_bytes",
    "state_hash",
    "state_to_bytes",
    "update_local_intel",
    "validate_action",
]
