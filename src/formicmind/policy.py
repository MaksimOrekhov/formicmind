"""Minimal stateless policy boundary.

A policy sees exactly ``(observation, mask, policy_rng)`` and returns
``PolicyLogitsV2``. It never receives ``SimState``, a snapshot, a side index, an
opponent id or a world seed. Random, open-loop, parametric and learned controllers
all implement this one signature and are decoded by the same
``decode_action_logits`` through ``decide``. ``policy_rng`` is the controller's own
stream from ``RunState`` (``policy:<policy_id>:<replicate>``), never a side stream.
"""

from __future__ import annotations

from typing import Protocol

from .actions import (
    ActionMaskV2,
    ValidatedActionV2,
    get_action_mask,
    validate_action,
)
from .decoder import PolicyLogitsV2, decode_action_logits
from .observation import ObservationV2
from .rng import Pcg32


class Policy(Protocol):
    def __call__(
        self, observation: ObservationV2, mask: ActionMaskV2, policy_rng: Pcg32
    ) -> PolicyLogitsV2: ...


def decide(policy: Policy, observation: ObservationV2, policy_rng: Pcg32) -> ValidatedActionV2:
    """observation -> mask -> policy -> shared decoder -> validator (the only path)."""
    mask = get_action_mask(observation)
    logits = PolicyLogitsV2.coerce(policy(observation, mask, policy_rng))
    action = decode_action_logits(logits, mask)
    return validate_action(observation, mask, action)
