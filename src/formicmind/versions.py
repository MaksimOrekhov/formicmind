"""Pinned contract references and version strings for the L0 foundation.

Linear documents are mutable, so each pin records the document URL and the
``updatedAt`` timestamp of the revision that was read. Precedence: the FOR-1
errata (A-G, appended to the FOR-1 document) and the FOR-2 errata v0.2.1 win over
any contradicting text in the document bodies.

``RULES_VERSION`` names the *prototype* rules (mechanics as written in FOR-1
v0.2.1). It is not a frozen balance: all numbers are candidates until the
economic pilot (FOR-1 errata F / FOR-4) picks them, see ``BALANCE_VERSION``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ContractPin:
    key: str
    title: str
    url: str
    updated_at: str
    role: str


CONTRACT_PINS: tuple[ContractPin, ...] = (
    ContractPin(
        key="FOR-1-v0.2.1",
        title="FOR-1 L0 World Specification v0.2.1 (body v0.2 + errata A-G)",
        url="https://linear.app/generation17/document/for-1-formicmind-l0-world-specification-v021-s-ispravleniyami-a-e-7257d16c6a93",
        updated_at="2026-11-02T12:31:31.981Z",
        role="world rules; errata A-G take precedence over the v0.2 body",
    ),
    ContractPin(
        key="FOR-2-v0.2",
        title="FOR-2 Agent Contract v0.2",
        url="https://linear.app/generation17/document/for-2-agent-contract-v02-draft-for-review-3d1167b99609",
        updated_at="2026-11-02T13:03:45.716Z",
        role="observation/action/decoder base; superseded where the errata says so",
    ),
    ContractPin(
        key="FOR-2-errata-v0.2.1",
        title="FOR-2 Agent Contract v0.2.1 (errata)",
        url="https://linear.app/generation17/document/for-2-agent-contract-v021-errata-ca924a5f9225",
        updated_at="2026-11-02T13:22:38.736Z",
        role="mandatory; takes precedence over FOR-2 v0.2",
    ),
)

# Strings taken verbatim from the FOR-4 manifest template.
OBSERVATION_VERSION = "FOR-2-v0.2+errata-v0.2.1"
ACTION_VERSION = "ActionV2"
DECODER_VERSION = "FOR-2-decoder-v0.2.1"

# Engineering choices of this repository (not decisions of FOR-1/FOR-2).
RULES_VERSION = "l0-proto-FOR1-v0.2.1"
BALANCE_VERSION = "candidate-FOR1-v0.2.1-unvalidated"
SERIALIZATION_VERSION = "formicmind-canonical-json-v1"
RNG_VERSION = "pcg32-xsh-rr-64-32-v1"
STREAM_DERIVATION_VERSION = "sha256-lenprefixed-v1"

# Domain-separation prefixes for SHA-256 (see canonical.domain_hash).
HASH_DOMAIN_CONFIG = "formicmind/config/v1"
HASH_DOMAIN_STATE = "formicmind/state/v1"
HASH_DOMAIN_CHECKPOINT = "formicmind/checkpoint/v1"
HASH_DOMAIN_RNG_STREAM = "formicmind/rng-stream/v1"
