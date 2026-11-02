"""Pinned contract references and version strings for the L0 foundation.

Linear documents are mutable, so each pin records the document URL and the
``updatedAt`` timestamp of the revision that was read. Precedence: the FOR-1
errata (A-G, appended to the FOR-1 document) and the FOR-2 errata v0.2.1 win over
any contradicting text in the document bodies.

``RULES_VERSION`` names the *prototype* rules: FOR-1 v0.2.1 plus routing errata R1
(equal-cost routes tie-break by the mover's side-local node order, FOR-11 review).
It is not a frozen balance: all numbers are candidates until the economic pilot
(FOR-1 errata F / FOR-4) picks them, see ``BALANCE_VERSION``. Observation, action and
decoder versions are unaffected by R1. FOR-13 (combat, terminal rules, runner) changes neither the rules nor the
balance version: it implements the phases FOR-1 v0.2.1 already specified. Only the checkpoint format moved to v3
(``HASH_DOMAIN_CHECKPOINT``) because the episode outcome is now part of the run state.
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
    ContractPin(
        key="FOR-1-routing-errata-R1",
        title="FOR-1 comment: routing errata R1 (side-local tie-break for equal-cost routes)",
        url="https://linear.app/generation17/issue/FOR-1#comment-b136cfe9-0917-4867-9ac0-5a6f20ecd318",
        updated_at="2026-11-02T17:20:56.487Z",
        role="topmost errata to FOR-1 v0.2.1; defines rules_version l0-proto-FOR1-v0.2.2",
    ),
    ContractPin(
        key="FOR-2-routing-errata-R1-sync",
        title="FOR-2 comment: sync of routing errata R1 (overrides world-id tie-break wording)",
        url="https://linear.app/generation17/issue/FOR-2#comment-3b5bf610-8b97-4ade-8592-b066da87fa77",
        updated_at="2026-11-02T17:21:10.542Z",
        role="overrides any world-id tie-break wording in FOR-2 v0.2/v0.2.1",
    ),
    ContractPin(
        key="FOR-1-FOR12-economy-decisions",
        title="FOR-1 comment: decisions on the FOR-12 interpretations (starvation order, debt reset, newborn tie-break)",
        url="https://linear.app/generation17/issue/FOR-1#comment-b5512302-d884-4541-adcf-1666a0b4767e",
        updated_at="2026-11-02T17:55:58.679Z",
        role="mandatory economy interpretations; implemented in FOR-12, unchanged by FOR-13",
    ),
)

# Strings taken verbatim from the FOR-4 manifest template.
OBSERVATION_VERSION = "FOR-2-v0.2+errata-v0.2.1"
ACTION_VERSION = "ActionV2"
DECODER_VERSION = "FOR-2-decoder-v0.2.1"

# Engineering choices of this repository (not decisions of FOR-1/FOR-2).
RULES_VERSION = "l0-proto-FOR1-v0.2.2"
BALANCE_VERSION = "candidate-FOR1-v0.2.1-unvalidated"
SERIALIZATION_VERSION = "formicmind-canonical-json-v1"
RNG_VERSION = "pcg32-xsh-rr-64-32-v1"
STREAM_DERIVATION_VERSION = "sha256-lenprefixed-v1"

# Domain-separation prefixes for SHA-256 (see canonical.domain_hash).
HASH_DOMAIN_CONFIG = "formicmind/config/v1"
HASH_DOMAIN_STATE = "formicmind/state/v1"
# v2: the checkpoint carries the per-side WindowStats (FOR-11). v3: it also carries the episode outcome
# (FOR-13); v1/v2 checkpoints are rejected by the schema tag.
HASH_DOMAIN_CHECKPOINT = "formicmind/checkpoint/v3"
HASH_DOMAIN_EVENTS = "formicmind/events/v1"
HASH_DOMAIN_RNG_STREAM = "formicmind/rng-stream/v1"

# Replay v1 (FOR-15) has its own schema version, independent of the rules, balance, observation, action, decoder
# and checkpoint versions. ``REPLAY_ENGINE_PIN`` is a static human-readable pin (no git hash, no timestamp).
REPLAY_FORMAT = "formicmind.replay"
REPLAY_SCHEMA_VERSION = "v1"
REPLAY_ENGINE_PIN = "formicmind-l0/0.0.1+for15"
HASH_DOMAIN_REPLAY_RECORD = "formicmind/replay-record/v1"
HASH_DOMAIN_REPLAY_VIEW = "formicmind/replay-view/v1"
HASH_DOMAIN_OBSERVATION = "formicmind/observation-f32le/v1"
