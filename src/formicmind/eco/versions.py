"""Versions and hash domains of Ecosystem v1 (E1). Independent of the L0 ``versions`` module (FOR-43 migration section 1)."""

from __future__ import annotations

ECO_RULES_VERSION = "eco1-proto-v0"
ECO_BALANCE_VERSION = "eco1-candidate-unvalidated"
OBSERVATION_VERSION = "ObservationE1/v1"  # contract only; the builder arrives in E2
ACTION_VERSION = "ActionE1/v1"  # contract only

CONFIG_SCHEMA = "formicmind.eco.config/1"
MAP_SCHEMA = "formicmind.eco.map/1"
STATE_SCHEMA = "formicmind.eco.state/1"
CHECKPOINT_SCHEMA = "formicmind.eco.checkpoint/1"
REPORT_SCHEMA = "formicmind.eco.fairness/1"
CLI_SCHEMA = "formicmind.eco.map-cli/1"
GENERATOR_ALGORITHM = "eco-mapgen-v1"

HASH_DOMAIN_CONFIG = "formicmind/eco/config/v1"
HASH_DOMAIN_MAP = "formicmind/eco/map/v1"
HASH_DOMAIN_STATE = "formicmind/eco/state/v1"
HASH_DOMAIN_CHECKPOINT = "formicmind/eco/checkpoint/v1"
HASH_DOMAIN_REPORT = "formicmind/eco/fairness/v1"

# RNG streams of E1: the map stream is the only consumer of randomness in E1-a; "world" is reserved and unused by the rules.
MAP_STREAM_ID = "map"
WORLD_STREAM_ID = "world"
