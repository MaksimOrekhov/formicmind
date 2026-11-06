"""Canonical bytes, strict loaders and hashes for MapE1, WorldStateE1 and the checkpoint (contracts sections 3, 4, 10.1)."""

from __future__ import annotations

from dataclasses import replace

from ..canonical import CanonicalError, canonical_dumps, canonical_loads, domain_hash
from ..rng import Pcg32
from .codec import from_canonical, to_canonical
from .config import ConfigE1, config_hash
from .model import MapE1, RunStateE1, StreamState, WorldStateE1
from .validate import validate_map, validate_run_state, validate_state
from .versions import CHECKPOINT_SCHEMA, HASH_DOMAIN_CHECKPOINT, HASH_DOMAIN_MAP, HASH_DOMAIN_STATE


# ---------------------------------------------------------------------------------------------------------------- map


def compute_map_hash(world_map: MapE1) -> str:
    """Hash of the map content with the ``map_hash`` field blanked (so the hash is not part of its own input)."""
    return domain_hash(HASH_DOMAIN_MAP, canonical_dumps(to_canonical(replace(world_map, map_hash=""))))


def seal_map(world_map: MapE1) -> MapE1:
    return replace(world_map, map_hash=compute_map_hash(world_map))


def map_to_bytes(world_map: MapE1) -> bytes:
    return canonical_dumps(to_canonical(world_map))


def map_from_canonical(data: object, config: ConfigE1) -> MapE1:
    world_map = from_canonical(MapE1, data)
    validate_map(world_map, config)
    if world_map.map_hash != compute_map_hash(world_map):
        raise CanonicalError("map_hash does not match the map content")
    return world_map


def map_from_bytes(data: bytes, config: ConfigE1) -> MapE1:
    return map_from_canonical(canonical_loads(data), config)


# ---------------------------------------------------------------------------------------------------------------- state


def state_to_bytes(state: WorldStateE1) -> bytes:
    return canonical_dumps(to_canonical(state))


def state_from_canonical(data: object, config: ConfigE1, world_map: MapE1) -> WorldStateE1:
    state = from_canonical(WorldStateE1, data)
    validate_state(state, config, world_map)
    return state


def state_from_bytes(data: bytes, config: ConfigE1, world_map: MapE1) -> WorldStateE1:
    return state_from_canonical(canonical_loads(data), config, world_map)


def state_hash(state: WorldStateE1) -> str:
    """State of the rules, no RNG and no policy state (as in L0). Binds the config and map hashes carried by the state."""
    body = {"config_hash": state.config_hash, "map_hash": state.map_hash, "state": to_canonical(state)}
    return domain_hash(HASH_DOMAIN_STATE, canonical_dumps(body))


# ---------------------------------------------------------------------------------------------------------------- run / checkpoint


def new_run_state(root_seed: int, streams: dict[str, Pcg32]) -> RunStateE1:
    ordered = [StreamState(sid, rng.state, rng.inc) for sid, rng in sorted(streams.items())]
    run = RunStateE1(root_seed, ordered)
    validate_run_state(run)
    return run


def checkpoint_to_canonical(config: ConfigE1, world_map: MapE1, state: WorldStateE1, run: RunStateE1) -> dict:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "config": to_canonical(config),
        "map": to_canonical(world_map),
        "state": to_canonical(state),
        "run": to_canonical(run),
    }


def checkpoint_hash(state: WorldStateE1, run: RunStateE1) -> str:
    """State + all RNG streams: everything needed to continue (contracts 10.1)."""
    return domain_hash(HASH_DOMAIN_CHECKPOINT, canonical_dumps({"state": to_canonical(state), "run": to_canonical(run)}))


def checkpoint_to_bytes(config: ConfigE1, world_map: MapE1, state: WorldStateE1, run: RunStateE1) -> bytes:
    return canonical_dumps(checkpoint_to_canonical(config, world_map, state, run))


def checkpoint_from_bytes(data: bytes) -> tuple[ConfigE1, MapE1, WorldStateE1, RunStateE1]:
    from .config import config_from_canonical

    doc = canonical_loads(data)
    if type(doc) is not dict or set(doc) != {"schema", "config", "map", "state", "run"} or doc["schema"] != CHECKPOINT_SCHEMA:
        raise CanonicalError("not an Ecosystem v1 checkpoint (bad schema or keys)")
    config = config_from_canonical(doc["config"])
    world_map = map_from_canonical(doc["map"], config)
    state = state_from_canonical(doc["state"], config, world_map)
    run = from_canonical(RunStateE1, doc["run"])
    validate_run_state(run)
    if state.config_hash != config_hash(config) or state.map_hash != world_map.map_hash:
        raise CanonicalError("checkpoint parts do not belong together (config/map hash mismatch)")
    return config, world_map, state, run
