"""Canonical serialization, round-trip and hashes for config, state and checkpoint.

Field order is fixed by the schema: JSON object keys are sorted (byte order);
arrays have a specified order: nodes by node id, colonies by side, units and
brood by unit id, ``forage_q`` by category, intel nodes by node id, resources by
node id, edges by ``(a, b)``. Loaders are strict: unknown or missing keys,
wrong schema tags, unsorted arrays and non-canonical bytes are errors.

Three hashes (all SHA-256, lowercase hex, domain-separated, see ``versions``):

* ``config_hash``      - canonical config bytes;
* ``state_hash``       - canonical ``SimState`` bytes (game model only; includes
  ``next_unit_id`` and the config hash, not the RNG);
* ``checkpoint_hash``  - canonical ``{state, run}`` bytes: the state plus every RNG
  stream state, both sides' ``WindowStats`` and the episode ``outcome`` (``{"finished": false}`` while it runs),
  i.e. everything needed to resume a run or to know that it is over.
  Format v3 (FOR-13; v2 added the windows in FOR-11); v1/v2 checkpoints fail fast on the schema tag. On load
  the outcome is cross-checked against the state (``terminal.validate_outcome``).
"""

from __future__ import annotations

from typing import Any

from .canonical import (
    CanonicalError,
    canonical_dumps,
    canonical_loads,
    domain_hash,
    u64_from_str,
    u64_to_str,
)
from .config import (
    CandidateBalance,
    EdgeSpec,
    NodeResource,
    NodeSpec,
    SimConfig,
    Topology,
)
from .invariants import InvariantError, validate_config, validate_state
from .model import BroodRecord, ColonyState, IntelMemory, NodeIntel, SimState, Unit
from .rng import Pcg32, check_stream_id
from .runstate import RunState
from .snapshot import WindowStats
from .terminal import outcome_from_canonical, validate_outcome
from .versions import (
    HASH_DOMAIN_CHECKPOINT,
    HASH_DOMAIN_CONFIG,
    HASH_DOMAIN_STATE,
    RNG_VERSION,
    SERIALIZATION_VERSION,
    STREAM_DERIVATION_VERSION,
)

SCHEMA_CONFIG = "formicmind.sim_config/1"
SCHEMA_STATE = "formicmind.sim_state/1"
SCHEMA_RUN = "formicmind.run_state/3"
SCHEMA_CHECKPOINT = "formicmind.checkpoint/3"


# --- strict readers ---------------------------------------------------------

def _obj(data: Any, keys: set[str], what: str) -> dict[str, Any]:
    if type(data) is not dict:
        raise CanonicalError(f"{what}: expected an object")
    if set(data) != keys:
        missing, extra = keys - set(data), set(data) - keys
        raise CanonicalError(f"{what}: missing keys {sorted(missing)}, unexpected keys {sorted(extra)}")
    return data


def _int(data: dict[str, Any], key: str, what: str) -> int:
    value = data[key]
    if type(value) is not int:
        raise CanonicalError(f"{what}.{key}: expected an int")
    return value


def _str(data: dict[str, Any], key: str, what: str) -> str:
    value = data[key]
    if type(value) is not str:
        raise CanonicalError(f"{what}.{key}: expected a string")
    return value


def _bool(data: dict[str, Any], key: str, what: str) -> bool:
    value = data[key]
    if type(value) is not bool:
        raise CanonicalError(f"{what}.{key}: expected a bool")
    return value


def _list(data: dict[str, Any], key: str, what: str) -> list[Any]:
    value = data[key]
    if type(value) is not list:
        raise CanonicalError(f"{what}.{key}: expected an array")
    return value


def _schema(data: dict[str, Any], expected: str) -> None:
    if data["schema"] != expected or data["serialization_version"] != SERIALIZATION_VERSION:
        raise CanonicalError(
            f"unsupported schema/version {data['schema']!r}/{data['serialization_version']!r}"
        )


# --- config -------------------------------------------------------------------

_BALANCE_INT_FIELDS = (
    "start_food", "start_workers", "start_soldiers", "worker_hp", "soldier_hp", "queen_hp",
    "worker_egg_cost", "soldier_egg_cost", "eggs_per_lay", "lay_interval_ticks",
    "worker_hatch_delay", "soldier_hatch_delay", "worker_upkeep_tenths", "soldier_upkeep_tenths",
    "starvation_unit_cost", "soldier_damage", "home_defense_num", "home_defense_den",
)


def config_to_canonical(config: SimConfig) -> dict[str, Any]:
    balance: dict[str, Any] = {name: getattr(config.balance, name) for name in _BALANCE_INT_FIELDS}
    balance["resources"] = [
        {"node_id": r.node_id, "start_food": r.start_food, "capacity": r.capacity,
         "regen": r.regen, "max_harvest": r.max_harvest}
        for r in config.balance.resources
    ]
    return {
        "schema": SCHEMA_CONFIG,
        "serialization_version": SERIALIZATION_VERSION,
        "rules_version": config.rules_version,
        "balance_version": config.balance_version,
        "policy_interval_k": config.policy_interval_k,
        "t_max": config.t_max,
        "topology": {
            "nodes": [{"node_id": n.node_id, "name": n.name, "home_side": n.home_side}
                      for n in config.topology.nodes],
            "edges": [{"a": e.a, "b": e.b, "length": e.length} for e in config.topology.edges],
        },
        "balance": balance,
    }


def config_from_canonical(data: Any) -> SimConfig:
    top = _obj(data, {"schema", "serialization_version", "rules_version", "balance_version",
                      "policy_interval_k", "t_max", "topology", "balance"}, "config")
    _schema(top, SCHEMA_CONFIG)
    topo = _obj(top["topology"], {"nodes", "edges"}, "config.topology")
    nodes = tuple(
        NodeSpec(_int(n, "node_id", "node"), _str(n, "name", "node"), _int(n, "home_side", "node"))
        for n in (_obj(x, {"node_id", "name", "home_side"}, "node") for x in _list(topo, "nodes", "topology"))
    )
    edges = tuple(
        EdgeSpec(_int(e, "a", "edge"), _int(e, "b", "edge"), _int(e, "length", "edge"))
        for e in (_obj(x, {"a", "b", "length"}, "edge") for x in _list(topo, "edges", "topology"))
    )
    bal = _obj(top["balance"], set(_BALANCE_INT_FIELDS) | {"resources"}, "config.balance")
    resources = tuple(
        NodeResource(
            _int(r, "node_id", "resource"), _int(r, "start_food", "resource"),
            _int(r, "capacity", "resource"), _int(r, "regen", "resource"),
            _int(r, "max_harvest", "resource"),
        )
        for r in (
            _obj(x, {"node_id", "start_food", "capacity", "regen", "max_harvest"}, "resource")
            for x in _list(bal, "resources", "balance")
        )
    )
    balance = CandidateBalance(
        resources=resources, **{name: _int(bal, name, "balance") for name in _BALANCE_INT_FIELDS}
    )
    config = SimConfig(
        rules_version=_str(top, "rules_version", "config"),
        balance_version=_str(top, "balance_version", "config"),
        topology=Topology(nodes=nodes, edges=edges),
        balance=balance,
        policy_interval_k=_int(top, "policy_interval_k", "config"),
        t_max=_int(top, "t_max", "config"),
    )
    validate_config(config)
    return config


def config_to_bytes(config: SimConfig) -> bytes:
    return canonical_dumps(config_to_canonical(config))


def config_from_bytes(data: bytes) -> SimConfig:
    return config_from_canonical(canonical_loads(data))


def config_hash(config: SimConfig) -> str:
    return domain_hash(HASH_DOMAIN_CONFIG, config_to_bytes(config))


# --- state --------------------------------------------------------------------

def _colony_to_canonical(c: ColonyState) -> dict[str, Any]:
    return {
        "side": c.side,
        "food": c.food,
        "queen_hp": c.queen_hp,
        "next_lay_tick": c.next_lay_tick,
        "upkeep_remainder10": c.upkeep_remainder10,
        "starvation_debt": c.starvation_debt,
        "lay_order": c.lay_order,
        "forage_q": list(c.forage_q),
        "army_choice": c.army_choice,
        "avoid_center": c.avoid_center,
        "units": [
            {"unit_id": u.unit_id, "caste": u.caste, "hp": u.hp, "node": u.node,
             "from_node": u.from_node, "to_node": u.to_node,
             "arrival_tick": u.arrival_tick, "target_node": u.target_node}
            for u in c.units
        ],
        "brood": [
            {"unit_id": b.unit_id, "caste": b.caste, "laid_tick": b.laid_tick,
             "hatch_tick": b.hatch_tick}
            for b in c.brood
        ],
        "intel": {
            "nodes": [
                {"seen_tick": n.seen_tick, "food_last": n.food_last,
                 "enemy_workers_last": n.enemy_workers_last,
                 "enemy_soldiers_last": n.enemy_soldiers_last}
                for n in c.intel.nodes
            ],
            "enemy_queen_hp_last": c.intel.enemy_queen_hp_last,
            "enemy_queen_seen_tick": c.intel.enemy_queen_seen_tick,
        },
    }


def state_to_canonical(state: SimState) -> dict[str, Any]:
    return {
        "schema": SCHEMA_STATE,
        "serialization_version": SERIALIZATION_VERSION,
        "rules_version": state.config.rules_version,
        "config_sha256": config_hash(state.config),
        "tick": state.tick,
        "next_unit_id": state.next_unit_id,
        "node_food": list(state.node_food),
        "colonies": [_colony_to_canonical(c) for c in state.colonies],
    }


_COLONY_KEYS = {
    "side", "food", "queen_hp", "next_lay_tick", "upkeep_remainder10", "starvation_debt",
    "lay_order", "forage_q", "army_choice", "avoid_center", "units", "brood", "intel",
}
_UNIT_KEYS = {"unit_id", "caste", "hp", "node", "from_node", "to_node", "arrival_tick", "target_node"}
_BROOD_KEYS = {"unit_id", "caste", "laid_tick", "hatch_tick"}
_INTEL_NODE_KEYS = {"seen_tick", "food_last", "enemy_workers_last", "enemy_soldiers_last"}


def _colony_from_canonical(data: Any) -> ColonyState:
    c = _obj(data, _COLONY_KEYS, "colony")
    units = [
        Unit(*(_int(u, k, "unit") for k in (
            "unit_id", "caste", "hp", "node", "from_node", "to_node", "arrival_tick", "target_node")))
        for u in (_obj(x, _UNIT_KEYS, "unit") for x in _list(c, "units", "colony"))
    ]
    brood = [
        BroodRecord(*(_int(b, k, "brood") for k in ("unit_id", "caste", "laid_tick", "hatch_tick")))
        for b in (_obj(x, _BROOD_KEYS, "brood") for x in _list(c, "brood", "colony"))
    ]
    intel_raw = _obj(c["intel"], {"nodes", "enemy_queen_hp_last", "enemy_queen_seen_tick"}, "intel")
    intel = IntelMemory(
        nodes=[
            NodeIntel(*(_int(n, k, "intel node") for k in (
                "seen_tick", "food_last", "enemy_workers_last", "enemy_soldiers_last")))
            for n in (_obj(x, _INTEL_NODE_KEYS, "intel node") for x in _list(intel_raw, "nodes", "intel"))
        ],
        enemy_queen_hp_last=_int(intel_raw, "enemy_queen_hp_last", "intel"),
        enemy_queen_seen_tick=_int(intel_raw, "enemy_queen_seen_tick", "intel"),
    )
    forage_q = _list(c, "forage_q", "colony")
    if any(type(v) is not int for v in forage_q):
        raise CanonicalError("colony.forage_q: expected ints")
    return ColonyState(
        side=_int(c, "side", "colony"),
        food=_int(c, "food", "colony"),
        queen_hp=_int(c, "queen_hp", "colony"),
        next_lay_tick=_int(c, "next_lay_tick", "colony"),
        upkeep_remainder10=_int(c, "upkeep_remainder10", "colony"),
        starvation_debt=_int(c, "starvation_debt", "colony"),
        lay_order=_int(c, "lay_order", "colony"),
        forage_q=list(forage_q),
        army_choice=_int(c, "army_choice", "colony"),
        avoid_center=_bool(c, "avoid_center", "colony"),
        units=units,
        brood=brood,
        intel=intel,
    )


def state_from_canonical(data: Any, config: SimConfig) -> SimState:
    top = _obj(data, {"schema", "serialization_version", "rules_version", "config_sha256",
                      "tick", "next_unit_id", "node_food", "colonies"}, "state")
    _schema(top, SCHEMA_STATE)
    if top["rules_version"] != config.rules_version or top["config_sha256"] != config_hash(config):
        raise CanonicalError("state was produced under a different config / rules_version")
    node_food = _list(top, "node_food", "state")
    if any(type(v) is not int for v in node_food):
        raise CanonicalError("state.node_food: expected ints")
    state = SimState(
        config=config,
        tick=_int(top, "tick", "state"),
        node_food=list(node_food),
        colonies=[_colony_from_canonical(c) for c in _list(top, "colonies", "state")],
        next_unit_id=_int(top, "next_unit_id", "state"),
    )
    validate_state(state)
    return state


def state_to_bytes(state: SimState) -> bytes:
    return canonical_dumps(state_to_canonical(state))


def state_from_bytes(data: bytes, config: SimConfig) -> SimState:
    return state_from_canonical(canonical_loads(data), config)


def state_hash(state: SimState) -> str:
    return domain_hash(HASH_DOMAIN_STATE, state_to_bytes(state))


# --- run state and checkpoint ---------------------------------------------------

def run_state_to_canonical(run: RunState) -> dict[str, Any]:
    return {
        "schema": SCHEMA_RUN,
        "serialization_version": SERIALIZATION_VERSION,
        "rng_version": RNG_VERSION,
        "stream_derivation_version": STREAM_DERIVATION_VERSION,
        "root_seed": u64_to_str(run.root_seed),
        "streams": {sid: stream.to_canonical() for sid, stream in run.streams.items()},
        "windows": [
            {"harvest": w.harvest, "upkeep_paid": w.upkeep_paid, "egg_spend": w.egg_spend,
             "starvation_debt_delta": w.starvation_debt_delta, "window_ticks": w.window_ticks}
            for w in run.windows
        ],
        "outcome": {"finished": False} if run.outcome is None else {"finished": True, **run.outcome.to_canonical()},
    }


def run_state_from_canonical(data: Any) -> RunState:
    top = _obj(data, {"schema", "serialization_version", "rng_version",
                      "stream_derivation_version", "root_seed", "streams", "windows", "outcome"}, "run")
    _schema(top, SCHEMA_RUN)
    if top["rng_version"] != RNG_VERSION or top["stream_derivation_version"] != STREAM_DERIVATION_VERSION:
        raise CanonicalError("unsupported RNG / stream derivation version")
    streams_raw = top["streams"]
    if type(streams_raw) is not dict:
        raise CanonicalError("run.streams: expected an object")
    streams: dict[str, Pcg32] = {}
    for sid in sorted(streams_raw):
        try:
            check_stream_id(sid)
        except ValueError as exc:
            raise CanonicalError(str(exc)) from exc
        streams[sid] = Pcg32.from_canonical(streams_raw[sid])
    windows_raw = _list(top, "windows", "run")
    if len(windows_raw) != 2:
        raise CanonicalError("run.windows must hold one entry per side")
    keys = {"harvest", "upkeep_paid", "egg_spend", "starvation_debt_delta", "window_ticks"}
    try:
        windows = [
            WindowStats(**{k: _int(w, k, "window") for k in keys})
            for w in (_obj(x, keys, "window") for x in windows_raw)
        ]
    except (TypeError, ValueError) as exc:
        raise CanonicalError(f"bad window stats: {exc}") from exc
    raw = top["outcome"]
    if type(raw) is dict and raw == {"finished": False}:  # no None in canonical data
        outcome = None
    elif type(raw) is dict and raw.get("finished") is True:
        outcome = outcome_from_canonical({k: v for k, v in raw.items() if k != "finished"})
    else:
        raise CanonicalError("run.outcome: expected {'finished': false} or a finished outcome object")
    return RunState(root_seed=u64_from_str(top["root_seed"]), streams=streams, windows=windows, outcome=outcome)


def checkpoint_to_bytes(state: SimState, run: RunState) -> bytes:
    return canonical_dumps({
        "schema": SCHEMA_CHECKPOINT,
        "serialization_version": SERIALIZATION_VERSION,
        "state": state_to_canonical(state),
        "run": run_state_to_canonical(run),
    })


def checkpoint_from_bytes(data: bytes, config: SimConfig) -> tuple[SimState, RunState]:
    top = _obj(canonical_loads(data), {"schema", "serialization_version", "state", "run"}, "checkpoint")
    _schema(top, SCHEMA_CHECKPOINT)
    state = state_from_canonical(top["state"], config)
    run = run_state_from_canonical(top["run"])
    try:
        validate_outcome(state, run.outcome)
    except InvariantError as exc:
        raise CanonicalError(str(exc)) from exc
    return state, run


def checkpoint_hash(state: SimState, run: RunState) -> str:
    return domain_hash(HASH_DOMAIN_CHECKPOINT, checkpoint_to_bytes(state, run))
