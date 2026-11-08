"""Versioned, offline replay file of a real E1 run for the viewer (FOR-47): ``formicmind.eco.replay/1``.

One file is one run of ``step_world_tick`` (no second simulation, no server). It is JSONL: one canonical-JSON record per line, each carrying
its ``seq`` and a SHA-256 ``chain`` over the previous chain value and the record, so a truncated, reordered or edited file is rejected.

Records, in order:

* ``header``  seed, roles, config (+hash), the complete map (+hash), cadence, the field layout of the compact rows;
* ``event``   every typed event of ``events.py`` in the order of execution (the whole log, nothing sampled);
* ``frame``   the state *after* ``tick`` completed ticks: true patch stocks, piles, colony vitals, every cohort (id, caste, count, cargo,
              mission, node or edge with the real ``depart``/``arrive`` ticks) and, per colony, its **private memory** (visited nodes, the
              nodes it only knows by name, met colonies, sightings). Frames are written every ``frame_every`` ticks plus at the start, the end
              and right after a queen death, a collapse or an extinction; nothing between frames is invented;
* ``result``  outcomes, ``events_hash`` (the chain hash of ``EventLog``), final state/checkpoint hashes, counts, ``chain_head``.

``read_replay`` is strict and fail-closed; ``verify_replay`` re-executes the run from the header and demands byte equality.
The file contains no wall-clock field, so the same seed, config and roles give the same bytes.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..canonical import CanonicalError, canonical_dumps, canonical_loads, domain_hash
from .codec import to_canonical
from .config import ConfigE1, config_from_canonical, config_hash
from .engine import RouteTable, step_world_tick
from .events import EVENT_KINDS, EventE1, EventLog
from .fixtures import fixture_controller, normalize_fixture
from .initial import build_world
from .model import MapE1, WorldStateE1
from .policy import PolicyRunSpec, build_policies, spec_from_canonical, spec_to_canonical
from .policy_run import PolicyController
from .serialization import checkpoint_hash, compute_map_hash, map_from_canonical, state_hash
from .versions import ECO_BALANCE_VERSION, ECO_RULES_VERSION

REPLAY_SCHEMA = "formicmind.eco.replay/1"
RECORD_DOMAIN = "formicmind/eco/replay-record/v1"
ZERO_CHAIN = "0" * 64

KIND_CODES = {"home": 0, "forage": 1, "scout": 2, "guard": 3, "attack": 4, "withdraw": 5}
PHASE_CODES = {"idle": 0, "go": 1, "work": 2, "return": 3}
STATUS_NAMES = ("alive", "orphan", "extinct")
COHORT_FIELDS = ("id", "colony", "caste", "count", "hp", "load", "kind", "phase", "where", "a", "b", "depart", "arrive",
                 "target_node", "target_colony")
VISITED_FIELDS = ("node", "seen_tick", "last_stock", "last_stock_tick", "queen_seen", "danger")
FOREIGN_FIELDS = ("colony", "first_contact", "last_seen", "workers_bucket", "soldiers_bucket", "nest_node", "they_attacked",
                  "i_attacked", "kills_mine", "kills_theirs")
SIGHTING_FIELDS = ("node", "colony", "tick", "workers_bucket", "soldiers_bucket", "aggression")
KEY_EVENTS = ("queen_dead", "collapse", "extinct")


class ReplayFormatError(CanonicalError):
    """The file is not a valid ``formicmind.eco.replay/1`` document."""


def _fail(message: str) -> None:
    raise ReplayFormatError(message)


# ------------------------------------------------------------------------------------------------ writing


class _Chain:
    def __init__(self) -> None:
        self.prev, self.seq, self.lines = ZERO_CHAIN, 0, []

    def add(self, record: dict) -> None:
        body = {**record, "seq": self.seq, "chain": ZERO_CHAIN}
        chain = domain_hash(RECORD_DOMAIN, self.prev.encode("ascii") + b"\n" + canonical_dumps(body))
        self.lines.append(canonical_dumps({**body, "chain": chain}))
        self.prev, self.seq = chain, self.seq + 1


def _cohort_row(cohort) -> list[int]:
    loc, mission = cohort.loc, cohort.mission
    where = 0 if loc.kind == "node" else 1
    return [cohort.cohort_id, cohort.colony_id, 0 if cohort.caste == "worker" else 1, cohort.count, cohort.hp_total, cohort.load,
            KIND_CODES[mission.kind], PHASE_CODES[mission.phase], where,
            loc.node_id if where == 0 else loc.edge_from, -1 if where == 0 else loc.edge_to,
            0 if where == 0 else loc.depart_tick, 0 if where == 0 else loc.arrive_tick, mission.target_node, mission.target_colony]


def _memory_record(colony) -> dict:
    memory = colony.memory
    visited, discovered = [], []
    for known in memory.nodes:
        if known.visited:
            visited.append([known.node_id, known.seen_tick, known.last_stock if known.last_stock_tick >= 0 else -1,
                            known.last_stock_tick, known.queen_seen, known.danger_q])
        else:
            discovered.append(known.node_id)
    nodes = memory.nodes
    return {
        "colony": colony.colony_id,
        "visited": sorted(visited),
        "discovered": sorted(discovered),
        "foreign": [[r.colony_id, r.first_contact_tick, r.last_seen_tick, r.last_seen_workers_bucket, r.last_seen_soldiers_bucket,
                     nodes[r.nest_node_slot].node_id if r.nest_node_slot >= 0 else -1, r.they_attacked_q, r.i_attacked_q,
                     r.kills_mine_q, r.kills_theirs_q] for r in memory.foreign],
        "sightings": [[nodes[s.node_slot].node_id, memory.foreign[s.colony_slot].colony_id, s.tick, s.workers_bucket,
                       s.soldiers_bucket, int(s.aggression_observed)] for s in memory.sightings],
    }


def _frame_record(state: WorldStateE1, reason: str) -> dict:
    colonies = []
    for colony in state.colonies:
        mine = [c for c in state.cohorts if c.colony_id == colony.colony_id]
        colonies.append({
            "id": colony.colony_id, "status": colony.status, "food": colony.food, "queen_hp": colony.queen_hp,
            "workers": sum(c.count for c in mine if c.caste == "worker"),
            "soldiers": sum(c.count for c in mine if c.caste == "soldier"),
            "brood_workers": sum(b.caste == "worker" for b in colony.brood),
            "brood_soldiers": sum(b.caste == "soldier" for b in colony.brood),
            "starvation_debt": colony.starvation_debt,
        })
    return {
        "type": "frame", "tick": state.tick, "reason": reason, "state_hash": state_hash(state),
        "patches": [patch.stock for patch in state.patches],
        "piles": [[pile.node_id, pile.stock] for pile in state.piles],
        "colonies": colonies,
        "cohorts": [_cohort_row(c) for c in state.cohorts],
        "memory": [_memory_record(c) for c in state.colonies],
    }


def _event_record(item: EventE1) -> dict:
    return {"type": "event", **to_canonical(item)}


def export_replay(config: ConfigE1, seed: int, fixture: str = "peaceful_fair", roles: dict[int, str] | None = None,
                  ticks: int | None = None, frame_every: int = 20) -> bytes:
    """Run the real engine and return the replay file bytes. Deterministic: no clock, no environment."""
    fixture = normalize_fixture(fixture)
    controller = fixture_controller(fixture, roles)
    return _export(config, seed, controller, fixture, {str(k): v for k, v in sorted(controller.roles.items())}, ticks, frame_every, None, None)


def export_policy_replay(config: ConfigE1, seed: int, spec: PolicyRunSpec, ticks: int | None = None,
                         frame_every: int = 20) -> tuple[bytes, bytes]:
    """Run the real engine with one independent ``PolicyE1`` per colony; returns ``(replay bytes, decision records as JSONL)``.

    The replay has the ordinary ``formicmind.eco.replay/1`` records (so the viewer shows the movements and events) plus the specification of the
    policies in the header and the count and chain hash of the decision records in the result; the decision records themselves are the sidecar.
    """
    controller = PolicyController(build_policies(spec, config), curriculum=spec.curriculum)
    roles ={str(i): (p.kind if not p.variant else f"{p.kind}:{p.variant}") for i, p in enumerate(spec.policies)}
    replay = _export(config, seed, controller, "policy", roles, ticks, frame_every, spec_to_canonical(spec), controller.log)
    return replay, controller.log.to_bytes()


def _export(config: ConfigE1, seed: int, controller, fixture: str, roles: dict[str, str], ticks: int | None, frame_every: int,
            policy: dict | None, decisions) -> bytes:
    duration = config.horizon if ticks is None else ticks
    if type(duration) is not int or not 0 <= duration <= config.horizon:
        raise ValueError("ticks must be an integer between zero and the configured horizon")
    if type(frame_every) is not int or frame_every < 1:
        raise ValueError("frame_every must be a positive integer")
    world = build_world(config, seed)
    routes = RouteTable(world.world_map)
    chain, log = _Chain(), EventLog()
    header = {
        "type": "header", "schema": REPLAY_SCHEMA, "rules_version": ECO_RULES_VERSION, "balance_version": ECO_BALANCE_VERSION,
        "seed": str(seed), "fixture": fixture, "roles": roles,
        "ticks_requested": duration, "frame_every": frame_every, "decision_k": config.decision_k,
        "alert_interval": config.alert_interval, "colonies": config.n_colonies,
        "config": to_canonical(config), "config_hash": config_hash(config),
        "map": to_canonical(world.world_map), "map_hash": world.world_map.map_hash,
        "cohort_fields": list(COHORT_FIELDS), "visited_fields": list(VISITED_FIELDS), "foreign_fields": list(FOREIGN_FIELDS),
        "sighting_fields": list(SIGHTING_FIELDS), "kind_codes": KIND_CODES, "phase_codes": PHASE_CODES,
    }
    if policy is not None:
        header["policy"] = policy
    chain.add(header)
    frames = 0
    chain.add(_frame_record(world.state, "start"))
    frames += 1
    for _ in range(duration):
        record = step_world_tick(world.state, world.run, config, world.world_map, controller, routes=routes,
                                 validate_input=world.state.tick == 0)
        log.extend(record.events)
        for item in record.events:
            chain.add(_event_record(item))
        finished = world.state.world_outcome.finished
        if world.state.tick == duration or finished:
            reason = "final"
        elif world.state.tick % frame_every == 0:
            reason = "interval"
        elif any(item.kind in KEY_EVENTS for item in record.events):
            reason = "key_event"
        else:
            reason = ""
        if reason:
            chain.add(_frame_record(world.state, reason))
            frames += 1
        if finished:
            break
    outcome = world.state.world_outcome
    result = {
        "type": "result", "ticks_completed": world.state.tick,
        "world_outcome": {"finished": outcome.finished, "reason": outcome.reason, "terminal_tick": outcome.terminal_tick},
        "colony_outcomes": [{"colony": i, "status": world.state.colonies[i].status, "reason": o.reason,
                             "terminal_tick": o.terminal_tick} for i, o in enumerate(world.state.outcomes)],
        "event_count": len(log.events), "events_hash": log.head, "frame_count": frames,
        "final_state_hash": state_hash(world.state), "final_checkpoint_hash": checkpoint_hash(world.state, world.run),
        "record_count": chain.seq + 1,
    }
    if decisions is not None:
        result["decision_count"], result["decisions_hash"] = decisions.count, decisions.head
    chain.add(result)
    return b"\n".join(chain.lines) + b"\n"


# ------------------------------------------------------------------------------------------------ reading


@dataclass(slots=True)
class ReplayDocument:
    header: dict
    config: ConfigE1
    world_map: MapE1
    events: list[EventE1]
    frames: list[dict]
    result: dict
    chain_head: str
    policy: PolicyRunSpec | None = None


def _need(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _is_int(value) -> bool:
    return type(value) is int


def _int_list(value, length: int | None, where: str) -> None:
    _need(type(value) is list and all(_is_int(x) for x in value) and (length is None or len(value) == length), f"{where}: bad integer row")


HEADER_KEYS = {"type", "seq", "chain", "schema", "rules_version", "balance_version", "seed", "fixture", "roles", "ticks_requested",
               "frame_every", "decision_k", "alert_interval", "colonies", "config", "config_hash", "map", "map_hash", "cohort_fields",
               "visited_fields", "foreign_fields", "sighting_fields", "kind_codes", "phase_codes"}
FRAME_KEYS = {"type", "seq", "chain", "tick", "reason", "state_hash", "patches", "piles", "colonies", "cohorts", "memory"}
EVENT_KEYS = {"type", "seq", "chain", "tick", "kind", "colony", "other", "node", "a", "b"}
RESULT_KEYS = {"type", "seq", "chain", "ticks_completed", "world_outcome", "colony_outcomes", "event_count", "events_hash",
               "frame_count", "final_state_hash", "final_checkpoint_hash", "record_count"}
COLONY_KEYS = {"id", "status", "food", "queen_hp", "workers", "soldiers", "brood_workers", "brood_soldiers", "starvation_debt"}
MEMORY_KEYS = {"colony", "visited", "discovered", "foreign", "sightings"}


POLICY_RESULT_KEYS = RESULT_KEYS | {"decision_count", "decisions_hash"}


def read_replay(data: bytes) -> ReplayDocument:
    """Parse and validate a replay file: canonical lines, chain, hashes, record shapes. Raises ``ReplayFormatError`` otherwise."""
    _need(isinstance(data, (bytes, bytearray)) and data.endswith(b"\n") and len(data) > 1, "a replay is non-empty JSONL ending with a newline")
    lines = bytes(data).split(b"\n")[:-1]
    prev, records = ZERO_CHAIN, []
    for seq, line in enumerate(lines):
        try:
            record = canonical_loads(line)
        except CanonicalError as exc:
            raise ReplayFormatError(f"line {seq + 1}: {exc}") from exc
        _need(type(record) is dict and record.get("seq") == seq, f"line {seq + 1}: seq must be {seq}")
        claimed = record.get("chain")
        _need(type(claimed) is str, f"line {seq + 1}: missing chain")
        expected = domain_hash(RECORD_DOMAIN, prev.encode("ascii") + b"\n" + canonical_dumps({**record, "chain": ZERO_CHAIN}))
        _need(claimed == expected, f"line {seq + 1}: the integrity chain is broken")
        prev = claimed
        records.append(record)
    _need(len(records) >= 3, "a replay needs a header, a frame and a result")
    header, result = records[0], records[-1]
    policy = None
    _need(header.get("type") == "header" and set(header) in (HEADER_KEYS, HEADER_KEYS | {"policy"}), "the first record must be a complete header")
    if "policy" in header:
        try:
            policy = spec_from_canonical(header["policy"])
        except CanonicalError as exc:
            raise ReplayFormatError(f"header.policy: {exc}") from exc
        _need(header["fixture"] == "policy", "a policy replay has the fixture name 'policy'")
    if header["schema"] != REPLAY_SCHEMA:
        _fail(f"unsupported replay schema {header['schema']!r}; this reader understands {REPLAY_SCHEMA!r}")
    _need(result.get("type") == "result" and set(result) == (POLICY_RESULT_KEYS if policy else RESULT_KEYS), "the last record must be a complete result")
    if policy is not None:
        _need(_is_int(result["decision_count"]) and result["decision_count"] >= 0 and type(result["decisions_hash"]) is str,
              "bad decision count or hash")
    try:
        config = config_from_canonical(header["config"])
        _need(config_hash(config) == header["config_hash"], "config_hash does not match the configuration in the file")
        world_map = map_from_canonical(header["map"], config)
    except ReplayFormatError:
        raise
    except (CanonicalError, ValueError) as exc:
        raise ReplayFormatError(f"header: {exc}") from exc
    _need(world_map.map_hash == header["map_hash"] == compute_map_hash(world_map), "map_hash does not match the map in the file")
    _need(header["colonies"] == config.n_colonies and type(header["seed"]) is str and header["seed"].isdigit(), "bad header")
    _need(header["cohort_fields"] == list(COHORT_FIELDS) and header["visited_fields"] == list(VISITED_FIELDS)
          and header["foreign_fields"] == list(FOREIGN_FIELDS) and header["sighting_fields"] == list(SIGHTING_FIELDS)
          and header["kind_codes"] == KIND_CODES and header["phase_codes"] == PHASE_CODES, "row layouts differ from this reader")
    patch_nodes = sum(1 for node in world_map.nodes if node.patch)
    events: list[EventE1] = []
    frames: list[dict] = []
    log = EventLog()
    last_event_tick, last_frame_tick = 0, -1
    for record in records[1:-1]:
        kind = record["type"]
        if kind == "event":
            _need(set(record) == EVENT_KEYS and record["kind"] in EVENT_KINDS, "bad event record")
            _need(all(_is_int(record[k]) for k in ("tick", "colony", "other", "node", "a", "b")), "bad event fields")
            _need(record["tick"] >= last_event_tick and record["tick"] >= last_frame_tick, "events must not go back in time")
            last_event_tick = record["tick"]
            item = EventE1(record["tick"], record["kind"], record["colony"], record["other"], record["node"], record["a"], record["b"])
            events.append(item)
            log.extend([item])
        elif kind == "frame":
            _validate_frame(record, config, world_map, patch_nodes, last_frame_tick)
            last_frame_tick = record["tick"]
            frames.append(record)
        else:
            _fail(f"unknown record type {kind!r}")
    _need(frames and frames[0]["tick"] == 0 and frames[0]["reason"] == "start", "the first frame must be the start state")
    _need(frames[-1]["reason"] == "final" and frames[-1]["tick"] == result["ticks_completed"], "the last frame must be the final state")
    _need(result["event_count"] == len(events) and result["events_hash"] == log.head, "events_hash does not match the event records")
    _need(result["frame_count"] == len(frames) and result["record_count"] == len(records), "record counts do not match")
    _need(result["final_state_hash"] == frames[-1]["state_hash"], "final_state_hash differs from the final frame")
    _need(all(e.tick < result["ticks_completed"] for e in events), "an event lies beyond the last completed tick")
    if policy is not None:
        _need(len(policy.policies) == config.n_colonies, "one policy per colony is required")
    return ReplayDocument(header, config, world_map, events, frames, result, prev, policy)


def _validate_frame(frame: dict, config: ConfigE1, world_map: MapE1, patch_nodes: int, last_tick: int) -> None:
    _need(set(frame) == FRAME_KEYS and _is_int(frame["tick"]) and frame["tick"] > last_tick, "bad or out-of-order frame")
    _need(frame["reason"] in ("start", "interval", "key_event", "final") and type(frame["state_hash"]) is str
          and len(frame["state_hash"]) == 64, "bad frame header fields")
    _int_list(frame["patches"], patch_nodes, "frame.patches")
    _need(all(type(p) is list and len(p) == 2 and all(_is_int(x) for x in p) for p in frame["piles"]), "bad piles")
    _need(type(frame["colonies"]) is list and len(frame["colonies"]) == config.n_colonies, "one colony row per colony")
    for index, row in enumerate(frame["colonies"]):
        _need(type(row) is dict and set(row) == COLONY_KEYS and row["id"] == index and row["status"] in STATUS_NAMES
              and all(_is_int(v) for k, v in row.items() if k != "status"), "bad colony row")
    seen_ids = set()
    node_count = len(world_map.nodes)
    for row in frame["cohorts"]:
        _int_list(row, len(COHORT_FIELDS), "cohort row")
        cid, colony, caste, count, hp, load, kind, phase, where, a, b, depart, arrive, target_node, target_colony = row
        _need(cid not in seen_ids and 0 <= colony < config.n_colonies and caste in (0, 1) and count >= 1 and hp >= 1 and load >= 0
              and 0 <= kind < len(KIND_CODES) and 0 <= phase < len(PHASE_CODES) and where in (0, 1)
              and 0 <= a < node_count and (b == -1 if where == 0 else 0 <= b < node_count) and (depart < arrive if where else True)
              and -1 <= target_node < node_count and -1 <= target_colony < config.n_colonies, "bad cohort row")
        seen_ids.add(cid)
    _need(type(frame["memory"]) is list and len({m.get("colony") for m in frame["memory"] if type(m) is dict}) == len(frame["memory"]),
          "bad memory list")
    for memory in frame["memory"]:
        _need(set(memory) == MEMORY_KEYS and 0 <= memory["colony"] < config.n_colonies, "bad memory record")
        for row in memory["visited"]:
            _int_list(row, len(VISITED_FIELDS), "memory.visited")
            _need(0 <= row[0] < node_count and row[4] in (0, 1, 2), "bad visited node")
        _need(all(_is_int(n) and 0 <= n < node_count for n in memory["discovered"]), "bad discovered nodes")
        visited_ids = {row[0] for row in memory["visited"]}
        _need(len(visited_ids) == len(memory["visited"]) and not visited_ids & set(memory["discovered"]), "visited and discovered overlap")
        for row in memory["foreign"]:
            _int_list(row, len(FOREIGN_FIELDS), "memory.foreign")
            _need(0 <= row[0] < config.n_colonies and row[0] != memory["colony"] and -1 <= row[5] < node_count, "bad foreign record")
        for row in memory["sightings"]:
            _int_list(row, len(SIGHTING_FIELDS), "memory.sightings")
            _need(row[0] in visited_ids and 0 <= row[1] < config.n_colonies and row[5] in (0, 1), "bad sighting")


def verify_replay(data: bytes, sidecar: bytes | None = None) -> ReplayDocument:
    """Read the file, then re-run the engine from its header and demand the identical bytes (the normative check)."""
    document = read_replay(data)
    header = document.header
    if document.policy is not None:
        regenerated, decisions = export_policy_replay(document.config, int(header["seed"]), document.policy, header["ticks_requested"],
                                                      header["frame_every"])
        _need(regenerated == bytes(data), "re-executing the simulation from the header does not reproduce this file")
        if sidecar is not None:
            _need(decisions == sidecar, "the decision records are not those this run produces")
        return document
    regenerated = export_replay(document.config, int(header["seed"]), header["fixture"],
                                {int(k): v for k, v in header["roles"].items()}, header["ticks_requested"], header["frame_every"])
    _need(regenerated == bytes(data), "re-executing the simulation from the header does not reproduce this file")
    return document
