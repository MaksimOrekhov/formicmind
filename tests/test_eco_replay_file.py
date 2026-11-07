"""FOR-47: the versioned replay file of a real E1 run: writing, strict reading, byte-exact re-execution, known/true separation."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from formicmind.canonical import canonical_dumps, canonical_loads, domain_hash
from formicmind.eco.cli import main
from formicmind.eco.config import default_config
from formicmind.eco.engine import RouteTable, step_world_tick
from formicmind.eco.fixtures import fixture_controller
from formicmind.eco.initial import build_world
from formicmind.eco.replay_file import (
    RECORD_DOMAIN, REPLAY_SCHEMA, ZERO_CHAIN, ReplayFormatError, _frame_record, export_replay, read_replay, verify_replay,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "viewer" / "public" / "samples" / "eco_seed50004_raider3.jsonl"
CONFIG = default_config(8, "moderate")


def _short(ticks=400, frame_every=20, roles=None):
    return export_replay(CONFIG, 50004, "peaceful_fair", {3: "raider"} if roles is None else roles, ticks, frame_every)


def _rechain(data: bytes, edit) -> bytes:
    """Rebuild a file after ``edit(record, index)`` so that only content checks (not the integrity chain) can notice the change."""
    previous, out = ZERO_CHAIN, []
    for index, line in enumerate(data.split(b"\n")[:-1]):
        record = edit(canonical_loads(line), index)
        if record is None:
            continue
        record = {**record, "seq": len(out), "chain": ZERO_CHAIN}
        chain = domain_hash(RECORD_DOMAIN, previous.encode("ascii") + b"\n" + canonical_dumps(record))
        out.append(canonical_dumps({**record, "chain": chain}))
        previous = chain
    return b"\n".join(out) + b"\n"


def test_a_short_run_round_trips_and_is_byte_for_byte_reproducible():
    first, second = _short(), _short()
    assert first == second                                        # no clock, no environment: same inputs, same bytes
    document = read_replay(first)
    assert document.header["schema"] == REPLAY_SCHEMA and document.header["seed"] == "50004"
    assert document.header["roles"] == {"3": "raider"} and document.header["colonies"] == 8
    ticks = [frame["tick"] for frame in document.frames]
    assert ticks[0] == 0 and ticks[-1] == 400 and ticks == sorted(set(ticks))
    assert [f["reason"] for f in document.frames[:1] + document.frames[-1:]] == ["start", "final"]
    assert all(t % 20 == 0 for t in ticks)                        # an interval frame every 20 ticks, nothing invented in between
    assert document.result["ticks_completed"] == 400 and document.result["event_count"] == len(document.events)
    assert document.chain_head == first.split(b"\n")[-2].decode().split('"chain":"')[1].split('"')[0]
    assert read_replay(_short(frame_every=50)).frames[1]["tick"] == 50
    verify_replay(first)


def test_frames_carry_the_true_state_of_the_engine():
    world = build_world(CONFIG, 50004)
    routes = RouteTable(world.world_map)
    controller = fixture_controller("peaceful_fair", {3: "raider"})
    for _ in range(300):
        step_world_tick(world.state, world.run, CONFIG, world.world_map, controller, routes=routes, validate_input=world.state.tick == 0)
    document = read_replay(_short(300, 300))
    frame = document.frames[-1]
    assert frame["tick"] == 300 and frame["state_hash"] == _frame_record(world.state, "final")["state_hash"]
    assert frame == {**_frame_record(world.state, "final"), "seq": frame["seq"], "chain": frame["chain"]}
    for row, colony in zip(frame["colonies"], world.state.colonies):
        mine = [c for c in world.state.cohorts if c.colony_id == colony.colony_id]
        assert row["workers"] == sum(c.count for c in mine if c.caste == "worker") and row["food"] == colony.food
        assert row["status"] == colony.status and row["queen_hp"] == colony.queen_hp
    on_edge = [row for row in frame["cohorts"] if row[8] == 1]
    assert on_edge and all(row[11] < row[12] for row in on_edge)  # real depart/arrive ticks for every moving cohort


def test_colony_memory_in_a_frame_is_the_colonys_own_knowledge_and_nothing_else():
    world = build_world(CONFIG, 50004)
    routes = RouteTable(world.world_map)
    controller = fixture_controller("peaceful_fair", {3: "raider"})
    for _ in range(900):
        step_world_tick(world.state, world.run, CONFIG, world.world_map, controller, routes=routes, validate_input=world.state.tick == 0)
    frame = _frame_record(world.state, "interval")
    n = len(world.world_map.nodes)
    for colony, record in zip(world.state.colonies, frame["memory"]):
        memory = colony.memory
        visited = {k.node_id for k in memory.nodes if k.visited}
        heard = {k.node_id for k in memory.nodes if not k.visited}
        assert {row[0] for row in record["visited"]} == visited and set(record["discovered"]) == heard
        assert len(visited) + len(heard) < n                        # the rest of the map is simply absent
        # the viewer derives known edges as "incident to a visited node"; that is exactly the engine's rule
        derived = {(e.a, e.b) for e in world.world_map.edges if e.a in visited or e.b in visited}
        engine_edges = {tuple(sorted((memory.nodes[e.slot_a].node_id, memory.nodes[e.slot_b].node_id))) for e in memory.edges}
        assert derived == engine_edges
        assert {row[0] for row in record["foreign"]} == {r.colony_id for r in memory.foreign}
        assert all(row[0] in visited for row in record["sightings"])
        for row in record["visited"]:                                # stocks are the last-seen copies, not the true stock
            known = next(k for k in memory.nodes if k.node_id == row[0])
            assert row[2] == (known.last_stock if known.last_stock_tick >= 0 else -1) and row[1] == known.seen_tick
    # a colony's record does not move when only facts it never observed change (another colony's stores, queen, army)
    twin = deepcopy(world.state)
    for other in twin.colonies[:3]:
        other.food, other.queen_hp = 1, 3
    twin.patches[0].stock = 0
    assert _frame_record(twin, "interval")["memory"][3] == frame["memory"][3]
    assert _frame_record(twin, "interval")["memory"][7] == frame["memory"][7]


def test_the_reader_is_strict_and_fails_closed_with_a_clear_message():
    data = _short(200)
    lines = data.split(b"\n")
    with pytest.raises(ReplayFormatError, match="ending with a newline"):
        read_replay(data[:-1])
    with pytest.raises(ReplayFormatError, match="line 1"):
        read_replay(b"not json\n" * 3)
    with pytest.raises(ReplayFormatError, match="canonical"):
        read_replay(b" " + data)                                     # leading space: valid JSON, not canonical
    tampered = lines[:]
    tampered[4] = tampered[4].replace(b'"tick":', b'"tick": ', 1)
    with pytest.raises(ReplayFormatError):
        read_replay(b"\n".join(tampered))
    swapped = lines[:]
    swapped[5], swapped[6] = swapped[6], swapped[5]
    with pytest.raises(ReplayFormatError, match="seq"):
        read_replay(b"\n".join(swapped))
    with pytest.raises(ReplayFormatError, match="result"):
        read_replay(b"\n".join(lines[:-2]) + b"\n")                  # truncated before the result
    edited = lines[:]
    record = canonical_loads(edited[7])
    record["seq"] += 0
    edited[7] = canonical_dumps({**record, "chain": "0" * 64})
    with pytest.raises(ReplayFormatError, match="chain is broken"):
        read_replay(b"\n".join(edited))


def test_a_valid_chain_does_not_excuse_unknown_versions_or_hash_mismatches():
    data = _short(200)
    with pytest.raises(ReplayFormatError, match="unsupported replay schema"):
        read_replay(_rechain(data, lambda r, i: {**r, "schema": "formicmind.eco.replay/2"} if r["type"] == "header" else r))
    with pytest.raises(ReplayFormatError, match="config_hash"):
        read_replay(_rechain(data, lambda r, i: {**r, "config": {**r["config"], "horizon": r["config"]["horizon"] - 1}} if r["type"] == "header" else r))

    def move_node(record, index):
        if record["type"] != "header":
            return record
        nodes = [dict(n) for n in record["map"]["nodes"]]
        nodes[2]["x"] += 1
        return {**record, "map": {**record["map"], "nodes": nodes}}
    with pytest.raises(ReplayFormatError, match="map"):
        read_replay(_rechain(data, move_node))
    first_death = {"done": False}

    def bump_event(record, index):
        if record["type"] == "event" and not first_death["done"]:
            first_death["done"] = True
            return {**record, "a": record["a"] + 1}
        return record
    with pytest.raises(ReplayFormatError, match="events_hash"):
        read_replay(_rechain(data, bump_event))
    with pytest.raises(ReplayFormatError, match="cohort"):
        read_replay(_rechain(data, lambda r, i: {**r, "cohorts": [[1, 99, 0, 1, 20, 0, 0, 0, 0, 0, -1, 0, 0, -1, -1]]} if r["type"] == "frame" and r["tick"] == 20 else r))


def test_verify_reruns_the_simulation_and_rejects_a_file_that_is_well_formed_but_not_what_the_engine_produced():
    data = _short(200)
    verify_replay(data)
    forged = _rechain(data, lambda r, i: {**r, "colonies": [{**r["colonies"][0], "food": r["colonies"][0]["food"] + 1}, *r["colonies"][1:]]}
                      if r["type"] == "frame" and r["tick"] == 100 else r)
    read_replay(forged)                                                # structurally fine, chain fine ...
    with pytest.raises(ReplayFormatError, match="re-executing"):
        verify_replay(forged)                                          # ... but not what the engine produces


def test_key_events_get_their_own_frame_so_the_viewer_can_show_the_exact_moment():
    document = read_replay(SAMPLE.read_bytes())
    key = [f["tick"] for f in document.frames if f["reason"] == "key_event"]
    queen = [e.tick + 1 for e in document.events if e.kind == "queen_dead"]
    extinct = [e.tick + 1 for e in document.events if e.kind == "extinct"]
    assert key == sorted(queen + extinct) == [2622, 4401]


def test_the_published_seed_50004_replay_is_exactly_what_the_engine_exports():
    data = SAMPLE.read_bytes()
    document = verify_replay(data)                                     # strict read + full re-execution, byte for byte
    published = json.loads((ROOT / "docs" / "ecosystem_examples" / "for46_raid_seed50004.json").read_text(encoding="utf-8"))
    assert document.result["events_hash"] == published["interaction"]["events_hash"]
    assert document.result["final_state_hash"] == published["hashes"]["state"]
    assert document.result["final_checkpoint_hash"] == published["hashes"]["checkpoint"]
    colonies = [c["status"] for c in document.frames[-1]["colonies"]]
    assert colonies.count("alive") == 7 and colonies[0] == "extinct"
    assert (document.header["seed"], document.header["roles"], document.header["colonies"]) == ("50004", {"3": "raider"}, 8)
    assert SAMPLE.stat().st_size < 4_000_000                           # a file a reviewer can download and open


def test_cli_exports_verifies_and_reports_bad_files(tmp_path, capsys):
    out = tmp_path / "run.jsonl"
    assert main(["replay-export", "--seed", "50004", "--role", "3=raider", "--ticks", "120", "--frame-every", "40", "--verify", "--out", str(out)]) == 0
    assert out.read_bytes() == export_replay(CONFIG, 50004, "peaceful_fair", {3: "raider"}, 120, 40)
    assert main(["replay-verify", str(out)]) == 0
    assert main(["replay-verify", "--read-only", str(out)]) == 0
    assert "byte-exact re-execution" in capsys.readouterr().out
    bad = tmp_path / "bad.jsonl"
    bad.write_bytes(out.read_bytes()[:-5] + b"xxxx\n")
    assert main(["replay-verify", str(bad)]) == 3
    assert main(["replay-export", "--seed", "1", "--role", "9=raider", "--ticks", "5", "--out", str(tmp_path / "x.jsonl")]) == 3
