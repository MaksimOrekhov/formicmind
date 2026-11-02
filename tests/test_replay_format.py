"""Replay v1 building blocks: hash chain, canonical lines, event schema pins, patches, RNG jump-ahead."""

import dataclasses

import pytest

from formicmind import default_config, events_hash, reset
from formicmind import events as events_module
from formicmind.canonical import canonical_dumps
from formicmind.events import Phase
from formicmind.match_smoke import MATCH_SMOKE_SEED, SCENARIOS
from formicmind.replay import frames
from formicmind.replay.format import (
    EVENT_SCHEMA,
    ZERO_CHAIN,
    EventsHasher,
    ReplayFormatError,
    SeatSpec,
    check_event,
    contract_versions,
    open_record,
    pcg_advance,
    seal_record,
    split_lines,
)
from formicmind.rng import new_stream
from formicmind.tick import ScriptedController, step_tick


# --- hash chain and lines -----------------------------------------------------------------------------------------

def test_seal_and_open_round_trip_and_chain_links_records():
    line1, c1 = seal_record({"type": "x", "seq": 0, "v": [1, 2]}, ZERO_CHAIN)
    line2, c2 = seal_record({"type": "x", "seq": 1, "v": []}, c1)
    assert line1.endswith(b"\n") and line1.count(b"\n") == 1 and c1 != c2
    rec1, got1 = open_record(line1[:-1], ZERO_CHAIN, 0)
    rec2, got2 = open_record(line2[:-1], c1, 1)
    assert (rec1, rec2, got1, got2) == ({"type": "x", "seq": 0, "v": [1, 2]}, {"type": "x", "seq": 1, "v": []}, c1, c2)
    with pytest.raises(ReplayFormatError, match="hash chain"):
        open_record(line2[:-1], ZERO_CHAIN, 1)  # a line is only valid behind its real predecessor


def test_changing_any_single_byte_of_a_line_is_detected():
    line, chain = seal_record({"type": "event", "seq": 5, "i": 3, "e": {"kind": "upkeep", "x": 12}}, "ab" * 32)
    body = line[:-1]
    for pos in range(len(body)):
        mutated = body[:pos] + bytes([body[pos] ^ 0x01]) + body[pos + 1:]
        with pytest.raises(ReplayFormatError):
            open_record(mutated, "ab" * 32, 0)


def test_non_canonical_spelling_of_a_valid_record_is_rejected():
    line, chain = seal_record({"type": "x", "seq": 0}, ZERO_CHAIN)
    spaced = line[:-1].replace(b",", b", ")
    with pytest.raises(ReplayFormatError, match="canonical"):
        open_record(spaced, ZERO_CHAIN, 0)


def test_a_record_cannot_smuggle_a_second_chain_field():
    with pytest.raises(ReplayFormatError):
        seal_record({"type": "header", "seq": 0, "meta": {"chain": "0" * 64}}, ZERO_CHAIN)
    with pytest.raises(ReplayFormatError):
        seal_record({"type": "x", "chain": "0" * 64}, ZERO_CHAIN)


@pytest.mark.parametrize("data, text", [
    (b"", "empty"), (b'{"a":1}', "truncated"), (b'{"a":1}\r\n', "carriage"), (b'{"a":1}\n\n{"b":2}\n', "blank"),
])
def test_split_lines_rejects_broken_files(data, text):
    with pytest.raises(ReplayFormatError, match=text):
        split_lines(data)


def test_events_hasher_equals_events_hash_for_real_events():
    state, run = reset(default_config(), MATCH_SMOKE_SEED)
    sc = SCENARIOS["raid_win"]
    controllers = (ScriptedController(sc.script_a), ScriptedController(sc.script_b))
    events, hasher = [], EventsHasher()
    assert hasher.hexdigest() == events_hash([])
    for _ in range(40):
        for e in step_tick(state, run, controllers).events:
            events.append(e)
            hasher.add(e.to_canonical())
    assert len(events) > 100 and hasher.count == len(events) and hasher.hexdigest() == events_hash(events)


# --- RNG jump-ahead ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("steps", [0, 1, 2, 3, 17, 255, 1000, 4096, 65537])
def test_pcg_advance_equals_stepping(steps):
    rng = new_stream(20261102, "policy:x:0")
    expect = rng.clone()
    for _ in range(steps):
        expect.next_u32()
    assert pcg_advance(rng.state, rng.inc, steps) == expect.state


# --- event schema pinned against events.py ------------------------------------------------------------------------

EVENT_CLASSES = [getattr(events_module, name) for name in dir(events_module)
                 if dataclasses.is_dataclass(getattr(events_module, name)) and hasattr(getattr(events_module, name), "kind")]


def test_event_schema_covers_exactly_the_engine_event_classes():
    assert {c.kind for c in EVENT_CLASSES} == set(EVENT_SCHEMA) and len(EVENT_CLASSES) == 15


@pytest.mark.parametrize("cls", EVENT_CLASSES, ids=lambda c: c.kind)
def test_event_schema_matches_the_dataclass_fields_and_rank(cls):
    schema = EVENT_SCHEMA[cls.kind]
    fields = {f.name: f.type for f in dataclasses.fields(cls)}
    declared = {name for name, _ in schema.fields}
    assert declared == set(fields)
    assert schema.rank == cls.kind_rank
    kinds = {"int": "i", "bool": "b", "str": "s", "tuple[int, ...]": "I", "tuple[int, int]": "J"}
    assert {name: kinds[fields[name]] for name in declared} == dict(schema.fields)


def test_event_schema_accepts_every_recorded_event_of_the_smoke_set(replays):
    from formicmind.replay import ReplayReader

    seen = set()
    for name in ("raid_win", "mutual_raid_draw", "collapse", "timeout_draw"):
        for ev in ReplayReader.from_bytes(replays[name].data).iter_events():
            check_event(ev, ev["tick"], "test")
            seen.add(ev["kind"])
    assert seen == set(EVENT_SCHEMA)  # the public smoke set exercises all 15 kinds


def test_check_event_rejects_unknown_kinds_extra_fields_wrong_types_and_impossible_phases():
    good = {"kind": "regenerate", "tick": 3, "phase": int(Phase.REGENERATION), "node": 1, "before": 5, "regen": 2,
            "added": 2, "after": 7, "capacity": 60}
    check_event(good, 3, "t")
    for bad_event, tick in [
        ({**good, "kind": "teleport"}, 3), ({**good, "extra": 1}, 3), ({**good, "after": 7.0}, 3),
        ({**good, "after": True}, 3), ({**good, "phase": int(Phase.HARVEST)}, 3), (good, 4),
        ({k: v for k, v in good.items() if k != "node"}, 3), ([good], 3),
    ]:
        with pytest.raises(ReplayFormatError):
            check_event(bad_event, tick, "t")


# --- seats and versions ----------------------------------------------------------------------------------------------

def test_seat_spec_is_data_only_and_validated():
    SeatSpec("scripted", "scripted", "", "script-sha256:ab")
    SeatSpec("rnd-a", "policy", "policy:rnd-a:0")
    for args in [("bad id",), ("x", "weird"), ("x", "policy", "world"), ("x", "policy", "no good"),
                 ("x", "other", "", "é"), ("",)]:
        with pytest.raises(ValueError):
            SeatSpec(*args)


def test_contract_versions_pin_every_dependency_of_a_replay():
    assert contract_versions() == {
        "rules_version": "l0-proto-FOR1-v0.2.2", "balance_version": "candidate-FOR1-v0.2.1-unvalidated",
        "observation_version": "FOR-2-v0.2+errata-v0.2.1", "action_version": "ActionV2",
        "decoder_version": "FOR-2-decoder-v0.2.1", "serialization_version": "formicmind-canonical-json-v1",
        "rng_version": "pcg32-xsh-rr-64-32-v1", "stream_derivation_version": "sha256-lenprefixed-v1",
        "hash_domain_config": "formicmind/config/v1", "hash_domain_state": "formicmind/state/v1",
        "hash_domain_checkpoint": "formicmind/checkpoint/v3", "hash_domain_events": "formicmind/events/v1",
        "hash_domain_rng_stream": "formicmind/rng-stream/v1",
        "hash_domain_observation": "formicmind/observation-f32le/v1",
        "hash_domain_view": "formicmind/replay-view/v1", "hash_domain_record": "formicmind/replay-record/v1"}


# --- frames: patches reproduce the engine's frames ------------------------------------------------------------------

def test_patches_reproduce_every_engine_frame_of_the_raid():
    state, run = reset(default_config(), MATCH_SMOKE_SEED)
    sc = SCENARIOS["raid_win"]
    controllers = (ScriptedController(sc.script_a), ScriptedController(sc.script_b))
    body = frames.build_body(state)
    patches = 0
    while run.outcome is None:
        step_tick(state, run, controllers)
        new = frames.build_body(state)
        patch = frames.diff_body(body, new)
        assert frames.apply_patch(body, patch) == new
        patches += bool(patch.get("nf") or any(patch["c"]))
        body = new
    assert patches > 150


def test_apply_patch_never_mutates_its_input_and_rejects_patches_that_do_not_fit():
    state, _ = reset(default_config(), 1)
    body = frames.build_body(state)
    snapshot = canonical_dumps(body)
    new = {**body, "node_food": [1, 2, 3, 4, 5, 6, 7]}
    out = frames.apply_patch(body, frames.diff_body(body, new))
    assert out["node_food"] == [1, 2, 3, 4, 5, 6, 7] and canonical_dumps(body) == snapshot
    bad_patches = [
        {"c": [{}, {}], "zz": 1}, {"c": [{}]}, {"nf": [[7, 1]], "c": [{}, {}]}, {"nf": [[2, 1], [1, 1]], "c": [{}, {}]},
        {"c": [{"food": -1}, {}]}, {"c": [{"stationed": {"set": [], "del": [[5, 0]]}}, {}]},  # deleting a missing row
        {"c": [{"stationed": {"set": [[0, 0, 1, 1]], "del": [[0, 0]]}}, {}]},  # set and deleted
        {"c": [{"stationed": {"set": [[0, 0, 1]], "del": []}}, {}]}, {"c": [{"queen_hp": "x"}, {}]},
    ]
    for patch in bad_patches:
        with pytest.raises(ReplayFormatError):
            frames.apply_patch(body, patch)


def test_initial_view_frame_matches_the_documented_start_position_literally():
    state, run = reset(default_config(), 1)
    frame = frames.view_from_state(state, run)
    assert (frame.tick, frame.boundary, frame.terminal) == (0, "start", None)
    assert frame.node_food == (0, 60, 40, 80, 60, 40, 0)
    a, b = frame.colonies
    # (node, caste, count, hp_sum): 12 workers x 20 hp and 4 soldiers x 30 hp at the home node, A=0 and B=6
    assert (a.food, a.queen_hp, a.stationed, a.transit, a.brood) == (60, 240, ((0, 0, 12, 240), (0, 1, 4, 120)), (), ())
    assert (b.food, b.queen_hp, b.stationed, b.transit, b.brood) == (60, 240, ((6, 0, 12, 240), (6, 1, 4, 120)), (), ())
    assert frames.ViewFrame.from_canonical(frame.to_canonical()) == frame
