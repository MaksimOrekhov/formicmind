"""FOR-15 Replay v1: record -> read -> seek -> verify over the public dev smoke matches (functional checks only).

Seeds are the arbitrary engineering constant of the smoke set, never a FOR-4 sealed seed.
"""

import dataclasses
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from formicmind import default_config, reset
from formicmind.canonical import canonical_loads
from formicmind.match import play
from formicmind.match_smoke import SCENARIOS
from formicmind.observation import build_observation, flatten_v2
from formicmind.replay import (
    BOUNDARY_START,
    BOUNDARY_TERMINAL,
    ReplayReader,
    ReplayVerifyError,
    ReplayWriteError,
    SeatSpec,
    SeekError,
    ViewFrame,
    iter_events,
    read_header,
    read_result,
    record_episode,
    record_play,
    seek,
    verify,
    view_from_state,
)
from formicmind.replay.format import ZERO_CHAIN
from formicmind.snapshot import begin_policy_phase
from formicmind.tick import ScriptedController, step_tick

from replay_helpers import PROBES, SEED, parse, scripted
from test_match import PINNED

SRC = Path(__file__).resolve().parents[1] / "src"
ALL = (*PROBES, "cutoff", "rng_policy")


def engine_frames(controllers, config=None):
    """Independent expectation: the frames of a real engine run, captured by the test itself."""
    state, run = reset(config or default_config(), SEED)
    out = {(0, BOUNDARY_START): view_from_state(state, run)}
    while run.outcome is None:
        step_tick(state, run, controllers)
        frame = view_from_state(state, run)
        out[(frame.tick, frame.boundary)] = frame
    return out


@pytest.fixture(scope="module")
def expected_frames():
    return {name: engine_frames(scripted(name)) for name in PROBES}


# --- exact hashes of the existing smoke pins -----------------------------------------------------------------------------

@pytest.mark.parametrize("name", PROBES)
def test_replay_carries_exactly_the_pinned_hashes_of_the_smoke_matches(replays, name):
    pin = PINNED[name]
    result = read_result(replays[name].path)
    assert (result.events_hash, result.final_state_hash, result.final_checkpoint_hash) == (
        pin["events_hash"], pin["state_hash"], pin["checkpoint_hash"])
    assert result.finished and result.outcome.winner.value == pin["winner"]
    assert (result.outcome.reason.value, result.outcome.terminal_tick, result.state_tick) == (
        pin["reason"], pin["terminal_tick"], pin["final_tick"])


@pytest.mark.parametrize("name", ALL)
def test_verify_reproduces_every_recording_through_step_tick(replays, name):
    report = verify(replays[name].path)
    rec = replays[name]
    assert report.events_hash == rec.result.events_hash and report.final_checkpoint_hash == rec.result.checkpoint_hash
    assert report.final_state_hash == rec.result.state_hash and report.events == len(rec.result.events)
    assert report.ticks == rec.result.ticks_executed and report.state_tick == rec.result.final_tick
    assert report.finished is (not rec.result.truncated)


def test_recording_does_not_change_the_match(replays):
    """Recording is transparent: the hashes equal an unrecorded ``run_episode`` of the same match."""
    from formicmind.match import run_episode

    for name in PROBES:
        plain = run_episode(default_config(), SEED, *scripted(name))
        rec = replays[name].result
        assert (plain.events_hash, plain.state_hash, plain.checkpoint_hash) == (
            rec.events_hash, rec.state_hash, rec.checkpoint_hash)


# --- the file itself -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL)
def test_file_is_plain_ascii_jsonl_with_unix_line_ends_and_one_record_per_line(replays, name):
    data = replays[name].data
    assert data.endswith(b"\n") and b"\r" not in data and data.isascii() and b"\n\n" not in data
    assert data.count(b"\n") == replays[name].summary.records == len(parse(data))
    assert parse(data)[0]["type"] == "header" and parse(data)[-1]["type"] == "result"


def test_header_is_self_describing_and_hides_nothing_dynamic(replays):
    head = read_header(replays["raid_win"].path)
    raw = head.raw
    assert raw["format"] == "formicmind.replay" and raw["replay_schema"] == "v1"
    assert raw["root_seed"] == str(SEED) and head.root_seed == SEED and type(raw["root_seed"]) is str
    assert raw["start"] == {"kind": "reset"} and raw["cursor"] == {"tick": 0, "event_index": 0}
    assert raw["contract"]["rules_version"] == "l0-proto-FOR1-v0.2.2"
    assert raw["config"]["t_max"] == 3000 and raw["config"]["policy_interval_k"] == 30
    assert head.config == default_config()
    assert [s["kind"] for s in raw["seats"]] == ["scripted", "scripted"] and raw["streams"] == []
    assert raw["source"] == "raid_win" and raw["meta"] == {}
    text = replays["raid_win"].data.decode("ascii")
    for dynamic in (str(Path.home()), "E:\\\\", "\\\\Users", "T00:", "+00:00", "wall"):
        assert dynamic not in text


@pytest.mark.parametrize("name", PROBES)
def test_the_same_match_records_to_identical_bytes_and_hashes_are_pinned(replays, name, tmp_path):
    second = tmp_path / "again.jsonl"
    record_episode(second, default_config(), SEED, *scripted(name), source=name)
    assert second.read_bytes() == replays[name].data
    assert hashlib.sha256(replays[name].data).hexdigest() == FILE_PINS[name]


# Byte-exact pins of the four recordings (source=<scenario>, no meta): a diff means the replay format or the
# match itself changed on purpose.
FILE_PINS = {
    "raid_win": "f71c5b31ac5e6cffb789d5fed35cbe9531f29729ea0689a026f365d4bb02c6c9",
    "mutual_raid_draw": "1c41648a69a27f28b57ff1102dc17b53c3da551bed9de6e83ef0f6b224b20e33",
    "collapse": "1b62dd58016f1ee2eb7075c9431b96f8d771ed0730bf031ce08420fee137822a",
    "timeout_draw": "c93b4fdfcebfa61e0703ea721d7150d3e48e24d7087ed6f3f0c4ea2759d7123b",
}


def test_events_in_the_file_equal_the_engine_journal_in_order(replays):
    for name in ALL:
        reader = ReplayReader.from_bytes(replays[name].data)
        assert list(reader.iter_events()) == [e.to_canonical() for e in replays[name].result.events]
    reader = ReplayReader.from_bytes(replays["raid_win"].data)
    assert [e["kind"] for e in reader.iter_events(tick=175)][-1] == "terminal"
    assert {e["kind"] for e in iter_events(replays["raid_win"].path, kind="queen_damage")} == {"queen_damage"}
    indices = [r["i"] for r in parse(replays["collapse"].data) if r["type"] == "event"]
    assert indices == list(range(len(replays["collapse"].result.events)))
    with pytest.raises(SeekError):
        list(reader.iter_events(kind="teleport"))


def test_iter_events_returns_independent_copies(replays):
    reader = ReplayReader.from_bytes(replays["collapse"].data)
    first = next(reader.iter_events())
    first["tick"] = 999
    assert next(reader.iter_events())["tick"] != 999


def test_decisions_are_recorded_exactly_at_policy_ticks_with_the_validated_actions(replays):
    reader = ReplayReader.from_bytes(replays["raid_win"].data)
    assert sorted(reader.decisions) == [0, 30, 60, 90, 120, 150]
    sc = SCENARIOS["raid_win"]
    for tick, decision in reader.decisions.items():
        for side, script in enumerate((sc.script_a, sc.script_b)):
            wanted = script[max(t for t in script if t <= tick)]
            assert decision.seats[side].action == wanted and decision.seats[side].stream_id == ""
    assert reader.decisions[150].seats[0].action.army_choice == 6  # the raid order, after the shared validator
    assert len(ReplayReader.from_bytes(replays["timeout_draw"].data).decisions) == 100


def test_recorded_observation_hash_is_the_hash_of_the_float32_bytes_a_policy_received(replays):
    state, _ = reset(default_config(), SEED)
    snapshot = begin_policy_phase(state)
    reader = ReplayReader.from_bytes(replays["raid_win"].data)
    for side in (0, 1):
        raw = flatten_v2(build_observation(snapshot, side))
        assert len(raw) == 223 * 4
        expect = hashlib.sha256(b"formicmind/observation-f32le/v1\x00" + raw).hexdigest()
        assert reader.decisions[0].seats[side].obs_hash == expect
    assert len(reader.decisions[0].seats[0].mask) == 30


# --- seek: whole ViewFrame against what the engine really had ---------------------------------------------------------

@pytest.mark.parametrize("name", PROBES)
def test_seek_returns_the_engines_frame_at_every_tick_and_the_terminal_frame(replays, expected_frames, name):
    reader = ReplayReader.from_bytes(replays[name].data)
    expected = expected_frames[name]
    for tick in range(reader.first_tick, reader.last_start_tick + 1):
        assert reader.seek(tick) == expected[(tick, BOUNDARY_START)], f"{name}: tick {tick}"
    terminal = reader.read_result().state_tick
    assert reader.seek(terminal, BOUNDARY_TERMINAL) == expected[(terminal, BOUNDARY_TERMINAL)]


def test_selected_ticks_0_1_29_30_31_raid_middle_and_terminal_compared_with_stored_engine_frames(replays, expected_frames):
    reader = ReplayReader.from_bytes(replays["raid_win"].data)
    engine = expected_frames["raid_win"]
    for tick in (0, 1, 29, 30, 31, 155, 160, 174, 175):
        frame = reader.seek(tick)
        assert frame == engine[(tick, BOUNDARY_START)] and (frame.tick, frame.boundary, frame.terminal) == (tick, "start", None)
    # raid middle: at 155 A's soldiers march on the edge C->N4 (arrival 156); at 170 they stand in B's home, B has
    # no soldiers left and its queen is being chewed through
    march, siege = reader.seek(155), reader.seek(170)
    assert march.colonies[0].transit == ((3, 4, 156, 1, 11, 330),) and march.colonies[1].queen_hp == 240
    assert (6, 1, 11, 321) in siege.colonies[0].stationed and all(row[1] == 0 for row in siege.colonies[1].stationed)
    assert siege.colonies[1].queen_hp == 116
    terminal = reader.seek(175, BOUNDARY_TERMINAL)
    assert terminal == engine[(175, BOUNDARY_TERMINAL)] and terminal.colonies[1].queen_hp == 0


def test_view_frame_groups_agree_with_a_naive_count_of_the_engine_units(replays):
    """The builder is checked against raw unit lists, not against itself."""
    state, run = reset(default_config(), SEED)
    controllers = scripted("raid_win")
    for _ in range(158):
        step_tick(state, run, controllers)
    frame = view_from_state(state, run)
    for colony, view in zip(state.colonies, frame.colonies):
        assert view.food == colony.food and view.queen_hp == colony.queen_hp
        assert sum(r[2] for r in view.stationed) + sum(r[4] for r in view.transit) == len(colony.units)
        assert sum(r[3] for r in view.stationed) + sum(r[5] for r in view.transit) == sum(u.hp for u in colony.units)
        for node, caste, count, hp in view.stationed:
            here = [u for u in colony.units if u.node == node and u.caste == caste]
            assert (count, hp) == (len(here), sum(u.hp for u in here))
        for src, dst, arrival, caste, count, hp in view.transit:
            moving = [u for u in colony.units if (u.from_node, u.to_node, u.arrival_tick, u.caste) == (src, dst, arrival, caste)]
            assert (count, hp) == (len(moving), sum(u.hp for u in moving)) and arrival > state.tick - 1
        assert sum(r[2] for r in view.brood) == len(colony.brood)
    assert frame.node_food == tuple(state.node_food) and any(c.transit for c in frame.colonies)


def test_seek_is_snapshot_plus_patches_and_never_asks_the_engine(replays, monkeypatch):
    """With every rule function booby-trapped, open/seek/iter_events/read_* still work; verify (the control) cannot."""
    from formicmind import combat, economy, movement, orders, routing, tick as tick_module

    def trap(*args, **kwargs):
        raise AssertionError("the reader executed game physics")

    for module, names in [
        (tick_module, ["step_tick", "resolve_combat", "harvest", "upkeep", "lay", "regenerate", "complete_hatching",
                       "arrive_units", "dispatch_available", "apply_joint_orders", "update_local_intel"]),
        (routing, ["shortest_path", "next_hop"]), (economy, ["harvest", "upkeep", "lay", "regenerate"]),
        (combat, ["resolve_combat"]), (movement, ["arrive_units", "dispatch_available"]),
        (orders, ["apply_joint_orders"]),
    ]:
        for name in names:
            monkeypatch.setattr(module, name, trap)
    reader = ReplayReader.from_bytes(replays["mutual_raid_draw"].data)
    reader.read_header(), reader.read_result()
    assert len(list(reader.iter_events())) == 676
    assert reader.seek(17).tick == 17 and reader.seek(39, "terminal").tick == 39
    with pytest.raises(ReplayVerifyError, match="AssertionError"):
        verify(reader)  # the control: verification does need the engine, so the traps are effective


def test_reader_modules_do_not_import_game_physics():
    import ast

    banned = {"tick", "routing", "economy", "movement", "combat", "orders", "match", "reset", "intel", "ledger"}
    for module in ("reader", "frames", "format"):
        tree = ast.parse((SRC / "formicmind" / "replay" / f"{module}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = set()
            if isinstance(node, ast.ImportFrom) and node.level:
                names = {(node.module or "").split(".")[0], *(a.name for a in node.names)} if node.level == 1 else {(node.module or "").split(".")[0]}
            elif isinstance(node, ast.ImportFrom):
                names = {(node.module or "").split(".")[-1]}
            elif isinstance(node, ast.Import):
                names = {a.name.split(".")[-1] for a in node.names}
            assert not (names & banned), f"replay/{module}.py imports {names & banned}"


def test_seek_rejects_unknown_ticks_and_boundaries(replays):
    reader = ReplayReader.from_bytes(replays["collapse"].data)
    for args in [(-1,), (47,), (1000,), (5, "middle"), (5, "terminal"), (46, "terminal"), (1.5,), ("3",)]:
        with pytest.raises(SeekError):
            reader.seek(*args)
    assert reader.seek(46).tick == 46 and reader.seek(47, "terminal").tick == 47


def test_seek_result_is_immutable_and_independent(replays):
    reader = ReplayReader.from_bytes(replays["mutual_raid_draw"].data)
    frame = reader.seek(10)
    with pytest.raises(dataclasses.FrozenInstanceError):
        frame.tick = 11
    assert isinstance(frame, ViewFrame) and reader.seek(10) == frame and reader.seek(10) is not frame
    assert seek(replays["mutual_raid_draw"].path, 10) == frame


# --- terminal cursor: where the match ends, per outcome ---------------------------------------------------------------

def test_queen_death_cuts_the_tick_after_five_phases_and_state_tick_stays(replays, expected_frames):
    reader = ReplayReader.from_bytes(replays["raid_win"].data)
    result = reader.read_result()
    out = result.outcome
    assert (out.terminal_tick, int(out.terminal_phase), out.phases_executed, out.tick_completed) == (175, 4, 5, False)
    assert result.state_tick == 175 and result.tick_completed is False
    start, end = reader.seek(175), reader.seek(175, BOUNDARY_TERMINAL)
    assert (start.tick, start.boundary, start.terminal) == (175, "start", None)
    assert (end.tick, end.boundary) == (175, "terminal")
    assert end.terminal.tick_completed is False and end.terminal.terminal_phase == 4 and end.terminal.terminal_tick == 175
    assert end.terminal.winner == "A" and end.terminal.reason == "queen_death"
    assert end.colonies[1].queen_hp == 0 and start.colonies[1].queen_hp > 0
    # no end-of-tick frame: nothing like tick 176 exists, and the last tick executed no upkeep/regeneration
    with pytest.raises(SeekError):
        reader.seek(176)
    last = [e for e in reader.iter_events(tick=175)]
    assert last[-1]["kind"] == "terminal" and last[-1]["phases_executed"] == 5
    assert {e["kind"] for e in last} <= {"order", "retarget", "depart", "arrive", "hatch", "combat", "wound", "death",
                                         "queen_damage", "terminal"}
    # the display state after the cut equals the engine's, with the food of the unexecuted Harvest/Upkeep untouched
    assert end == expected_frames["raid_win"][(175, BOUNDARY_TERMINAL)]
    assert end.colonies[0].food == start.colonies[0].food and end.node_food == start.node_food


def test_simultaneous_double_queen_kill_is_a_draw_cut_at_queen_terminal(replays):
    reader = ReplayReader.from_bytes(replays["mutual_raid_draw"].data)
    end = reader.seek(39, BOUNDARY_TERMINAL)
    assert (end.terminal.winner, end.terminal.tick_completed, end.terminal.terminal_phase) == ("draw", False, 4)
    assert [c.queen_hp for c in end.colonies] == [0, 0] and reader.read_result().outcome.sides == (0, 1)
    assert reader.last_start_tick == 39 and reader.read_result().state_tick == 39


def test_end_of_tick_terminals_complete_the_tick_and_advance_state_tick(replays):
    collapse = ReplayReader.from_bytes(replays["collapse"].data)
    r = collapse.read_result()
    assert (r.outcome.terminal_tick, int(r.outcome.terminal_phase), r.outcome.phases_executed) == (46, 10, 11)
    assert (r.state_tick, r.tick_completed, collapse.last_start_tick) == (47, True, 46)
    end = collapse.seek(47, BOUNDARY_TERMINAL)
    assert (end.tick, end.terminal.terminal_tick, end.terminal.tick_completed) == (47, 46, True)
    assert end.colonies[0].stationed == () and end.colonies[0].food == 0  # A starved out
    timeout = ReplayReader.from_bytes(replays["timeout_draw"].data)
    t = timeout.read_result()
    assert (t.outcome.terminal_tick, t.state_tick, t.tick_completed) == (2999, 3000, True)
    assert timeout.seek(3000, BOUNDARY_TERMINAL).terminal.reason == "timeout" and timeout.last_start_tick == 2999
    with pytest.raises(SeekError):
        timeout.seek(3000)  # the horizon is not the start of any tick


def test_a_queen_death_on_a_policy_tick_keeps_both_the_start_and_the_cut_frame(tmp_path):
    """K=13 turns the lethal tick 39 into a policy tick: decision, start snapshot, events, cut, terminal snapshot."""
    config = dataclasses.replace(default_config(), policy_interval_k=13)
    path = tmp_path / "k13.jsonl"
    result, _ = record_episode(path, config, SEED, *scripted("mutual_raid_draw"))
    assert result.outcome.terminal_tick == 39 and 39 % 13 == 0
    reader = ReplayReader.from_bytes(path.read_bytes())
    records = parse(path.read_bytes())
    assert 39 in reader.decisions and reader.read_result().state_tick == 39
    types = [r["type"] for r in records]
    snaps = [(r["frame"]["tick"], r["frame"]["boundary"]) for r in records if r["type"] == "snapshot"]
    assert (39, "start") in snaps and snaps[-1] == (39, "terminal")
    # order: snapshot(39,start) -> decision(39) -> events -> delta -> snapshot(39,terminal) -> result
    first_tick39 = next(i for i, r in enumerate(records) if r["type"] == "decision" and r["tick"] == 39)
    assert records[first_tick39 - 1]["type"] == "snapshot" and records[first_tick39 - 1]["frame"]["boundary"] == "start"
    assert types[-3:] == ["delta", "snapshot", "result"]
    assert reader.seek(39).colonies[0].queen_hp > 0 and reader.seek(39, BOUNDARY_TERMINAL).colonies[0].queen_hp == 0
    assert verify(path).terminal_tick == 39


def test_a_truncated_harness_run_has_no_outcome_and_a_cutoff_frame(replays):
    reader = ReplayReader.from_bytes(replays["cutoff"].data)
    r = reader.read_result()
    assert (r.finished, r.outcome, r.tick_completed, r.state_tick, r.ticks_executed) == (False, None, None, 45, 45)
    assert r.raw["outcome"] == {"finished": False} and reader.last_start_tick == 45
    assert reader.seek(45).terminal is None
    with pytest.raises(SeekError):
        reader.seek(45, BOUNDARY_TERMINAL)
    assert verify(reader).finished is False and verify(reader).winner == ""


def test_zero_tick_recording_is_a_valid_truncated_replay(tmp_path):
    path = tmp_path / "zero.jsonl"
    result, summary = record_episode(path, default_config(), SEED, *scripted("raid_win"), max_ticks=0)
    assert result.truncated and summary.ticks == 0 and summary.records == 3
    reader = ReplayReader.open(path)
    assert reader.seek(0) == view_from_state(*reset(default_config(), SEED))
    assert verify(path).ticks == 0


def test_a_cutoff_on_a_policy_boundary_has_one_snapshot_not_two(tmp_path):
    path = tmp_path / "cut30.jsonl"
    record_episode(path, default_config(), SEED, *scripted("raid_win"), max_ticks=30)
    snaps = [r["frame"]["tick"] for r in parse(path.read_bytes()) if r["type"] == "snapshot"]
    assert snaps == [0, 30] and verify(path).state_tick == 30


# --- sizes (measured, reported in the FOR-15 comment and docs; the bounds are a regression guard) --------------------------

SIZE_BOUNDS = {"mutual_raid_draw": 230_000, "collapse": 170_000, "raid_win": 760_000, "timeout_draw": 11_500_000}


@pytest.mark.parametrize("name", list(SIZE_BOUNDS))
def test_file_sizes_stay_within_the_documented_budget(replays, name):
    assert len(replays[name].data) <= SIZE_BOUNDS[name]


def test_display_data_is_a_small_share_of_a_long_file(replays):
    records = parse(replays["timeout_draw"].data)
    sizes = {t: 0 for t in ("snapshot", "delta", "event", "decision")}
    for rec in records:
        if rec["type"] in sizes:
            sizes[rec["type"]] += len(str(rec))
    assert sizes["snapshot"] + sizes["delta"] < 0.2 * sum(sizes.values())  # events dominate, not the view patches


# --- recorder contract --------------------------------------------------------------------------------------------------------

def test_a_fresh_reset_starts_with_reset_and_a_restored_state_embeds_its_checkpoint(tmp_path):
    state, run = reset(default_config(), SEED)
    path = tmp_path / "fresh.jsonl"
    record_play(path, state, run, scripted("raid_win"), 5)
    assert read_header(path).start_kind == "reset"
    state2, run2 = reset(default_config(), SEED)
    path2 = tmp_path / "forced.jsonl"
    record_play(path2, state2, run2, scripted("raid_win"), 5, embed_checkpoint=True)
    head = read_header(path2)
    assert head.start_kind == "checkpoint" and head.start_checkpoint_hash and verify(path2).ticks == 5
    state3, run3 = reset(default_config(), SEED)
    play(state3, run3, scripted("raid_win"), 7)
    path3 = tmp_path / "mid.jsonl"
    record_play(path3, state3, run3, scripted("raid_win"), 5)
    assert read_header(path3).start_kind == "checkpoint" and read_header(path3).start_tick == 7


def test_recording_failures_leave_no_file_and_never_become_outcomes(tmp_path):
    class Boom(ScriptedController):
        def act(self, tick, observation, run):
            if tick >= 30:
                raise RuntimeError("policy crashed")
            return super().act(tick, observation, run)

    path = tmp_path / "boom.jsonl"
    a, b = scripted("raid_win")
    with pytest.raises(RuntimeError, match="policy crashed"):
        record_episode(path, default_config(), SEED, Boom(), b)
    assert not path.exists() and not (tmp_path / "boom.jsonl.part").exists()


def test_the_writer_refuses_misuse(tmp_path):
    import io

    from formicmind.replay import ReplayWriter

    config = default_config()
    seats = (SeatSpec("a", "scripted"), SeatSpec("b", "scripted"))
    state, run = reset(config, SEED)
    # decisions are only seen through wrapped controllers
    writer = ReplayWriter(io.BytesIO(), config, seats)
    writer.start(state, run)
    with pytest.raises(ReplayWriteError, match="policy tick"):
        writer.record_tick(state, run, step_tick(state, run, scripted("raid_win")))
    with pytest.raises(ReplayWriteError, match="not open"):
        ReplayWriter(io.BytesIO(), config, seats).record_tick(state, run, None)
    # development modes are not matches
    writer = ReplayWriter(io.BytesIO(), config, seats)
    state, run = reset(config, SEED)
    writer.start(state, run)
    with pytest.raises(ReplayWriteError, match="FULL"):
        writer.record_tick(state, run, step_tick(state, run, writer.wrap(scripted("raid_win")), movement_only=True))
    with pytest.raises(ReplayWriteError):
        ReplayWriter(io.BytesIO(), config, (seats[0],))
    with pytest.raises(ReplayWriteError, match="share"):
        ReplayWriter(io.BytesIO(), config, (SeatSpec("a", "policy", "policy:a:0"), SeatSpec("b", "policy", "policy:a:0")))
    with pytest.raises(ReplayWriteError, match="chain"):
        ReplayWriter(io.BytesIO(), config, seats, meta={"chain": "x"})
    with pytest.raises(ReplayWriteError, match="registered"):
        w = ReplayWriter(io.BytesIO(), config, (SeatSpec("a", "policy", "policy:a:0"), SeatSpec("b", "scripted")))
        w.start(*reset(config, SEED))


def test_meta_and_source_are_recorded_but_do_not_change_the_replayed_content(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    record_episode(a, default_config(), SEED, *scripted("mutual_raid_draw"), source="x")
    record_episode(b, default_config(), SEED, *scripted("mutual_raid_draw"), source="y", meta={"note": "dev"})
    assert read_header(b).meta == {"note": "dev"} and read_header(b).source == "y"
    ra, rb = read_result(a), read_result(b)
    assert (ra.events_hash, ra.final_state_hash, ra.final_checkpoint_hash) == (
        rb.events_hash, rb.final_state_hash, rb.final_checkpoint_hash)
    assert verify(a).events_hash == verify(b).events_hash


def test_replay_bytes_do_not_depend_on_pythonhashseed(replays, tmp_path):
    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "from formicmind import default_config\n"
        "from formicmind.match_smoke import SCENARIOS\n"
        "from formicmind.replay import record_episode\n"
        "from formicmind.tick import ScriptedController\n"
        "sc = SCENARIOS['mutual_raid_draw']\n"
        "record_episode(sys.argv[1], default_config(), 20261102, ScriptedController(sc.script_a),"
        " ScriptedController(sc.script_b), source='mutual_raid_draw')\n" % str(SRC))
    for seed in ("0", "4294967295", "random"):
        out = tmp_path / f"h{seed}.jsonl"
        proc = subprocess.run([sys.executable, "-c", script, str(out)], capture_output=True, text=True, check=False,
                              env={**os.environ, "PYTHONHASHSEED": seed})
        assert proc.returncode == 0, proc.stderr
        assert out.read_bytes() == replays["mutual_raid_draw"].data


def test_zero_chain_constant_is_64_zeros():
    assert ZERO_CHAIN == "0" * 64 and canonical_loads(b'{"a":1}') == {"a": 1}
