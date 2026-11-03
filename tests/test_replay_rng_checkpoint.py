"""FOR-15: RNG-consuming policies (post-decision stream restoration) and checkpoint/restore + replayed suffix."""

import io
from pathlib import Path

import pytest

from formicmind import checkpoint_from_bytes, checkpoint_hash, checkpoint_to_bytes, default_config, reset, state_hash
from formicmind.actions import get_action_mask, validate_action
from formicmind.decoder import PolicyLogitsV2
from formicmind.events import events_hash
from formicmind.match import play
from formicmind.replay import (
    ReplayFormatError,
    ReplayReader,
    ReplayVerifyError,
    ReplayWriteError,
    SeatSpec,
    SeekError,
    read_header,
    record_episode,
    record_play,
    verify,
)
from formicmind.replay.format import EventsHasher, pcg_advance
from formicmind.replay.writer import ReplayWriter
from formicmind.rng import Pcg32, policy_stream_id
from formicmind.tick import PolicyController, ScriptedController, step_tick

from replay_helpers import (
    RNG_TICKS,
    SEED,
    find,
    forge,
    parse,
    quiet_stream_policy,
    random_policy,
    rng_controllers,
    scripted,
)

A_STREAM, B_STREAM = policy_stream_id("rnd-a", 0), policy_stream_id("rnd-b", 0)
RNG_257_A, RNG_257_B = policy_stream_id("rng-257-a", 0), policy_stream_id("rng-257-b", 0)


def _policy_257(observation, mask, policy_rng):
    for _ in range(257):
        policy_rng.next_u32()
    return PolicyLogitsV2([0.0] * 17)


# --- the recorded stream states are the engine's --------------------------------------------------------------------

def test_recorded_post_decision_states_are_exactly_the_live_streams_after_each_policy_phase(replays):
    """Independent expectation: the test replays the match itself and reads the streams after every policy tick."""
    state, run = reset(default_config(), SEED, [A_STREAM, B_STREAM])
    controllers = rng_controllers()
    live = {}
    initial = {sid: run.streams[sid].clone() for sid in (A_STREAM, B_STREAM)}
    while run.outcome is None and state.tick < RNG_TICKS:
        tick = state.tick
        step_tick(state, run, controllers)
        if tick % 30 == 0:
            live[tick] = {sid: run.streams[sid].clone() for sid in (A_STREAM, B_STREAM)}
    reader = ReplayReader.from_bytes(replays["rng_policy"].data)
    assert sorted(reader.decisions) == sorted(live) and len(live) >= 9
    previous = initial
    for tick, decision in sorted(reader.decisions.items()):
        for seat, sid in zip(decision.seats, (A_STREAM, B_STREAM)):
            assert seat.stream_id == sid and seat.draws == 17  # random_policy draws 17 logits per decision
            assert seat.state_after == live[tick][sid]
            assert pcg_advance(previous[sid].state, previous[sid].inc, 17) == seat.state_after.state
        previous = live[tick]


def test_header_names_the_policies_and_their_streams_as_data_only(replays):
    head = read_header(replays["rng_policy"].path)
    assert [(s.policy_id, s.kind, s.stream_id) for s in head.seats] == [
        ("rnd-a", "policy", A_STREAM), ("rnd-b", "policy", B_STREAM)]
    assert head.streams == (A_STREAM, B_STREAM)
    assert "random_policy" not in replays["rng_policy"].data.decode("ascii")  # no executable object, no code name


def test_verify_reproduces_an_rng_consuming_match_including_the_checkpoint_hash(replays):
    report = verify(replays["rng_policy"].path)
    assert report.final_checkpoint_hash == replays["rng_policy"].result.checkpoint_hash
    assert report.decisions >= 9 and report.finished


def test_python_generated_257_draw_fixture_is_readable_and_verifiable(tmp_path):
    generated = tmp_path / "rng_257.jsonl"
    record_episode(
        generated,
        default_config(),
        SEED,
        PolicyController(_policy_257, RNG_257_A),
        PolicyController(_policy_257, RNG_257_B),
        max_ticks=1,
        source="cross-language:rng-257",
    )
    fixture = Path(__file__).parents[1] / "viewer" / "test-fixtures" / "rng_257.jsonl"
    assert generated.read_bytes() == fixture.read_bytes()

    reader = ReplayReader.open(fixture)
    assert [seat.draws for seat in reader.decisions[0].seats] == [257, 257]
    report = verify(reader)
    assert report.decisions == 1 and not report.finished


def test_replaying_the_actions_without_restoring_the_streams_breaks_only_the_checkpoint_hash(replays):
    """The edge case of the task: a naive scripted substitute reproduces state and events, not checkpoint_hash."""
    reader = ReplayReader.from_bytes(replays["rng_policy"].data)
    recorded = reader.read_result()
    state, run = reset(default_config(), SEED, reader.read_header().streams)

    class Naive:
        def __init__(self, side):
            self.side = side

        def act(self, tick, observation, run):
            return validate_action(observation, get_action_mask(observation), reader.decisions[tick].seats[self.side].action)

    controllers = (Naive(0), Naive(1))
    hasher = EventsHasher()
    for record in reader.tick_records:
        for e in step_tick(state, run, controllers).events:
            hasher.add(e.to_canonical())
    assert state_hash(state) == recorded.final_state_hash       # the game is the same ...
    assert hasher.hexdigest() == recorded.events_hash            # ... the journal is the same ...
    assert checkpoint_hash(state, run) != recorded.final_checkpoint_hash  # ... but the policy streams never moved
    assert run.streams[A_STREAM] == reset(default_config(), SEED, [A_STREAM])[1].streams[A_STREAM]


def test_a_policy_that_owns_a_stream_but_never_draws_records_zero_draws(tmp_path):
    path = tmp_path / "quiet.jsonl"
    result, _ = record_episode(path, default_config(), SEED, *rng_controllers(quiet_stream_policy), max_ticks=100)
    reader = ReplayReader.open(path)
    assert {s.draws for d in reader.decisions.values() for s in d.seats} == {0}
    assert verify(reader).final_checkpoint_hash == result.checkpoint_hash
    initial = reset(default_config(), SEED, [A_STREAM, B_STREAM])[1]
    last = max(reader.decisions)
    assert last > 0 and reader.decisions[last].seats[0].state_after == initial.streams[A_STREAM]


def test_a_mixed_match_scripted_seat_without_stream_against_a_random_policy(tmp_path):
    path = tmp_path / "mixed.jsonl"
    scripted_a = scripted("raid_win")[0]
    controller_b = PolicyController(random_policy, B_STREAM)
    result, _ = record_episode(path, default_config(), SEED, scripted_a, controller_b, max_ticks=200)
    reader = ReplayReader.open(path)
    head = reader.read_header()
    assert head.streams == (B_STREAM,) and head.seats[0].stream_id == "" and head.seats[1].stream_id == B_STREAM
    assert all(d.seats[0].stream_id == "" and d.seats[1].draws == 17 for d in reader.decisions.values())
    assert verify(reader).final_checkpoint_hash == result.checkpoint_hash


# --- the recorder refuses controllers that break the stream contract -----------------------------------------------------

def _record_with(controller_a, controller_b, seats, tmp_path, extra=()):
    return record_episode(tmp_path / "x.jsonl", default_config(), SEED, controller_a, controller_b, max_ticks=40,
                          seats=seats, extra_streams=extra)


def test_a_controller_that_draws_from_the_world_stream_is_refused(tmp_path):
    class Greedy(ScriptedController):
        def act(self, tick, observation, run):
            run.stream("world").next_u32()
            return super().act(tick, observation, run)

    a, b = scripted("raid_win")
    with pytest.raises(ReplayWriteError, match="declares no RNG stream"):
        _record_with(Greedy(), b, None, tmp_path)
    assert not (tmp_path / "x.jsonl").exists()


def test_a_controller_that_draws_from_the_other_seats_stream_is_refused(tmp_path):
    class Thief(PolicyController):
        def act(self, tick, observation, run):
            run.stream(B_STREAM).next_u32()
            return super().act(tick, observation, run)

    with pytest.raises(ReplayWriteError, match="other than"):
        _record_with(Thief(random_policy, A_STREAM), PolicyController(random_policy, B_STREAM), None, tmp_path)


def test_a_seat_declared_without_stream_cannot_hide_a_consuming_policy(tmp_path):
    seats = (SeatSpec("rnd-a", "policy", ""), SeatSpec("rnd-b", "policy", B_STREAM))
    with pytest.raises(ReplayWriteError, match="declares no RNG stream"):
        _record_with(PolicyController(random_policy, A_STREAM), PolicyController(random_policy, B_STREAM), seats, tmp_path)


def test_a_controller_that_rewrites_its_stream_state_arbitrarily_is_refused(tmp_path):
    class Rewriter(PolicyController):
        def act(self, tick, observation, run):
            run.stream(self.stream_id).state = 12345
            return super().act(tick, observation, run)

    with pytest.raises(ReplayWriteError, match="not reachable"):
        _record_with(Rewriter(random_policy, A_STREAM), PolicyController(random_policy, B_STREAM), None, tmp_path)

    class Replacer(PolicyController):
        def act(self, tick, observation, run):
            run.streams[self.stream_id] = Pcg32(1, 3)
            return super().act(tick, observation, run)

    with pytest.raises(ReplayWriteError, match="identity"):
        _record_with(Replacer(random_policy, A_STREAM), PolicyController(random_policy, B_STREAM), None, tmp_path)


# --- tampered RNG metadata -----------------------------------------------------------------------------------------------

def _rng_entry(records, tick, seat):
    decision = next(r for r in records if r["type"] == "decision" and r["tick"] == tick)
    return decision["seats"][seat]["rng"][0]


def test_an_unreachable_state_after_is_a_format_error(replays):
    def edit(records):
        _rng_entry(records, 60, 0)["state_after"]["state"] = "7"

    with pytest.raises(ReplayFormatError, match="not reachable"):
        ReplayReader.from_bytes(forge(replays["rng_policy"].data, edit))


def test_a_changed_stream_identity_unknown_stream_or_wrong_owner_is_a_format_error(replays):
    def inc(records):
        _rng_entry(records, 30, 1)["state_after"]["inc"] = "3"

    def unknown(records):
        _rng_entry(records, 30, 1)["stream_id"] = "policy:ghost:0"

    def swapped(records):
        _rng_entry(records, 30, 0)["stream_id"] = B_STREAM

    for edit, text in [(inc, "inc"), (unknown, "not the seat"), (swapped, "not the seat")]:
        with pytest.raises(ReplayFormatError, match=text):
            ReplayReader.from_bytes(forge(replays["rng_policy"].data, edit))


def test_missing_or_surplus_rng_entries_and_contradictory_headers_are_format_errors(replays):
    def drop(records):
        next(r for r in records if r["type"] == "decision")["seats"][0]["rng"] = []

    with pytest.raises(ReplayFormatError, match="stream entries"):
        ReplayReader.from_bytes(forge(replays["rng_policy"].data, drop))

    def add_to_scripted(records):
        next(r for r in records if r["type"] == "decision")["seats"][0]["rng"] = [
            {"stream_id": A_STREAM, "draws": 0, "state_after": {"state": "1", "inc": "3"}}]

    with pytest.raises(ReplayFormatError, match="stream entries"):
        ReplayReader.from_bytes(forge(replays["collapse"].data, add_to_scripted))

    def shared(records):
        find(records, "header")["seats"][1]["stream_id"] = A_STREAM

    def not_registered(records):
        find(records, "header")["streams"] = [A_STREAM]

    for edit in (shared, not_registered):
        with pytest.raises(ReplayFormatError, match="header.seats"):
            ReplayReader.from_bytes(forge(replays["rng_policy"].data, edit))


def test_a_consistent_forgery_of_the_last_decisions_stream_state_is_caught_by_verify(replays):
    """draws and state_after shifted together pass every file check; only the replayed checkpoint_hash notices."""
    def edit(records):
        last = max((r for r in records if r["type"] == "decision"), key=lambda r: r["tick"])
        entry = last["seats"][0]["rng"][0]
        entry["draws"] += 1
        entry["state_after"]["state"] = str(pcg_advance(int(entry["state_after"]["state"]), int(entry["state_after"]["inc"]), 1))

    forged = forge(replays["rng_policy"].data, edit)
    ReplayReader.from_bytes(forged)  # well-formed
    with pytest.raises(ReplayVerifyError, match="checkpoint_hash"):
        verify(forged)


# --- checkpoint / restore in the middle of a match + replay of the suffix --------------------------------------------------

def _split_run(controllers, streams, split_tick):
    state, run = reset(default_config(), SEED, streams)
    prefix, _ = play(state, run, controllers, stop_before_tick=split_tick)
    return state, run, prefix


@pytest.mark.parametrize("kind", ["scripted", "rng"])
def test_a_suffix_replay_from_a_restored_checkpoint_continues_the_same_journal(kind, tmp_path):
    controllers = scripted("raid_win") if kind == "scripted" else rng_controllers()
    streams = () if kind == "scripted" else (A_STREAM, B_STREAM)
    full_state, full_run = reset(default_config(), SEED, streams)
    full_events, _ = play(full_state, full_run, controllers, None if kind == "scripted" else RNG_TICKS)

    state, run, prefix = _split_run(controllers, streams, 90)
    blob = checkpoint_to_bytes(state, run)
    restored_state, restored_run = checkpoint_from_bytes(blob, default_config())  # a different process would do this
    suffix_path = tmp_path / "suffix.jsonl"
    result, summary = record_play(suffix_path, restored_state, restored_run, controllers,
                                  None if kind == "scripted" else RNG_TICKS - 90, event_index_base=len(prefix),
                                  source="suffix")
    reader = ReplayReader.open(suffix_path)
    head = reader.read_header()
    assert (head.start_kind, head.start_tick, head.start_event_index) == ("checkpoint", 90, len(prefix))
    assert head.start_checkpoint_hash == checkpoint_hash(state, run)
    # the journal: same events, global indices continue where the prefix stopped
    suffix_events = list(reader.iter_events())
    assert suffix_events == [e.to_canonical() for e in full_events[len(prefix):]]
    indices = [r["i"] for r in parse(suffix_path.read_bytes()) if r["type"] == "event"]
    assert indices == list(range(len(prefix), len(full_events)))
    assert events_hash(prefix + list(result.events)) == events_hash(full_events)
    # the end of the match is the same match
    assert (result.state_hash, result.checkpoint_hash) == (state_hash(full_state), checkpoint_hash(full_state, full_run))
    assert verify(reader).final_checkpoint_hash == checkpoint_hash(full_state, full_run)
    # views: the suffix starts at the checkpoint tick and its frames equal the full match's frames
    full_reader = ReplayReader.from_bytes(_full_bytes(kind, tmp_path))
    assert reader.first_tick == 90 and reader.seek(90) == full_reader.seek(90)
    for tick in (91, 119, 120, reader.last_start_tick):
        assert reader.seek(tick) == full_reader.seek(tick)
    with pytest.raises(SeekError):
        reader.seek(89)


def _full_bytes(kind, tmp_path):
    controllers = scripted("raid_win") if kind == "scripted" else rng_controllers()
    path = tmp_path / f"full-{kind}.jsonl"
    record_episode(path, default_config(), SEED, *controllers, max_ticks=None if kind == "scripted" else RNG_TICKS,
                   source="suffix")
    return path.read_bytes()


def test_the_streams_in_a_suffix_replay_start_from_the_checkpoints_live_states(tmp_path):
    state, run, prefix = _split_run(rng_controllers(), (A_STREAM, B_STREAM), 90)
    live_a = run.streams[A_STREAM].clone()
    path = tmp_path / "suffix.jsonl"
    record_play(path, state, run, rng_controllers(), 60, event_index_base=len(prefix))
    first = ReplayReader.open(path).decisions[90]
    assert first.seats[0].state_after.state == pcg_advance(live_a.state, live_a.inc, 17)  # continues, not reseeds
    assert live_a != reset(default_config(), SEED, [A_STREAM])[1].streams[A_STREAM]


def test_a_forged_cursor_or_checkpoint_in_a_suffix_header_is_rejected(tmp_path):
    state, run, prefix = _split_run(scripted("raid_win"), (), 90)
    path = tmp_path / "suffix.jsonl"
    record_play(path, state, run, scripted("raid_win"), 40, event_index_base=len(prefix))
    data = path.read_bytes()
    ReplayReader.from_bytes(data)

    def cursor_event(records):
        find(records, "header")["cursor"]["event_index"] += 5

    def cursor_tick(records):
        find(records, "header")["cursor"]["tick"] = 60

    def checkpoint_hash_(records):
        find(records, "header")["start"]["checkpoint_hash"] = "0" * 64

    def checkpoint_tampered(records):
        find(records, "header")["start"]["checkpoint"]["state"]["colonies"][0]["food"] += 1

    def reset_at_90(records):
        find(records, "header")["start"] = {"kind": "reset"}

    for edit in (cursor_event, cursor_tick, checkpoint_hash_, checkpoint_tampered, reset_at_90):
        with pytest.raises(ReplayFormatError):
            ReplayReader.from_bytes(forge(data, edit))


def test_a_finished_run_cannot_start_a_replay(tmp_path):
    state, run = reset(default_config(), SEED)
    play(state, run, scripted("mutual_raid_draw"))
    assert run.outcome is not None
    writer = ReplayWriter(io.BytesIO(), default_config(), (SeatSpec("a", "scripted"), SeatSpec("b", "scripted")))
    with pytest.raises(ReplayWriteError, match="running episode"):
        writer.start(state, run)
