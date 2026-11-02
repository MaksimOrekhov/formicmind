"""FOR-15: strict format and hash checks. Every corruption must fail with the *right kind* of error:
``ReplayFormatError`` (not a valid v1 file) or ``ReplayVerifyError`` (valid file the engine does not reproduce)."""

import random

import pytest

from formicmind.canonical import canonical_dumps
from formicmind.replay import ReplayFormatError, ReplayReader, ReplayVerifyError, ReplayVersionError, verify
from formicmind.replay.format import ZERO_CHAIN, seal_record, split_lines
from formicmind.replay import frames
from formicmind.replay.frames import frame_hash
from formicmind.serialization import config_from_canonical, config_hash

from replay_helpers import build, find, find_event, fix_event_totals, forge, index_of, parse


@pytest.fixture(scope="module")
def small(replays):
    return replays["mutual_raid_draw"].data  # queen draw at tick 39 (cut at QueenTerminal), decisions at 0 and 30


@pytest.fixture(scope="module")
def collapse(replays):
    return replays["collapse"].data  # an end-of-tick terminal


def reads(data):
    return ReplayReader.from_bytes(data)


def fails_format(data, match=None, exc=ReplayFormatError):
    with pytest.raises(exc, match=match or None):
        reads(data)


def fails_verify(data, match=None):
    reads(data)  # the file itself is well-formed ...
    with pytest.raises(ReplayVerifyError, match=match):  # ... the engine does not reproduce it
        verify(data)


def test_the_untouched_files_pass_both_checks(small, collapse):
    assert verify(small).terminal_tick == 39 and verify(collapse).terminal_tick == 46
    assert build(parse(small)) == small  # the forging helper itself is lossless


# --- bytes and lines (no re-sealing: the hash chain and the strict codec must notice) ------------------------------------

def test_truncated_files_are_refused(small):
    fails_format(small[:-1], "truncated")  # the last line lost its newline
    fails_format(small[:-60], "truncated")  # cut inside the result line
    fails_format(small[: small.rfind(b"\n", 0, len(small) - 1) + 1], "missing result")  # result line removed cleanly
    fails_format(small[: len(small) // 2], "truncated|canonical|JSON")
    fails_format(b"", "empty")
    lines = split_lines(small)
    fails_format(b"\n".join(lines[:40]) + b"\n", "missing result")


def test_a_replay_that_was_never_closed_has_no_result_and_is_refused(small):
    records = parse(small)
    fails_format(build(records[:-1]), "missing result")
    fails_format(build(records[:1] + records[-1:]), "")  # header + result only: the grammar is violated


def test_lines_that_are_dropped_inserted_duplicated_or_swapped_break_the_chain(small):
    lines = small.split(b"\n")[:-1]
    n = len(lines)
    cases = {
        "dropped": lines[:100] + lines[101:],
        "duplicated": lines[:100] + [lines[100]] + lines[100:],
        "swapped": lines[:100] + [lines[101], lines[100]] + lines[102:],
        "inserted": lines[:100] + [lines[5]] + lines[100:],
        "moved_to_end": lines[:100] + lines[101:] + [lines[100]],
    }
    for name, new in cases.items():
        assert len(new) in (n - 1, n, n + 1)
        fails_format(b"\n".join(new) + b"\n", "chain|seq|records after")


def test_every_sampled_single_bit_flip_is_a_format_error_never_silently_accepted(small):
    rng = random.Random(15)
    positions = sorted(rng.sample(range(len(small)), 400)) + [0, 1, len(small) - 2, len(small) - 1]
    for pos in positions:
        mutated = bytearray(small)
        mutated[pos] ^= 1 << rng.randrange(8)
        with pytest.raises(ReplayFormatError):
            reads(bytes(mutated))


def test_framing_defects_are_refused(small):
    fails_format(small.replace(b"\n", b"\r\n"), "carriage")
    fails_format(b"\xef\xbb\xbf" + small, "")
    fails_format(small + b"\n", "blank")
    fails_format(small.replace(b'{"', b'{ "', 1), "canonical")
    fails_format(b"[]\n" + small, "")
    fails_format(b"not json\n", "")
    fails_format(b'{"a":1.5}\n', "")


def test_records_after_the_result_are_refused(small):
    records = parse(small)
    extra = {"type": "event", "seq": 0, "i": 0, "e": {}}
    fails_format(build(records + [extra]), "after the result")
    line, _ = seal_record({"type": "snapshot", "seq": len(records)}, ZERO_CHAIN)
    fails_format(small + line, "chain|after the result")


# --- versions and header (re-sealed: the chain is fine, the content is not) --------------------------------------------------

def test_unknown_replay_schema_and_contract_versions_fail_fast(small):
    def schema(r):
        find(r, "header")["replay_schema"] = "v2"

    def rules(r):
        find(r, "header")["contract"]["rules_version"] = "l0-proto-other"

    def domain(r):
        find(r, "header")["contract"]["hash_domain_checkpoint"] = "formicmind/checkpoint/v4"

    def missing(r):
        del find(r, "header")["contract"]["rng_version"]

    fails_format(forge(small, schema), "schema", ReplayVersionError)
    fails_format(forge(small, rules), "contract", ReplayVersionError)
    fails_format(forge(small, domain), "contract", ReplayVersionError)
    fails_format(forge(small, missing), "contract", ReplayVersionError)


def test_header_defects_are_refused(small):
    def tag(r):
        find(r, "header")["format"] = "other.replay"

    def extra(r):
        find(r, "header")["note"] = "x"

    def drop(r):
        del find(r, "header")["source"]

    def config(r):
        find(r, "header")["config"]["t_max"] = 2999

    def seed(r):
        find(r, "header")["root_seed"] = "0x10"

    def seed_big(r):
        find(r, "header")["root_seed"] = str(2**64)

    def seats(r):
        find(r, "header")["seats"][0]["kind"] = "wizard"

    def meta(r):
        find(r, "header")["meta"] = {"k": 1}

    def not_header(r):
        r[0], r[1] = r[1], r[0]

    for edit, text in [(tag, "not a FormicMind"), (extra, "unexpected"), (drop, "missing"), (config, "config_hash"),
                       (seed, "root_seed"), (seed_big, "root_seed"), (seats, "seat"), (meta, "meta"),
                       (not_header, "header")]:
        fails_format(forge(small, edit), text)


# --- events -------------------------------------------------------------------------------------------------------------------

def test_malformed_impossible_or_foreign_events_are_refused(small):
    def kind(r):
        find_event(r, "harvest")["e"]["kind"] = "teleport"

    def extra(r):
        find_event(r, "harvest")["e"]["bonus"] = 1

    def wrong_type(r):
        find_event(r, "harvest")["e"]["harvested"] = "1"

    def bool_as_int(r):
        find_event(r, "harvest")["e"]["blocked"] = 0

    def wrong_tick(r):
        find_event(r, "harvest", 3)["e"]["tick"] += 1

    def wrong_phase(r):
        find_event(r, "harvest")["e"]["phase"] = 0

    def bad_side(r):
        find_event(r, "harvest")["e"]["side"] = 2

    def not_an_object(r):
        find_event(r, "harvest")["e"] = [1, 2]

    def terminal_phase(r):
        e = find_event(r, "terminal")["e"]
        e["phase"], e["phases_executed"] = 10, 11

    def terminal_flag(r):
        find_event(r, "terminal")["e"]["tick_completed"] = True

    for edit in (kind, extra, wrong_type, bool_as_int, wrong_tick, wrong_phase, bad_side, not_an_object,
                 terminal_phase, terminal_flag):
        fails_format(forge(small, edit))


def test_dropped_duplicated_and_reordered_events_are_refused(small):
    def drop(r):
        del r[index_of(r, find_event(r, "harvest", 5))]

    def duplicate(r):
        e = find_event(r, "harvest", 5)
        r.insert(index_of(r, e), dict(e))

    def reorder(r):
        a, b = find_event(r, "upkeep", 0, tick=4), find_event(r, "harvest", 0, tick=4)  # same tick, wrong phase order
        i, j = index_of(r, a), index_of(r, b)
        r[i], r[j] = r[j], r[i]

    def renumbered_reorder(r):
        reorder(r)
        fix_event_totals(r)

    fails_format(forge(small, drop), "event index")
    fails_format(forge(small, duplicate), "event index")
    fails_format(forge(small, reorder), "event index|engine order|phase")
    fails_format(forge(small, renumbered_reorder), "engine order")


def test_an_event_that_is_lost_or_forged_with_all_totals_repaired_is_caught_by_verify(small):
    """A careful forger renumbers the events and repairs count and events_hash; only the engine can disagree."""
    def drop(r):
        del r[index_of(r, find_event(r, "harvest", 5))]
        fix_event_totals(r)

    def change(r):
        find_event(r, "harvest", 5)["e"]["harvested"] += 1
        fix_event_totals(r)

    def swap_kind_value(r):
        find_event(r, "combat")["e"]["damage"] += 2
        fix_event_totals(r)

    for edit in (drop, change, swap_kind_value):
        fails_verify(forge(small, edit), "tick|event")


def test_an_event_changed_without_repairing_the_totals_fails_the_events_hash(small):
    def change(r):
        find_event(r, "harvest", 5)["e"]["harvested"] += 1

    fails_format(forge(small, change), "events_hash")


# --- decisions ----------------------------------------------------------------------------------------------------------------

def test_missing_duplicated_misplaced_and_malformed_decisions_are_refused(small):
    def missing(r):
        del r[index_of(r, find(r, "decision", 1))]

    def duplicated(r):
        d = find(r, "decision", 0)
        r.insert(index_of(r, d), dict(d))

    def wrong_tick(r):
        find(r, "decision", 1)["tick"] = 31

    def at_a_non_policy_tick(r):
        d = dict(find(r, "decision", 0))
        e = find_event(r, "order", 2)  # in the middle of tick 30; a decision is only legal at the start of a policy tick
        r.insert(index_of(r, e), d)

    def seat_order(r):
        find(r, "decision", 0)["seats"].reverse()

    def action_sum(r):
        find(r, "decision", 0)["seats"][0]["action"]["forage_share_q"][0] += 1

    def action_support(r):
        find(r, "decision", 0)["seats"][0]["action"]["forage_share_q"] = [16384, 16384, 16384, 16383, 0, 0]

    def action_type(r):
        find(r, "decision", 0)["seats"][0]["action"]["avoid_center"] = 1

    def short_mask(r):
        find(r, "decision", 0)["seats"][0]["mask"] = "1" * 29

    def bad_mask_chars(r):
        find(r, "decision", 0)["seats"][0]["mask"] = "2" * 30

    def bad_hash(r):
        find(r, "decision", 0)["seats"][0]["obs_hash"] = "XYZ"

    def extra_key(r):
        find(r, "decision", 0)["seats"][0]["bonus"] = 1

    for edit in (missing, duplicated, wrong_tick, at_a_non_policy_tick, seat_order, action_sum, action_support,
                 action_type, short_mask, bad_mask_chars, bad_hash, extra_key):
        fails_format(forge(small, edit))


def test_a_tampered_observation_hash_mask_or_action_is_caught_by_verify(small):
    def obs(r):
        find(r, "decision", 1)["seats"][1]["obs_hash"] = "0" * 64

    def mask(r):
        seat = find(r, "decision", 1)["seats"][0]
        seat["mask"] = "".join("0" if c == "1" else "1" for c in seat["mask"][:3]) + seat["mask"][3:]

    def action(r):
        seat = find(r, "decision", 0)["seats"][0]
        seat["action"]["lay_order"] = 1  # a legal action - just not the one that produced the recorded events

    fails_verify(forge(small, obs), "ObservationV2 hash")
    fails_verify(forge(small, mask), "mask")
    fails_verify(forge(small, action), "tick|event|differ")


# --- ticks, deltas, snapshots, boundaries ---------------------------------------------------------------------------------------

def test_tick_order_boundaries_and_patches_are_checked(small):
    def swap_delta_ticks(r):
        a, b = find(r, "delta", 3), find(r, "delta", 4)
        a["tick"], b["tick"] = b["tick"], a["tick"]

    def duplicate_tick(r):
        find(r, "delta", 4)["tick"] = 3

    def skip_tick(r):
        find(r, "delta", 4)["tick"] = 5

    def to_tick(r):
        find(r, "delta", 4)["to_tick"] += 1

    def last_boundary(r):
        find(r, "delta", 39)["to_boundary"] = "start"

    def boundary_name(r):
        find(r, "delta", 4)["to_boundary"] = "middle"

    def patch_without_hash(r):
        find(r, "delta", 4)["patch"]["nf"] = [[1, 999]]

    def patch_garbage(r):
        find(r, "delta", 4)["patch"] = {"c": "no"}

    def hash_only(r):
        find(r, "delta", 4)["frame_hash"] = "1" * 64

    def delta_missing(r):
        del r[index_of(r, find(r, "delta", 7))]

    for edit in (swap_delta_ticks, duplicate_tick, skip_tick, to_tick, last_boundary, boundary_name,
                 patch_without_hash, patch_garbage, hash_only, delta_missing):
        fails_format(forge(small, edit))


def test_snapshots_must_exist_where_the_grammar_says_and_match_the_patches(small, collapse):
    def drop_boundary_30(r):
        del r[index_of(r, find(r, "snapshot", 1))]

    def drop_terminal(r):
        del r[index_of(r, find(r, "snapshot", 2))]

    def drop_first(r):
        del r[index_of(r, find(r, "snapshot", 0))]

    def extra_snapshot(r):
        snap = dict(find(r, "snapshot", 1))
        r.insert(index_of(r, find(r, "delta", 5)) + 1, snap)

    def tamper_with_hash_repaired(r):
        snap = find(r, "snapshot", 1)
        snap["frame"]["colonies"][0]["food"] += 1
        snap["frame_hash"] = frame_hash(snap["frame"])

    def tamper_without_repair(r):
        find(r, "snapshot", 1)["frame"]["node_food"][1] += 1

    def terminal_object(r):
        snap = find(r, "snapshot", 2)
        snap["frame"]["terminal"]["winner"] = "A"
        snap["frame_hash"] = frame_hash(snap["frame"])

    def terminal_boundary(r):
        snap = find(r, "snapshot", 2)
        snap["frame"]["boundary"] = "start"
        snap["frame"]["terminal"] = {"finished": False}
        snap["frame_hash"] = frame_hash(snap["frame"])

    for edit in (drop_boundary_30, drop_terminal, drop_first, extra_snapshot, tamper_with_hash_repaired,
                 tamper_without_repair, terminal_object, terminal_boundary):
        fails_format(forge(small, edit))
    # collapse is an end-of-tick terminal: its terminal snapshot sits at state.tick = terminal_tick + 1
    def wrong_tick(r):
        snap = find(r, "snapshot", -1)
        snap["frame"]["tick"] -= 1
        snap["frame_hash"] = frame_hash(snap["frame"])

    fails_format(forge(collapse, wrong_tick))


def test_a_display_patch_forged_with_a_matching_frame_hash_is_caught_by_verify(small):
    """The file is internally consistent (patch, hash, snapshots) but shows a state the engine never had."""
    def edit(r):
        records = r
        # make node 1's stock different from tick 9 on, consistently: patch, delta hashes and snapshots
        body = frames.body_of(find(records, "snapshot", 0)["frame"])
        deltas = [x for x in records if x["type"] == "delta"]
        snapshots = {x["frame"]["tick"]: x for x in records if x["type"] == "snapshot" and x["frame"]["boundary"] == "start"}
        first = deltas[0]
        first["patch"].setdefault("nf", [])
        first["patch"]["nf"] = sorted([[1, 777]] + [c for c in first["patch"]["nf"] if c[0] != 1])
        # replay the (now different) patches to repair every later hash and snapshot
        body = frames.apply_patch(body, first["patch"])
        terminal = {"finished": False}
        for d in deltas:
            if d is not first:
                new = frames.apply_patch(body, d["patch"])
                # the stock at node 1 is restored by the original patch only if it changed; keep ours otherwise
                body = new
            end = d["to_boundary"] == "terminal"
            ev_terminal = find_event(records, "terminal")["e"] if end else None
            terminal = frames.terminal_object_from_event(ev_terminal) if end else {"finished": False}
            frame = frames.frame_dict(d["to_tick"], d["to_boundary"], terminal, body)
            d["frame_hash"] = frame_hash(frame)
            snap = snapshots.get(d["to_tick"]) if not end else next(x for x in records if x["type"] == "snapshot" and x["frame"]["boundary"] == "terminal")
            if snap is not None:
                snap["frame"], snap["frame_hash"] = frame, frame_hash(frame)

    forged = forge(small, edit)
    reads(forged)
    with pytest.raises(ReplayVerifyError, match="display|snapshot|frame"):
        verify(forged)


# --- result and hashes --------------------------------------------------------------------------------------------------------

def test_result_defects_are_refused(small):
    def field(name, value):
        def edit(r):
            r[-1][name] = value
        return edit

    for name, value in [("finished", False), ("ticks_executed", 3), ("decision_count", 5), ("event_count", 1),
                        ("state_tick", 38), ("events_hash", "0" * 64), ("outcome", {"finished": False}),
                        ("bonus", 1), ("record_count", 5), ("chain_head", "0" * 64), ("final_state_hash", "nope")]:
        fails_format(forge(small, field(name, value), renumber=False), "")

    def winner(r):
        r[-1]["outcome"]["winner"] = "A"

    def phase(r):
        r[-1]["outcome"]["terminal_phase"] = 10

    def completed(r):
        r[-1]["outcome"]["tick_completed"] = True

    for edit in (winner, phase, completed):
        fails_format(forge(small, edit))


def test_wrong_final_hashes_are_caught_by_verify(small):
    for name in ("final_state_hash", "final_checkpoint_hash"):
        def edit(r, name=name):
            r[-1][name] = "a" * 64

        fails_verify(forge(small, edit), name.replace("final_", "").replace("_hash", "_hash"))


def test_a_consistent_change_of_the_config_or_the_seed_is_caught_by_verify(small):
    def balance(r):
        header = find(r, "header")
        cfg = header["config"]
        cfg["balance"]["start_food"] = 61
        header["config_hash"] = config_hash(config_from_canonical(cfg))

    def seed(r):
        find(r, "header")["root_seed"] = "7"

    fails_verify(forge(small, balance), "first snapshot|event|differ")
    fails_verify(forge(small, seed), "checkpoint_hash")


def test_the_terminal_event_is_what_ends_the_replay(small, collapse):
    def lose_terminal(r):
        i = index_of(r, find_event(r, "terminal"))
        del r[i]
        fix_event_totals(r)

    def terminal_early(r):
        e = find_event(r, "terminal")
        r.insert(index_of(r, find_event(r, "queen_damage", 3)), dict(e))
        del r[index_of(r, e)]
        fix_event_totals(r)

    def reason(r):
        find_event(r, "terminal")["e"]["reason"] = "timeout"
        fix_event_totals(r)

    def tick_after_terminal(r):
        delta = dict(find(r, "delta", -1))
        r.insert(len(r) - 2, delta)

    for edit in (lose_terminal, terminal_early, reason, tick_after_terminal):
        fails_format(forge(small, edit))

    def timeout_early(r):  # a timeout is only legal in the last tick of the horizon
        e = find_event(r, "terminal")
        e["e"].update(reason="timeout", winner="draw", sides=[], phase=10, phases_executed=11, tick_completed=True)
        fix_event_totals(r)

    fails_format(forge(collapse, timeout_early))


def test_the_canonical_checkpoint_bytes_of_a_fresh_start_are_not_hidden_in_the_header(small):
    assert canonical_dumps({"kind": "reset"}) == b'{"kind":"reset"}'
    assert parse(small)[0]["start"] == {"kind": "reset"}


# --- Python's True == 1 must not be accepted as an integer --------------------------------------------------------------------

def test_bools_and_other_lookalikes_are_not_accepted_as_recorded_integers(small, tmp_path):
    def seq(r):
        r[1]["seq"] = True

    def event_index(r):
        find_event(r, "order", 0)  # the first events of tick 0
        events = [x for x in r if x["type"] == "event"]
        events[1]["i"] = True  # index 1

    def delta_to_tick(r):
        find(r, "delta", 0)["to_tick"] = True  # 1 == True

    def delta_tick(r):
        find(r, "delta", 0)["tick"] = False  # 0 == False

    def decision_tick(r):
        find(r, "decision", 0)["tick"] = False

    for edit in (seq, event_index, delta_to_tick, delta_tick, decision_tick):
        fails_format(forge(small, edit, renumber=False))
    from formicmind import default_config
    from formicmind.replay import record_episode
    from replay_helpers import SEED, scripted

    zero = tmp_path / "zero.jsonl"
    record_episode(zero, default_config(), SEED, *scripted("raid_win"), max_ticks=0)

    def ticks_executed(r):
        r[-1]["ticks_executed"] = False  # 0 == False

    def event_count(r):
        r[-1]["event_count"] = False

    for edit in (ticks_executed, event_count):
        fails_format(forge(zero.read_bytes(), edit), "result")
