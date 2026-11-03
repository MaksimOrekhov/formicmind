"""FOR-14 reliability regression: checkpoint hardening, multi-scenario determinism, stitched-journal restore,
full-trace side swap, privacy on real mid-game states, invariants over eggs_per_lay 1/2/4, version pins, CI file.

Correctness regression only. Nothing here measures performance or balance, and every seed is a developer-only
engineering constant (never a FOR-4 sealed seed).
"""

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from formicmind import (
    checkpoint_from_bytes,
    checkpoint_hash,
    checkpoint_to_bytes,
    default_config,
    reset,
    state_hash,
)
from formicmind.canonical import CanonicalError, canonical_dumps
from formicmind.events import Phase, event_sort_key, events_hash
from formicmind.invariants import InvariantError
from formicmind.match import play, run_episode
from formicmind.match_smoke import SCENARIOS
from formicmind.observation import build_observation, flatten_v2
from formicmind.rules import MIRROR
from formicmind.snapshot import begin_policy_phase
from formicmind.terminal import validate_outcome
from formicmind.tick import EpisodeFinishedError, ScriptedController, step_tick

from combat_helpers import A, B, C, N1, N4, N5, act, arena, cfg, ctrl, soldiers, workers
from helpers import transit
from test_match import PINNED

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEV_SEED = 20261102     # developer-only engineering constant


def controllers(scenario):
    return ScriptedController(scenario.script_a), ScriptedController(scenario.script_b)


# =================================================================================================
# 1. Checkpoint hardening: a running episode may not stand on/after the horizon
# =================================================================================================

def _finished_timeout(t_max=5):
    state, run = arena(config=cfg(t_max=t_max))
    workers(state, 0, 1, A)
    workers(state, 1, 1, B)
    for _ in range(t_max):
        step_tick(state, run, ctrl())
    assert run.outcome is not None and state.tick == t_max
    return state, run


def _doc(state, run):
    return json.loads(checkpoint_to_bytes(state, run))


def _load(doc, config, **kw):
    return checkpoint_from_bytes(canonical_dumps(doc), config, **kw)


def test_corrupted_finished_false_at_the_horizon_is_rejected_by_the_loader():
    state, run = _finished_timeout()
    doc = _doc(state, run)
    _load(doc, state.config)                                           # the honest one loads
    doc["run"]["outcome"] = {"finished": False}                          # ... the forged "still running" one not
    with pytest.raises(CanonicalError, match="T_max"):
        _load(doc, state.config)
    with pytest.raises(CanonicalError, match="T_max"):                   # a default-config twin, too
        state2, run2 = arena()
        state2.tick = state2.config.t_max
        _load(_doc(state2, run2), state2.config)


def test_tick_equal_to_t_max_without_outcome_is_rejected_and_one_before_is_fine():
    state, run = arena(config=cfg(t_max=5), tick=3)
    workers(state, 0, 1, A)
    workers(state, 1, 1, B)
    ok = _doc(state, run)
    _load(ok, state.config)
    last = json.loads(json.dumps(ok))
    last["state"]["tick"] = 4
    _load(last, state.config)                                           # T_max - 1 is a normal running tick
    doc = json.loads(json.dumps(ok))
    doc["state"]["tick"] = 5
    with pytest.raises(CanonicalError, match="T_max"):
        _load(doc, state.config)


def test_tick_beyond_t_max_is_rejected_even_in_dev_mode_and_with_any_outcome_field():
    state, run = arena(config=cfg(t_max=5), tick=3)
    workers(state, 0, 1, A)
    workers(state, 1, 1, B)
    doc = _doc(state, run)
    doc["state"]["tick"] = 6
    for kw in ({}, {"dev_mode": True}):
        with pytest.raises(CanonicalError, match="horizon"):
            _load(doc, state.config, **kw)
    finished, frun = _finished_timeout()
    fdoc = _doc(finished, frun)
    fdoc["state"]["tick"] = 6                                            # a finished outcome pins state.tick too
    with pytest.raises(CanonicalError):
        _load(fdoc, finished.config)


def test_development_mode_run_at_the_horizon_restores_only_with_the_explicit_flag_and_gets_no_outcome():
    state, run = arena(config=cfg(t_max=5))
    workers(state, 0, 1, A)
    workers(state, 1, 1, B)
    for _ in range(5):
        r = step_tick(state, run, ctrl(), partial_economy=True)
        assert r.physics_complete is False and r.outcome is None
    assert state.tick == 5 and run.outcome is None                       # nothing is decided in a dev mode
    data = checkpoint_to_bytes(state, run)
    with pytest.raises(CanonicalError, match="dev_mode"):
        checkpoint_from_bytes(data, state.config)                        # a FULL-episode loader refuses it
    state2, run2 = checkpoint_from_bytes(data, state.config, dev_mode=True)
    assert run2.outcome is None and state_hash(state2) == state_hash(state)
    assert checkpoint_hash(state2, run2) == checkpoint_hash(state, run)
    with pytest.raises(ValueError, match="horizon"):                     # and it still cannot step on
        step_tick(state2, run2, ctrl(), partial_economy=True)
    # dev-mode checkpoints before the horizon never needed the flag
    s3, r3 = arena(config=cfg(t_max=5))
    step_tick(s3, r3, ctrl(), movement_only=True)
    checkpoint_from_bytes(checkpoint_to_bytes(s3, r3), s3.config)


def test_validate_outcome_unit_level_horizon_contract():
    state, _ = arena(config=cfg(t_max=5), tick=5)
    with pytest.raises(InvariantError, match="T_max"):
        validate_outcome(state, None)
    validate_outcome(state, None, dev_mode=True)
    state.tick = 6
    for dev in (False, True):
        with pytest.raises(InvariantError, match="horizon"):
            validate_outcome(state, None, dev_mode=dev)
    state.tick = 4
    validate_outcome(state, None)


def _tampered(doc_fn, state, run):
    doc = _doc(state, run)
    doc_fn(doc)
    with pytest.raises(CanonicalError):
        _load(doc, state.config)


def test_terminal_outcome_boundaries_are_pinned_for_timeout_collapse_and_queen_terminal():
    # timeout: terminal_tick must be T_max - 1, state.tick T_max, phase Terminal/Log, 11 phases, no sides
    state, run = _finished_timeout()
    for mutate in (
        lambda d: d["run"]["outcome"].update(terminal_tick=3),
        lambda d: d["run"]["outcome"].update(terminal_tick=5),           # outside 0..T_max-1
        lambda d: d["run"]["outcome"].update(terminal_phase=int(Phase.QUEEN_TERMINAL), phases_executed=5),
        lambda d: d["run"]["outcome"].update(phases_executed=10),
        lambda d: d["run"]["outcome"].update(winner="A"),
        lambda d: d["run"]["outcome"].update(sides=[0]),
        lambda d: d["state"].update(tick=4),                              # outcome says the tick completed
    ):
        _tampered(mutate, state, run)

    # economic collapse: complete tick (state.tick == terminal_tick + 1), exactly the collapsed colony loses
    cstate, crun = arena(tick=1)
    cstate.colonies[0].food = 9
    workers(cstate, 1, 1, B)
    result = step_tick(cstate, crun, ctrl(), check_invariants=True)
    assert result.terminal == "economic_collapse" and cstate.tick == 2
    _load(_doc(cstate, crun), cstate.config)
    for mutate in (
        lambda d: d["state"].update(tick=1),                              # as if the tick were cut mid-way
        lambda d: d["run"]["outcome"].update(terminal_phase=int(Phase.QUEEN_TERMINAL), phases_executed=5),
        lambda d: d["run"]["outcome"].update(sides=[1], winner="A"),      # B is not collapsed
        lambda d: d["run"]["outcome"].update(reason="queen_death"),       # both queens are alive
        lambda d: d["state"]["colonies"][0].update(food=500),             # A is no longer collapsed
    ):
        _tampered(mutate, cstate, crun)


def test_mid_tick_queen_terminal_keeps_state_tick_t_and_cannot_be_forged_into_a_completed_tick():
    state, run = arena(config=cfg(t_max=5), tick=4)                      # the very last tick of the episode
    soldiers(state, 0, 4, B)
    state.colonies[1].queen_hp = 8
    result = step_tick(state, run, ctrl({0: act(army_choice=6)}), check_invariants=True)
    out = result.outcome
    assert (out.reason.value, out.terminal_tick, out.tick_completed, state.tick) == ("queen_death", 4, False, 4)
    assert out.phases_executed == 5 and result.events[-1].tick_completed is False
    s2, r2 = checkpoint_from_bytes(checkpoint_to_bytes(state, run), state.config)    # honest restore
    assert r2.outcome == out and s2.tick == 4
    with pytest.raises(EpisodeFinishedError):
        step_tick(s2, r2, ctrl())
    for mutate in (
        lambda d: d["state"].update(tick=5),                              # "advance" the incomplete tick
        lambda d: d["run"]["outcome"].update(terminal_phase=int(Phase.TERMINAL_LOG), phases_executed=11),
        lambda d: d["run"]["outcome"].update(phases_executed=11),
        lambda d: d["run"].update(outcome={"finished": False}),           # queen is dead but "running"
        lambda d: d["state"]["colonies"][1].update(queen_hp=8),           # queen alive, outcome says she died
    ):
        _tampered(mutate, state, run)


# =================================================================================================
# 2. Determinism of the whole reset -> run_episode path
# =================================================================================================

SEEDS = (0, 1, DEV_SEED, 2**53 + 1, 2**64 - 1)          # incl. seeds that are not JS-safe integers


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("name", ["raid_win", "mutual_raid_draw", "collapse"])
def test_rules_consume_no_randomness_so_the_seed_changes_only_the_run_state(name, seed):
    r = run_episode(default_config(), seed, *controllers(SCENARIOS[name]))
    pin = PINNED[name]
    assert (r.winner, r.reason, r.terminal_tick) == (pin["winner"], pin["reason"], pin["terminal_tick"])
    assert (r.events_hash, r.state_hash) == (pin["events_hash"], pin["state_hash"])      # mechanics: seed-free
    if seed == DEV_SEED:
        assert r.checkpoint_hash == pin["checkpoint_hash"]
    else:
        assert r.checkpoint_hash != pin["checkpoint_hash"]                                # root_seed is run state


@pytest.mark.parametrize("seed", [2**53 + 1, 2**64 - 1])
def test_64_bit_seeds_are_serialised_as_strings_and_survive_a_checkpoint(seed):
    state, run = reset(default_config(), seed)
    play(state, run, controllers(SCENARIOS["raid_win"]), 40)
    data = checkpoint_to_bytes(state, run)
    doc = json.loads(data)
    assert doc["run"]["root_seed"] == str(seed) and type(doc["run"]["root_seed"]) is str
    state2, run2 = checkpoint_from_bytes(data, state.config)
    assert run2.root_seed == seed and checkpoint_to_bytes(state2, run2) == data      # byte-identical round trip


def test_repeated_full_runs_are_identical_down_to_the_event_objects_and_stable_ordering():
    for name in ("raid_win", "mutual_raid_draw", "collapse"):
        first = run_episode(default_config(), DEV_SEED, *controllers(SCENARIOS[name]))
        second = run_episode(default_config(), DEV_SEED, *controllers(SCENARIOS[name]))
        assert first.events == second.events and first.outcome == second.outcome
        assert first.events_hash == second.events_hash == events_hash(list(second.events))
        per_tick = {}
        for e in first.events:
            per_tick.setdefault(e.tick, []).append(e)
        for events in per_tick.values():                                 # execution order == canonical sort order
            assert events == sorted(events, key=event_sort_key)


_HASHSEED_SCRIPT = f"""
import sys; sys.path.insert(0, {str(SRC)!r})
from formicmind.match_smoke import run_match_smoke
for name in ('raid_win', 'mutual_raid_draw', 'collapse', 'timeout_draw'):
    r = run_match_smoke(name, check_invariants=False).result
    print(name, r.events_hash, r.state_hash, r.checkpoint_hash)
"""


def _run_with_hashseed(value):
    proc = subprocess.run([sys.executable, "-c", _HASHSEED_SCRIPT], capture_output=True, text=True,
                          env={**os.environ, "PYTHONHASHSEED": value}, check=False)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip().splitlines()


@pytest.mark.parametrize("hashseed", ["0", "random"])
def test_all_four_smoke_matches_have_the_pinned_hashes_under_any_pythonhashseed(hashseed):
    expected = [f"{n} {PINNED[n]['events_hash']} {PINNED[n]['state_hash']} {PINNED[n]['checkpoint_hash']}"
                for n in ("raid_win", "mutual_raid_draw", "collapse", "timeout_draw")]
    assert _run_with_hashseed(hashseed) == expected


# =================================================================================================
# 3. checkpoint -> restore at dangerous boundaries, with the *stitched* event journal
# =================================================================================================

CUTS = {
    "raid_win": [1, 29, 30, 31, 155, 156, 174, 175],       # K boundary, mid-movement, first combat, lethal tick
    "mutual_raid_draw": [1, 9, 10, 38, 39],
    "collapse": [1, 45, 46],
    "timeout_draw": [29, 30, 2998, 2999],
}


@pytest.fixture(scope="module")
def full_runs():
    return {n: run_episode(default_config(), DEV_SEED, *controllers(SCENARIOS[n])) for n in SCENARIOS}


@pytest.mark.parametrize("name,cut", [(n, c) for n, cuts in CUTS.items() for c in cuts])
def test_stitched_journal_after_restore_equals_the_uninterrupted_journal(full_runs, name, cut):
    full = full_runs[name]
    ctrls = controllers(SCENARIOS[name])
    state, run = reset(default_config(), DEV_SEED)
    head, _ = play(state, run, ctrls, stop_before_tick=cut)
    assert state.tick == cut and run.outcome is None
    data = checkpoint_to_bytes(state, run)
    state2, run2 = checkpoint_from_bytes(data, state.config)
    assert checkpoint_to_bytes(state2, run2) == data and run2.windows == run.windows   # WindowStats included
    tail, _ = play(state2, run2, ctrls)
    stitched = head + tail
    assert stitched == list(full.events)                                  # whole journal, not just the suffix
    assert events_hash(stitched) == full.events_hash == PINNED[name]["events_hash"]
    assert run2.outcome == full.outcome
    assert (state_hash(state2), checkpoint_hash(state2, run2)) == (full.state_hash, full.checkpoint_hash)
    assert head and head[-1].tick == cut - 1 and tail[0].tick == cut        # no tick split in two


def test_restore_before_policy_tick_30_gives_byte_identical_observations_and_window_stats():
    ctrls = controllers(SCENARIOS["raid_win"])
    state, run = reset(default_config(), DEV_SEED)
    play(state, run, ctrls, stop_before_tick=30)
    assert [w.window_ticks for w in run.windows] == [30, 30]
    state2, run2 = checkpoint_from_bytes(checkpoint_to_bytes(state, run), state.config)

    def obs(s, r):
        return [_obs_bytes(s, r, side) for side in (0, 1)]

    assert obs(state, run) == obs(state2, run2)
    step_tick(state, run, ctrls)
    step_tick(state2, run2, ctrls)
    assert [w.window_ticks for w in run.windows] == [1, 1] == [w.window_ticks for w in run2.windows]
    assert checkpoint_hash(state, run) == checkpoint_hash(state2, run2)


# =================================================================================================
# 4. Side swap over whole traces (events compared with sides swapped and nodes mirrored)
# =================================================================================================

_SIDE_KEYS = {"side", "attacker_side", "defender_side"}
_NODE_KEYS = {"node", "from_node", "to_node", "target_node", "army_target_node", "old_target", "new_target"}
_DROP = {"unit_id", "unit_ids"}                  # unit ids are allocated A-first, so they differ under a swap


def signature(event, mirror):
    out = []
    for key, value in event.to_canonical().items():
        if key in _DROP:
            continue
        if mirror:
            if key in _SIDE_KEYS:
                value = 1 - value
            elif key in _NODE_KEYS and value >= 0:
                value = MIRROR[value]
            elif key == "sides":
                value = sorted(1 - s for s in value)
            elif key == "queen_hp":
                value = value[::-1]
            elif key == "winner":
                value = {"A": "B", "B": "A", "draw": "draw"}[value]
        out.append((key, tuple(value) if isinstance(value, list) else value))
    return tuple(sorted(out))


def trace(events, mirror):
    return Counter((e.tick, int(e.phase), e.kind, signature(e, mirror)) for e in events)


def _play_scripts(sa, sb, seed=DEV_SEED, **kw):
    return run_episode(default_config(), seed, ScriptedController(sa), ScriptedController(sb), **kw)


@pytest.mark.parametrize("pair", ["raid_vs_calm", "centre_vs_raid", "economy_vs_hunger"])
def test_swapping_the_seats_mirrors_the_complete_event_trace(pair):
    from formicmind.match_smoke import _act
    scripts = {
        "raid_vs_calm": (SCENARIOS["raid_win"].script_a, SCENARIOS["raid_win"].script_b),
        "centre_vs_raid": ({0: _act(lay=2, army=3), 90: _act(lay=0, army=3)}, {0: _act(army=6, avoid=True)}),
        "economy_vs_hunger": (SCENARIOS["collapse"].script_a, SCENARIOS["collapse"].script_b),
    }[pair]
    x, y = scripts
    ab = _play_scripts(x, y, max_ticks=400)
    ba = _play_scripts(y, x, max_ticks=400)
    assert trace(ab.events, mirror=True) == trace(ba.events, mirror=False)
    assert (ab.outcome is None) == (ba.outcome is None)
    if ab.outcome is not None:
        assert signature(ab.events[-1], True) == signature(ba.events[-1], False)
        assert ab.score == tuple(reversed(ba.score))


@pytest.mark.parametrize("name", ["mutual_raid_draw", "timeout_draw"])
def test_symmetric_matches_are_their_own_mirror_image(full_runs, name):
    events = full_runs[name].events
    assert trace(events, mirror=True) == trace(events, mirror=False)
    assert full_runs[name].outcome.winner.value == "draw" and full_runs[name].score == (0.5, 0.5)


def test_equal_cost_routes_use_the_side_local_tiebreak_in_a_full_run():
    # N1 -> N2 (via A) and the mirrored N4 -> N5 (via B) are the equal-cost cases R1 is about; the swapped
    # whole-run trace above would diverge if the tie-break depended on the seat.
    from formicmind.match_smoke import _act
    shares_n1 = (0, 65535, 0, 0, 0, 0)
    shares_n2 = (0, 0, 65535, 0, 0, 0)
    mover = {0: _act(shares=shares_n1), 30: _act(shares=shares_n2)}
    ab = _play_scripts(mover, {0: _act(shares=(65535, 0, 0, 0, 0, 0))}, max_ticks=60)
    ba = _play_scripts({0: _act(shares=(65535, 0, 0, 0, 0, 0))}, mover, max_ticks=60)
    assert trace(ab.events, mirror=True) == trace(ba.events, mirror=False)
    departs = [e for e in ab.events if e.kind == "depart" and e.tick == 30 and e.side == 0]
    assert departs and {(e.from_node, e.to_node) for e in departs} == {(N1, A)}   # side-local rank: via home


# =================================================================================================
# 5. Privacy on real mid-game states
# =================================================================================================

def _state_at(name, tick):
    state, run = reset(default_config(), DEV_SEED)
    play(state, run, controllers(SCENARIOS[name]), stop_before_tick=tick)
    return state, run


def _obs_bytes(state, run, side):
    snap = begin_policy_phase(state.clone(), tuple(run.windows))
    return flatten_v2(build_observation(snap, side))


def test_hidden_enemy_state_never_changes_the_observation_bytes_of_a_real_mid_game_state():
    state, run = _state_at("raid_win", 30)
    base = _obs_bytes(state, run, 0)
    b = state.colonies[1]
    twin = state.clone()
    tb = twin.colonies[1]
    tb.food += 77
    tb.upkeep_remainder10 = (tb.upkeep_remainder10 + 3) % 10
    tb.starvation_debt += 2
    tb.next_lay_tick += 7
    tb.lay_order, tb.avoid_center, tb.army_choice = 2, True, 6
    tb.forage_q = [0, 0, 0, 0, 0, 65535]
    tb.queen_hp -= 100                                                   # A has nobody in B's home: unseen
    tb.brood.clear()
    for u in tb.units:
        if u.node in (N4, N5):
            u.hp = max(1, u.hp - 7)                                      # hp of units in nodes A cannot see
    edge = next(u for u in tb.units if u.caste == 0 and u.node == N4)
    transit(edge, N4, C, state.tick + 3)                                 # a unit on an edge, invisible to A
    twin.node_food[N4] = max(0, twin.node_food[N4] - 33)                 # stocks of nodes A does not stand in
    twin.node_food[C] = max(0, twin.node_food[C] - 11)
    assert not any(u.node in (N4, N5, C) for u in state.colonies[0].units)
    assert _obs_bytes(twin, run, 0) == base
    # positive controls: what A *can* see does change the bytes
    seen = state.clone()
    seen.node_food[N1] -= 9
    assert _obs_bytes(seen, run, 0) != base
    own = state.clone()
    own.colonies[0].units[0].hp -= 1
    assert _obs_bytes(own, run, 0) != base
    assert b.queen_hp == 240


def test_the_enemy_queen_hp_is_visible_only_while_a_unit_stands_in_her_home():
    state, run = _state_at("mutual_raid_draw", 30)                        # A's raiders are inside B's home
    seen_a = _obs_bytes(state, run, 0)
    wounded = state.clone()
    wounded.colonies[1].queen_hp -= 20
    assert _obs_bytes(wounded, run, 0) != seen_a                          # sensor present: the new value shows
    blind = state.clone()
    for u in blind.colonies[0].units:
        if u.node == B:
            u.node = N5                                                   # the raiders leave: no sensor in B
    base_blind = _obs_bytes(blind, run, 0)
    blind2 = blind.clone()
    blind2.colonies[1].queen_hp -= 20
    assert _obs_bytes(blind2, run, 0) == base_blind


def test_own_home_is_observed_even_with_no_adults_and_shows_the_raiders():
    state, run = _state_at("mutual_raid_draw", 30)
    state.colonies[0].units.clear()                                       # A has no adult anywhere
    obs = build_observation(begin_policy_phase(state.clone(), tuple(run.windows)), 0)
    assert obs.block("enemy_known")[0] == 1.0 and obs.block("enemy_current")[0] == 1.0
    assert obs.block("enemy_count_last")[1] > 0                           # [workers, soldiers] of home: B's raiders
    assert obs.block("enemy_queen_current")[0] == 0.0                     # B's queen is only a stale memory now


# =================================================================================================
# 6. Invariants where combat and starvation remove units in the same tick, eggs_per_lay 1/2/4
# =================================================================================================

@pytest.mark.parametrize("eggs", [1, 2, 4])
def test_invariants_hold_when_combat_and_starvation_remove_units_in_the_same_ticks(eggs):
    state, run = arena(config=cfg(eggs_per_lay=eggs), tick=1, food=3)    # A is hungry while it raids
    soldiers(state, 0, 20, B)
    workers(state, 1, 4, B)
    soldiers(state, 1, 3, B)
    state.colonies[1].food = 200
    both = 0
    for _ in range(15):
        if run.outcome is not None:
            break
        r = step_tick(state, run, ctrl({0: act(army_choice=6)}, {0: act(lay_order=1)}), check_invariants=True)
        kinds = {e.kind for e in r.events}
        both += ("death" in kinds and "starvation" in kinds)
    assert both >= 1                                                       # the awkward tick really happened
    ids = [u.unit_id for c in state.colonies for u in c.units] + [b.unit_id for c in state.colonies for b in c.brood]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("eggs", [1, 2, 4])
@pytest.mark.parametrize("name", ["raid_win", "mutual_raid_draw", "collapse"])
def test_full_scenarios_pass_check_invariants_for_every_eggs_per_lay(name, eggs):
    config = cfg(eggs_per_lay=eggs)
    r = run_episode(config, DEV_SEED, *controllers(SCENARIOS[name]), max_ticks=400, check_invariants=True)
    assert r.ticks_executed >= 1 and r.final_tick >= 1
    if r.outcome is not None:
        assert r.outcome.terminal_tick <= config.t_max - 1


# =================================================================================================
# 7. Version pins and the CI file
# =================================================================================================

def test_contract_and_format_versions_are_pinned():
    from formicmind import serialization, versions
    assert versions.RULES_VERSION == "l0-proto-FOR1-v0.2.2"
    assert versions.BALANCE_VERSION == "candidate-FOR1-v0.2.1-unvalidated"
    assert versions.SERIALIZATION_VERSION == "formicmind-canonical-json-v1"
    assert versions.HASH_DOMAIN_CHECKPOINT == "formicmind/checkpoint/v3"
    assert (serialization.SCHEMA_CHECKPOINT, serialization.SCHEMA_RUN) == (
        "formicmind.checkpoint/3", "formicmind.run_state/3")
    state, run = reset(default_config(), DEV_SEED)
    assert state.config.rules_version == versions.RULES_VERSION and state.config.t_max == 3000
    doc = _doc(state, run)
    assert doc["schema"] == "formicmind.checkpoint/3" and doc["state"]["rules_version"] == versions.RULES_VERSION


def test_ci_workflow_is_windows_python_312_read_only_and_has_no_linux_job():
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    active = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    # two Windows jobs only: the current tree, and the pinned historical FOR-25 checkout (docs/starvation_debt_verification.md)
    runs_on = [line.strip() for line in active.splitlines() if "runs-on:" in line]
    assert runs_on == ["runs-on: windows-latest"] * 2
    assert active.count("timeout-minutes") == 2 and "historical-for25:" in active
    for forbidden in ("ubuntu", "macos", "matrix", "secrets.", "upload-artifact"):
        assert forbidden not in active, forbidden
    assert 'python-version: "3.12"' in active and "permissions:\n  contents: read" in active
    assert "pip install -e" in active and "python -m pytest" in active and "timeout-minutes" in active
    for trigger in ("push:", "pull_request:", "for-*", "main"):
        assert trigger in active
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.12"' in pyproject
