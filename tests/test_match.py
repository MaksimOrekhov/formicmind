"""FOR-13 run_episode and the full-match smoke scenarios (functional checks, not balance results)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from formicmind import checkpoint_from_bytes, checkpoint_hash, checkpoint_to_bytes, default_config, state_hash
from formicmind.decoder import PolicyLogitsV2
from formicmind.match import EpisodeResult, play, run_episode, summarize
from formicmind.match_smoke import N1_N2, SCENARIOS, run_match_smoke
from formicmind.rng import policy_stream_id
from formicmind.tick import PolicyController, ScriptedController, step_tick

from combat_helpers import act as _act


def act(**kw):
    """Economy orders like the smoke scenarios: 8 workers on N1, 4 on N2 (the helper default keeps them home)."""
    kw.setdefault("forage_share_q", N1_N2)
    return _act(**kw)

SRC = Path(__file__).resolve().parents[1] / "src"

# Regression pins of the four scripted scenarios on the default candidate balance (seed 20261102). They are
# produced by this implementation; a diff means the rules or the event/hash contract changed on purpose.
PINNED = {
    "raid_win": dict(
        winner="A", reason="queen_death", terminal_tick=175, final_tick=175, queen_hp=(240, 0), food=(122, 752),
        workers=(18, 11), soldiers=(12, 0), brood=(0, 0),
        events_hash="2f6547ef1653a3c95b8cf3bad376d5e2aaba6cff00ea1c6cab7b91a8d61708a6",
        state_hash="3fafbc8406661d3b66c3a895fffaf6e6ea29bcc90c56628600495be6aad32526",
        checkpoint_hash="746208b8e9d9d3d1fc19260a863149a4899d79b650f0d7e666c03b58af0829da"),
    "mutual_raid_draw": dict(
        winner="draw", reason="queen_death", terminal_tick=39, final_tick=39, queen_hp=(0, 0), food=(184, 184),
        workers=(12, 12), soldiers=(4, 4), brood=(0, 0),
        events_hash="93b401faab189be2c6ab6d851614c8ce19c8ef7abb830e156e44449d87c1805b",
        state_hash="55bcde4380745b4bad8595698ff6d19c86261b747eefd7dfcd15676d62e2dba0",
        checkpoint_hash="c455dccfe16c56b713bb3242f2120d0b367d8b504780897ee2a5b231e5ad71fa"),
    "collapse": dict(
        winner="B", reason="economic_collapse", terminal_tick=46, final_tick=47, queen_hp=(240, 240), food=(0, 220),
        workers=(0, 12), soldiers=(0, 4), brood=(0, 0),
        events_hash="d7944c1025eb64ddc8270315b8889221a677f05dd76a0c55805b7a7965c45bc8",
        state_hash="ab7744c86ef99ffc623eb1e39753124f7c4b31ae41ea01f373d4c6bf296af87a",
        checkpoint_hash="46631e5393e5b5b3a8c31003bf868130195167e4bd04315b071d13d06e81aab3"),
    "timeout_draw": dict(
        winner="draw", reason="timeout", terminal_tick=2999, final_tick=3000, queen_hp=(240, 240),
        food=(10169, 10169), workers=(21, 21), soldiers=(4, 4), brood=(0, 0),
        events_hash="ee1fa9cf817a8b2a30a0b0c6f6c6f20ebbe6efda0d276837cd42ab0f89331933",
        state_hash="8baea6d0ea6282cad03a33c375010faf5420257316eb00ad6f5f3ccb3b88189c",
        checkpoint_hash="6bb95f36c52aa4a4643e99934defb4af5ebf517dd0394446240ddb9ea6c2b4d1"),
}


@pytest.fixture(scope="module")
def reports():
    return {name: run_match_smoke(name) for name in SCENARIOS}      # check_invariants=True on every tick


@pytest.mark.parametrize("name", list(PINNED))
def test_smoke_scenario_outcome_and_hashes_are_pinned(reports, name):
    r = reports[name].result
    pin = PINNED[name]
    assert (r.winner, r.reason, r.terminal_tick, r.final_tick) == (
        pin["winner"], pin["reason"], pin["terminal_tick"], pin["final_tick"])
    assert (r.queen_hp, r.food, r.workers, r.soldiers, r.brood) == (
        pin["queen_hp"], pin["food"], pin["workers"], pin["soldiers"], pin["brood"])
    assert (r.events_hash, r.state_hash, r.checkpoint_hash) == (
        pin["events_hash"], pin["state_hash"], pin["checkpoint_hash"])
    assert r.truncated is False and r.outcome is not None


def test_the_smoke_set_has_a_decisive_combat_a_timeout_draw_a_simultaneous_double_kill_and_a_collapse(reports):
    kinds = {n: (rep.result.winner, rep.result.reason) for n, rep in reports.items()}
    assert kinds == {"raid_win": ("A", "queen_death"), "mutual_raid_draw": ("draw", "queen_death"),
                     "collapse": ("B", "economic_collapse"), "timeout_draw": ("draw", "timeout")}
    raid = reports["raid_win"].result
    counts = dict(raid.event_counts)
    assert counts["combat"] > 0 and counts["death"] > 0 and counts["queen_damage"] == 12 and counts["terminal"] == 1
    assert raid.score == (1.0, 0.0) and reports["timeout_draw"].result.score == (0.5, 0.5)
    mutual = reports["mutual_raid_draw"].result
    assert mutual.terminal_tick == 39 and mutual.queen_hp == (0, 0)      # 30 combat phases, t = 10..39
    assert reports["timeout_draw"].result.terminal_tick == default_config().t_max - 1


def test_every_smoke_ends_with_exactly_one_terminal_event_last_in_the_log_and_all_ticks_are_ordered(reports):
    for report in reports.values():
        events = report.result.events
        terminals = [e for e in events if e.kind == "terminal"]
        assert len(terminals) == 1 and events[-1] is terminals[0]
        assert [e.tick for e in events] == sorted(e.tick for e in events)
        assert all(e.tick <= terminals[0].tick for e in events)
        assert terminals[0].tick == report.result.terminal_tick


def test_smoke_reports_render_and_say_it_is_only_a_functional_check(reports, capsys):
    from formicmind.match_smoke import main
    main(["raid_win"])
    out = capsys.readouterr().out
    assert "FUNCTIONAL CHECK" in out and "raid_win" in out and "TERMINAL t=175: queen_death" in out
    assert "combat A at B -> soldiers" in out and "queen B: hit t=164..175" in out


def test_smoke_matches_are_reproducible_in_process():
    again = run_match_smoke("mutual_raid_draw")
    assert again.result.events_hash == PINNED["mutual_raid_draw"]["events_hash"]


# --- run_episode contract -----------------------------------------------------------------------------

def test_run_episode_returns_a_structured_result():
    r = run_episode(default_config(), 1, ScriptedController({0: act(army_choice=6, avoid_center=True)}),
                    ScriptedController({0: act(army_choice=6, avoid_center=True)}), check_invariants=True)
    assert isinstance(r, EpisodeResult)
    assert (r.winner, r.reason, r.terminal_tick, r.ticks_executed) == ("draw", "queen_death", 39, 40)
    assert r.final_tick == 39 and r.score == (0.5, 0.5) and r.outcome.sides == (0, 1)
    assert len(r.config_hash) == len(r.events_hash) == len(r.state_hash) == len(r.checkpoint_hash) == 64
    assert dict(r.event_counts)["terminal"] == 1


def test_max_ticks_is_a_harness_cutoff_not_an_outcome_and_never_exceeds_t_max():
    quiet = (ScriptedController({}), ScriptedController({}))
    r = run_episode(default_config(), 1, *quiet, max_ticks=10)
    assert r.truncated and r.outcome is None and r.score is None and r.winner is None
    assert (r.ticks_executed, r.final_tick) == (10, 10)
    with pytest.raises(ValueError):
        run_episode(default_config(), 1, *quiet, max_ticks=3001)
    with pytest.raises(ValueError):
        run_episode(default_config(), 1, *quiet, max_ticks=-1)


def test_a_match_always_terminates_by_t_max_even_with_passive_idle_controllers():
    r = run_episode(default_config(), 1, ScriptedController({}), ScriptedController({}))
    assert r.outcome is not None and r.final_tick <= 3000 and r.ticks_executed <= 3000
    assert r.reason in ("timeout", "economic_collapse")              # idle colonies starve; never infinite


def test_run_episode_continues_from_a_checkpoint_to_the_same_result():
    controllers = (ScriptedController({0: act(army_choice=6, avoid_center=True)}),
                   ScriptedController({0: act(army_choice=6, avoid_center=True)}))
    full = run_episode(default_config(), 5, *controllers)
    from formicmind.reset import reset
    state, run = reset(default_config(), 5)
    head, _ = play(state, run, controllers, 25)
    state2, run2 = checkpoint_from_bytes(checkpoint_to_bytes(state, run), state.config)
    tail_a, _ = play(state, run, controllers)
    tail_b, _ = play(state2, run2, controllers)
    assert summarize(state2, run2, tail_b, 0).events_hash == summarize(state, run, tail_a, 0).events_hash
    assert (state_hash(state2), checkpoint_hash(state2, run2)) == (full.state_hash, full.checkpoint_hash)
    assert len(head) + len(tail_b) == len(full.events)


def test_a_controller_failure_is_a_technical_error_not_a_loss_and_leaves_the_run_unfinished():
    class Boom:
        def act(self, tick, observation, run):
            if tick >= 30:
                raise RuntimeError("controller crashed")
            return ScriptedController({}).act(tick, observation, run)

    from formicmind.reset import reset
    state, run = reset(default_config(), 3)
    with pytest.raises(RuntimeError, match="crashed"):
        play(state, run, (Boom(), ScriptedController({})))
    assert run.outcome is None and state.tick == 30                    # tick 30 rolled back, nobody "lost"
    before = (state_hash(state), checkpoint_hash(state, run))
    with pytest.raises(RuntimeError):
        step_tick(state, run, (Boom(), ScriptedController({})))
    assert (state_hash(state), checkpoint_hash(state, run)) == before


def test_an_invalid_controller_answer_is_a_technical_error_never_a_game_outcome():
    class Forger:
        def act(self, tick, observation, run):
            return "raid everything"                                   # not a ValidatedActionV2

    from formicmind.reset import reset
    state, run = reset(default_config(), 3)
    before = (state_hash(state), checkpoint_hash(state, run))
    with pytest.raises(TypeError):
        play(state, run, (Forger(), ScriptedController({})))
    assert run.outcome is None and state.tick == 0
    assert (state_hash(state), checkpoint_hash(state, run)) == before


def deterministic_policy(observation, mask, policy_rng):
    return PolicyLogitsV2([0.0] * 17)


def test_rng_streams_belong_to_policies_not_seats_and_a_shared_stream_is_refused():
    pa = PolicyController(deterministic_policy, policy_stream_id("alpha", 0))
    pb = PolicyController(deterministic_policy, policy_stream_id("beta", 0))
    r = run_episode(default_config(), 9, pa, pb)
    assert r.outcome is not None and r.winner == "draw"               # the do-nothing policy starves both sides
    with pytest.raises(ValueError, match="shared"):
        run_episode(default_config(), 9, pa, PolicyController(deterministic_policy, pa.stream_id), max_ticks=1)
    # swapping the seats keeps each policy's stream (role-keyed): the run state is identical either way
    r_swap = run_episode(default_config(), 9, pb, pa)
    assert (r.winner, r.reason, r.terminal_tick, r.food, r.workers) == (
        r_swap.winner, r_swap.reason, r_swap.terminal_tick, r_swap.food, r_swap.workers)


def test_full_matches_consume_no_rng_the_world_stream_is_untouched():
    from formicmind.reset import reset
    state, run = reset(default_config(), 11, [policy_stream_id("alpha", 0)])
    before = {sid: s.to_canonical() for sid, s in run.streams.items()}
    controllers = (ScriptedController({0: act(army_choice=6, avoid_center=True)}),) * 2
    play(state, run, controllers)
    assert {sid: s.to_canonical() for sid, s in run.streams.items()} == before


def test_side_swapped_scripted_matches_have_mirrored_summaries():
    raid = {0: act(lay_order=1), 60: act(lay_order=2), 150: act(army_choice=6)}
    calm = {0: act()}
    ab = run_episode(default_config(), 1, ScriptedController(raid), ScriptedController(calm))
    ba = run_episode(default_config(), 1, ScriptedController(calm), ScriptedController(raid))
    assert (ab.winner, ba.winner) == ("A", "B")
    assert (ab.reason, ab.terminal_tick, ab.final_tick) == (ba.reason, ba.terminal_tick, ba.final_tick)
    for field in ("queen_hp", "food", "workers", "soldiers", "brood"):
        assert getattr(ab, field) == tuple(reversed(getattr(ba, field))), field
    swapped = {k: v for k, v in dict(ba.event_counts).items()}
    assert dict(ab.event_counts) == swapped


# --- PYTHONHASHSEED --------------------------------------------------------------------------------------

SCRIPT = f"""
import sys; sys.path.insert(0, {str(SRC)!r})
from formicmind.match_smoke import run_match_smoke
r = run_match_smoke('raid_win').result
print(r.events_hash, r.state_hash, r.checkpoint_hash)
"""


def _run(hashseed):
    proc = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, check=False,
                          env={**os.environ, "PYTHONHASHSEED": hashseed})
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


@pytest.mark.parametrize("other", ["1", "4294967295", "random"])
def test_full_match_hashes_do_not_depend_on_pythonhashseed(other):
    expected = (f"{PINNED['raid_win']['events_hash']} {PINNED['raid_win']['state_hash']} "
                f"{PINNED['raid_win']['checkpoint_hash']}")
    assert _run("0") == expected == _run(other)
