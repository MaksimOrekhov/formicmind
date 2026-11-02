"""FOR-13 QueenTerminal, economic collapse, timeout, EpisodeOutcome and the checkpoint contract."""

import json

import pytest

from formicmind import checkpoint_from_bytes, checkpoint_hash, checkpoint_to_bytes, state_hash
from formicmind.canonical import CanonicalError, canonical_dumps
from formicmind.events import Phase, events_hash
from formicmind.invariants import InvariantError
from formicmind.match import play
from formicmind.model import BroodRecord
from formicmind.snapshot import WindowStats
from formicmind.terminal import (
    EpisodeOutcome,
    Reason,
    Winner,
    collapsed_sides,
    evaluate_terminal,
    is_collapsed,
    validate_outcome,
)
from formicmind.tick import EpisodeFinishedError, TickMode, step_tick

from combat_helpers import A, B, N1, act, arena, cfg, ctrl, of_kind, soldiers, step, workers
from helpers import transit


def raid_arena(n=4, queen_b=240, queen_a=240, **kw):
    """A holds n soldiers inside B's empty home (orders say raid, so the first Policy keeps them there)."""
    state, run = arena(**kw)
    soldiers(state, 0, n, B)
    state.colonies[1].queen_hp, state.colonies[0].queen_hp = queen_b, queen_a
    return state, run, ctrl({0: act(army_choice=6)})


# --- QueenTerminal -------------------------------------------------------------------------------------

def test_one_queen_dies_the_other_side_wins_in_that_very_tick_after_30_combat_phases():
    state, run, controllers = raid_arena()
    for _ in range(29):
        r = step_tick(state, run, controllers, check_invariants=True)
        assert r.outcome is None and r.terminal is None
    assert state.colonies[1].queen_hp == 8 and state.tick == 29
    last = step_tick(state, run, controllers, check_invariants=True)
    assert last.tick == 29 and last.terminal == "queen_death"
    out = last.outcome
    assert (out.winner, out.reason, out.terminal_tick, out.sides) == (Winner.A, Reason.QUEEN_DEATH, 29, (1,))
    assert (out.terminal_phase, out.phases_executed, out.tick_completed) == (Phase.QUEEN_TERMINAL, 5, False)
    assert run.outcome == out and state.colonies[1].queen_hp == 0


def test_mirrored_raid_gives_the_mirrored_outcome():
    state, run = arena()
    soldiers(state, 1, 4, A)
    out = None
    controllers = ctrl(b={0: act(army_choice=6)})
    while out is None:
        out = step_tick(state, run, controllers).outcome
    assert (out.winner, out.sides, out.terminal_tick) == (Winner.B, (0,), 29)


def test_a_queen_terminal_tick_runs_exactly_five_phases_and_nothing_after_combat_happens():
    state, run, controllers = raid_arena(queen_b=8, tick=1)
    b = state.colonies[1]
    b.lay_order, b.next_lay_tick = 1, 0                            # a lay is due and affordable ...
    stationary, = workers(state, 0, 1, A, target=N1)               # ... a unit would depart in AutomaticDepart
    hungry = state.colonies[0].food = 5                            # ... upkeep would be due ...
    state.node_food[N1] -= 20                                      # ... and regeneration would refill N1
    snapshot = (list(state.node_food), b.food, list(b.brood), state.colonies[0].food)
    in_transit, = soldiers(state, 1, 1, N1)
    transit(in_transit, N1, 3, 1)                                  # arrives in Arrival, would depart again later
    result = step_tick(state, run, controllers, check_invariants=True)
    assert result.terminal == "queen_death" and state.tick == 1    # state.tick is NOT advanced
    assert [r.phase for r in result.phases] == [Phase.POLICY, Phase.COMPLETION, Phase.ARRIVAL, Phase.COMBAT,
                                                Phase.QUEEN_TERMINAL]
    assert result.outcome.phases_executed == len(result.phases) == 5
    assert max(int(e.phase) for e in result.events) == Phase.QUEEN_TERMINAL
    assert not any(e.kind in ("harvest", "upkeep", "starvation", "lay", "regenerate", "depart")
                   for e in result.events if int(e.phase) > Phase.ARRIVAL)
    assert (list(state.node_food), b.food, list(b.brood), state.colonies[0].food) == (
        snapshot[0], snapshot[1], snapshot[2], hungry)
    assert stationary.node == A and run.windows == [WindowStats(), WindowStats()]    # no phantom window tick
    assert result.physics_complete is True and result.mode is TickMode.FULL


def test_both_queens_destroyed_by_one_combat_is_a_draw():
    state, run = arena()
    soldiers(state, 0, 4, B)
    soldiers(state, 1, 4, A)
    state.colonies[0].queen_hp = state.colonies[1].queen_hp = 8
    result = step(state, run, ctrl({0: act(army_choice=6)}, {0: act(army_choice=6)}))
    out = result.outcome
    assert (out.winner, out.reason, out.sides) == (Winner.DRAW, Reason.QUEEN_DEATH, (0, 1))
    assert out.score == (0.5, 0.5)
    assert [e.queen_hp for e in of_kind(result.events, "terminal")] == [(0, 0)]


def test_a_dead_queen_whose_killer_is_economically_collapsed_still_loses():
    state, run = arena(tick=1)
    soldiers(state, 0, 4, B)
    state.colonies[1].queen_hp = 8
    state.colonies[0].food = 0                          # A has no worker and no food: it would collapse ...
    for u in state.colonies[0].units:
        u.target_node = B
    result = step_tick(state, run, ctrl(), check_invariants=True)
    assert is_collapsed(state, 0)
    out = result.outcome
    assert (out.reason, out.winner) == (Reason.QUEEN_DEATH, Winner.A)        # ... but the queen outranks it


def test_no_step_after_a_terminal_outcome_and_nothing_changes():
    state, run, controllers = raid_arena(queen_b=8)
    step_tick(state, run, controllers)
    before = (state_hash(state), checkpoint_hash(state, run), state.tick)
    for kwargs in ({}, {"movement_only": True}, {"partial_economy": True}):
        with pytest.raises(EpisodeFinishedError):
            step_tick(state, run, controllers, **kwargs)
    assert (state_hash(state), checkpoint_hash(state, run), state.tick) == before
    assert play(state, run, controllers) == ([], 0)


# --- economic collapse ----------------------------------------------------------------------------------

def make_colony(state, side, workers_n=0, food=0, brood_worker=False, brood_soldier=False, soldiers_n=0):
    colony = state.colonies[side]
    colony.units.clear()
    colony.food = food
    colony.brood.clear()
    if workers_n:
        workers(state, side, workers_n, side * 6)
    if soldiers_n:
        soldiers(state, side, soldiers_n, side * 6)
    for flag, caste in ((brood_worker, 0), (brood_soldier, 1)):
        if flag:
            colony.brood.append(BroodRecord(state.allocate_unit_id(), caste, 0, 20 if caste == 0 else 30))
    colony.brood.sort(key=lambda b: b.unit_id)


@pytest.mark.parametrize("eggs,threshold", [(1, 10), (2, 20), (4, 40)])
def test_collapse_threshold_is_the_cost_of_one_whole_worker_batch_not_a_hardcoded_10(eggs, threshold):
    state, _ = arena(config=cfg(eggs_per_lay=eggs))
    make_colony(state, 0, food=threshold - 1)
    assert is_collapsed(state, 0)
    make_colony(state, 0, food=threshold)
    assert not is_collapsed(state, 0)                    # a full batch is affordable: it can respawn


def test_four_eggs_per_lay_zero_workers_food_25_without_paid_worker_brood_collapses():
    state, run = arena(config=cfg(eggs_per_lay=4), tick=1)
    make_colony(state, 0, food=25)                       # 25 >= 10 (the old hardcoded value) but < 4 * 10
    make_colony(state, 1, workers_n=12, food=25, soldiers_n=4)
    assert collapsed_sides(state) == (0,)
    result = step_tick(state, run, ctrl(), check_invariants=True)
    out = result.outcome
    assert (out.reason, out.winner, out.sides, out.terminal_phase) == (
        Reason.ECONOMIC_COLLAPSE, Winner.B, (0,), Phase.TERMINAL_LOG)
    assert out.tick_completed and state.tick == 2        # the tick was complete: the clock advanced


def test_a_paid_worker_in_the_brood_prevents_collapse_a_soldier_in_the_brood_does_not():
    state, run = arena(config=cfg(eggs_per_lay=4), tick=1)
    make_colony(state, 0, food=25, brood_worker=True)
    make_colony(state, 1, workers_n=12, food=25)
    assert collapsed_sides(state) == ()
    assert step_tick(state, run, ctrl(), check_invariants=True).outcome is None
    state2, _ = arena(config=cfg(eggs_per_lay=4), tick=1)
    make_colony(state2, 0, food=25, brood_soldier=True)   # a paid soldier cannot become a worker
    assert is_collapsed(state2, 0)


def test_zero_workers_with_food_for_a_whole_batch_is_not_a_collapse_and_present_workers_never_are():
    state, run = arena(config=cfg(eggs_per_lay=4), tick=1)
    make_colony(state, 0, food=40)
    make_colony(state, 1, workers_n=1, food=0)
    assert collapsed_sides(state) == ()
    assert step_tick(state, run, ctrl(), check_invariants=True).outcome is None


def test_both_colonies_collapsing_in_the_same_tick_is_a_draw():
    state, run = arena(tick=1)
    make_colony(state, 0, food=9)
    make_colony(state, 1, food=0)
    out = step(state, run).outcome
    assert (out.reason, out.winner, out.sides) == (Reason.ECONOMIC_COLLAPSE, Winner.DRAW, (0, 1))


def test_collapse_is_evaluated_after_upkeep_and_lay_the_last_worker_starving_ends_it():
    state, run = arena(tick=1)
    make_colony(state, 0, workers_n=1, food=0)           # one worker, no food: upkeep starves it this tick
    make_colony(state, 1, workers_n=12, food=500)
    state.colonies[0].upkeep_remainder10 = 6             # 6 + 4 = 10 -> due 1 of 0 food: debt 1 (< 5)
    out = step(state, run).outcome
    assert out is None                                    # the worker is still alive: no collapse yet
    state.colonies[0].starvation_debt = 4                # the next shortage reaches 5 and kills it
    state.colonies[0].upkeep_remainder10 = 6
    result = step(state, run)
    assert any(e.kind == "starvation" for e in result.events)
    assert result.outcome.reason is Reason.ECONOMIC_COLLAPSE and result.outcome.sides == (0,)


# --- priority queen death > collapse > timeout --------------------------------------------------------

def test_evaluate_terminal_orders_queen_then_collapse_then_timeout():
    state, _ = arena(config=cfg(t_max=10), tick=9)
    make_colony(state, 0, food=0)
    make_colony(state, 1, food=0)
    state.colonies[0].queen_hp = 0
    assert evaluate_terminal(state, 9, Phase.TERMINAL_LOG).reason is Reason.QUEEN_DEATH
    assert evaluate_terminal(state, 9, Phase.QUEEN_TERMINAL).reason is Reason.QUEEN_DEATH
    state.colonies[0].queen_hp = 240
    assert evaluate_terminal(state, 9, Phase.QUEEN_TERMINAL) is None       # collapse waits for Terminal/Log
    out = evaluate_terminal(state, 9, Phase.TERMINAL_LOG)
    assert (out.reason, out.winner) == (Reason.ECONOMIC_COLLAPSE, Winner.DRAW)
    make_colony(state, 0, workers_n=1, food=0)
    make_colony(state, 1, workers_n=1, food=0)
    assert evaluate_terminal(state, 9, Phase.TERMINAL_LOG).reason is Reason.TIMEOUT
    assert evaluate_terminal(state, 8, Phase.TERMINAL_LOG) is None


def test_queen_death_beats_collapse_in_the_same_tick_and_the_loser_is_the_dead_queen():
    state, run = arena(tick=1)
    make_colony(state, 0, food=0)                        # A: collapsed already
    soldiers(state, 1, 4, A)                             # B kills A's queen (8 hp) right now
    state.colonies[0].queen_hp = 8
    state.colonies[1].food = 0                           # B has no workers and no food either: also "collapsed"
    assert collapsed_sides(state) == (0, 1)
    out = step(state, run, ctrl(b={0: act(army_choice=6)})).outcome
    assert (out.reason, out.winner, out.sides) == (Reason.QUEEN_DEATH, Winner.B, (0,))


# --- timeout and the T_max boundary ---------------------------------------------------------------------

def test_timeout_after_the_last_tick_t_max_minus_one_is_a_draw_and_state_tick_reaches_t_max():
    state, run = arena(config=cfg(t_max=5), tick=0)
    workers(state, 0, 1, A)
    workers(state, 1, 1, B)
    controllers = ctrl()
    for t in range(4):
        r = step_tick(state, run, controllers, check_invariants=True)
        assert r.outcome is None and state.tick == t + 1
    last = step_tick(state, run, controllers, check_invariants=True)
    out = last.outcome
    assert (out.reason, out.winner, out.terminal_tick, out.sides) == (Reason.TIMEOUT, Winner.DRAW, 4, ())
    assert (state.tick, out.state_tick, out.score) == (5, 5, (0.5, 0.5))
    assert out.tick_completed and last.terminal == "timeout"
    with pytest.raises(EpisodeFinishedError):
        step_tick(state, run, controllers)
    assert [w.window_ticks for w in run.windows] == [5, 5]            # includes the final tick


def test_default_t_max_3000_ends_exactly_at_tick_2999_with_a_full_tick_and_never_steps_beyond():
    state, run = arena(tick=2999)
    workers(state, 0, 1, A)
    workers(state, 1, 1, B)
    result = step_tick(state, run, ctrl(), check_invariants=True)
    out = result.outcome
    assert (out.reason, out.terminal_tick, out.phases_executed, state.tick) == (Reason.TIMEOUT, 2999, 11, 3000)
    with pytest.raises(EpisodeFinishedError):
        step_tick(state, run, ctrl())
    state2, run2 = arena(tick=3000)                       # a state past the horizon without an outcome
    with pytest.raises(ValueError, match="horizon"):
        step_tick(state2, run2, ctrl())


def test_queen_death_in_the_last_tick_outranks_the_timeout_and_collapse_outranks_it_too():
    state, run = arena(tick=2999)
    soldiers(state, 0, 4, B)
    state.colonies[1].queen_hp = 8
    out = step(state, run, ctrl({0: act(army_choice=6)})).outcome
    assert (out.reason, out.terminal_tick, out.tick_completed) == (Reason.QUEEN_DEATH, 2999, False)
    state, run = arena(tick=2999)
    make_colony(state, 0, food=0)
    workers(state, 1, 1, B)
    out = step(state, run).outcome
    assert (out.reason, out.winner, out.terminal_tick) == (Reason.ECONOMIC_COLLAPSE, Winner.B, 2999)


# --- window stats ---------------------------------------------------------------------------------------

def test_windows_count_every_completed_tick_including_the_final_one_and_no_phantom_tick():
    state, run = arena(config=cfg(t_max=40), tick=30)     # a Policy tick: a fresh window starts at 30
    workers(state, 0, 1, A)
    workers(state, 1, 1, B)
    for _ in range(10):
        r = step_tick(state, run, ctrl())
    assert r.outcome.reason is Reason.TIMEOUT
    assert [w.window_ticks for w in run.windows] == [10, 10]          # ticks 30..39, the last one counted
    state, run, controllers = raid_arena(queen_b=8, tick=31)
    step_tick(state, run, controllers)
    assert [w.window_ticks for w in run.windows] == [0, 0] and state.tick == 31


# --- outcome in the checkpoint ---------------------------------------------------------------------------

def finished_run():
    state, run, controllers = raid_arena(queen_b=8)
    step_tick(state, run, controllers)
    return state, run, controllers


def test_outcome_round_trips_through_the_checkpoint_and_keeps_the_hashes():
    state, run, controllers = finished_run()
    data = checkpoint_to_bytes(state, run)
    state2, run2 = checkpoint_from_bytes(data, state.config)
    assert run2.outcome == run.outcome and run2.outcome is not None
    assert (state_hash(state2), checkpoint_hash(state2, run2)) == (state_hash(state), checkpoint_hash(state, run))
    with pytest.raises(EpisodeFinishedError):
        step_tick(state2, run2, controllers)


def test_a_running_episode_serialises_finished_false_and_state_hash_ignores_the_outcome():
    state, run, controllers = raid_arena()
    doc = json.loads(checkpoint_to_bytes(state, run))
    assert doc["run"]["outcome"] == {"finished": False} and doc["schema"] == "formicmind.checkpoint/3"
    assert doc["run"]["schema"] == "formicmind.run_state/3"
    twin = run.clone()
    twin.outcome = EpisodeOutcome(Winner.DRAW, Reason.TIMEOUT, 0, int(Phase.TERMINAL_LOG), 11, ())
    before = state_hash(state)
    assert checkpoint_hash(state, twin) != checkpoint_hash(state, run)   # the run state differs ...
    assert state_hash(state) == before                                   # ... the mechanics hash does not know it


def test_tampered_or_stale_outcomes_and_old_checkpoint_formats_are_rejected():
    state, run, _ = finished_run()
    doc = json.loads(checkpoint_to_bytes(state, run))

    def load(mutate):
        d = json.loads(json.dumps(doc))
        mutate(d)
        return checkpoint_from_bytes(canonical_dumps(d), state.config)

    for mutate in (
        lambda d: d["run"]["outcome"].update(winner="B"),                 # wrong winner for the dead queen
        lambda d: d["run"]["outcome"].update(terminal_tick=5),            # state.tick disagrees
        lambda d: d["run"]["outcome"].update(reason="timeout"),
        lambda d: d["run"]["outcome"].update(sides=[0]),
        lambda d: d["run"]["outcome"].update(phases_executed=11),
        lambda d: d["run"].update(outcome={"finished": False}),           # queen dead but "still running"
        lambda d: d["run"].update(outcome={"finished": True}),
        lambda d: d["run"]["outcome"].update(extra=1),
        lambda d: d["run"]["outcome"].update(winner="C"),
    ):
        with pytest.raises(CanonicalError):
            load(mutate)
    with pytest.raises(CanonicalError):
        load(lambda d: d.update(schema="formicmind.checkpoint/2"))
    with pytest.raises(CanonicalError):
        load(lambda d: d["run"].update(schema="formicmind.run_state/2"))
    old = json.loads(checkpoint_to_bytes(*arena()))
    old["schema"], old["run"]["schema"] = "formicmind.checkpoint/2", "formicmind.run_state/2"
    del old["run"]["outcome"]
    with pytest.raises(CanonicalError):
        checkpoint_from_bytes(canonical_dumps(old), state.config)


def test_validate_outcome_pins_the_boundary_for_each_reason():
    state, run = arena(config=cfg(t_max=5), tick=0)
    workers(state, 0, 1, A)
    workers(state, 1, 1, B)
    for _ in range(5):
        step_tick(state, run, ctrl())
    validate_outcome(state, run.outcome)
    log = int(Phase.TERMINAL_LOG)
    for bad in (
        EpisodeOutcome(Winner.DRAW, Reason.TIMEOUT, 3, log, 11, ()),                # timeout before the last tick
        EpisodeOutcome(Winner.A, Reason.ECONOMIC_COLLAPSE, 4, log, 11, (1,)),       # nobody is collapsed
        EpisodeOutcome(Winner.A, Reason.QUEEN_DEATH, 4, int(Phase.QUEEN_TERMINAL), 5, (1,)),   # queens alive
        EpisodeOutcome(Winner.DRAW, Reason.TIMEOUT, 4, log, 10, ()),                # phases_executed off by one
        EpisodeOutcome(Winner.B, Reason.TIMEOUT, 4, log, 11, ()),                   # a timeout has no winner
        None,                                                                       # finished state, no outcome
    ):
        if bad is None:
            state_running = state.clone()
            state_running.colonies[0].queen_hp = 0                                   # dead queen, still "running"
            with pytest.raises(InvariantError):
                validate_outcome(state_running, None)
        else:
            with pytest.raises(InvariantError):
                validate_outcome(state, bad)


# --- checkpoint before a deadly fight / before a terminal tick --------------------------------------------

def tail(state, run, controllers):
    events, executed = play(state, run, controllers)
    return events_hash(events), executed, run.outcome, state_hash(state), checkpoint_hash(state, run), state.tick


def test_checkpoint_right_before_the_lethal_combat_restores_the_same_outcome_and_hashes():
    state, run, controllers = raid_arena()
    play(state, run, controllers, stop_before_tick=29)
    assert state.tick == 29 and state.colonies[1].queen_hp == 8 and run.outcome is None
    data = checkpoint_to_bytes(state, run)
    state2, run2 = checkpoint_from_bytes(data, state.config)
    straight = tail(state, run, controllers)
    restored = tail(state2, run2, controllers)
    assert straight == restored
    assert straight[2].terminal_tick == 29 and straight[1] == 1 and straight[5] == 29


def test_checkpoint_before_collapse_and_before_timeout_ticks_restores_the_same_outcome():
    for build in ("collapse", "timeout"):
        if build == "collapse":
            state, run = arena(tick=1)
            make_colony(state, 0, food=9)
            workers(state, 1, 1, B)
        else:
            state, run = arena(config=cfg(t_max=50), tick=49)
            workers(state, 0, 1, A)
            workers(state, 1, 1, B)
        controllers = ctrl()
        state2, run2 = checkpoint_from_bytes(checkpoint_to_bytes(state, run), state.config)
        assert tail(state, run, controllers) == tail(state2, run2, controllers)
        assert run.outcome == run2.outcome and run.outcome is not None


def test_checkpoint_in_the_middle_of_a_raid_matches_an_uninterrupted_run():
    state, run, controllers = raid_arena()
    state_u, run_u, controllers_u = raid_arena()
    play(state, run, controllers, 12)
    state2, run2 = checkpoint_from_bytes(checkpoint_to_bytes(state, run), state.config)
    play(state2, run2, controllers)
    play(state_u, run_u, controllers_u)
    assert (state_hash(state2), checkpoint_hash(state2, run2), run2.outcome) == (
        state_hash(state_u), checkpoint_hash(state_u, run_u), run_u.outcome)


def test_side_swapped_local_policies_give_mirrored_outcomes():
    def play_out(attacker):
        state, run = arena()
        soldiers(state, attacker, 4, B if attacker == 0 else A)
        scripts = [{}, {}]
        scripts[attacker] = {0: act(army_choice=6)}
        events, _ = play(state, run, ctrl(*scripts))
        return state, run.outcome, events

    sa, oa, ea = play_out(0)
    sb, ob, eb = play_out(1)
    assert (oa.winner, ob.winner) == (Winner.A, Winner.B)
    assert (oa.terminal_tick, oa.phases_executed, oa.reason) == (ob.terminal_tick, ob.phases_executed, ob.reason)
    assert oa.sides == (1,) and ob.sides == (0,)
    assert [c.queen_hp for c in sa.colonies] == [c.queen_hp for c in reversed(sb.colonies)]
    assert [e.kind for e in ea] == [e.kind for e in eb]
    assert [(e.damage, e.queen_hp_after) for e in of_kind(ea, "queen_damage")] == [
        (e.damage, e.queen_hp_after) for e in of_kind(eb, "queen_damage")]
