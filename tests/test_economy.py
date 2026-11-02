"""FOR-12 economy and brood: Harvest, Upkeep/Starvation, Lay, Completion/Hatch, Regeneration.

Every expected number below is derived by hand from FOR-1 v0.2.1 (+ errata B-F), not read back from
the engine. Combat does not exist: tests with opposing soldiers mutate the fixture state and never
imitate a fight.
"""

import itertools

import pytest

from formicmind import default_config, reset
from formicmind.economy import (
    complete_hatching,
    harvest,
    lay,
    newborn_worker_target,
    regenerate,
    split_contested,
    starvation_victim_key,
    upkeep,
)
from formicmind.events import Phase
from formicmind.model import BroodRecord
from formicmind.runstate import RunState
from formicmind.snapshot import WindowStats
from formicmind.tick import (
    ECONOMY_IMPLEMENTED_PHASES,
    ECONOMY_NOT_IMPLEMENTED_PHASES,
    PARTIAL_ECONOMY_PHASES,
    IncompleteRulesError,
    TickMode,
    step_tick,
)

from econ_helpers import FORAGE_N1_N2, HOME_Q, Q, act, bare, cfg, new_run, of_kind, tick
from helpers import A, B, C, N1, N2, N4, N5, fresh_state, place, transit, unit_of



# --- mode, phases, reporting -------------------------------------------------------------------

def test_partial_economy_mode_fills_exactly_the_non_combat_non_terminal_phases():
    assert ECONOMY_NOT_IMPLEMENTED_PHASES == (Phase.COMBAT, Phase.QUEEN_TERMINAL)
    assert [p.name for p in ECONOMY_IMPLEMENTED_PHASES] == [
        "POLICY", "COMPLETION", "ARRIVAL", "HARVEST", "UPKEEP", "LAY", "REGENERATION",
        "AUTOMATIC_DEPART", "TERMINAL_LOG"]
    state, run, ctrl = new_run()
    result = step_tick(state, run, ctrl, partial_economy=True)
    assert result.mode is TickMode.PARTIAL_ECONOMY and result.mode.value == "PARTIAL (NO COMBAT/TERMINAL)"
    assert result.missing_phases == (Phase.COMBAT, Phase.QUEEN_TERMINAL)
    assert result.physics_complete is False and result.terminal is None
    assert [r.phase for r in result.phases] == list(Phase)


def test_a_tick_still_refuses_to_run_without_naming_a_development_mode():
    state, run, ctrl = new_run()
    with pytest.raises(IncompleteRulesError) as info:
        step_tick(state, run, ctrl)
    assert "partial_economy" in str(info.value) and "Combat" in str(info.value)
    with pytest.raises(ValueError):
        step_tick(state, run, ctrl, movement_only=True, partial_economy=True)
    assert state.tick == 0


def test_movement_only_mode_is_unchanged_and_runs_no_economy():
    state, run, ctrl = new_run({0: act(lay_order=1, forage_share_q=FORAGE_N1_N2)})
    events = []
    for _ in range(25):
        result = step_tick(state, run, ctrl, movement_only=True)
        assert result.mode is TickMode.MOVEMENT_ONLY and result.physics_complete is False
        events.extend(result.events)
    assert {e.kind for e in events} <= {"order", "retarget", "depart", "arrive"}
    assert [c.food for c in state.colonies] == [60, 60] and state.node_food[N1] == 60
    assert not state.colonies[0].brood and run.windows[0].harvest == 0


def test_phases_run_in_the_specified_order_and_regeneration_follows_harvest_upkeep_and_lay():
    state, run, ctrl = new_run({0: act(lay_order=1, forage_share_q=FORAGE_N1_N2)})
    seen = []

    def spy(phase):
        func = PARTIAL_ECONOMY_PHASES[phase]
        return lambda ctx: (seen.append(phase), func(ctx))

    phases = {p: (spy(p) if f is not None else None) for p, f in PARTIAL_ECONOMY_PHASES.items()}
    step_tick(state, run, ctrl, partial_economy=True, phases=phases)
    assert seen == [Phase.POLICY, Phase.COMPLETION, Phase.ARRIVAL, Phase.HARVEST, Phase.UPKEEP,
                    Phase.LAY, Phase.REGENERATION, Phase.AUTOMATIC_DEPART, Phase.TERMINAL_LOG]


# --- manual golden: first four ticks, lay_order=none (FOR-1 section 10.1) --------------------------

def test_manual_golden_four_ticks_without_laying_and_mirror_symmetry():
    script = {0: act(forage_share_q=FORAGE_N1_N2)}
    state, run, ctrl = new_run(script, script)
    # (food, harvest this tick, N1, N2) after ticks 0..3, derived by hand in FOR-1 v0.2.1 section 10.1
    expected = [(52, 0, 68, 48), (44, 0, 76, 56), (44, 8, 76, 60), (48, 12, 76, 60)]
    for t, (food, got, near, far) in enumerate(expected):
        events = tick(state, run, ctrl)
        assert [c.food for c in state.colonies] == [food, food]
        assert (state.node_food[N1], state.node_food[N2]) == (near, far)
        assert (state.node_food[N4], state.node_food[N5]) == (near, far)
        assert sum(e.harvested for e in of_kind(events, "harvest") if e.side == 0) == got
        assert sum(e.harvested for e in of_kind(events, "harvest") if e.side == 1) == got
        assert state.tick == t + 1
    assert all(not c.brood for c in state.colonies) and len(state.colonies[0].units) == 16
    assert [c.queen_hp for c in state.colonies] == [240, 240]
    assert state.node_food[C] == 120 and state.node_food[A] == state.node_food[B] == 0
    # windows: harvest 0+0+8+12 = 20, upkeep 8 * 4 = 32, no eggs, no debt
    assert run.windows[0] == run.windows[1] == WindowStats(20, 32, 0, 0, 4)


# --- Harvest: individual limit, max_harvest, presence -------------------------------------------------

def test_only_stationed_workers_harvest_and_each_brings_at_most_one_unit_per_tick():
    state = fresh_state(5)
    run = RunState.new(1)
    for uid in range(1, 4):
        place(unit_of(state, 0, uid), N1)             # 3 workers stand in N1
    transit(unit_of(state, 0, 4), A, N1, 6)           # in transit: no harvest
    place(unit_of(state, 0, 13), N1)                  # a soldier does not harvest
    events = harvest(state, run)
    (e,) = events
    assert (e.side, e.node, e.workers, e.demand, e.harvested) == (0, N1, 3, 3, 3)
    assert state.colonies[0].food == 63 and state.node_food[N1] == 57
    assert run.windows[0].harvest == 3 and run.windows[1].harvest == 0


def test_max_harvest_caps_a_colony_per_node_and_homes_give_no_food():
    state = fresh_state(5)
    run = RunState.new(1)
    for uid in range(1, 13):
        place(unit_of(state, 0, uid), N1)             # 12 workers, node cap is 8
    events = harvest(state, run)
    assert [(e.node, e.workers, e.demand, e.harvested) for e in events] == [(N1, 12, 8, 8)]
    assert state.colonies[0].food == 68 and state.node_food[N1] == 52
    # the other 4 workers in the home node A harvest nothing and emit nothing
    state2 = fresh_state(5)
    assert harvest(state2, RunState.new(1)) == [] and state2.colonies[0].food == 60


def test_harvest_never_exceeds_the_physical_stock():
    state = fresh_state(5)
    run = RunState.new(1)
    for uid in range(1, 9):
        place(unit_of(state, 0, uid), N1)
    state.node_food[N1] = 5
    (e,) = harvest(state, run)
    assert (e.demand, e.harvested, e.stock_before, e.stock_after) == (8, 5, 5, 0)
    assert state.node_food[N1] == 0 and state.colonies[0].food == 65
    state.node_food[N1] = 0
    (e,) = harvest(state, run)
    assert e.harvested == 0 and state.colonies[0].food == 65


# --- contested split (errata C) --------------------------------------------------------------------------

@pytest.mark.parametrize("stock, claims, expected", [
    (120, (12, 12), (12, 12)),        # plenty: everybody is paid in full
    (10, (8, 8), (5, 5)),             # 1:1
    (5, (2, 4), (2, 3)),              # 1:2: 5/3 = 1.67 and 10/3 = 3.33 -> the larger fraction gets the unit
    (3, (3, 1), (2, 1)),              # 3:1: 2.25 / 0.75 -> the unit goes to the 0.75 side
    (3, (1, 3), (1, 2)),              # FOR-1 errata C example: 1 + 2, nothing left over
    (3, (2, 2), (1, 1)),              # FOR-1 errata C example: 1 + 1, one unit stays in the node
    (0, (8, 8), (0, 0)),              # empty node
    (7, (7, 0), (7, 0)),              # one claimant only
    (4, (0, 9), (0, 4)),
])
def test_split_contested_examples(stock, claims, expected):
    assert split_contested(stock, claims) == expected
    assert split_contested(stock, claims[::-1]) == expected[::-1]       # no side priority


def test_split_contested_exhaustive_properties():
    for stock, a, b in itertools.product(range(0, 26), range(0, 14), range(0, 14)):
        ga, gb = split_contested(stock, (a, b))
        assert (gb, ga) == split_contested(stock, (b, a))               # side-swap symmetry
        assert 0 <= ga <= a and 0 <= gb <= b and ga + gb <= stock       # never more than claimed or held
        if a + b <= stock:
            assert (ga, gb) == (a, b)
        else:
            assert stock - ga - gb in (0, 1)                             # at most one unit stays
            if stock - ga - gb == 1:
                # only an exact tie of the fractional parts may leave a unit behind
                assert (stock * a) % (a + b) == (stock * b) % (a + b)


def _contested(stock, wa, wb, node=C):
    state = fresh_state(5)
    run = RunState.new(1)
    state.node_food[node] = stock
    for i in range(wa):
        place(unit_of(state, 0, 1 + i), node)
    for i in range(wb):
        place(unit_of(state, 1, 17 + i), node)
    return state, run


@pytest.mark.parametrize("stock, wa, wb, expected_left", [
    (10, 8, 8, 0), (5, 2, 4, 0), (3, 3, 1, 0), (0, 5, 5, 0), (3, 2, 2, 1), (9, 6, 6, 1), (11, 6, 6, 1),
])
def test_contested_node_in_the_engine_matches_the_split_and_does_not_depend_on_the_side(
        stock, wa, wb, expected_left):
    state, run = _contested(stock, wa, wb)
    harvest(state, run)
    got = [state.colonies[0].food - 60, state.colonies[1].food - 60]
    assert tuple(got) == split_contested(stock, (wa, wb))
    assert state.node_food[C] == expected_left
    assert got[0] + got[1] + state.node_food[C] == stock
    # swap the roles: B gets A's workers and vice versa -> mirrored result
    state2, run2 = _contested(stock, wb, wa)
    harvest(state2, run2)
    assert [state2.colonies[1].food - 60, state2.colonies[0].food - 60] == got
    assert state2.node_food[C] == expected_left
    assert [w.harvest for w in run.windows] == got


# --- blocking by enemy soldiers (errata B) ----------------------------------------------------------------

def test_enemy_soldiers_block_harvest_even_when_own_soldiers_are_present():
    state, run = _contested(100, 4, 0)
    place(unit_of(state, 0, 13), C)                   # own soldier does not help
    place(unit_of(state, 1, 29), C)                   # enemy soldier stands in C
    events = harvest(state, run)
    (e,) = events
    assert (e.side, e.blocked, e.claim, e.harvested) == (0, True, 0, 0)
    assert state.node_food[C] == 100 and state.colonies[0].food == 60


def test_a_blocked_side_does_not_compete_and_the_unblocked_one_gets_its_full_demand():
    state, run = _contested(4, 3, 3)                  # claims 3 + 3 would split a stock of 4
    place(unit_of(state, 0, 13), C)                   # only A has soldiers: B is blocked, A is not
    events = harvest(state, run)
    by_side = {e.side: e for e in events}
    assert (by_side[0].blocked, by_side[0].harvested) == (False, 3)
    assert (by_side[1].blocked, by_side[1].claim, by_side[1].harvested) == (True, 0, 0)
    assert state.node_food[C] == 1


def test_soldiers_on_both_sides_block_both_and_enemy_workers_alone_do_not_block():
    state, run = _contested(100, 3, 3)
    place(unit_of(state, 0, 13), C)
    place(unit_of(state, 1, 29), C)
    events = harvest(state, run)
    assert [e.blocked for e in events] == [True, True] and state.node_food[C] == 100
    state2, run2 = _contested(100, 3, 3)              # workers only
    assert [e.blocked for e in harvest(state2, run2)] == [False, False]
    assert state2.node_food[C] == 94


def test_an_enemy_soldier_in_transit_does_not_block():
    state, run = _contested(100, 3, 0)
    transit(unit_of(state, 1, 29), N4, C, 6)
    (e,) = harvest(state, run)
    assert not e.blocked and e.harvested == 3


# --- Regeneration -------------------------------------------------------------------------------------------

def test_regeneration_is_capped_never_negative_and_uses_the_post_harvest_stock():
    state = fresh_state(5)
    state.node_food = [0, 79, 60, 5, 0, 52, 0]
    events = regenerate(state)
    assert state.node_food == [0, 80, 60, 17, 8, 60, 0]
    by_node = {e.node: e for e in events}
    assert sorted(by_node) == [N1, N2, C, N4, N5]                  # homes (capacity 0) are skipped
    assert (by_node[N1].before, by_node[N1].added, by_node[N1].after) == (79, 1, 80)
    assert (by_node[N2].added, by_node[N5].added, by_node[C].added) == (0, 8, 12)
    assert all(0 <= s <= r.capacity for s, r in zip(state.node_food, state.config.balance.resources))


def test_regeneration_in_a_tick_starts_from_what_harvest_left():
    state = fresh_state(5)
    run = RunState.new(1)
    for uid in range(1, 9):
        place(unit_of(state, 0, uid), N1)
    state.node_food[N1] = 79
    harvest(state, run)                                              # 79 - 8 = 71
    (e,) = [e for e in regenerate(state) if e.node == N1]
    assert (e.before, e.after) == (71, 79)


# --- Upkeep in tenths with a carried remainder -------------------------------------------------------------------

def test_start_population_costs_exactly_eight_food_per_tick_and_brood_and_queen_pay_nothing():
    state = fresh_state(5)
    run = RunState.new(1)
    state.colonies[0].brood.append(BroodRecord(33, 0, 0, 20))
    state.colonies[0].brood.append(BroodRecord(34, 1, 0, 30))
    state.next_unit_id = 35
    events = upkeep(state, run)
    by_side = {e.side: e for e in events}
    assert (by_side[0].adult_workers, by_side[0].adult_soldiers) == (12, 4)
    assert (by_side[0].due, by_side[0].paid, by_side[0].remainder_after) == (8, 8, 0)
    assert by_side[0].due == by_side[1].due == 8                       # brood did not change side A's bill
    assert [c.food for c in state.colonies] == [52, 52]
    assert [w.upkeep_paid for w in run.windows] == [8, 8]


def test_fractional_upkeep_carries_a_remainder_across_ticks_without_floats():
    state = fresh_state(0)
    run = RunState.new(1)
    bare(state, 0, workers=1)
    bare(state, 1, workers=0)
    state.colonies[0].food = 100
    dues, rems = [], []
    for _ in range(10):
        (e, _other) = upkeep(state, run)
        dues.append(e.due)
        rems.append(state.colonies[0].upkeep_remainder10)
    # 0.4 food per tick: 4, 8, 12 -> pay 1 (rem 2), 6, 10 -> pay 1 (rem 0), 4, 8, 12 ...
    assert dues == [0, 0, 1, 0, 1, 0, 0, 1, 0, 1]
    assert rems == [4, 8, 2, 6, 0, 4, 8, 2, 6, 0]
    assert state.colonies[0].food == 96 and sum(dues) == 4              # 10 ticks * 0.4 exactly


def test_mixed_castes_pay_four_and_eight_tenths():
    state = fresh_state(0)
    run = RunState.new(1)
    bare(state, 0, workers=3, soldiers=1)                               # 3*4 + 8 = 20 tenths = 2 food
    bare(state, 1, workers=0, soldiers=1)                               # 8 tenths per tick
    state.colonies[0].food = state.colonies[1].food = 10
    first = upkeep(state, run)
    assert [(e.due, e.remainder_after) for e in first] == [(2, 0), (0, 8)]
    second = upkeep(state, run)
    assert [(e.due, e.remainder_after) for e in second] == [(2, 0), (1, 6)]


# --- Starvation (errata D) ------------------------------------------------------------------------------------------

def _starving_side(debt=0, food=0, remainder=0):
    state = fresh_state(5)
    run = RunState.new(1)
    c = state.colonies[0]
    c.food, c.starvation_debt, c.upkeep_remainder10 = food, debt, remainder
    return state, run, c


def test_partial_payment_charges_only_the_missing_part_and_five_unpaid_units_cost_one_adult():
    state, run, c = _starving_side(food=3)                              # due 8, paid 3, missing 5
    events = upkeep(state, run)
    up = of_kind(events, "upkeep")[0]
    assert (up.due, up.paid, up.missing, up.deaths) == (8, 3, 5, 1)
    (death,) = [e for e in of_kind(events, "starvation") if e.side == 0]
    assert (death.caste, death.unit_id, death.debt_before, death.debt_after) == (1, 13, 5, 0)
    assert c.food == 0 and c.starvation_debt == 0 and len(c.units) == 15
    assert run.windows[0].upkeep_paid == 3 and run.windows[0].starvation_debt_delta == 0


def test_debt_seven_kills_one_adult_and_leaves_two():
    state, run, c = _starving_side(debt=4, food=5)                      # due 8, paid 5, missing 3 -> 7
    events = upkeep(state, run)
    assert [e.unit_id for e in of_kind(events, "starvation") if e.side == 0] == [13]
    assert c.starvation_debt == 2 and run.windows[0].starvation_debt_delta == -2   # 4 -> 2
    assert len(c.units) == 15


def test_several_deaths_in_one_tick_follow_the_victim_order_and_pay_the_debt_after_each():
    state, run, c = _starving_side(debt=14)                             # + missing 8 = 22 -> four deaths
    transit(unit_of(state, 0, 13), A, N1, 6)                            # soldier in transit
    place(unit_of(state, 0, 14), N1)                                    # soldier in a node
    # soldiers 15 and 16 stay at home; workers are never needed
    events = upkeep(state, run)
    deaths = [e for e in of_kind(events, "starvation") if e.side == 0]
    assert [e.unit_id for e in deaths] == [15, 16, 14, 13]              # home, home, node, transit
    assert [(e.seq, e.debt_before, e.debt_after) for e in deaths] == [(0, 22, 17), (1, 17, 12), (2, 12, 7), (3, 7, 2)]
    assert c.starvation_debt == 2 and [u.caste for u in c.units] == [0] * 12
    assert of_kind(events, "upkeep")[0].deaths == 4
    assert run.windows[0].starvation_debt_delta == 2 - 14               # signed net change
    # next victims are workers: home first, then by unit id
    assert starvation_victim_key(A, unit_of(state, 0, 3)) < starvation_victim_key(A, unit_of(state, 0, 4)) \
        < (2, 0, 0)


def test_workers_are_killed_only_after_all_soldiers_and_home_workers_before_remote_ones():
    state, run, c = _starving_side(debt=60)                             # 60 + 8 owed = 68 -> 13 deaths in one tick: 4 S + 9 W
    transit(unit_of(state, 0, 1), A, N1, 6)
    place(unit_of(state, 0, 2), N1)
    events = upkeep(state, run)
    victims = [e.unit_id for e in of_kind(events, "starvation") if e.side == 0]
    assert victims[:4] == [13, 14, 15, 16]
    # workers 3..12 stand at home; 2 is in a node, 1 in transit -> the order would be 3..12, 2, 1
    assert victims[4:] == [3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert len(victims) == 13 and c.starvation_debt == 60 + 8 - 65      # floor(68 / 5) = 13 deaths


def test_a_fully_paid_upkeep_clears_the_leftover_debt():
    state, run, c = _starving_side(debt=4, food=20)
    up = of_kind(upkeep(state, run), "upkeep")[0]
    assert (up.missing, up.debt_before, up.debt_after, up.debt_reset) == (0, 4, 0, True)
    assert c.starvation_debt == 0 and run.windows[0].starvation_debt_delta == -4


def test_a_tick_with_nothing_due_neither_charges_nor_clears_the_debt():
    state, run, c = _starving_side(debt=3, food=0)
    bare(state, 0, workers=1)                                           # 4 tenths: due 0 this tick
    up = of_kind(upkeep(state, run), "upkeep")[0]
    assert (up.due, up.missing, up.debt_after, up.debt_reset) == (0, 0, 3, False)
    assert c.starvation_debt == 3 and c.upkeep_remainder10 == 4


def test_debt_after_the_last_adults_die_does_not_loop_and_survives():
    state, run, c = _starving_side(debt=100, remainder=9)
    bare(state, 0, workers=1)                                           # due 1, unpaid -> 101, one death -> 96
    events = upkeep(state, run)
    assert len(of_kind(events, "starvation")) == 1 and c.units == []
    assert c.starvation_debt == 96
    again = upkeep(state, run)                                          # nobody left: nothing due, no crash
    assert of_kind(again, "starvation") == [] and c.starvation_debt == 96


def test_starvation_is_the_only_way_an_adult_disappears_and_does_not_touch_brood():
    state, run, c = _starving_side(debt=0, food=0)
    c.brood.append(BroodRecord(33, 0, 0, 20))
    state.next_unit_id = 34
    upkeep(state, run)
    assert [b.unit_id for b in c.brood] == [33]
