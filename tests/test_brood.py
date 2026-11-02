"""FOR-12 Lay / Completion (hatch): whole paid batches, cooldown, ids, errata E accounting."""

import pytest

from formicmind import default_config, reset
from formicmind.economy import complete_hatching, lay, newborn_worker_target
from formicmind.events import Phase
from formicmind.model import BroodRecord

from econ_helpers import FORAGE_N1_N2, Q, act, bare, cfg, new_run, of_kind, tick
from helpers import A, B, N1, N2, N4, unit_of


# --- Lay: whole paid batches, cooldown, queen, no-lay ------------------------------------------------------

def _lay_state(config=None, order=1, tick=0, food=None):
    state, run = reset(config or default_config(), 1)
    state.tick = tick
    c = state.colonies[0]
    c.lay_order = order
    if food is not None:
        c.food = food
    return state, run, c


def test_lay_none_does_nothing_and_charges_nothing():
    state, run, c = _lay_state(order=0)
    assert lay(state, run) == [] and c.food == 60 and c.brood == [] and c.next_lay_tick == 0
    assert run.windows[0].egg_spend == 0 and state.next_unit_id == 33


def test_cooldown_and_a_dead_queen_prevent_laying_without_any_charge():
    state, run, c = _lay_state(tick=5)
    c.next_lay_tick = 10
    assert lay(state, run) == [] and c.food == 60 and c.brood == []
    state, run, c = _lay_state()
    c.queen_hp = 0
    assert lay(state, run) == [] and c.food == 60 and c.brood == []


@pytest.mark.parametrize("eggs, caste, cost, delay", [
    (1, 0, 10, 20), (2, 0, 20, 20), (4, 0, 40, 20), (1, 1, 20, 30), (2, 1, 40, 30), (4, 1, 80, 30),
])
def test_one_batch_costs_eggs_times_unit_cost_once_and_hatches_together(eggs, caste, cost, delay):
    state, run, c = _lay_state(cfg(eggs_per_lay=eggs), order=1 + caste, tick=7, food=100)
    (e,) = lay(state, run)
    assert (e.laid, e.eggs, e.total_cost, e.food_before, e.food_after) == (True, eggs, cost, 100, 100 - cost)
    assert c.food == 100 - cost and run.windows[0].egg_spend == cost
    assert [(b.unit_id, b.caste, b.laid_tick, b.hatch_tick) for b in c.brood] == [
        (33 + i, caste, 7, 7 + delay) for i in range(eggs)]
    assert e.unit_ids == tuple(33 + i for i in range(eggs)) and e.hatch_tick == 7 + delay
    assert c.next_lay_tick == 17 and state.next_unit_id == 33 + eggs      # cooldown 10 ticks
    assert lay(state, run) == [] and c.food == 100 - cost                 # same tick again: cooling down


def test_a_batch_that_cannot_be_paid_in_full_lays_nothing_and_keeps_the_right_to_lay():
    state, run, c = _lay_state(cfg(eggs_per_lay=4), order=1, food=39)     # 4 * 10 = 40 needed
    (e,) = lay(state, run)
    assert (e.laid, e.total_cost, e.food_before, e.food_after, e.unit_ids) == (False, 40, 39, 39, ())
    assert c.food == 39 and c.brood == [] and c.next_lay_tick == 0 and state.next_unit_id == 33
    assert run.windows[0].egg_spend == 0
    c.food = 40                                                           # the next tick it can pay
    (e,) = lay(state, run)
    assert e.laid and c.food == 0 and len(c.brood) == 4 and c.next_lay_tick == 10


def test_a_soldier_batch_is_priced_per_soldier_not_per_worker():
    state, run, c = _lay_state(cfg(eggs_per_lay=2), order=2, food=39)     # 2 * 20 = 40 needed
    assert not lay(state, run)[0].laid
    c.food = 40
    assert lay(state, run)[0].laid and c.food == 0


def test_both_colonies_lay_in_the_same_tick_with_ids_allocated_in_side_order():
    state, run = reset(cfg(eggs_per_lay=2), 1)
    for c in state.colonies:
        c.lay_order = 1
    a, b = lay(state, run)
    assert (a.side, a.unit_ids, b.side, b.unit_ids) == (0, (33, 34), 1, (35, 36))


# --- Completion / hatch ------------------------------------------------------------------------------------

def _hatch_state(tick=20):
    state, run = reset(default_config(), 1)
    state.tick = tick
    return state, run


def test_hatching_keeps_the_id_gives_full_hp_at_home_and_removes_exactly_the_ripe_brood():
    state, run = _hatch_state()
    c = state.colonies[0]
    c.brood[:] = [BroodRecord(33, 0, 0, 20), BroodRecord(34, 1, 0, 30), BroodRecord(35, 0, 10, 30)]
    state.next_unit_id = 36
    events = complete_hatching(state)
    assert [(e.unit_id, e.caste, e.node, e.hp, e.laid_tick) for e in events] == [(33, 0, A, 20, 0)]
    assert [b.unit_id for b in c.brood] == [34, 35]
    new = unit_of(state, 0, 33)
    assert (new.caste, new.hp, new.node, new.from_node) == (0, 20, A, -1)
    assert [u.unit_id for u in c.units] == sorted(u.unit_id for u in c.units)
    assert len(c.units) == 17 and state.next_unit_id == 36               # ids are not reallocated


def test_a_new_soldier_takes_the_standing_army_order_on_both_sides():
    state, run = _hatch_state(30)
    state.colonies[0].army_choice = 1                                     # defend local N1
    state.colonies[1].army_choice = 1                                     # B defends its own N4
    state.colonies[0].brood[:] = [BroodRecord(33, 1, 0, 30)]
    state.colonies[1].brood[:] = [BroodRecord(34, 1, 0, 30)]
    state.next_unit_id = 35
    events = complete_hatching(state)
    assert [(e.side, e.unit_id, e.node, e.target_node, e.hp) for e in events] == [
        (0, 33, A, N1, 30), (1, 34, B, N4, 30)]


def test_a_missed_hatch_tick_is_an_error_and_changes_nothing():
    state, run = _hatch_state(25)
    state.colonies[0].brood[:] = [BroodRecord(33, 0, 0, 20)]
    state.next_unit_id = 34
    with pytest.raises(ValueError):
        complete_hatching(state)
    assert len(state.colonies[0].units) == 16 and len(state.colonies[0].brood) == 1


def _assigned(state, side=0):
    c = state.colonies[side]
    for i, u in enumerate(u for u in c.units if u.caste == 0):
        u.target_node = (N1 if side == 0 else N4) if i < 8 else (N2 if side == 0 else N4 + 1)
    c.forage_q[:] = list(FORAGE_N1_N2)


def test_new_workers_fill_the_largest_deficit_of_the_current_forage_weights():
    state, run = _hatch_state()
    _assigned(state)                                                      # 8 on N1, 4 on N2
    state.colonies[0].brood[:] = [BroodRecord(33, 0, 0, 20), BroodRecord(34, 0, 0, 20)]
    state.next_unit_id = 35
    assert newborn_worker_target(state, state.colonies[0]) == N1          # quotas for 13: 9 / 4
    events = complete_hatching(state)
    # the 13th worker -> N1 (9/4); the 14th sees 9/4 and quotas 9/5 for 14 workers -> N2
    assert [(e.unit_id, e.target_node) for e in events] == [(33, N1), (34, N2)]


def test_equal_deficits_break_ties_in_the_side_local_order_so_the_rule_is_mirror_symmetric():
    state, run = _hatch_state()
    for side in (0, 1):
        bare(state, side, workers=1)
        c = state.colonies[side]
        c.forage_q[:] = [0, 32768, 0, 32767, 0, 0]                        # local N1 and local C, about half each
        c.brood[:] = [BroodRecord(33 + side, 0, 0, 20)]
        assert c.units[0].target_node == (A if side == 0 else B)          # the lone worker is at "home"
    state.next_unit_id = 35
    # quotas for 2 workers: N1 = 1, C = 1 -> deficits N1 +1, C +1, home -1: a tie between N1 and C
    events = complete_hatching(state)
    assert [(e.side, e.target_node) for e in events] == [(0, N1), (1, N4)]   # local N1 for both; world-id would give C to B


def test_malformed_forage_weights_leave_the_new_worker_at_home():
    state, run = _hatch_state()
    state.colonies[0].forage_q[:] = [0, 0, 0, 0, 0, 0]
    assert newborn_worker_target(state, state.colonies[0]) == A
    state.colonies[1].forage_q[:] = [-1, Q + 1, 0, 0, 0, 0]
    assert newborn_worker_target(state, state.colonies[1]) == B


def test_a_hatch_on_a_policy_tick_uses_the_orders_just_issued_and_leaves_in_automatic_depart():
    state, run, ctrl = new_run({30: act(forage_share_q=FORAGE_N1_N2)})
    state.tick = 30
    state.colonies[0].brood[:] = [BroodRecord(33, 0, 10, 30)]
    state.next_unit_id = 34
    events = tick(state, run, ctrl)
    (hatch,) = of_kind(events, "hatch")
    assert (hatch.tick, hatch.phase, hatch.target_node) == (30, Phase.COMPLETION, N1)
    (dep,) = [e for e in of_kind(events, "depart") if e.unit_id == 33]
    assert (dep.phase, dep.from_node, dep.to_node, dep.arrival_tick) == (Phase.AUTOMATIC_DEPART, A, N1, 32)
    assert not [e for e in of_kind(events, "retarget") if e.unit_id == 33]   # the policy phase never saw it
    assert unit_of(state, 0, 33).node == -1                                  # already on its way


# --- laid@0 -> hatch@20 / @30, same-tick accounting -----------------------------------------------------------

def test_worker_laid_at_0_hatches_at_20_pays_upkeep_that_tick_and_leaves_for_its_node_at_once():
    state, run, ctrl = new_run({0: act(lay_order=1, forage_share_q=FORAGE_N1_N2)},
                               {0: act(lay_order=2, forage_share_q=FORAGE_N1_N2)})
    events = tick(state, run, ctrl, 21)                                       # ticks 0..20
    laid = [e for e in of_kind(events, "lay") if e.side == 0 and e.laid]
    assert [(e.tick, e.unit_ids, e.hatch_tick) for e in laid] == [(0, (33,), 20), (10, (35,), 30), (20, (37,), 40)]
    (hatch,) = [e for e in of_kind(events, "hatch") if e.side == 0]
    assert (hatch.tick, hatch.unit_id, hatch.caste, hatch.node, hatch.hp) == (20, 33, 0, A, 20)
    up = {e.tick: e for e in of_kind(events, "upkeep") if e.side == 0}
    assert (up[19].adult_workers, up[20].adult_workers) == (12, 13)           # the newborn pays in its hatch tick
    (dep,) = [e for e in of_kind(events, "depart") if e.unit_id == 33]
    assert (dep.tick, dep.phase, dep.from_node, dep.to_node, dep.arrival_tick) == (
        20, Phase.AUTOMATIC_DEPART, A, N1, 22)
    assert not [e for e in of_kind(events, "arrive") if e.unit_id == 33]
    more = tick(state, run, ctrl, 2)                                          # ticks 21, 22
    assert [(e.tick, e.node) for e in of_kind(more, "arrive") if e.unit_id == 33] == [(22, N1)]
    assert unit_of(state, 0, 33).node == N1


def test_each_egg_is_paid_exactly_once_and_hatching_costs_nothing():
    state, run, ctrl = new_run({0: act(lay_order=1, forage_share_q=FORAGE_N1_N2)})
    events = tick(state, run, ctrl, 41)
    batches = [e for e in of_kind(events, "lay") if e.side == 0 and e.laid]
    assert len(batches) == 5                                                  # t = 0, 10, 20, 30, 40
    assert sum(e.total_cost for e in batches) == 50
    assert run.windows[0].egg_spend == 20                                     # window since the t=30 policy tick: lays at 30, 40
    for h in of_kind(events, "hatch"):
        up = [e for e in of_kind(events, "upkeep") if e.tick == h.tick and e.side == h.side][0]
        assert up.food_after == up.food_before - up.paid                      # hatching itself changed no food


def test_soldier_laid_at_0_hatches_at_30_and_a_hatch_and_a_lay_can_share_a_tick():
    script = {0: act(lay_order=2, forage_share_q=FORAGE_N1_N2)}
    state, run, ctrl = new_run(script, script)
    events = tick(state, run, ctrl, 31)
    first = [e for e in of_kind(events, "lay") if e.side == 0 and e.laid][0]
    (hatch,) = [e for e in of_kind(events, "hatch") if e.side == 0]
    assert (first.tick, first.hatch_tick, hatch.tick, hatch.caste, hatch.hp, hatch.unit_id) == (0, 30, 30, 1, 30, 33)
    same = [e for e in events if e.tick == 30 and e.side == 0 and e.kind in ("hatch", "lay")]
    assert [e.kind for e in same] == ["hatch", "lay"]                          # Completion before Lay
    assert [e.phase for e in same] == [Phase.COMPLETION, Phase.LAY]
    brood = state.colonies[0].brood
    assert 33 not in [b.unit_id for b in brood] and len(brood) == 3            # the eggs of ticks 10, 20 and 30


def test_there_is_no_population_or_brood_cap():
    state, run, ctrl = new_run({0: act(lay_order=1)}, None, config=cfg(eggs_per_lay=8))
    state.colonies[0].food = 10 ** 9
    tick(state, run, ctrl, 200)
    c = state.colonies[0]
    workers = sum(1 for u in c.units if u.caste == 0)
    assert workers == 12 + 8 * 18 == 156                                       # 18 batches hatched by tick 199
    assert len(c.brood) == 16 and workers + 4 > 120                            # far above the old cap of 120
    assert c.starvation_debt == 0


def test_a_colony_that_cannot_pay_for_a_batch_keeps_its_cooldown():
    state, run, ctrl = new_run({0: act(lay_order=2)})
    state.colonies[0].food = 25                                               # upkeep 8 -> 17 < 20: no soldier egg
    events = tick(state, run, ctrl, 1)
    (ev,) = [e for e in of_kind(events, "lay") if e.side == 0]
    assert not ev.laid and state.colonies[0].next_lay_tick == 0
