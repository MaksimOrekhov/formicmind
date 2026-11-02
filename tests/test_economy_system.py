"""FOR-12 system level: food ledger over long runs, checkpoint/restore, mirror symmetry,
information boundary, determinism. Combat does not exist, so soldiers that meet simply block each other."""

import copy
import dataclasses

import pytest

from formicmind import (
    build_observation,
    checkpoint_from_bytes,
    checkpoint_hash,
    checkpoint_to_bytes,
    default_config,
    reset,
    state_hash,
)
from formicmind.decoder import PolicyLogitsV2
from formicmind.events import Phase, event_sort_key, events_hash
from formicmind.invariants import InvariantError
from formicmind.ledger import EconomyTotals, check_transition, economy_totals, total_food
from formicmind.observation import flatten_v2
from formicmind.rng import policy_stream_id
from formicmind.rules import MIRROR, local_to_world
from formicmind.snapshot import WindowStats, begin_policy_phase
from formicmind.tick import PARTIAL_ECONOMY_PHASES, PolicyController, ScriptedController, step_tick

from econ_helpers import FORAGE_N1_N2, act, cfg, new_run, of_kind, tick
from helpers import N4

BOOM = {0: act(lay_order=1, forage_share_q=FORAGE_N1_N2),
        60: act(lay_order=2, forage_share_q=FORAGE_N1_N2),
        90: act(lay_order=0, forage_share_q=FORAGE_N1_N2)}
BARRACKS = {0: act(lay_order=2, forage_share_q=FORAGE_N1_N2),
            60: act(lay_order=1, forage_share_q=FORAGE_N1_N2)}
GREEDY_C = {0: act(lay_order=1, forage_share_q=(0, 21845, 0, 43690, 0, 0)),
            30: act(lay_order=1, forage_share_q=(0, 0, 0, 65535, 0, 0))}
RAIDER_FORAGER = {0: act(lay_order=2, forage_share_q=(0, 0, 0, 32768, 0, 32767)),
                  30: act(lay_order=1, forage_share_q=(0, 0, 0, 32768, 0, 32767))}


def random_policy(observation, mask, policy_rng):
    return PolicyLogitsV2([policy_rng.next_u32() / 2**31 - 1 for _ in range(17)])


def summary(state, run, side):
    """Everything of one colony in its own frame, ids excluded (ids differ between seats)."""
    c = state.colonies[side]
    loc = lambda n: n if n == -1 else local_to_world(side, n)  # noqa: E731
    units = sorted((u.caste, u.hp, loc(u.node), loc(u.from_node), loc(u.to_node), u.arrival_tick,
                    loc(u.target_node)) for u in c.units)
    brood = sorted((b.caste, b.laid_tick, b.hatch_tick) for b in c.brood)
    return (c.food, c.queen_hp, c.next_lay_tick, c.upkeep_remainder10, c.starvation_debt, c.lay_order,
            tuple(c.forage_q), c.army_choice, c.avoid_center, units, brood, run.windows[side])


def obs_pair(state, run):
    snap = begin_policy_phase(state.clone(), tuple(run.windows))
    return [flatten_v2(build_observation(snap, s)) for s in (0, 1)]


# --- ledger over long runs ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("eggs", [1, 2, 4])
def test_food_is_conserved_over_150_ticks_with_two_scripted_policies(eggs):
    state, run, ctrl = new_run(BOOM, BARRACKS, config=cfg(eggs_per_lay=eggs))
    start = total_food(state)
    events = tick(state, run, ctrl, 150)                      # check_transition runs after every tick
    totals = economy_totals(events)
    assert total_food(state) == totals.expected_total_food(start)
    assert totals.regen_added > 0 and sum(s.upkeep_paid for s in totals.sides) > 0
    assert state.tick == 150


def test_contested_centre_run_keeps_the_ledger_and_the_side_symmetry_of_a_shared_stock():
    state, run, ctrl = new_run(GREEDY_C, RAIDER_FORAGER)
    start = total_food(state)
    events = tick(state, run, ctrl, 200)
    assert total_food(state) == economy_totals(events).expected_total_food(start)
    contested = [e for e in of_kind(events, "harvest") if e.node == 3 and e.stock_before < e.demand * 2
                 and not e.blocked]
    assert contested, "the scenario must exercise a scarce centre"
    for e in contested:
        assert 0 <= e.harvested <= e.demand and e.stock_after >= 0


@pytest.mark.parametrize("seed", [3, 11, 29])
def test_random_policies_keep_every_invariant_for_300_ticks(seed):
    ids = [policy_stream_id("alpha", 0), policy_stream_id("beta", 0)]
    state, run = reset(default_config(), seed, ids)
    ctrl = (PolicyController(random_policy, ids[0]), PolicyController(random_policy, ids[1]))
    start = total_food(state)
    all_events = []
    for _ in range(300):
        all_events.extend(step_tick(state, run, ctrl, partial_economy=True, check_invariants=True).events)
    totals = economy_totals(all_events)
    assert total_food(state) == totals.expected_total_food(start)
    assert all_events == sorted(all_events, key=lambda e: (e.tick, *event_sort_key(e)))
    ids_seen = [u.unit_id for c in state.colonies for u in c.units] + \
        [b.unit_id for c in state.colonies for b in c.brood]
    assert len(ids_seen) == len(set(ids_seen)) and max(ids_seen) < state.next_unit_id
    assert all(c.food >= 0 and 0 <= c.upkeep_remainder10 <= 9 and c.starvation_debt >= 0
               for c in state.colonies)


def test_every_phase_changes_food_in_the_one_way_the_ledger_allows():
    state, run, ctrl = new_run(BOOM, BARRACKS)
    deltas = {}

    def spy(phase):
        func = PARTIAL_ECONOMY_PHASES[phase]

        def wrapped(ctx):
            before = total_food(ctx.state)
            n = len(ctx.events)
            func(ctx)
            deltas.setdefault(phase, []).append((total_food(ctx.state) - before, ctx.events[n:]))
        return wrapped

    phases = {p: (spy(p) if f is not None else None) for p, f in PARTIAL_ECONOMY_PHASES.items()}
    for _ in range(60):
        step_tick(state, run, ctrl, partial_economy=True, phases=phases)
    for delta, _ in deltas[Phase.HARVEST]:
        assert delta == 0                                               # food only moves node -> warehouse
    for delta, evs in deltas[Phase.UPKEEP]:
        assert delta == -sum(e.paid for e in evs if e.kind == "upkeep")
    for delta, evs in deltas[Phase.LAY]:
        assert delta == -sum(e.total_cost for e in evs if e.laid)
    for delta, evs in deltas[Phase.REGENERATION]:
        assert delta == sum(e.added for e in evs) >= 0
    for phase in (Phase.POLICY, Phase.COMPLETION, Phase.ARRIVAL, Phase.AUTOMATIC_DEPART, Phase.TERMINAL_LOG):
        assert all(d == 0 for d, _ in deltas[phase])


def test_a_starving_colony_loses_adults_deterministically_and_the_ledger_still_holds():
    state, run, ctrl = new_run({0: act(forage_share_q=FORAGE_N1_N2)})
    state.colonies[1].food = 0                                          # B has nothing and does not forage
    start = total_food(state)
    events = tick(state, run, ctrl, 60)
    deaths = [e for e in of_kind(events, "starvation") if e.side == 1]
    assert deaths and all(e.side == 1 for e in of_kind(events, "starvation") if e.tick < 5)
    assert [e.caste for e in deaths[:4]] == [1, 1, 1, 1]                # soldiers go first
    assert [e.unit_id for e in deaths[:4]] == [29, 30, 31, 32]
    assert not [e for e in of_kind(events, "starvation") if e.side == 0]
    assert total_food(state) == economy_totals(events).expected_total_food(start)
    assert len(state.colonies[1].units) < 16 and len(state.colonies[0].units) == 16


def test_window_counters_equal_the_sums_of_the_window_events():
    state, run, ctrl = new_run(BOOM, BARRACKS)
    events = tick(state, run, ctrl, 45)                                 # window restarted at t=30: ticks 30..44
    window = [e for e in events if e.tick >= 30]
    totals = economy_totals(window)
    for side in (0, 1):
        s = totals.sides[side]
        assert run.windows[side] == WindowStats(s.harvested, s.upkeep_paid, s.egg_spend,
                                                run.windows[side].starvation_debt_delta, 15)
    debts = [e for e in of_kind(window, "upkeep")]
    for side in (0, 1):
        assert run.windows[side].starvation_debt_delta == sum(
            e.debt_after - e.debt_before for e in debts if e.side == side)


# --- the validator catches tampering ----------------------------------------------------------------------------------

def _one_step(ticks=25):
    state, run, ctrl = new_run(BOOM, BARRACKS)
    tick(state, run, ctrl, ticks - 1, check=False)
    before = state.clone()
    events = list(step_tick(state, run, ctrl, partial_economy=True).events)
    return before, state, events


def test_check_transition_accepts_an_honest_tick_and_rejects_forged_ones():
    before, after, events = _one_step()
    check_transition(before, after, events)
    forged = after.clone()
    forged.colonies[0].food += 1                                        # food from nowhere
    with pytest.raises(InvariantError):
        check_transition(before, forged, events)
    forged = after.clone()
    forged.node_food[1] -= 1                                            # food that vanished from a node
    with pytest.raises(InvariantError):
        check_transition(before, forged, events)
    forged = after.clone()
    del forged.colonies[0].units[3]                                     # an adult removed without starvation
    with pytest.raises(InvariantError):
        check_transition(before, forged, events)
    forged = after.clone()
    forged.colonies[1].units[0].hp -= 1                                 # damage without combat
    with pytest.raises(InvariantError):
        check_transition(before, forged, events)
    with pytest.raises(InvariantError):
        check_transition(before, after, [e for e in events if e.kind != "regenerate"])


def test_check_transition_rejects_a_hatch_without_ripe_brood():
    state, run, ctrl = new_run({0: act(lay_order=1, forage_share_q=FORAGE_N1_N2)})
    tick(state, run, ctrl, 19, check=False)
    before = state.clone()
    events = list(step_tick(state, run, ctrl, partial_economy=True).events)
    assert not of_kind(events, "hatch")                                 # tick 19: nothing hatches yet
    tick(state, run, ctrl, 1, check=False)                              # tick 20 hatches unit 33 ...
    forged = state.clone()
    with pytest.raises(InvariantError):
        check_transition(before, forged, events)                        # ... which this tick's events do not show


# --- checkpoint / restore ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("cut", [30, 45, 60])
def test_checkpoint_restore_continues_with_identical_events_state_and_observation_bytes(cut):
    state, run, ctrl = new_run(BOOM, BARRACKS)
    tick(state, run, ctrl, cut)
    assert state.colonies[0].brood and state.colonies[1].brood          # brood is part of the checkpoint
    assert run.windows[0].harvest > 0 and run.windows[0].upkeep_paid > 0
    state2, run2 = checkpoint_from_bytes(checkpoint_to_bytes(state, run), state.config)
    assert checkpoint_hash(state2, run2) == checkpoint_hash(state, run)
    assert state_hash(state2) == state_hash(state)
    if cut % 30 == 0:                                                   # observations exist on policy ticks only
        assert obs_pair(state2, run2) == obs_pair(state, run)
    tail_full = tick(state, run, ctrl, 120 - cut)                       # always ends on the policy tick 120
    tail_restored = tick(state2, run2, ctrl, 120 - cut)
    assert events_hash(tail_full) == events_hash(tail_restored)
    assert state_hash(state) == state_hash(state2)
    assert checkpoint_hash(state, run) == checkpoint_hash(state2, run2)
    assert obs_pair(state, run) == obs_pair(state2, run2)


def test_observation_at_t30_is_byte_identical_after_restore_and_depends_on_the_economy_windows():
    state, run, ctrl = new_run(BOOM, BARRACKS)
    tick(state, run, ctrl, 30)
    state2, run2 = checkpoint_from_bytes(checkpoint_to_bytes(state, run), state.config)
    assert obs_pair(state2, run2) == obs_pair(state, run)
    forgot = begin_policy_phase(state2.clone(), (WindowStats(), WindowStats()))
    assert flatten_v2(build_observation(forgot, 0)) != obs_pair(state2, run2)[0]   # F1 hazard stays visible
    own = build_observation(begin_policy_phase(state2.clone(), tuple(run2.windows)), 0).block("own_global")
    assert own[17] == 1.0 and any(v > 0 for v in own[13:16])           # window ticks / K and harvest.. > 0


# --- mirror symmetry and independence of the side order ------------------------------------------------------------------

@pytest.mark.parametrize("script", [BOOM, GREEDY_C, RAIDER_FORAGER])
def test_identical_local_policies_give_mirror_identical_economies(script):
    state, run, ctrl = new_run(script, script)
    for _ in range(150):
        tick(state, run, ctrl)
        assert summary(state, run, 0) == summary(state, run, 1)
        assert all(state.node_food[i] == state.node_food[MIRROR[i]] for i in range(7))


@pytest.mark.parametrize("x, y", [(GREEDY_C, RAIDER_FORAGER), (BOOM, BARRACKS), (BARRACKS, GREEDY_C)])
def test_swapping_the_seats_of_two_different_policies_swaps_the_results(x, y):
    s1, r1, c1 = new_run(x, y)
    s2, r2, c2 = new_run(y, x)
    for _ in range(150):
        tick(s1, r1, c1)
        tick(s2, r2, c2)
    assert summary(s1, r1, 0) == summary(s2, r2, 1)
    assert summary(s1, r1, 1) == summary(s2, r2, 0)
    assert s1.node_food == [s2.node_food[MIRROR[i]] for i in range(7)]


# --- information boundary and determinism -----------------------------------------------------------------------------------------

def test_the_enemys_warehouse_brood_orders_and_unseen_stocks_never_reach_the_observation():
    state, run, ctrl = new_run(BOOM, BARRACKS)
    tick(state, run, ctrl, 60)
    base = obs_pair(state, run)[0]
    changed = state.clone()
    changed.colonies[1].food += 37                                      # enemy warehouse
    changed.colonies[1].upkeep_remainder10 = (changed.colonies[1].upkeep_remainder10 + 3) % 10
    changed.colonies[1].starvation_debt += 2
    changed.colonies[1].lay_order = 0
    changed.node_food[N4] = max(0, changed.node_food[N4] - 20)         # a stock A is not standing in
    assert not any(u.node == N4 for u in changed.colonies[0].units)
    snap = begin_policy_phase(changed, tuple(run.windows))
    assert flatten_v2(build_observation(snap, 0)) == base
    # ... and the enemy's own economy windows are not visible either
    other = tuple(run.windows)
    swapped = (other[0], WindowStats(harvest=999))
    snap2 = begin_policy_phase(state.clone(), swapped)
    assert flatten_v2(build_observation(snap2, 0)) == base


def test_the_economy_consumes_no_randomness():
    ids = [policy_stream_id("alpha", 0)]
    state, run, ctrl = new_run(BOOM, BARRACKS, streams=ids)
    before = copy.deepcopy({k: v.to_canonical() for k, v in run.streams.items()})
    tick(state, run, ctrl, 120)
    assert {k: v.to_canonical() for k, v in run.streams.items()} == before


def test_same_inputs_give_the_same_hashes_twice_and_different_scripts_give_different_ones():
    def go(a, b):
        state, run, ctrl = new_run(a, b)
        events = tick(state, run, ctrl, 100)
        return events_hash(events), state_hash(state), checkpoint_hash(state, run)
    assert go(BOOM, BARRACKS) == go(BOOM, BARRACKS)
    assert go(BOOM, BARRACKS) != go(BARRACKS, BOOM)


def test_reset_hashes_are_untouched_by_the_economy_work():
    from formicmind import config_hash
    state, run = reset(default_config(), 20261102)
    assert config_hash(state.config) == "3f6aaa36a0d3d15888fa42f142df1151376794e62b553ab8e5f7659c9794145d"
    assert state_hash(state) == "675116b1ac021868d41fd234ed9642c7bae5b59377daa239355913602555f3d3"
    assert checkpoint_hash(state, run) == "5a122bd4993d878b10fc9798bd9c90016ec8e95a091644b945226ebc5eac525d"


def test_new_event_kinds_are_canonical_json_safe_and_sorted_per_tick():
    import json
    from formicmind.canonical import canonical_dumps
    from formicmind.events import events_to_canonical
    state, run, ctrl = new_run(BOOM, BARRACKS)
    events = tick(state, run, ctrl, 35)
    kinds = {e.kind for e in events}
    assert {"hatch", "harvest", "upkeep", "lay", "regenerate", "order", "depart", "arrive"} <= kinds
    canon = events_to_canonical(events)
    assert json.loads(canonical_dumps(canon)) == canon
    for t in range(35):
        mine = [e for e in events if e.tick == t]
        assert mine == sorted(mine, key=event_sort_key)
