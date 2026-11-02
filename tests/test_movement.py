"""Departure/arrival timing, multi-hop routes, retargeting, avoid_center, blocking."""

import pytest

from formicmind import default_config, reset
from formicmind.actions import ActionV2
from formicmind.events import ArriveEvent, DepartEvent, Phase
from formicmind.model import NOWHERE
from formicmind.movement import arrive_units, dispatch_available
from formicmind.rules import MIRROR
from formicmind.tick import ScriptedController, step_tick

from helpers import A, B, C, N1, N2, N4, N5, fresh_state, place, transit, unit_of

Q = 65535
HOME_Q = (Q, 0, 0, 0, 0, 0)


def act(**kw):
    base = dict(lay_order=0, forage_share_q=HOME_Q, army_choice=0, avoid_center=False)
    base.update(kw)
    return ActionV2(**base)


def new_run(a=None, b=None, seed=1):
    state, run = reset(default_config(), seed)
    ctrl = (ScriptedController(a or {}), ScriptedController(b or {}))
    return state, run, ctrl


def advance(state, run, ctrl, n):
    events = []
    for _ in range(n):
        events.extend(step_tick(state, run, ctrl, movement_only=True).events)
    return events


def moves(events, unit_id):
    out = []
    for e in events:
        if e.unit_id != unit_id:
            continue
        if e.kind == "depart":
            out.append(("depart", e.tick, e.phase, e.from_node, e.to_node, e.arrival_tick))
        elif e.kind == "arrive":
            out.append(("arrive", e.tick, e.phase, e.node))
    return out


def test_departure_at_t_arrives_exactly_at_t_plus_d():
    state, run, ctrl = new_run({0: act(army_choice=1)})        # defend(N1): edge A-N1 has length 2
    step_tick(state, run, ctrl, movement_only=True)             # tick 0
    soldier = unit_of(state, 0, 13)
    assert state.tick == 1
    assert (soldier.node, soldier.from_node, soldier.to_node, soldier.arrival_tick) == (NOWHERE, A, N1, 2)
    step_tick(state, run, ctrl, movement_only=True)             # tick 1: still on the edge
    assert soldier.node == NOWHERE and soldier.arrival_tick == 2
    result = step_tick(state, run, ctrl, movement_only=True)    # tick 2: Arrival
    assert soldier.node == N1 and soldier.arrival_tick == NOWHERE
    arrivals = [e for e in result.events if isinstance(e, ArriveEvent) and e.unit_id == 13]
    assert [(e.tick, e.phase, e.node) for e in arrivals] == [(2, Phase.ARRIVAL, N1)]


def test_a_to_b_through_the_centre_takes_eight_ticks_without_transfer_delay():
    state, run, ctrl = new_run({0: act(army_choice=6)})
    events = advance(state, run, ctrl, 10)
    assert moves(events, 13) == [
        ("depart", 0, Phase.POLICY, A, N1, 2),
        ("arrive", 2, Phase.ARRIVAL, N1),
        ("depart", 2, Phase.AUTOMATIC_DEPART, N1, C, 4),
        ("arrive", 4, Phase.ARRIVAL, C),
        ("depart", 4, Phase.AUTOMATIC_DEPART, C, N4, 6),
        ("arrive", 6, Phase.ARRIVAL, N4),
        ("depart", 6, Phase.AUTOMATIC_DEPART, N4, B, 8),
        ("arrive", 8, Phase.ARRIVAL, B),
    ]
    assert unit_of(state, 0, 13).node == B


def test_avoid_center_route_takes_ten_ticks():
    state, run, ctrl = new_run({0: act(army_choice=6, avoid_center=True)})
    events = advance(state, run, ctrl, 12)
    assert moves(events, 13) == [
        ("depart", 0, Phase.POLICY, A, N2, 3),
        ("arrive", 3, Phase.ARRIVAL, N2),
        ("depart", 3, Phase.AUTOMATIC_DEPART, N2, N5, 7),
        ("arrive", 7, Phase.ARRIVAL, N5),
        ("depart", 7, Phase.AUTOMATIC_DEPART, N5, B, 10),
        ("arrive", 10, Phase.ARRIVAL, B),
    ]


def test_mirrored_b_to_a_routes_take_eight_and_ten_ticks():
    state, run, ctrl = new_run(b={0: act(army_choice=6)})
    events = advance(state, run, ctrl, 10)
    assert moves(events, 29) == [
        ("depart", 0, Phase.POLICY, B, N4, 2), ("arrive", 2, Phase.ARRIVAL, N4),
        ("depart", 2, Phase.AUTOMATIC_DEPART, N4, C, 4), ("arrive", 4, Phase.ARRIVAL, C),
        ("depart", 4, Phase.AUTOMATIC_DEPART, C, N1, 6), ("arrive", 6, Phase.ARRIVAL, N1),
        ("depart", 6, Phase.AUTOMATIC_DEPART, N1, A, 8), ("arrive", 8, Phase.ARRIVAL, A)]
    state, run, ctrl = new_run(b={0: act(army_choice=6, avoid_center=True)})
    events = advance(state, run, ctrl, 12)
    assert [m[1] for m in moves(events, 29) if m[0] == "arrive"] == [3, 7, 10]
    assert unit_of(state, 1, 29).node == A


def test_mirrored_scenarios_give_mirrored_events():
    def run_side(side):
        script = {0: act(forage_share_q=(21845, 43690, 0, 0, 0, 0), army_choice=6)}
        state, run, ctrl = new_run(a=script if side == 0 else None, b=script if side == 1 else None)
        return advance(state, run, ctrl, 12)

    def mirror(e, offset):
        d = e.to_canonical()
        d["side"] = 1 - d["side"]
        for key in ("from_node", "to_node", "node", "target_node", "army_target_node", "old_target", "new_target"):
            if key in d:
                d[key] = MIRROR[d[key]]
        if "unit_id" in d and d["unit_id"] > 0:
            d["unit_id"] += offset
        return d

    ev_a, ev_b = run_side(0), run_side(1)
    keep = lambda evs: [e for e in evs if e.kind != "order"]  # noqa: E731
    mirrored = [mirror(e, 16) for e in keep(ev_a)]
    got = [e.to_canonical() for e in keep(ev_b)]
    key = lambda d: (d["tick"], d["phase"], d["kind"], d["unit_id"])  # noqa: E731
    assert sorted(mirrored, key=key) == sorted(got, key=key)


def test_stationary_units_never_travel_and_standing_orders_persist():
    state, run, ctrl = new_run()
    events = advance(state, run, ctrl, 70)
    assert [e for e in events if e.kind in ("depart", "arrive", "retarget")] == []
    assert state.tick == 70
    assert all(u.node == (0 if u.unit_id <= 16 else 6) for c in state.colonies for u in c.units)


def test_units_stop_at_their_target_and_do_not_move_again():
    state, run, ctrl = new_run({0: act(army_choice=1)})
    events = advance(state, run, ctrl, 70)
    assert [e.tick for e in events if e.kind == "arrive" and e.unit_id == 13] == [2]
    assert [e.tick for e in events if e.kind == "depart" and e.unit_id == 13] == [0]


def test_workers_return_home_when_the_order_changes():
    script = {0: act(forage_share_q=(0, Q, 0, 0, 0, 0)), 30: act()}
    state, run, ctrl = new_run(script)
    events = advance(state, run, ctrl, 40)
    assert [(e.kind, e.tick, e.to_node if e.kind == "depart" else e.node) for e in events
            if e.kind in ("depart", "arrive") and e.unit_id == 1] == [
        ("depart", 0, N1), ("arrive", 2, N1), ("depart", 30, A), ("arrive", 32, A)]
    assert unit_of(state, 0, 1).node == A


def test_retargeting_mid_edge_keeps_the_edge_and_continues_from_the_next_node():
    state, run, ctrl = new_run({30: act(forage_share_q=(0, 0, Q, 0, 0, 0))})
    for uid in range(1, 13):                                    # workers are on edge A->N1, arriving at 31
        transit(unit_of(state, 0, uid), A, N1, 31)
    state.tick = 30
    events = []
    result = step_tick(state, run, ctrl, movement_only=True)    # policy tick 30: targets change, edge stays
    events += result.events
    w = unit_of(state, 0, 1)
    assert (w.node, w.from_node, w.to_node, w.arrival_tick, w.target_node) == (NOWHERE, A, N1, 31, N2)
    assert not [e for e in result.events if e.kind == "depart" and e.unit_id == 1]
    result = step_tick(state, run, ctrl, movement_only=True)    # tick 31: arrives at N1, leaves at once
    assert [(e.kind, e.phase, e.from_node if e.kind == "depart" else e.node, getattr(e, "arrival_tick", None))
            for e in result.events if e.unit_id == 1] == [
        ("arrive", Phase.ARRIVAL, N1, None), ("depart", Phase.AUTOMATIC_DEPART, N1, 33)]
    assert w.to_node == A                                       # N1->N2: equal cost, lexicographic via A


def test_avoid_center_does_not_ban_the_current_or_arrival_node_c():
    # soldiers stand in C when the raid-with-avoid order arrives
    state, run, ctrl = new_run({30: act(army_choice=6, avoid_center=True)})
    for uid in range(13, 17):
        place(unit_of(state, 0, uid), C)
    state.tick = 30
    result = step_tick(state, run, ctrl, movement_only=True)
    departs = [e for e in result.events if e.kind == "depart" and e.unit_id == 13]
    assert [(e.phase, e.from_node, e.to_node, e.arrival_tick) for e in departs] == [(Phase.POLICY, C, N4, 32)]

    # soldiers still on the edge N1->C when the order arrives: they reach C and continue from it
    state, run, ctrl = new_run({30: act(army_choice=6, avoid_center=True)})
    for uid in range(13, 17):
        transit(unit_of(state, 0, uid), N1, C, 31)
    state.tick = 30
    step_tick(state, run, ctrl, movement_only=True)
    result = step_tick(state, run, ctrl, movement_only=True)
    assert [(e.kind, e.phase) for e in result.events if e.unit_id == 14] == [
        ("arrive", Phase.ARRIVAL), ("depart", Phase.AUTOMATIC_DEPART)]
    s = unit_of(state, 0, 14)
    assert (s.from_node, s.to_node, s.arrival_tick) == (C, N4, 33)


# --- blocking ----------------------------------------------------------------------------------

def test_enemy_soldiers_block_departure_until_the_state_changes_then_release_in_automatic_depart():
    state, run, ctrl = new_run({0: act(forage_share_q=(0, 0, 0, 0, Q, 0))})   # workers target local N4
    state.tick = 1                                                           # non-policy tick
    worker = unit_of(state, 0, 1)
    place(worker, C)
    worker.target_node = N4
    place(unit_of(state, 1, 29), C)                                          # B soldier in C
    first = step_tick(state, run, ctrl, movement_only=True)
    assert [e for e in first.events if e.unit_id == 1] == [] and worker.node == C
    place(unit_of(state, 1, 29), N5)                                         # fixture: the blocker is gone
    result = step_tick(state, run, ctrl, movement_only=True)                 # tick 2
    assert [(e.tick, e.phase, e.from_node, e.to_node, e.arrival_tick)
            for e in result.events if e.unit_id == 1] == [(2, Phase.AUTOMATIC_DEPART, C, N4, 4)]  # same tick


def test_own_soldiers_and_enemy_workers_do_not_block():
    state, run, ctrl = new_run()
    state.tick = 1
    w = unit_of(state, 0, 1)
    place(w, C)
    w.target_node = N4
    place(unit_of(state, 0, 13), C)          # own soldier
    place(unit_of(state, 1, 17), C)          # enemy worker
    events = dispatch_available(state, Phase.AUTOMATIC_DEPART)
    assert [(e.unit_id, e.to_node) for e in events if e.unit_id == 1] == [(1, N4)]


def test_blocking_applies_at_policy_dispatch_and_is_evaluated_simultaneously_for_both_sides():
    state, run, ctrl = new_run({0: act(army_choice=3)}, {0: act(army_choice=3)})
    place(unit_of(state, 0, 13), N1)
    place(unit_of(state, 1, 29), N1)
    # an A soldier and a B soldier share N1: neither may leave, regardless of processing order
    result = step_tick(state, run, ctrl, movement_only=True)
    assert [e for e in result.events if e.kind == "depart" and e.unit_id in (13, 29)] == []
    assert unit_of(state, 0, 13).node == N1 and unit_of(state, 1, 29).node == N1
    others = [e.unit_id for e in result.events if e.kind == "depart"]
    assert 14 in others and 30 in others                                     # the rest dispatch normally


def test_two_raiding_armies_meet_in_c_and_both_stay_blocked_without_combat():
    """Known limitation of the movement-only skeleton (documented): nobody dies, nobody leaves."""
    state, run, ctrl = new_run({0: act(army_choice=6)}, {0: act(army_choice=6)})
    events = advance(state, run, ctrl, 20)
    assert all(unit_of(state, 0, uid).node == C for uid in range(13, 17))
    assert all(unit_of(state, 1, uid).node == C for uid in range(29, 33))
    assert [e.tick for e in events if e.kind == "depart" and e.unit_id == 13] == [0, 2]


def test_arrive_and_dispatch_primitives_validate_the_state():
    state = fresh_state(5)
    transit(unit_of(state, 0, 1), A, N1, 4)               # missed arrival
    with pytest.raises(ValueError):
        arrive_units(state)
    state = fresh_state(5)
    assert arrive_units(state) == [] and dispatch_available(state, Phase.AUTOMATIC_DEPART) == []
