"""Worker quotas (FOR-2 section 4.2), unit targets and atomic order application."""

import dataclasses

import pytest

from formicmind import default_config, reset, state_hash
from formicmind.actions import ActionV2, ValidatedActionV2
from formicmind.events import OrderEvent, RetargetEvent
from formicmind.model import NOWHERE
from formicmind.orders import apply_joint_orders, worker_quotas
from formicmind.rng import new_stream
from formicmind.routing import path_cost, shortest_path

from helpers import A, B, C, N1, N2, N4, N5, fresh_state, place, transit, unit_of

Q = 65535


def valid(**kw):
    base = dict(lay_order=0, forage_share_q=(Q, 0, 0, 0, 0, 0), army_choice=0, avoid_center=False)
    base.update(kw)
    return ValidatedActionV2(ActionV2(**base))


def both(a=None, b=None):
    return (a or valid(), b or valid())


# --- quotas -----------------------------------------------------------------------------

def test_newborn_quota_golden_n13():
    # FOR-2 section 5.3: q_home=16384, q_N1=49151, N=13 -> home=3, N1=10
    assert worker_quotas(13, (16384, 49151, 0, 0, 0, 0)) == (3, 10, 0, 0, 0, 0)
    assert worker_quotas(12, (16384, 49151, 0, 0, 0, 0)) == (3, 9, 0, 0, 0, 0)  # 12 workers: N=12 case


def test_float_error_golden_large_n():
    assert worker_quotas(10_000_000, (32767, 32768, 0, 0, 0, 0)) == (4_999_924, 5_000_076, 0, 0, 0, 0)


def test_quota_edge_cases_and_ties():
    assert worker_quotas(0, (Q, 0, 0, 0, 0, 0)) == (0,) * 6
    assert worker_quotas(5, (Q, 0, 0, 0, 0, 0)) == (5, 0, 0, 0, 0, 0)
    # 2 workers, shares 32768/32767: ideal 1.0000153/0.99998 -> (1, 1)
    assert worker_quotas(2, (32768, 32767, 0, 0, 0, 0)) == (1, 1, 0, 0, 0, 0)
    # 1 worker on three equal shares: equal remainders -> lowest category index
    assert worker_quotas(1, (21845, 21845, 21845, 0, 0, 0)) == (1, 0, 0, 0, 0, 0)
    assert worker_quotas(2, (21845, 21845, 21845, 0, 0, 0)) == (1, 1, 0, 0, 0, 0)
    for bad in ((Q - 1, 0, 0, 0, 0, 0), (Q, 0, 0, 0, 0), (Q + 1, -1, 0, 0, 0, 0)):
        with pytest.raises(ValueError):
            worker_quotas(3, bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        worker_quotas(-1, (Q, 0, 0, 0, 0, 0))


def test_quota_properties_on_random_shares():
    rng = new_stream(7, "test:quotas")
    for _ in range(2000):
        support = [i for i in range(6) if rng.below(2)] or [0]
        cuts = sorted(rng.below(Q + 1) for _ in range(len(support) - 1))
        parts = [b - a for a, b in zip([0, *cuts], [*cuts, Q])]
        q = [0] * 6
        for i, p in zip(support, parts):
            q[i] = p
        n = rng.below(500)
        quota = worker_quotas(n, tuple(q))
        assert sum(quota) == n and min(quota) >= 0
        assert all(quota[i] == 0 for i in range(6) if q[i] == 0)
        for i in range(6):  # never more than one away from the ideal share
            assert abs(quota[i] - n * q[i] / Q) < 1


# --- assignment -------------------------------------------------------------------------

def targets(state, side):
    return {u.unit_id: u.target_node for u in state.colonies[side].units}


def test_reset_orders_leave_every_target_at_home():
    state = fresh_state(0)
    events = apply_joint_orders(state, both())
    assert [type(e) for e in events] == [OrderEvent, OrderEvent]
    assert all(u.target_node == 0 for u in state.colonies[0].units)
    assert all(u.target_node == 6 for u in state.colonies[1].units)


def test_workers_fill_categories_in_order_and_soldiers_follow_the_army_choice():
    state = fresh_state(0)
    apply_joint_orders(state, both(valid(forage_share_q=(21845, 43690, 0, 0, 0, 0), army_choice=2)))
    t = targets(state, 0)
    assert [t[i] for i in range(1, 13)] == [A] * 4 + [N1] * 8       # unit ids ascending inside a place
    assert {t[i] for i in range(13, 17)} == {N2}                    # defend(N2)
    assert state.colonies[0].army_choice == 2 and state.colonies[0].forage_q == [21845, 43690, 0, 0, 0, 0]


def test_side_local_orders_are_translated_to_world_ids_for_side_b():
    state = fresh_state(0)
    # B's local N1 is world N4, local N2 is world N5, local 6 (raid) is world A
    apply_joint_orders(state, both(b=valid(forage_share_q=(0, 32768, 32767, 0, 0, 0), army_choice=6)))
    t = targets(state, 1)
    assert [t[i] for i in range(17, 29)] == [N4] * 6 + [N5] * 6
    assert {t[i] for i in range(29, 33)} == {A}
    apply_joint_orders(state, both(b=valid(army_choice=3)))
    assert {targets(state, 1)[i] for i in range(29, 33)} == {C}
    apply_joint_orders(state, both(b=valid(army_choice=1)))
    assert {targets(state, 1)[i] for i in range(29, 33)} == {N4}    # defend local N1 = world N4


def test_assignment_order_stationed_by_local_node_then_transit_by_next_node_and_eta():
    state = fresh_state(10)
    for uid in (1, 2, 3):
        place(unit_of(state, 0, uid), C)                    # local 3
    for uid in (4, 5, 6):
        place(unit_of(state, 0, uid), N2)                   # local 2
    place(unit_of(state, 0, 12), N1)                        # local 1 (7, 8 stay at home: local 0)
    transit(unit_of(state, 0, 9), A, N1, 12)                # transit, next node N1, ETA 2
    transit(unit_of(state, 0, 10), A, N1, 11)               # transit, next node N1, ETA 1: before 9
    transit(unit_of(state, 0, 11), N2, A, 13)               # transit, next node A: before N1 transits
    # stationed first (A, N1, N2, C), then transit by (next node, ETA): 11, 10, 9; inside a place by id
    expected = [7, 8, 12, 4, 5, 6, 1, 2, 3, 11, 10, 9]
    apply_joint_orders(state, both(valid(forage_share_q=(21845, 21845, 21845, 0, 0, 0))))
    t = targets(state, 0)
    assert [t[i] for i in expected] == [A] * 4 + [N1] * 4 + [N2] * 4
    # side B uses its own canonical order: home, local N1 (= world N4), local N2 (= world N5), ...
    for uid in (17, 18):
        place(unit_of(state, 1, uid), N5)
    place(unit_of(state, 1, 19), N4)
    apply_joint_orders(state, both(b=valid(forage_share_q=(21845, 21845, 21845, 0, 0, 0))))
    tb = targets(state, 1)
    assert [tb[i] for i in (20, 21, 22, 23)] == [B] * 4
    assert [tb[i] for i in (24, 25, 26, 27)] == [N4] * 4
    assert [tb[i] for i in (28, 19, 17, 18)] == [N5] * 4


def test_assignment_follows_the_literal_sequence_and_pins_audit_d5():
    """D5 is a rules question: the literal section 4.2 order costs 48 worker-ticks, the minimum is 12."""
    state = fresh_state(0)
    for uid in range(1, 7):
        place(unit_of(state, 0, uid), N2)
    for uid in range(7, 13):
        place(unit_of(state, 0, uid), C)
    apply_joint_orders(state, both(valid(forage_share_q=(0, 32768, 32767, 0, 0, 0))))
    t = targets(state, 0)
    topo = state.config.topology
    moves = [(unit_of(state, 0, uid).node, t[uid]) for uid in range(1, 13)]
    assert moves == [(N2, N1)] * 6 + [(C, N2)] * 6                  # the first six (in N2) go to N1
    literal = sum(path_cost(topo, shortest_path(topo, a, b, side=0)) for a, b in moves)
    assert literal == 6 * 5 + 6 * 3 == 48
    minimal = 6 * path_cost(topo, shortest_path(topo, C, N1, side=0))       # C workers to N1, N2 workers stay
    assert minimal == 12


def test_units_in_transit_keep_their_edge_and_eta_when_retargeted():
    state = fresh_state(5)
    u = unit_of(state, 0, 1)
    transit(u, A, N1, 7)
    apply_joint_orders(state, both(valid(forage_share_q=(0, 0, Q, 0, 0, 0))))
    assert (u.node, u.from_node, u.to_node, u.arrival_tick) == (NOWHERE, A, N1, 7)
    assert u.target_node == N2


def test_retarget_events_only_for_changed_targets_sorted_by_unit_id():
    state = fresh_state(0)
    events = apply_joint_orders(state, both(valid(army_choice=3)))
    retargets = [e for e in events if isinstance(e, RetargetEvent)]
    assert [e.unit_id for e in retargets] == [13, 14, 15, 16]
    assert all((e.old_target, e.new_target, e.caste, e.side) == (A, C, 1, 0) for e in retargets)
    assert apply_joint_orders(state, both(valid(army_choice=3))) [2:] == []   # nothing left to retarget


def test_orders_are_conservative_and_keep_ids_hp_and_one_location():
    state = fresh_state(0)
    before = {u.unit_id: (u.caste, u.hp, u.node, u.from_node, u.to_node, u.arrival_tick)
              for c in state.colonies for u in c.units}
    apply_joint_orders(state, both(valid(forage_share_q=(0, 0, 21845, 43690, 0, 0), army_choice=6, avoid_center=True)))
    after = {u.unit_id: (u.caste, u.hp, u.node, u.from_node, u.to_node, u.arrival_tick)
             for c in state.colonies for u in c.units}
    assert after == before                                        # orders never move or heal anybody


def test_apply_is_atomic_on_bad_input():
    state = fresh_state(0)
    before = state_hash(state)
    good = valid(army_choice=3)
    for bad in ((good,), (good, good, good), (good, ActionV2(0, (Q, 0, 0, 0, 0, 0), 0, False)),
                [good, good], (good, None)):
        with pytest.raises(TypeError):
            apply_joint_orders(state, bad)  # type: ignore[arg-type]
        assert state_hash(state) == before


def test_quota_uses_all_living_workers_including_blocked_or_travelling_ones():
    state = fresh_state(0)
    transit(unit_of(state, 0, 1), A, N1, 2)
    apply_joint_orders(state, both(valid(forage_share_q=(0, Q, 0, 0, 0, 0))))
    assert sum(1 for u in state.colonies[0].units if u.caste == 0 and u.target_node == N1) == 12
