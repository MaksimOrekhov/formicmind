"""ObservationV2: layout, golden G0, purity, mirror and information boundary."""

import dataclasses
import json
import struct

import pytest

from formicmind import default_config, state_hash
from formicmind.intel import IntelPhase, update_local_intel
from formicmind.model import NOWHERE
from formicmind.observation import (
    OBS_V2_BLOCKS,
    OBS_V2_LENGTH,
    OBS_V2_OFFSETS,
    ObservationConfigError,
    ObservationV2,
    build_observation,
    check_observation_config,
    f32,
    flatten_v2,
    observation_from_bytes,
)
from formicmind.snapshot import (
    PolicySnapshot,
    WindowStats,
    begin_policy_phase,
    freeze_policy_snapshot,
)

from helpers import A, B, C, N1, N2, N4, N5, busy_state, fresh_state, mirror_state, place, transit, unit_of

# FOR-2 errata v0.2.1 / v0.2 section 3.4: every non-zero element of flatten_v2 for
# side A at the FOR-1 start, as [index, value]; all other elements are exactly 0.
G0_NONZERO = json.loads(
    "[[1,1],[2,0.00999999978],[3,0.5],[4,0.0909090936],[5,0.166666672],[6,0.111111112],[7,0.25],"
    "[8,0.25],[9,0.333333343],[10,0.400000006],[11,0.800000012],[12,0.193548381],[13,0.157894731],"
    "[14,0.0588235296],[15,0.166666672],[16,0.0909090936],[17,1],[30,1],[33,1],[39,1],[47,0.5],"
    "[48,0.75],[54,0.5],[58,0.75],[60,1],[62,0.5],[63,0.75],[66,0.5],[67,0.75],[69,0.666666687],"
    "[70,0.5],[71,1],[72,0.666666687],[73,0.5],[76,0.666666687],[77,0.666666687],[78,1],"
    "[79,0.666666687],[80,0.666666687],[82,0.157894731],[83,0.0588235296],[96,0.166666672],"
    "[97,0.0909090936],[163,1],[170,1],[178,1],[179,1],[180,1],[181,1],[182,1],[183,1],[198,1],"
    "[205,1],[213,1],[214,1],[215,1],[216,1],[217,1],[218,1],[222,1]]"
)


def observe(state, side, windows=None):
    return build_observation(begin_policy_phase(state, windows), side)


def block(obs, name, k=None):
    vals = obs.block(name)
    return vals if k is None else vals[k]


# --- layout -------------------------------------------------------------------

def test_layout_is_contiguous_and_totals_223():
    assert OBS_V2_LENGTH == 223
    assert sum(length for _, length in OBS_V2_BLOCKS) == 223
    expected_offsets = {
        "time": 0, "public_rules": 2, "own_global": 12, "current_orders": 30, "public_edges": 47,
        "public_food_capacity": 68, "public_food_regen": 75, "own_node_count": 82,
        "own_node_hp_sum": 96, "nearest_arrival_eta": 110, "nearest_arrival_count": 124,
        "nearest_arrival_hp": 138, "transit_total_count": 152, "transit_total_hp": 154,
        "last_food_ratio": 156, "food_known": 163, "food_current": 170, "food_age_norm": 177,
        "enemy_count_last": 184, "enemy_known": 198, "enemy_current": 205, "enemy_age_norm": 212,
        "enemy_queen_hp_last": 219, "enemy_queen_known": 220, "enemy_queen_current": 221,
        "enemy_queen_age_norm": 222,
    }
    assert {k: v[0] for k, v in OBS_V2_OFFSETS.items()} == expected_offsets
    end = 0
    for name, length in OBS_V2_BLOCKS:
        assert OBS_V2_OFFSETS[name] == (end, length)
        end += length
    assert end == 223


# --- golden G0 ------------------------------------------------------------------

def test_g0_golden_vector_side_a_and_b():
    state = fresh_state(0)
    snapshot = begin_policy_phase(state)
    expected = {i: f32(float(v)) for i, v in G0_NONZERO}
    for side in (0, 1):  # B is the mirror image, so it sees the very same vector
        obs = build_observation(snapshot, side)
        assert len(obs.values) == 223
        assert list(obs.values) == [expected.get(i, 0.0) for i in range(223)]
        assert flatten_v2(obs) == struct.pack("<223f", *[expected.get(i, 0.0) for i in range(223)])


def test_g0_literal_values_round_trip_through_float32():
    # the document prints 9 significant digits "sufficient for IEEE float32 round-trip"
    for _, v in G0_NONZERO:
        x = f32(float(v))
        assert f32(float(f"{x:.9g}")) == x


def test_flatten_is_little_endian_float32_and_round_trips():
    obs = observe(fresh_state(0), 0)
    data = flatten_v2(obs)
    assert len(data) == 892
    assert data[4:8] == struct.pack("<f", 1.0)  # index 1 = ticks_left_norm = 1
    assert observation_from_bytes(data) == obs
    with pytest.raises(ValueError):
        observation_from_bytes(data[:-1])


def test_observation_type_is_strict_and_immutable():
    obs = observe(fresh_state(0), 0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        obs.values = ()  # type: ignore[misc]
    for bad in (obs.values[:-1], tuple(obs.values[:-1]) + (float("nan"),),
                tuple(obs.values[:-1]) + (float("inf"),), tuple(obs.values[:-1]) + (0.1,),
                tuple(obs.values[:-1]) + (1,), list(obs.values)):
        with pytest.raises(ValueError):
            ObservationV2(bad)  # type: ignore[arg-type]


def test_own_home_is_known_current_age_zero_even_without_adults():
    # FOR-2 errata P0-1: permanent home sensor
    state = fresh_state(0)
    for u in state.colonies[0].units:
        place(u, N1)  # every adult leaves home
    obs = observe(state, 0)
    assert block(obs, "own_node_count", 0) == 0.0 and block(obs, "own_node_count", 1) == 0.0
    assert (block(obs, "food_known", 0), block(obs, "food_current", 0), block(obs, "food_age_norm", 0)) == (1.0, 1.0, 0.0)
    assert (block(obs, "enemy_known", 0), block(obs, "enemy_current", 0), block(obs, "enemy_age_norm", 0)) == (1.0, 1.0, 0.0)


# --- purity and repeatability -------------------------------------------------------

def test_build_observation_is_pure_and_repeatable():
    state = busy_state()
    snapshot = begin_policy_phase(state)
    before = state_hash(state)
    first = [flatten_v2(build_observation(snapshot, s)) for s in (0, 1)]
    for _ in range(3):
        assert [flatten_v2(build_observation(snapshot, s)) for s in (0, 1)] == first
    assert state_hash(state) == before  # nothing in the state (incl. intel) changed


def test_snapshot_is_immutable_and_detached_from_state():
    state = busy_state()
    snapshot = begin_policy_phase(state)
    obs_before = [flatten_v2(build_observation(snapshot, s)) for s in (0, 1)]

    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.tick = 1  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.view(0).food = 1  # type: ignore[misc]

    def walk(value):
        assert not isinstance(value, (list, dict, set)), "mutable container in snapshot"
        if dataclasses.is_dataclass(value):
            for f in dataclasses.fields(value):
                walk(getattr(value, f.name))
        elif isinstance(value, tuple):
            for item in value:
                walk(item)

    walk(snapshot.views)
    # mutate the live state: the snapshot-based observations must not move
    state.colonies[0].food = 1
    state.colonies[0].units[0].hp = 1
    state.colonies[1].intel.nodes[N2].seen_tick = 60
    state.node_food[N1] = 0
    assert [flatten_v2(build_observation(snapshot, s)) for s in (0, 1)] == obs_before


def test_snapshot_views_contain_only_own_data():
    snapshot = begin_policy_phase(busy_state())
    own_ids = {s: {u.unit_id for u in snapshot.view(s).units} for s in (0, 1)}
    assert own_ids[0].isdisjoint(own_ids[1])
    assert not hasattr(snapshot, "colonies") and not hasattr(snapshot.view(0), "enemy")


def test_snapshot_only_at_policy_ticks():
    with pytest.raises(ValueError):
        freeze_policy_snapshot(fresh_state(7))
    freeze_policy_snapshot(fresh_state(30))


def test_both_observations_come_from_one_pre_action_state_regardless_of_order():
    s1, s2 = busy_state(), busy_state()
    for side in (0, 1):
        update_local_intel(s1, side, IntelPhase.PRE_POLICY)
    for side in (1, 0):
        update_local_intel(s2, side, IntelPhase.PRE_POLICY)
    assert state_hash(s1) == state_hash(s2)
    p1, p2 = freeze_policy_snapshot(s1), freeze_policy_snapshot(s2)
    assert p1 == p2


# --- hand-checked values on the asymmetric fixture -------------------------------------

def test_busy_state_hand_checked_values():
    obs = observe(busy_state(60), 0, (WindowStats(harvest=100, upkeep_paid=300, egg_spend=0,
                                                  starvation_debt_delta=-300, window_ticks=30), WindowStats()))
    g = obs.block("own_global")
    u = lambda x, s: f32(x / (x + s))  # noqa: E731
    assert g[0] == u(130, 250)
    assert g[1] == u(12, 64) and g[2] == u(4, 64)
    assert g[3] == u(11 * 20 + 11, 1200) and g[4] == u(3 * 30 + 17, 1200)   # unit 7 has 11 hp, unit 15 has 17
    assert g[5] == f32(200 / 240)
    assert (g[6], g[7]) == (u(1, 32), u(1, 32))
    assert g[8] == f32(10 / 3000) and g[9] == f32(15 / 3000)   # earliest hatch ETAs
    assert g[10] == f32(4 / 10)                                  # lay cooldown 4 of 10 ticks
    assert g[11] == f32(0.6) and g[12] == u(3, 10)
    assert g[13] == u(100, 300) and g[14] == u(300, 300) and g[15] == 0.0
    assert g[16] == f32(-0.5) and g[17] == 1.0
    # own_node_count rows: A home, N1, N2, C ... (workers, soldiers)
    counts = obs.block("own_node_count")
    assert counts[2 * N1:2 * N1 + 2] == (f32(3 / 67), 0.0)
    assert counts[2 * N2:2 * N2 + 2] == (0.0, f32(2 / 66))
    assert counts[2 * C:2 * C + 2] == (f32(1 / 65), 0.0)
    # transit: workers 5,6,7 (to C) and soldier 15 (to N2)
    assert obs.block("transit_total_count") == (f32(3 / 67), f32(1 / 65))
    assert obs.block("transit_total_hp") == (u(20 + 20 + 11, 1200), u(17, 1200))
    # current orders: lay worker, forage shares, army defend(C), no avoid
    assert obs.block("current_orders")[:3] == (0.0, 1.0, 0.0)
    assert obs.block("current_orders")[3:9] == tuple(f32(q / 65535) for q in (10000, 30000, 25535, 0, 0, 0))
    assert obs.block("current_orders")[9:16] == (0, 0, 0, 1, 0, 0, 0)


def test_busy_state_intel_values():
    state = busy_state(60)
    obs = observe(state, 0)
    # observed now: A (home), N1, N2, C; stale: N5 (seen at 30); unknown: N4, B
    cur = obs.block("food_current")
    assert [cur[k] for k in range(7)] == [1, 1, 1, 1, 0, 0, 0]
    known = obs.block("food_known")
    assert [known[k] for k in range(7)] == [1, 1, 1, 1, 0, 1, 0]
    assert list(obs.block("enemy_current")) == [1, 1, 1, 1, 0, 0, 0]
    assert list(obs.block("enemy_known")) == [1, 1, 1, 1, 0, 1, 0]
    age = obs.block("food_age_norm")
    assert age[N5] == f32(30 / 3000) and age[4] == 1.0 and age[B] == 1.0 and age[N1] == 0.0
    assert obs.block("last_food_ratio")[N1] == f32(33 / 80)
    assert obs.block("last_food_ratio")[C] == f32(77 / 120)
    assert obs.block("last_food_ratio")[N5] == f32(25 / 60)
    # enemy soldiers 29, 30 sit in C and are seen there; N5 sighting: 2 workers, 1 soldier
    enemy = obs.block("enemy_count_last")
    assert enemy[2 * C:2 * C + 2] == (0.0, f32(2 / 66))
    assert enemy[2 * N5:2 * N5 + 2] == (f32(2 / 66), f32(1 / 65))
    # the enemy queen is invisible: nothing of A is at B
    assert obs.block("enemy_queen_hp_last") == (0.0,) and obs.block("enemy_queen_known") == (0.0,)
    assert obs.block("enemy_queen_age_norm") == (1.0,)


def test_enemy_queen_is_seen_only_with_an_observer_at_the_enemy_home():
    state = busy_state(60)
    place(unit_of(state, 0, 13), B)
    obs = observe(state, 0)
    assert obs.block("enemy_queen_hp_last") == (f32(90 / 240),)
    assert obs.block("enemy_queen_known") == (1.0,) and obs.block("enemy_queen_current") == (1.0,)
    assert obs.block("enemy_queen_age_norm") == (0.0,)


# --- nearest arrival (FOR-2 errata P1-3, golden R-1) ---------------------------------------

def test_r1_nearest_arrival_is_keyed_by_next_edge_destination():
    state = fresh_state(60)
    # route A -> N1 -> C -> N4 -> B; the group is on edge N1 -> C, two ticks from C
    group = [unit_of(state, 0, uid) for uid in (1, 2)]
    for u in group:
        transit(u, N1, C, 62)

    def arrivals(obs):
        eta, cnt, hp = (obs.block(n) for n in ("nearest_arrival_eta", "nearest_arrival_count", "nearest_arrival_hp"))
        return {(k, c): (eta[2 * k + c], cnt[2 * k + c], hp[2 * k + c])
                for k in range(7) for c in (0, 1) if cnt[2 * k + c]}

    obs = observe(state, 0)
    assert arrivals(obs) == {(C, 0): (f32(2 / 4), f32(2 / 66), f32(40 / 1240))}  # keyed under C, not B
    # AutomaticDepart has sent them on C -> N4, arrival 2 ticks later
    for u in group:
        transit(u, C, N4, 62)
    keys = arrivals(observe(state, 0))
    assert keys == {(N4, 0): (f32(2 / 4), f32(2 / 66), f32(40 / 1240))}
    # arrival_tick == tick: ETA 0, the group is still counted
    for u in group:
        transit(u, C, N4, 60)
    keys = arrivals(observe(state, 0))
    assert keys == {(N4, 0): (0.0, f32(2 / 66), f32(40 / 1240))}


def test_nearest_arrival_uses_only_the_minimum_eta_group_per_destination_and_caste():
    state = fresh_state(60)
    transit(unit_of(state, 0, 1), N1, C, 61)
    transit(unit_of(state, 0, 2), N1, C, 61)
    transit(unit_of(state, 0, 3), N1, C, 62)      # later group, same destination/caste: not listed
    unit_of(state, 0, 1).hp = 5
    obs = observe(state, 0)
    assert obs.block("nearest_arrival_eta")[2 * C] == f32(1 / 4)
    assert obs.block("nearest_arrival_count")[2 * C] == f32(2 / 66)
    assert obs.block("nearest_arrival_hp")[2 * C] == f32(25 / 1225)
    assert obs.block("transit_total_count")[0] == f32(3 / 67)   # but all three count in transit totals


def test_units_standing_in_a_node_are_not_in_nearest_arrival():
    obs = observe(fresh_state(0), 0)
    assert not any(obs.block("nearest_arrival_count")) and not any(obs.block("nearest_arrival_eta"))


# --- information boundary -----------------------------------------------------------------

def perturb_hidden(state):
    """Change everything side A cannot sense in busy_state (A observes A, N1, N2, C only)."""
    s = state.clone()
    s.node_food[4], s.node_food[5] = 7, 60          # unobserved nodes N4, N5
    b = s.colonies[1]
    b.queen_hp = 17                                  # enemy queen at unobserved B
    b.food, b.next_lay_tick, b.starvation_debt, b.upkeep_remainder10 = 1, 99, 4, 3
    b.lay_order, b.army_choice, b.avoid_center, b.forage_q = 0, 0, False, [65535, 0, 0, 0, 0, 0]
    for u in b.units:
        if u.node in (4, 5, 6):                      # B units at unobserved nodes
            u.hp = max(1, u.hp - 5)
        if u.node == NOWHERE:                        # B units on edges are invisible
            u.hp = max(1, u.hp - 3)
            u.arrival_tick += 1
    place(b.units[2], 5)                             # move one B worker between unobserved nodes
    b.brood.clear()
    s.next_unit_id += 100
    return s


def test_hidden_enemy_state_does_not_leak_into_observation_or_mask():
    from formicmind.actions import flatten_mask_v2, get_action_mask

    base = busy_state()
    other = perturb_hidden(base)
    assert state_hash(base) != state_hash(other)
    o1, o2 = observe(base, 0), observe(other, 0)
    assert flatten_v2(o1) == flatten_v2(o2)
    assert flatten_mask_v2(get_action_mask(o1)) == flatten_mask_v2(get_action_mask(o2))


def test_enemy_changes_become_visible_only_through_a_sensor():
    base = busy_state()
    seen = base.clone()
    # enemy worker steps into node N2, where A has soldiers: now it must show up
    place(unit_of(seen, 1, 17), N2)
    o_base, o_seen = observe(base, 0), observe(seen, 0)
    assert o_base != o_seen
    assert o_seen.block("enemy_count_last")[2 * N2] == f32(1 / 65)
    # the same unit one step further away (N5, unobserved) is invisible
    away = base.clone()
    place(unit_of(away, 1, 17), N5)
    assert observe(away, 0) == o_base


def test_enemy_in_transit_toward_my_node_is_invisible():
    base = busy_state()
    moving = base.clone()
    transit(unit_of(moving, 1, 18), N4, C, 61)
    assert observe(moving, 0) == observe(base, 0)


# --- mirror ------------------------------------------------------------------------

@pytest.mark.parametrize("make", [lambda: fresh_state(0), lambda: busy_state(60)])
def test_mirrored_states_give_side_canonical_observations(make):
    state = make()
    mirrored = mirror_state(state)
    win_a, win_b = WindowStats(7, 8, 9, -4, 12), WindowStats(1, 2, 3, 4, 5)
    o_a = observe(state, 0, (win_a, win_b))
    o_b = observe(state, 1, (win_a, win_b))
    m_a = observe(mirrored, 0, (win_b, win_a))
    m_b = observe(mirrored, 1, (win_b, win_a))
    assert flatten_v2(m_a) == flatten_v2(o_b)
    assert flatten_v2(m_b) == flatten_v2(o_a)


# --- misc ---------------------------------------------------------------------------

def test_window_stats_feed_own_global_and_are_validated():
    obs = observe(fresh_state(0), 0, (WindowStats(30, 60, 20, 5, 15), WindowStats()))
    g = obs.block("own_global")
    assert g[13:18] == (f32(30 / 330), f32(60 / 360), f32(20 / 320), f32(5 / 305), f32(15 / 30))
    for bad in (dict(harvest=-1), dict(window_ticks=-1), dict(harvest=1.5)):
        with pytest.raises((ValueError, TypeError)):
            WindowStats(**bad)


def test_tick_and_time_normalizers():
    obs = observe(fresh_state(2970), 0)
    assert obs.block("time") == (f32(2970 / 2999), f32(30 / 3000))


def test_config_outside_the_normalizers_is_rejected():
    config = default_config()
    bad_balance = dataclasses.replace(config.balance, queen_hp=250)
    with pytest.raises(ObservationConfigError):
        check_observation_config(dataclasses.replace(config, balance=bad_balance))
    big = dataclasses.replace(config.balance.resources[3], capacity=121)
    resources = config.balance.resources[:3] + (big,) + config.balance.resources[4:]
    with pytest.raises(ObservationConfigError):
        check_observation_config(
            dataclasses.replace(config, balance=dataclasses.replace(config.balance, resources=resources)))


def test_snapshot_type():
    assert isinstance(begin_policy_phase(fresh_state(0)), PolicySnapshot)
