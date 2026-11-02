"""update_local_intel: private, deterministic, sensor-only (FOR-2 errata P0-1, P1-4; goldens H-1, I-1)."""

import pytest

from formicmind import state_hash
from formicmind.intel import IntelPhase, observed_nodes, update_local_intel
from formicmind.model import NOWHERE
from formicmind.observation import build_observation, f32, flatten_v2
from formicmind.serialization import state_to_canonical
from formicmind.snapshot import begin_policy_phase

from helpers import A, B, C, N1, N2, N4, N5, busy_state, fresh_state, place, transit, unit_of


def without_intel(state):
    doc = state_to_canonical(state)
    for colony in doc["colonies"]:
        colony.pop("intel")
    return doc


def test_reset_intel_is_empty_and_first_write_is_the_home_sensor():
    state = fresh_state(0)
    assert all(n.seen_tick == NOWHERE for c in state.colonies for n in c.intel.nodes)
    update_local_intel(state, 0, IntelPhase.PRE_POLICY)
    seen = [i for i, n in enumerate(state.colonies[0].intel.nodes) if n.seen_tick != NOWHERE]
    assert seen == [A]
    assert state.colonies[0].intel.nodes[A].seen_tick == 0
    assert state.colonies[0].intel.enemy_queen_seen_tick == NOWHERE
    # the other side's memory is untouched
    assert all(n.seen_tick == NOWHERE for n in state.colonies[1].intel.nodes)


def test_update_changes_only_the_callers_intel():
    state = busy_state()
    before = without_intel(state)
    other_intel = state.colonies[1].intel.clone()
    update_local_intel(state, 0, IntelPhase.PRE_POLICY)
    assert without_intel(state) == before
    assert state.colonies[1].intel.nodes == other_intel.nodes


@pytest.mark.parametrize("phase", list(IntelPhase))
def test_both_phases_apply_the_same_sensor_rule_and_are_idempotent(phase):
    s1, s2 = busy_state(), busy_state()
    update_local_intel(s1, 0, IntelPhase.PRE_POLICY)
    update_local_intel(s2, 0, phase)
    assert state_hash(s1) == state_hash(s2)
    h = state_hash(s2)
    update_local_intel(s2, 0, phase)
    assert state_hash(s2) == h


def test_bad_arguments():
    state = fresh_state(0)
    with pytest.raises(TypeError):
        update_local_intel(state, 0, "PRE_POLICY")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        update_local_intel(state, 2, IntelPhase.PRE_POLICY)


def test_observed_nodes_are_home_plus_nodes_with_stationed_adults():
    state = busy_state()
    assert observed_nodes(state, 0) == [A, N1, N2, C]
    assert observed_nodes(state, 1) == [C, N4, 6]


def test_adults_in_transit_do_not_observe():
    state = fresh_state(0)
    for u in state.colonies[0].units:
        transit(u, A, N1, 2)
    assert observed_nodes(state, 0) == [A]
    state.colonies[1].units[0].node = N1  # enemy at N1 while A's units are still on the edge
    update_local_intel(state, 0, IntelPhase.PRE_POLICY)
    assert state.colonies[0].intel.nodes[N1].seen_tick == NOWHERE


def test_unobserved_nodes_are_never_read():
    state = busy_state()
    update_local_intel(state, 0, IntelPhase.PRE_POLICY)
    snapshot = state.colonies[0].intel.clone()
    for u in state.colonies[1].units:       # the enemy shuffles around unobserved nodes
        if u.node in (N4, 6):
            place(u, N5)
    state.node_food[N5] = 3
    state.colonies[1].queen_hp = 1
    update_local_intel(state, 0, IntelPhase.POST_ARRIVAL_PRE_COMBAT)
    assert state.colonies[0].intel.nodes[N5] == snapshot.nodes[N5]
    assert state.colonies[0].intel.nodes[N4].seen_tick == NOWHERE
    assert state.colonies[0].intel.enemy_queen_seen_tick == NOWHERE


def test_seen_tick_cannot_go_backwards():
    state = busy_state(60)
    state.colonies[0].intel.nodes[N1].seen_tick = 90
    with pytest.raises(ValueError):
        update_local_intel(state, 0, IntelPhase.PRE_POLICY)


def test_h1_home_without_adults_sees_invaders_and_own_queen_hp():
    state = fresh_state(60)
    for u in state.colonies[0].units:
        place(u, N1)                         # all adults of A are away from home
    for uid in range(29, 33):                # B's four soldiers enter A's home
        place(unit_of(state, 1, uid), A)
    state.colonies[0].queen_hp = 112
    # B's private memory must not reach A's observation, whatever it holds
    state.colonies[1].intel.nodes[A].seen_tick = 30
    state.colonies[1].intel.nodes[A].enemy_workers_last = 9

    snapshot = begin_policy_phase(state)
    obs = build_observation(snapshot, 0)
    assert obs.block("own_node_count")[:2] == (0.0, 0.0)
    assert obs.block("enemy_count_last")[:2] == (0.0, f32(4 / 68))
    assert obs.block("enemy_current")[0] == 1.0 and obs.block("enemy_age_norm")[0] == 0.0
    assert obs.block("own_global")[5] == f32(112 / 240)
    # no other node became visible
    assert [obs.block("enemy_current")[k] for k in range(7)] == [1, 1, 0, 0, 0, 0, 0]

    # B's observation is unaffected by A's private memory
    reference = flatten_v2(build_observation(snapshot, 1))
    state2 = state.clone()
    state2.colonies[0].intel.nodes[N2].seen_tick = 0
    state2.colonies[0].intel.nodes[N2].enemy_soldiers_last = 40
    assert flatten_v2(build_observation(begin_policy_phase(state2), 1)) == reference


def test_i1_last_seen_survives_the_death_of_the_last_observer_and_ages():
    state = fresh_state(61)                  # a non-policy tick: only the intel updater runs
    for u in state.colonies[0].units:
        place(u, A)
    place(unit_of(state, 0, 1), N2)          # a lone observer arrives at N2
    for uid in (29, 30, 31):
        place(unit_of(state, 1, uid), N2)
    state.node_food[N2] = 40
    update_local_intel(state, 0, IntelPhase.POST_ARRIVAL_PRE_COMBAT)
    seen = state.colonies[0].intel.nodes[N2]
    assert (seen.seen_tick, seen.food_last, seen.enemy_soldiers_last) == (61, 40, 3)

    # Combat kills the observer; things at N2 change afterwards
    state.colonies[0].units = [u for u in state.colonies[0].units if u.unit_id != 1]
    place(unit_of(state, 1, 32), N2)
    state.node_food[N2] = 10
    state.tick = 62
    update_local_intel(state, 0, IntelPhase.POST_ARRIVAL_PRE_COMBAT)
    assert state.colonies[0].intel.nodes[N2] == seen

    state.tick = 90
    obs = build_observation(begin_policy_phase(state), 0)
    assert obs.block("food_known")[N2] == 1.0 and obs.block("food_current")[N2] == 0.0
    assert obs.block("food_age_norm")[N2] == f32(29 / 3000)
    assert obs.block("last_food_ratio")[N2] == f32(40 / 60)
    assert obs.block("enemy_count_last")[2 * N2:2 * N2 + 2] == (0.0, f32(3 / 67))
    assert obs.block("enemy_age_norm")[N2] == f32(29 / 3000)
    assert obs.block("enemy_known")[N2] == 1.0 and obs.block("enemy_current")[N2] == 0.0
    state.tick = 150
    obs = build_observation(begin_policy_phase(state), 0)
    assert obs.block("food_age_norm")[N2] == f32(89 / 3000)


def test_last_seen_is_refreshed_when_an_observer_is_present_again():
    state = busy_state(60)
    update_local_intel(state, 0, IntelPhase.PRE_POLICY)
    assert state.colonies[0].intel.nodes[N5].seen_tick == 30
    place(unit_of(state, 0, 8), N5)
    state.node_food[N5] = 44
    state.tick = 90
    update_local_intel(state, 0, IntelPhase.PRE_POLICY)
    entry = state.colonies[0].intel.nodes[N5]
    assert (entry.seen_tick, entry.food_last) == (90, 44)


def test_enemy_queen_hp_is_recorded_only_at_the_enemy_home():
    state = fresh_state(0)
    place(unit_of(state, 0, 13), 6)
    state.colonies[1].queen_hp = 77
    update_local_intel(state, 0, IntelPhase.PRE_POLICY)
    intel = state.colonies[0].intel
    assert (intel.enemy_queen_hp_last, intel.enemy_queen_seen_tick) == (77, 0)
    state.tick = 30
    state.colonies[1].queen_hp = 50
    place(unit_of(state, 0, 13), N4)  # observer left; the record becomes stale, hp not refreshed
    update_local_intel(state, 0, IntelPhase.PRE_POLICY)
    assert (intel.enemy_queen_hp_last, intel.enemy_queen_seen_tick) == (77, 0)
