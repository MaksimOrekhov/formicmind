"""FOR-13 Combat: simultaneous damage, home defence, target priority, no carry-over, events, ledger checks."""

import dataclasses

import pytest

from formicmind.combat import attack_damage, resolve_combat
from formicmind.events import Phase, event_sort_key, events_hash, events_to_canonical
from formicmind.invariants import InvariantError
from formicmind.ledger import check_transition, combat_totals
from formicmind.rules import SOLDIER, WORKER, default_config
from formicmind.serialization import state_hash
from formicmind.tick import FULL_PHASES, step_tick

from combat_helpers import A, B, C, N1, N2, N4, add, arena, ctrl, hp_of, of_kind, soldiers, step, workers
from helpers import transit


# --- exact integer arithmetic -----------------------------------------------------------------------

@pytest.mark.parametrize("n,expected", [(1, 2), (2, 5), (3, 7), (4, 10), (5, 12), (6, 15), (7, 17), (8, 20)])
def test_home_defence_rounds_the_total_down_exactly(n, expected):
    bal = default_config().balance
    base, dmg = attack_damage(n, bal.soldier_damage, True, bal.home_defense_num, bal.home_defense_den)
    assert (base, dmg) == (2 * n, expected)           # floor(1.25 * 2n), not 2.5 per soldier
    assert attack_damage(n, bal.soldier_damage, False, 5, 4) == (2 * n, 2 * n)   # no bonus for attackers


# --- the empty-home raid fixtures (FOR-1 section 6) ----------------------------------------------------

@pytest.mark.parametrize("n,per_phase,phases", [(1, 2, 120), (4, 8, 30), (8, 16, 15)])
@pytest.mark.parametrize("attacker", [0, 1])
def test_raiders_kill_an_unguarded_queen_after_exactly_the_specified_number_of_phases(n, per_phase, phases, attacker):
    state, _ = arena()
    victim_home = B if attacker == 0 else A
    soldiers(state, attacker, n, victim_home)
    last = None
    for phase_no in range(1, phases + 1):
        events = resolve_combat(state)
        (summary,) = of_kind(events, "combat")
        (hit,) = of_kind(events, "queen_damage")
        assert (summary.target, summary.home_bonus, summary.damage, summary.wasted) == ("queen", False, per_phase, 0)
        assert hit.queen_hp_before == 240 - per_phase * (phase_no - 1)
        assert hit.queen_hp_after == 240 - per_phase * phase_no          # absolute value for a spectator
        last = hit
    assert state.colonies[1 - attacker].queen_hp == 0 and last.queen_hp_after == 0
    assert state.colonies[attacker].queen_hp == 240                      # the queen never strikes back


def test_four_raiders_leave_the_queen_with_8_hp_after_29_phases():
    state, _ = arena()
    soldiers(state, 0, 4, B)
    for _ in range(29):
        resolve_combat(state)
    assert state.colonies[1].queen_hp == 8
    assert resolve_combat(state)[-1].queen_hp_after == 0


def test_the_queen_is_hit_only_in_her_own_home_never_from_another_node():
    state, _ = arena()
    soldiers(state, 0, 8, C)            # many soldiers, in the centre
    soldiers(state, 0, 8, N4)           # next to B's home, but not in it
    assert resolve_combat(state) == []
    assert state.colonies[1].queen_hp == 240
    soldiers(state, 0, 1, A)            # own home with nobody to hit
    assert resolve_combat(state) == []


# --- simultaneity ------------------------------------------------------------------------------------

def test_spec_fixture_three_against_two_soldiers_in_the_centre():
    state, _ = arena()
    soldiers(state, 0, 3, C)
    soldiers(state, 1, 2, C)
    events = resolve_combat(state)
    a_hits, b_hits = [e for e in of_kind(events, "combat") if e.attacker_side == 0][0], \
        [e for e in of_kind(events, "combat") if e.attacker_side == 1][0]
    assert (a_hits.damage, b_hits.damage) == (6, 4) and not a_hits.home_bonus and not b_hits.home_bonus
    assert sum(hp for _, hp in hp_of(state, 0)) == 86 and len(hp_of(state, 0)) == 3
    assert sum(hp for _, hp in hp_of(state, 1)) == 54 and len(hp_of(state, 1)) == 2


def test_both_sides_strike_from_the_pre_combat_state_even_when_both_soldiers_die():
    state, _ = arena()
    soldiers(state, 0, 1, C, hp=2)
    soldiers(state, 1, 1, C, hp=2)
    events = resolve_combat(state)
    assert [e.side for e in of_kind(events, "death")] == [0, 1]      # both fall in the same phase
    assert state.colonies[0].units == [] and state.colonies[1].units == []
    assert {e.damage for e in of_kind(events, "combat")} == {2} and all(e.wasted == 0 for e in of_kind(events, "combat"))


def test_a_soldier_killed_in_the_phase_still_deals_its_damage():
    state, _ = arena()
    s_a, = soldiers(state, 0, 1, C, hp=2)
    soldiers(state, 1, 2, C, hp=30)
    resolve_combat(state)
    assert state.colonies[0].units == []                              # A's lone soldier died ...
    assert sum(hp for _, hp in hp_of(state, 1)) == 58                 # ... but still wounded B for 2
    assert s_a.hp == 0 or s_a not in state.colonies[0].units


def test_processing_order_of_the_sides_does_not_matter_mirror_state_gives_mirrored_events():
    state, _ = arena()
    soldiers(state, 0, 5, C, hp=11)
    workers(state, 0, 2, C)
    soldiers(state, 1, 3, C, hp=17)
    mirror, _ = arena()
    soldiers(mirror, 1, 5, C, hp=11)
    workers(mirror, 1, 2, C)
    soldiers(mirror, 0, 3, C, hp=17)

    def signature(events, swap):
        out = []
        for e in events:
            if e.kind == "combat":
                att = 1 - e.attacker_side if swap else e.attacker_side
                out.append((e.kind, e.node, att, e.damage, e.applied, e.wasted, e.target))
            elif e.kind in ("wound", "death"):
                owner = 1 - e.side if swap else e.side
                out.append((e.kind, e.node, owner, e.caste, e.hp_before, e.damage, e.hp_after))
        return sorted(out)

    e1, e2 = resolve_combat(state), resolve_combat(mirror)
    assert signature(e1, swap=False) == signature(e2, swap=True)
    for side in (0, 1):
        assert sorted(hp for _, hp in hp_of(state, side)) == sorted(hp for _, hp in hp_of(mirror, 1 - side))


# --- the home-defence bonus --------------------------------------------------------------------------

def test_defending_soldiers_get_five_quarters_and_the_attackers_do_not():
    state, _ = arena()
    soldiers(state, 0, 4, B)            # raiders in B's home
    soldiers(state, 1, 4, B)            # the defenders
    events = resolve_combat(state)
    attack_a = next(e for e in of_kind(events, "combat") if e.attacker_side == 0)
    attack_b = next(e for e in of_kind(events, "combat") if e.attacker_side == 1)
    assert (attack_a.damage, attack_a.home_bonus) == (8, False)
    assert (attack_b.damage, attack_b.base_damage, attack_b.home_bonus) == (10, 8, True)
    assert hp_of(state, 1)[0][1] == 22 and hp_of(state, 0)[0][1] == 20   # first soldier absorbs the pour


def test_the_same_soldiers_deal_the_bonus_in_their_own_home_and_not_elsewhere():
    state, _ = arena()
    soldiers(state, 0, 4, A)
    workers(state, 1, 1, A, hp=100)      # an enemy worker in A's home is hit by the defenders with the bonus
    (e,) = of_kind(resolve_combat(state), "combat")
    assert (e.damage, e.home_bonus, e.target) == (10, True, "worker")
    state2, _ = arena()
    soldiers(state2, 0, 4, N1)
    workers(state2, 1, 1, N1, hp=100)
    (e2,) = of_kind(resolve_combat(state2), "combat")
    assert (e2.damage, e2.home_bonus) == (8, False)


# --- target priority, wounds, death order ------------------------------------------------------------

def test_damage_is_poured_over_units_in_unit_id_order_with_individual_hp():
    state, _ = arena()
    soldiers(state, 0, 35, C)                                 # 70 damage
    victims = soldiers(state, 1, 3, C, hp=30)
    events = resolve_combat(state)
    summary = next(e for e in of_kind(events, "combat") if e.attacker_side == 0)
    assert (summary.damage, summary.target, summary.applied, summary.wasted) == (70, "soldier", 70, 0)
    assert [(e.unit_id, e.damage, e.hp_after) for e in of_kind(events, "death")] == [
        (victims[0].unit_id, 30, 0), (victims[1].unit_id, 30, 0)]
    assert [(e.unit_id, e.hp_before, e.damage, e.hp_after) for e in of_kind(events, "wound")
            if e.side == 1] == [(victims[2].unit_id, 30, 10, 20)]


def test_overkill_is_wasted_and_never_carried_to_workers_or_the_queen():
    state, _ = arena()
    soldiers(state, 0, 10, B)                                 # 20 damage into B's home
    lone, = soldiers(state, 1, 1, B, hp=5)
    ws = workers(state, 1, 3, B)
    events = resolve_combat(state)
    summary = next(e for e in of_kind(events, "combat") if e.attacker_side == 0)
    assert (summary.target, summary.target_hp_total, summary.applied, summary.wasted) == ("soldier", 5, 5, 15)
    assert [e.unit_id for e in of_kind(events, "death") if e.side == 1] == [lone.unit_id]
    assert [(u.unit_id, u.hp) for u in state.colonies[1].units] == [(w.unit_id, 20) for w in ws]
    assert state.colonies[1].queen_hp == 240 and not of_kind(events, "queen_damage")
    # the next phase turns on the workers; the phase after that on the queen
    events = resolve_combat(state)
    assert next(e for e in of_kind(events, "combat") if e.attacker_side == 0).target == "worker"
    assert state.colonies[1].queen_hp == 240


def test_workers_shield_the_queen_until_the_phase_after_the_last_one_dies():
    state, _ = arena()
    soldiers(state, 0, 4, B)                                  # 8 damage per phase
    w1, w2 = workers(state, 1, 2, B, hp=20)
    w2.hp = 4                                                 # 20 + 4 = 24 hp of shield: exactly 3 phases
    seen = []
    for _ in range(4):
        events = resolve_combat(state)
        seen.append((next(e for e in of_kind(events, "combat")).target, state.colonies[1].queen_hp))
    assert seen == [("worker", 240), ("worker", 240), ("worker", 240), ("queen", 232)]
    assert state.colonies[1].units == []


def test_the_phase_that_kills_the_last_defender_wastes_the_rest_instead_of_hitting_the_queen():
    state, _ = arena()
    soldiers(state, 0, 4, B)
    workers(state, 1, 1, B, hp=3)
    events = resolve_combat(state)
    (summary,) = of_kind(events, "combat")
    assert (summary.target, summary.applied, summary.wasted) == ("worker", 3, 5)
    assert state.colonies[1].queen_hp == 240 and not of_kind(events, "queen_damage")


def test_soldiers_in_a_neutral_node_hit_enemy_workers_but_workers_never_hit_back():
    state, _ = arena()
    soldiers(state, 0, 2, N1)
    at_n1 = workers(state, 1, 5, N1)
    own = workers(state, 0, 5, N2)
    at_n2 = workers(state, 1, 5, N2)                          # workers alone: no combat at all
    events = resolve_combat(state)
    assert [(e.node, e.attacker_side, e.target) for e in of_kind(events, "combat")] == [(N1, 0, "worker")]
    assert [u.hp for u in at_n1] == [16, 20, 20, 20, 20]      # 4 damage poured into the lowest unit_id
    assert [u.hp for u in own + at_n2] == [20] * 10


def test_units_on_edges_neither_fight_nor_are_hurt():
    state, _ = arena(tick=1)
    soldiers(state, 0, 4, C)
    (travelling,) = soldiers(state, 1, 1, C)
    transit(travelling, C, N4, 3)                              # enemy soldier moving away along an edge
    assert resolve_combat(state) == []
    assert travelling.hp == 30


def test_a_node_with_soldiers_of_one_side_and_nobody_else_has_no_combat():
    state, _ = arena()
    soldiers(state, 0, 4, N2)
    assert resolve_combat(state) == []


def test_a_newborn_standing_in_a_raided_home_takes_damage_in_the_tick_it_hatches():
    from formicmind.model import BroodRecord
    state, run = arena(tick=20)
    state.colonies[1].brood.append(BroodRecord(state.allocate_unit_id(), WORKER, 0, 20))
    soldiers(state, 0, 1, B)                                   # raider already standing in B's home
    soldiers(state, 1, 2, N1)                                  # B keeps some soldiers away
    result = step_tick(state, run, ctrl(), check_invariants=True)
    (hatch,) = of_kind(result.events, "hatch")
    (wound,) = [e for e in of_kind(result.events, "wound") if e.side == 1]
    assert wound.unit_id == hatch.unit_id and (wound.hp_before, wound.hp_after) == (20, 18)
    assert next(u for u in state.colonies[1].units if u.unit_id == hatch.unit_id).hp == 18


# --- event contract ------------------------------------------------------------------------------------

def test_combat_events_are_ordered_typed_and_canonical():
    state, run = arena(tick=1)
    soldiers(state, 0, 4, B)
    workers(state, 1, 2, B)
    soldiers(state, 1, 2, B)
    soldiers(state, 0, 3, C)
    soldiers(state, 1, 1, C)
    result = step(state, run)
    combat_events = [e for e in result.events if int(e.phase) == Phase.COMBAT]
    assert [e.kind for e in combat_events] == sorted((e.kind for e in combat_events), key=lambda k: {
        "combat": 0, "wound": 1, "death": 2, "queen_damage": 3}[k])
    assert [e.kind for e in combat_events][:4] == ["combat"] * 4
    assert result.events == tuple(sorted(result.events, key=event_sort_key))
    canon = events_to_canonical(result.events)
    assert all(type(v) in (int, str, bool, list) for d in canon for v in d.values())
    assert events_hash(result.events) == events_hash(list(result.events))
    first = of_kind(combat_events, "combat")[0]
    assert (first.node, first.attacker_side) == (C, 0)         # (node, attacker) order: C before B's home


def test_combat_totals_summarise_who_hurt_whom():
    state, _ = arena()
    soldiers(state, 0, 35, C)
    soldiers(state, 1, 3, C)
    totals = combat_totals(resolve_combat(state))
    assert (totals.sides[0].soldiers_killed, totals.sides[0].wounds_dealt, totals.sides[0].applied) == (2, 1, 70)
    assert totals.sides[1].lost_soldiers == 2 and totals.sides[1].damage == 6 and totals.sides[0].wasted == 0


# --- the phase inside the tick -------------------------------------------------------------------------

def test_arrival_and_sensor_come_first_then_combat_then_queen_terminal_then_harvest():
    state, run = arena(tick=1)
    soldiers(state, 0, 2, N1, target=N4)
    ws = workers(state, 0, 1, N1)
    soldiers(state, 1, 1, N1)
    transit(add(state, 1, SOLDIER, 1, N1)[0], N4, N1, 1)       # arrives in N1 in this very tick
    order = []
    table = dict(FULL_PHASES)
    for phase, func in FULL_PHASES.items():
        def wrapped(ctx, phase=phase, func=func):
            order.append((phase, sum(1 for u in ctx.state.colonies[1].units if u.node == N1),
                          ctx.state.colonies[0].intel.nodes[N1].enemy_soldiers_last))
            func(ctx)
        table[phase] = wrapped
    step_tick(state, run, ctrl(), phases=table)
    assert [p for p, _, _ in order] == list(Phase)
    by_phase = {p: (n, seen) for p, n, seen in order}
    assert by_phase[Phase.ARRIVAL][0] == 1 and by_phase[Phase.COMBAT][0] == 2    # B's second soldier arrived
    assert by_phase[Phase.COMBAT][1] == 2                       # the sensor ran after Arrival, before Combat
    assert ws[0].node == N1


def test_sensor_record_is_pre_combat_and_combat_deaths_do_not_leak_into_private_intel():
    state, run = arena(tick=1)
    soldiers(state, 0, 1, C, hp=2)
    soldiers(state, 1, 5, C, hp=2)
    step(state, run)
    intel = state.colonies[0].intel.nodes[C]
    assert intel.enemy_soldiers_last == 5 and intel.seen_tick == 1      # not updated after the fight
    assert state.colonies[0].units == [] and len(state.colonies[1].units) == 4


def test_blocked_harvest_exactly_while_an_enemy_soldier_survives_the_combat_phase():
    # A holds N1 with 8 workers and a soldier; B's single soldier is too weak to survive the same phase.
    state, run = arena(tick=1)
    workers(state, 0, 8, N1)
    soldiers(state, 0, 1, N1, hp=2)
    soldiers(state, 1, 1, N1, hp=2)
    result = step(state, run)
    harvest, = [e for e in of_kind(result.events, "harvest") if e.side == 0]
    assert harvest.blocked is False and harvest.harvested == 8          # the enemy died in Combat first

    state, run = arena(tick=1)
    workers(state, 0, 8, N1)
    soldiers(state, 0, 1, N1, hp=2)
    soldiers(state, 1, 1, N1, hp=30)                                   # survives (A deals 2)
    result = step(state, run)
    harvest, = [e for e in of_kind(result.events, "harvest") if e.side == 0]
    assert harvest.blocked is True and harvest.harvested == 0           # own soldiers do not help (errata B)


def test_automatic_depart_releases_units_in_the_same_tick_the_blocker_dies():
    def run_one(**flags):
        state, run = arena(tick=1)
        w, = workers(state, 0, 1, N1, target=C)
        soldiers(state, 0, 1, N1, hp=2)
        soldiers(state, 1, 1, N1, hp=2)
        step_tick(state, run, ctrl(), check_invariants=flags.get("check", False), **{
            k: v for k, v in flags.items() if k != "check"})
        return w

    freed = run_one(check=True)
    assert freed.node == -1 and freed.to_node == C                      # left N1 in AutomaticDepart
    held = run_one(partial_economy=True)                               # no Combat: still blocked
    assert held.node == N1


# --- ledger: damage conservation is verified, forgeries are caught --------------------------------------

def _fight():
    state, run = arena(tick=1)
    soldiers(state, 0, 4, B)
    workers(state, 1, 1, B, hp=3)
    soldiers(state, 1, 2, A)                                   # B raids A's unguarded home at the same time
    before = state.clone()
    result = step_tick(state, run, ctrl())
    return before, state, result


def test_check_transition_accepts_the_real_combat_and_rejects_each_forgery():
    before, after, result = _fight()
    check_transition(before, after, result.events)                      # legitimate

    events = list(result.events)
    combat = next(i for i, e in enumerate(events) if e.kind == "combat")

    def forged(index, **changes):
        out = list(events)
        out[index] = dataclasses.replace(events[index], **changes)
        return out

    with pytest.raises(InvariantError, match="damage"):
        check_transition(before, after, forged(combat, damage=events[combat].damage + 1,
                                              wasted=events[combat].wasted + 1))
    with pytest.raises(InvariantError):                                # damage carried over: applied > pool
        check_transition(before, after, forged(combat, applied=events[combat].applied + 1,
                                              wasted=events[combat].wasted - 1))
    with pytest.raises(InvariantError, match="home bonus"):
        check_transition(before, after, forged(combat, home_bonus=True))
    with pytest.raises(InvariantError, match="base damage"):
        check_transition(before, after, forged(combat, attacker_soldiers=5))


def test_check_transition_rejects_healing_unrecorded_damage_and_a_regenerating_queen():
    before, after, result = _fight()
    healed = after.clone()
    healed.colonies[0].units[0].hp -= 1
    with pytest.raises(InvariantError, match="hp changed without recorded combat damage"):
        check_transition(before, healed, result.events)
    assert after.colonies[0].queen_hp == 236                  # 2 raiders * 2
    queen = after.clone()
    queen.colonies[0].queen_hp = 240
    with pytest.raises(InvariantError, match="queen_hp does not match queen_damage"):
        check_transition(before, queen, result.events)
    revived = after.clone()
    revived.colonies[0].queen_hp = 239
    before.colonies[0].queen_hp = 238
    with pytest.raises(InvariantError, match="queen_hp may only fall"):
        check_transition(before, revived, result.events)


def test_a_vanished_unit_needs_a_death_or_a_starvation_event():
    before, after, result = _fight()
    no_death = [e for e in result.events if e.kind != "death"]
    with pytest.raises(InvariantError):
        check_transition(before, after, no_death)
    ghost = after.clone()
    ghost.colonies[0].units.pop()
    with pytest.raises(InvariantError, match="vanish"):
        check_transition(before, ghost, result.events)


def test_combat_death_is_not_mistaken_for_starvation_and_both_ledgers_hold_in_one_tick():
    state, run = arena(tick=1, food=3)                                  # A is hungry too
    soldiers(state, 0, 20, B)                                          # 40 damage kills both 20-hp workers
    victims = workers(state, 1, 2, B)
    before = state.clone()
    result = step_tick(state, run, ctrl(), check_invariants=True)
    kinds = {e.kind for e in result.events}
    assert "death" in kinds and "starvation" in kinds
    check_transition(before, state, result.events)
    died = {e.unit_id for e in of_kind(result.events, "death")}
    starved = {e.unit_id for e in of_kind(result.events, "starvation")}
    assert died == {v.unit_id for v in victims} and not (died & starved)


def test_no_rng_is_consumed_by_combat_or_the_terminal_rules():
    state, run = arena(tick=1, streams=["policy:alpha:0"])
    soldiers(state, 0, 4, B)
    before = {sid: s.to_canonical() for sid, s in run.streams.items()}
    for _ in range(3):
        step(state, run)
    assert {sid: s.to_canonical() for sid, s in run.streams.items()} == before


def test_resolve_combat_is_pure_in_the_inputs_same_state_same_events_and_hashes():
    def build():
        state, _ = arena()
        soldiers(state, 0, 6, C)
        workers(state, 0, 3, C)
        soldiers(state, 1, 5, C, hp=13)
        return state

    s1, s2 = build(), build()
    assert events_hash(resolve_combat(s1)) == events_hash(resolve_combat(s2))
    assert state_hash(s1) == state_hash(s2)


# --- information boundary (FOR-2 errata P0-1 / P1-4) ----------------------------------------------------

def test_own_home_is_sensed_even_with_zero_adults_but_the_record_is_pre_combat():
    state, run = arena(tick=1)
    soldiers(state, 1, 4, A)                                   # B raids A's home; A has no adult at all
    workers(state, 1, 1, B)
    state.colonies[0].queen_hp = 100
    before_b = state.colonies[1].intel.enemy_queen_hp_last
    step(state, run)
    a_view = state.colonies[0].intel.nodes[A]
    assert (a_view.seen_tick, a_view.enemy_soldiers_last) == (1, 4)    # permanent home sensor, no adults needed
    assert state.colonies[0].queen_hp == 92                            # the real queen was hit in Combat ...
    assert state.colonies[1].intel.enemy_queen_hp_last == 100          # ... the raiders' record is the pre-combat value
    assert before_b == 0
    assert state.colonies[1].intel.enemy_queen_seen_tick == 1


def test_a_node_whose_last_observer_dies_keeps_its_old_record_and_ages():
    state, run = arena(tick=1)
    soldiers(state, 0, 1, N1, hp=2)
    soldiers(state, 1, 3, N1, hp=30)
    step(state, run)                                           # A's only unit dies in Combat
    assert state.colonies[0].units == []
    n1 = state.colonies[0].intel.nodes[N1]
    assert (n1.seen_tick, n1.enemy_soldiers_last) == (1, 3)
    step(state, run)                                           # nobody observes N1 any more: nothing is updated
    assert state.colonies[0].intel.nodes[N1].seen_tick == 1


def test_observations_never_expose_enemy_hp_brood_food_or_an_unseen_queen_after_a_fight():
    from formicmind.model import BroodRecord
    from formicmind.observation import build_observation, flatten_v2
    from formicmind.snapshot import begin_policy_phase
    state, run = arena(tick=29)
    soldiers(state, 0, 6, N1)
    soldiers(state, 1, 2, N1, hp=27)
    state.colonies[1].brood.append(BroodRecord(state.allocate_unit_id(), WORKER, 10, 30))
    step(state, run)
    assert [u.hp for u in state.colonies[1].units] == [15, 27]         # 12 damage poured into the lowest id

    def observe(s):
        return list(flatten_v2(build_observation(begin_policy_phase(s.clone(), tuple(run.windows)), 0)))

    twin = state.clone()
    for u in twin.colonies[1].units:
        u.hp = 1                                                       # other enemy hp
    twin.colonies[1].queen_hp = 77                                     # A has no unit in B's home: unseen
    twin.colonies[1].brood.append(BroodRecord(twin.allocate_unit_id(), SOLDIER, 5, 35))   # other enemy brood
    twin.colonies[1].food += 500
    assert observe(state) == observe(twin)
    killed = state.clone()
    killed.colonies[1].units.pop()                                     # control: a visible change does show up
    assert observe(state) != observe(killed)
