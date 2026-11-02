"""Tick skeleton: phases, policy cadence, atomic failure, windows/checkpoint, intel, determinism."""

import dataclasses
import json

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
from formicmind.actions import ActionV2, InvalidAction
from formicmind.canonical import CanonicalError, canonical_dumps
from formicmind.decoder import PolicyLogitsV2, PolicyOutputError
from formicmind.events import Phase, event_sort_key, events_hash, events_to_canonical
from formicmind.invariants import validate_state
from formicmind.observation import flatten_v2
from formicmind.policy import decide
from formicmind.rng import policy_stream_id
from formicmind.routing import edge_length
from formicmind.snapshot import WindowStats, begin_policy_phase
from formicmind.tick import (
    DEFAULT_PHASES,
    IMPLEMENTED_PHASES,
    NOT_IMPLEMENTED_PHASES,
    TickMode,
    PolicyController,
    ScriptedController,
    step_tick,
)

from helpers import A, B, C, N1, N2, N4, N5, place, unit_of

Q = 65535
HOME_Q = (Q, 0, 0, 0, 0, 0)


def act(**kw):
    base = dict(lay_order=0, forage_share_q=HOME_Q, army_choice=0, avoid_center=False)
    base.update(kw)
    return ActionV2(**base)


def new_run(a=None, b=None, seed=1, streams=()):
    state, run = reset(default_config(), seed, streams)
    return state, run, (ScriptedController(a or {}), ScriptedController(b or {}))


def advance(state, run, ctrl, n):
    events = []
    for _ in range(n):
        events.extend(step_tick(state, run, ctrl, movement_only=True).events)
    return events


def obs_pair(state, run):
    snap = begin_policy_phase(state.clone(), tuple(run.windows))
    return [flatten_v2(build_observation(snap, s)) for s in (0, 1)]


class Recorder:
    """Controller that records what it was shown and returns a fixed action."""

    def __init__(self, action=None):
        self.calls, self.action = [], action or act()

    def act(self, tick, observation, run):
        from formicmind.actions import get_action_mask, validate_action
        self.calls.append((tick, flatten_v2(observation)))
        return validate_action(observation, get_action_mask(observation), self.action)


# --- phases and boundaries ------------------------------------------------------------------

def test_phase_order_and_explicitly_unimplemented_phases():
    assert [p.name for p in Phase] == [
        "POLICY", "COMPLETION", "ARRIVAL", "COMBAT", "QUEEN_TERMINAL", "HARVEST", "UPKEEP", "LAY",
        "REGENERATION", "AUTOMATIC_DEPART", "TERMINAL_LOG"]
    assert [int(p) for p in Phase] == list(range(11))
    assert IMPLEMENTED_PHASES == (Phase.POLICY, Phase.ARRIVAL, Phase.AUTOMATIC_DEPART, Phase.TERMINAL_LOG)
    assert {p.name for p in NOT_IMPLEMENTED_PHASES} == {
        "COMPLETION", "COMBAT", "QUEEN_TERMINAL", "HARVEST", "UPKEEP", "LAY", "REGENERATION"}
    state, run, ctrl = new_run()
    result = step_tick(state, run, ctrl, movement_only=True)
    assert [r.phase for r in result.phases] == list(Phase)
    assert [(r.phase, r.implemented) for r in result.phases if not r.implemented] == [
        (p, False) for p in NOT_IMPLEMENTED_PHASES]
    assert result.physics_complete is False and result.terminal is None


def test_step_without_flags_is_the_full_match_tick_and_the_dev_modes_stay_incomplete():
    # FOR-13: no flags -> every phase is implemented and physics is complete; the old modes are explicit
    # development modes (no Combat/QueenTerminal/terminal check) and are never "complete".
    state, run, ctrl = new_run()
    result = step_tick(state, run, ctrl)
    assert result.mode is TickMode.FULL and result.physics_complete is True
    assert result.missing_phases == () and result.terminal is None and result.outcome is None
    assert [r.phase for r in result.phases] == list(Phase) and all(r.implemented for r in result.phases)
    assert state.tick == 1
    for kw in ({"movement_only": True}, {"partial_economy": True}):
        s2, r2, c2 = new_run()
        dev = step_tick(s2, r2, c2, **kw)
        assert dev.physics_complete is False and dev.mode is not TickMode.FULL


def test_tick_boundary_state_tick_is_start_of_tick_and_events_carry_the_executed_tick():
    state, run, ctrl = new_run({0: act(army_choice=1)})
    assert state.tick == 0
    r0 = step_tick(state, run, ctrl, movement_only=True)
    assert (r0.tick, state.tick) == (0, 1)
    assert {e.tick for e in r0.events} == {0}
    r1 = step_tick(state, run, ctrl, movement_only=True)
    r2 = step_tick(state, run, ctrl, movement_only=True)
    assert (r1.tick, r2.tick, state.tick) == (1, 2, 3)
    assert {e.tick for e in r2.events} == {2}
    validate_state(state)


def test_horizon_guard():
    state, run, ctrl = new_run()
    state.tick = state.config.t_max
    with pytest.raises(ValueError):
        step_tick(state, run, ctrl, movement_only=True)


def test_terminal_stops_mid_tick_without_running_the_remaining_phases():
    state, run, ctrl = new_run({0: act(army_choice=1)})

    def terminate(ctx):
        ctx.terminal = "test-terminal"

    phases = dict(DEFAULT_PHASES)
    phases[Phase.ARRIVAL] = terminate
    result = step_tick(state, run, ctrl, movement_only=True, phases=phases)
    assert result.terminal == "test-terminal"
    assert [r.phase for r in result.phases] == [Phase.POLICY, Phase.COMPLETION, Phase.ARRIVAL]
    assert state.tick == 0                                    # not advanced
    assert run.windows == [WindowStats(), WindowStats()]      # Terminal/Log never ran
    assert not any(e.kind == "arrive" for e in result.events)
    assert result.events                                      # the Policy phase had already happened


# --- policy cadence and the single snapshot -----------------------------------------------------

def test_policy_runs_only_at_multiples_of_k_and_orders_persist_between():
    rec_a, rec_b = Recorder(act(army_choice=1)), Recorder()
    state, run = reset(default_config(), 1)
    events = advance(state, run, (rec_a, rec_b), 65)
    assert [t for t, _ in rec_a.calls] == [0, 30, 60] == [t for t, _ in rec_b.calls]
    orders = [e for e in events if e.kind == "order"]
    assert sorted({e.tick for e in orders}) == [0, 30, 60] and len(orders) == 6
    assert state.colonies[0].army_choice == 1                 # standing between policy ticks


def test_both_observations_are_built_from_one_pre_action_snapshot():
    rec_a, rec_b = Recorder(act(army_choice=6, forage_share_q=(0, Q, 0, 0, 0, 0))), Recorder()
    state, run = reset(default_config(), 1)
    advance(state, run, (rec_a, rec_b), 1)
    reference = obs_pair(reset(default_config(), 1)[0], reset(default_config(), 1)[1])
    assert [rec_a.calls[0][1], rec_b.calls[0][1]] == reference
    # what A orders does not change what B was shown at the same policy tick
    rec_a2, rec_b2 = Recorder(act(army_choice=1)), Recorder()
    state2, run2 = reset(default_config(), 1)
    advance(state2, run2, (rec_a2, rec_b2), 1)
    assert rec_b2.calls[0][1] == rec_b.calls[0][1]


# --- atomic failure ---------------------------------------------------------------------------------

def random_policy(observation, mask, policy_rng):
    return PolicyLogitsV2([policy_rng.next_u32() / 2**31 - 1 for _ in range(17)])


def nan_policy(observation, mask, policy_rng):
    return [float("nan")] * 17


@pytest.mark.parametrize("bad_b, error", [
    (ScriptedController({0: ActionV2(0, (16384, 16384, 16384, 16383, 0, 0), 0, False)}), InvalidAction),
    (ScriptedController({0: ActionV2(3, HOME_Q, 0, False)}), InvalidAction),
    (ScriptedController({0: ActionV2(0, HOME_Q, 0, True)}), InvalidAction),
    (PolicyController(nan_policy, policy_stream_id("beta", 0)), PolicyOutputError),
])
def test_a_failing_side_leaves_no_partial_orders_or_departures(bad_b, error):
    ids = [policy_stream_id("alpha", 0), policy_stream_id("beta", 0)]
    state, run = reset(default_config(), 1, ids)
    good_a = PolicyController(random_policy, ids[0])        # A decides first and consumes its RNG
    before = (state_hash(state), checkpoint_hash(state, run), state.tick, list(run.windows))
    intel_before = [c.intel.clone() for c in state.colonies]
    with pytest.raises(error):
        step_tick(state, run, (good_a, bad_b), movement_only=True)
    assert (state_hash(state), checkpoint_hash(state, run), state.tick, list(run.windows)) == before
    assert [c.intel for c in state.colonies] == intel_before
    assert all(u.node != -1 for c in state.colonies for u in c.units)   # nobody left


def test_controller_must_return_a_validated_action():
    state, run = reset(default_config(), 1)

    class Raw:
        def act(self, tick, observation, run):
            return act()

    before = state_hash(state)
    with pytest.raises(TypeError):
        step_tick(state, run, (Raw(), Raw()), movement_only=True)
    assert state_hash(state) == before


# --- windows (F1) and checkpoint -------------------------------------------------------------------

def test_window_counters_accumulate_between_policy_ticks_and_restart_at_each_policy_tick():
    state, run, ctrl = new_run()
    assert run.windows == [WindowStats(), WindowStats()]
    advance(state, run, ctrl, 30)
    assert [w.window_ticks for w in run.windows] == [30, 30]
    obs = [build_observation(begin_policy_phase(state.clone(), tuple(run.windows)), s) for s in (0, 1)]
    assert [o.block("own_global")[17] for o in obs] == [1.0, 1.0]       # window_ticks / K
    advance(state, run, ctrl, 1)                                         # policy tick 30 itself
    assert [w.window_ticks for w in run.windows] == [1, 1]
    advance(state, run, ctrl, 29)
    assert [w.window_ticks for w in run.windows] == [30, 30]


def test_checkpoint_restore_at_a_policy_tick_gives_identical_observation_bytes():
    state, run, ctrl = new_run({0: act(forage_share_q=(21845, 43690, 0, 0, 0, 0), army_choice=3)})
    advance(state, run, ctrl, 30)
    assert state.tick == 30 and run.windows[0].window_ticks == 30
    data = checkpoint_to_bytes(state, run)
    state2, run2 = checkpoint_from_bytes(data, state.config)
    assert checkpoint_hash(state2, run2) == checkpoint_hash(state, run)
    assert obs_pair(state2, run2) == obs_pair(state, run)
    # the live-run observation really contains the non-zero window...
    obs = build_observation(begin_policy_phase(state2.clone(), tuple(run2.windows)), 0)
    assert obs.block("own_global")[17] == 1.0
    # ...and a restore that forgot the windows would silently build a different one (the F1 hazard)
    forgot = (WindowStats(), WindowStats())
    assert flatten_v2(build_observation(begin_policy_phase(state2.clone(), forgot), 0)) != obs_pair(state2, run2)[0]


def test_continuing_after_restore_equals_the_uninterrupted_run():
    script = {0: act(forage_share_q=(21845, 43690, 0, 0, 0, 0), army_choice=6),
              30: act(forage_share_q=(0, 0, Q, 0, 0, 0), army_choice=6, avoid_center=True)}
    state, run, ctrl = new_run(script)
    advance(state, run, ctrl, 30)
    state2, run2 = checkpoint_from_bytes(checkpoint_to_bytes(state, run), state.config)
    tail_full = advance(state, run, ctrl, 45)
    tail_restored = advance(state2, run2, ctrl, 45)
    assert events_hash(tail_full) == events_hash(tail_restored)
    assert (state_hash(state), checkpoint_hash(state, run)) == (state_hash(state2), checkpoint_hash(state2, run2))


def test_golden_hashes_after_routing_errata_r1_and_checkpoint_v2():
    from formicmind import config_hash
    state, run = reset(default_config(), 20261102)
    assert state.config.rules_version == "l0-proto-FOR1-v0.2.2"
    # Intentional diffs: rules_version bumped to v0.2.2 (R1) changes config/state hashes;
    # checkpoint format v2 (domain checkpoint/v2, schema /2, windows) changes the checkpoint hash.
    assert config_hash(state.config) == "3f6aaa36a0d3d15888fa42f142df1151376794e62b553ab8e5f7659c9794145d"
    assert state_hash(state) == "675116b1ac021868d41fd234ed9642c7bae5b59377daa239355913602555f3d3"
    # FOR-13: checkpoint format v3 (schema /3, domain checkpoint/v3, outcome) changes the checkpoint hash only.
    assert checkpoint_hash(state, run) == "bd2418474f225b83a9c81ee067ba65a05d72b39b8b11731508a0ac92419bd0b9"


def test_state_and_config_from_the_previous_rules_version_are_rejected():
    from formicmind.config import SimConfig  # noqa: F401
    from formicmind.invariants import InvariantError
    from formicmind.serialization import config_from_bytes, config_to_canonical
    state, run = reset(default_config(), 1)
    old_rules = "l0-proto-FOR1-v0.2.1"
    doc = json.loads(checkpoint_to_bytes(state, run))
    doc["state"]["rules_version"] = old_rules
    doc["state"]["config_sha256"] = "c93de10ef90cc2b77ad11f635dac5cf2c5d9a8984524b0898caaaa26f63f6274"
    with pytest.raises(CanonicalError):
        checkpoint_from_bytes(canonical_dumps(doc), state.config)
    cfg = config_to_canonical(state.config)
    cfg["rules_version"] = old_rules
    with pytest.raises(InvariantError):
        config_from_bytes(canonical_dumps(cfg))


def test_contract_pins_record_routing_errata_r1():
    from formicmind.versions import CONTRACT_PINS, RULES_VERSION, BALANCE_VERSION, OBSERVATION_VERSION
    keys = [p.key for p in CONTRACT_PINS]
    assert "FOR-1-routing-errata-R1" in keys and "FOR-2-routing-errata-R1-sync" in keys
    assert "FOR-1-FOR12-economy-decisions" in keys
    assert all(p.url.startswith("https://linear.app/") and p.updated_at.endswith("Z") for p in CONTRACT_PINS)
    assert RULES_VERSION == "l0-proto-FOR1-v0.2.2"
    # R1 touches neither balance nor the agent contract versions
    assert BALANCE_VERSION == "candidate-FOR1-v0.2.1-unvalidated"
    assert OBSERVATION_VERSION == "FOR-2-v0.2+errata-v0.2.1"


def test_old_checkpoint_format_fails_fast_and_bad_windows_are_rejected():
    state, run = reset(default_config(), 1)
    doc = json.loads(checkpoint_to_bytes(state, run))
    v1 = json.loads(json.dumps(doc))
    v1["schema"] = "formicmind.checkpoint/1"
    v1["run"]["schema"] = "formicmind.run_state/1"
    del v1["run"]["windows"]
    with pytest.raises(CanonicalError):
        checkpoint_from_bytes(canonical_dumps(v1), state.config)
    for edit in (lambda d: d["run"].pop("windows"),
                 lambda d: d["run"]["windows"].pop(),
                 lambda d: d["run"]["windows"][0].__setitem__("window_ticks", -1),
                 lambda d: d["run"]["windows"][0].__setitem__("harvest", 1.5),
                 lambda d: d["run"]["windows"][1].pop("egg_spend"),
                 lambda d: d["run"]["windows"][1].__setitem__("extra", 1)):
        broken = json.loads(json.dumps(doc))
        edit(broken)
        with pytest.raises(CanonicalError):
            checkpoint_from_bytes(canonical_dumps(broken), state.config)


def test_windows_are_part_of_the_checkpoint_hash_but_not_the_state_hash():
    state, run = reset(default_config(), 1)
    h_state, h_ckpt = state_hash(state), checkpoint_hash(state, run)
    run.windows[1] = WindowStats(harvest=5, window_ticks=3)
    assert state_hash(state) == h_state and checkpoint_hash(state, run) != h_ckpt
    state2, run2 = checkpoint_from_bytes(checkpoint_to_bytes(state, run), state.config)
    assert run2.windows == run.windows
    assert run.clone().windows == run.windows and run.clone().windows is not run.windows


# --- intel and information boundary ----------------------------------------------------------------------

def test_arrival_is_observed_at_the_next_policy_tick():
    state, run, ctrl = new_run({0: act(army_choice=1)})        # soldiers defend N1 and stay there
    advance(state, run, ctrl, 30)
    obs = build_observation(begin_policy_phase(state.clone(), tuple(run.windows)), 0)
    assert obs.block("food_current")[1] == 1.0
    assert obs.block("last_food_ratio")[1] == pytest.approx(60 / 80, rel=1e-6)   # no regeneration yet
    assert obs.block("own_node_count")[2 * 1 + 1] > 0
    assert state.colonies[0].intel.nodes[N1].seen_tick == 29    # refreshed by every POST_ARRIVAL sensor


def test_a_visited_node_becomes_last_seen_and_ages_after_the_units_leave():
    script = {0: act(forage_share_q=(0, 0, Q, 0, 0, 0)), 30: act()}      # workers to N2, home again at t=30
    state, run, ctrl = new_run(script)
    advance(state, run, ctrl, 60)
    obs = build_observation(begin_policy_phase(state.clone(), tuple(run.windows)), 0)
    assert obs.block("food_known")[2] == 1.0 and obs.block("food_current")[2] == 0.0
    assert obs.block("food_age_norm")[2] == pytest.approx(30 / 3000, rel=1e-6)   # seen at tick 30, PRE_POLICY
    assert unit_of(state, 0, 1).node == A


def test_own_home_stays_known_and_current_when_no_adult_is_home():
    script = {0: act(forage_share_q=(0, Q, 0, 0, 0, 0), army_choice=1)}
    state, run, ctrl = new_run(script)
    advance(state, run, ctrl, 30)
    obs = build_observation(begin_policy_phase(state.clone(), tuple(run.windows)), 0)
    assert obs.block("own_node_count")[:2] == (0.0, 0.0)
    assert (obs.block("enemy_known")[0], obs.block("enemy_current")[0], obs.block("enemy_age_norm")[0]) == (1.0, 1.0, 0.0)
    assert (obs.block("food_known")[0], obs.block("food_current")[0]) == (1.0, 1.0)


def test_enemy_movement_is_invisible_without_a_sensor():
    outcomes = []
    for army in (1, 2):                                       # B defends its N4 or its N5 (A observes neither)
        state, run, ctrl = new_run(b={0: act(army_choice=army, forage_share_q=(0, Q, 0, 0, 0, 0))})
        advance(state, run, ctrl, 30)
        outcomes.append(obs_pair(state, run)[0])
    assert outcomes[0] == outcomes[1]


def test_enemy_in_transit_toward_my_node_is_not_shown():
    state, run, ctrl = new_run(b={0: act(army_choice=6)})     # B raids A, arriving at t=8
    advance(state, run, ctrl, 5)
    assert any(u.node == -1 for u in state.colonies[1].units)
    state_quiet, run_quiet, ctrl_quiet = new_run()
    advance(state_quiet, run_quiet, ctrl_quiet, 5)
    # A's intel and its would-be observation at tick 5 equal the quiet run's apart from the clock
    assert state.colonies[0].intel == state_quiet.colonies[0].intel


# --- conservation / invariants fuzz -------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [3, 11, 29])
def test_random_policies_conserve_units_and_respect_travel_times(seed):
    ids = [policy_stream_id("alpha", 0), policy_stream_id("beta", 0)]
    state, run = reset(default_config(), seed, ids)
    ctrl = (PolicyController(random_policy, ids[0]), PolicyController(random_policy, ids[1]))
    topo = state.config.topology
    start = {u.unit_id: (u.caste, u.hp) for c in state.colonies for u in c.units}
    pending = {}                                              # unit -> (arrival_tick, to_node)
    all_events = []
    for _ in range(150):
        result = step_tick(state, run, ctrl, movement_only=True)
        all_events.extend(result.events)
        for e in result.events:
            if e.kind == "depart":
                assert e.arrival_tick == e.tick + edge_length(topo, e.from_node, e.to_node)
                assert e.unit_id not in pending
                pending[e.unit_id] = (e.arrival_tick, e.to_node)
            elif e.kind == "arrive":
                assert pending.pop(e.unit_id) == (e.tick, e.node)
        validate_state(state)
        now = {u.unit_id: (u.caste, u.hp) for c in state.colonies for u in c.units}
        assert now == start                                   # ids, castes and HP never change
        for c in state.colonies:
            for u in c.units:
                assert (u.node != -1) != (u.unit_id in pending)   # exactly one location
    assert {u for u in pending} == {u.unit_id for c in state.colonies for u in c.units if u.node == -1}
    assert all_events == sorted(all_events, key=lambda e: (e.tick, *event_sort_key(e)))


# --- events -------------------------------------------------------------------------------------------------------

def test_events_are_ordered_canonical_and_hashable():
    state, run, ctrl = new_run({0: act(forage_share_q=(0, Q, 0, 0, 0, 0), army_choice=6)})
    events = advance(state, run, ctrl, 12)
    assert events == sorted(events, key=lambda e: (e.tick, *event_sort_key(e)))
    first_tick = [e for e in events if e.tick == 0]
    kinds = [e.kind for e in first_tick]
    assert kinds == sorted(kinds, key=["order", "retarget", "depart"].index)
    assert [e.side for e in first_tick if e.kind == "order"] == [0, 1]
    canon = events_to_canonical(events)
    assert json.loads(canonical_dumps(canon)) == canon
    assert events_hash(events) == events_hash(list(events)) and len(events_hash(events)) == 64
    order = canon[0]
    assert order["kind"] == "order" and order["worker_quota"] == [0, 12, 0, 0, 0, 0]
    assert order["army_target_node"] == B


def test_live_tick_path_feeds_runstate_windows_into_the_policy_observation():
    rec_a, rec_b = Recorder(), Recorder()
    state, run = reset(default_config(), 1)
    advance(state, run, (rec_a, rec_b), 30)
    expected = obs_pair(state, run)                       # built from the real windows at t=30
    zero = begin_policy_phase(state.clone(), (WindowStats(), WindowStats()))
    assert expected[0] != flatten_v2(build_observation(zero, 0))
    advance(state, run, (rec_a, rec_b), 1)                # the engine's own policy call at t=30
    assert [rec_a.calls[1][1], rec_b.calls[1][1]] == expected
