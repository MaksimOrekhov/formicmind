"""FOR-48: ObservationE1 / ActionE1 / masks / decoder / atomic application / independent policies on the real E1 world."""

from __future__ import annotations

import copy
import itertools
import math
import random
from dataclasses import replace

import pytest

from test_eco_policy_helpers import AttackerPolicy, Recorder, Run, explorer_policies, first_observation
from formicmind.canonical import CanonicalError
from formicmind.eco import default_config
from formicmind.eco.action import (
    ATTACK_SIGHTING_AGE, ActionError, ActionE1, SquadActionE1, action_masks, attack_candidates, validate_action,
)
from formicmind.eco.decoder import LATENT_SIZE, LAYOUT, PolicyOutputError, check_latents, decode
from formicmind.eco.engine import RouteTable, step_world_tick
from formicmind.eco.events import EventLog
from formicmind.eco.initial import build_world
from formicmind.eco.model import ColonyOutcome
from formicmind.eco.observation import (
    EVENT_DIM, FOREIGN_DIM, GLOBAL_DIM, COHORT_DIM, EDGE_DIM, NODE_DIM, SIGHTING_DIM, ObservationError, build_observation, next_rolling,
    observation_bytes, observation_hash,
)
from formicmind.eco.policy import (
    ObservationHeuristicPolicy, PolicyRunSpec, PolicySpec, RandomValidPolicy, build_policies, spec_from_canonical, spec_to_canonical, uniform_spec,
)
from formicmind.eco.policy_run import PolicyApplyError, PolicyController, run_policies
from formicmind.eco.replay_file import ReplayFormatError, export_policy_replay, read_replay, verify_replay
from formicmind.eco.serialization import checkpoint_hash, state_hash
from formicmind.eco.versions import ACTION_VERSION, OBSERVATION_VERSION
from formicmind.rng import new_stream


def rand_latents(seed: int, scale: float = 6.0) -> tuple[float, ...]:
    rng = random.Random(seed)
    return tuple(rng.uniform(-scale, scale) for _ in range(LATENT_SIZE))


# ------------------------------------------------------------------------------------------------ ObservationE1


def test_first_observation_shapes_and_special_cases_before_any_contact():
    obs, masks = first_observation(colony=2)
    assert obs.version == OBSERVATION_VERSION == "ObservationE1/v1"
    assert len(obs.global_features) == GLOBAL_DIM == 40
    assert all(len(row) == NODE_DIM for row in obs.node_features) and 0 < len(obs.node_features) <= 128
    assert len(obs.node_ints) == len(obs.node_features)
    assert all(len(row) == EDGE_DIM for row in obs.edge_features) and 0 < len(obs.edge_features) <= 320
    assert obs.foreign_features == () and obs.foreign_nest_slots == () and obs.sighting_features == () and obs.event_features == ()
    assert all(len(row) == COHORT_DIM for row in obs.cohort_features)
    assert len(obs.cohort_features) == 2  # the starting workers and soldiers, as two cohorts
    assert all(len(row) == SIGHTING_DIM for row in obs.sighting_features) and all(len(row) == EVENT_DIM for row in obs.event_features)
    # slot 0 is the own nest, visited, 0 ticks from home; every edge and cohort refers to local slots only
    assert obs.node_ints[0] == (2, 1, 1, 0) or obs.node_ints[0][:2] == (2, 1)
    assert obs.node_features[0][2] == 1.0 and obs.node_features[0][4] == 1.0
    n = len(obs.node_features)
    assert all(0 <= a < b < n or 0 <= b < a < n for a, b in obs.edge_slots)
    assert all(0 <= slot < n for slot, _target in obs.cohort_slots)
    # everything is a float32 in [-1, 1] (unit-scale features)
    assert all(-1.0 <= x <= 1.0 for x in obs.global_features)
    assert all(-1.0 <= x <= 1.0 for row in obs.node_features for x in row)
    assert obs.global_features[0] == 0.0 and obs.global_features[3] == 1.0
    assert obs.facts.idle_workers_home == obs.facts.workers == 10 and obs.facts.idle_soldiers_home == obs.facts.soldiers == 2
    # first tick: nobody to attack, nothing to scout further than the frontier of the 5-tick known radius
    assert masks.attack_pairs == () and not any(masks.guard_target[:0])
    assert any(masks.forage_target) and any(masks.scout_target)


def test_observation_is_deterministic_byte_for_byte_and_hash_is_stable():
    a, _ = first_observation(colony=5)
    b, _ = first_observation(colony=5)
    assert observation_bytes(a) == observation_bytes(b) and observation_hash(a) == observation_hash(b)
    c, _ = first_observation(colony=6)
    assert observation_hash(a) != observation_hash(c)  # another colony sits elsewhere, so it knows another map


def test_observation_refuses_a_colony_without_a_queen_and_foreign_cohorts():
    run = Run(seed=50012)
    state = run.state
    colony = state.colonies[1]
    own = [c for c in state.cohorts if c.colony_id == 1]
    with pytest.raises(ObservationError):
        build_observation(colony, [*own, state.cohorts[0]], 0, run.config, decision_index=0, ticks_since_last_decision=0)
    dead = copy.deepcopy(colony)
    dead.queen_hp, dead.status = 0, "orphan"
    with pytest.raises(ObservationError):
        build_observation(dead, own, 0, run.config, decision_index=0, ticks_since_last_decision=0)


def test_rolling_window_is_closed_with_integer_ema_only():
    run = Run(seed=50012)
    colony = run.state.colonies[0]
    window = replace(colony.window, harvested=60, upkeep_paid=30)
    rolling = next_rolling(colony.rolling, window, 10, 2, 30)
    assert rolling.income_ema_q == (3 * colony.rolling.income_ema_q + 60 * 65536 // 30) // 4
    assert next_rolling(colony.rolling, window, 10, 2, 0) == colony.rolling


# ------------------------------------------------------------------------------------------------ masks (pure functions of the observation)


def test_masks_follow_the_observation_and_never_name_an_unknown_colony():
    obs, masks = first_observation(colony=0)
    n = len(obs.node_features)
    assert len(masks.forage_target) == len(masks.scout_target) == len(masks.guard_target) == n
    for slot, (kind, visited, age, path) in enumerate(obs.node_ints):
        if masks.forage_target[slot]:
            assert kind == 1 and visited == 1 and path >= 0 and slot != 0
        if masks.scout_target[slot]:
            assert slot != 0 and path >= 0 and (visited == 0 or age >= 90)
        if masks.guard_target[slot]:
            assert visited == 1 and path >= 0
    assert masks.workers_idle_home == 10 and masks.soldiers_idle_home == 2 and masks.soldiers_total == 2


def test_attack_masks_use_fresh_sightings_and_found_nests_only():
    run = Run(seed=50012).advance(900)
    seen_pairs = 0
    for colony in range(8):
        if run.state.colonies[colony].status != "alive":
            continue
        own = [c for c in run.state.cohorts if c.colony_id == colony]
        obs = build_observation(run.state.colonies[colony], own, run.state.tick, run.config, decision_index=1, ticks_since_last_decision=30)
        masks = action_masks(obs)
        for node_slot, colony_slot in masks.attack_pairs:
            seen_pairs += 1
            assert 0 <= colony_slot < len(obs.foreign_nest_slots)  # a slot exists only for a colony that was met
            nest_known = obs.foreign_nest_slots[colony_slot] == node_slot
            fresh = any((s[0], s[1]) == (node_slot, colony_slot) and s[2] <= ATTACK_SIGHTING_AGE for s in obs.sighting_ints)
            assert nest_known or fresh
            assert obs.node_ints[node_slot][3] >= 0  # the route is over known edges
        # an unmet colony has no foreign slot, hence cannot be a target
        assert max((c for _, c in masks.attack_pairs), default=-1) < len(obs.foreign_features)
    assert seen_pairs > 0


def test_mask_edge_cases_by_editing_the_observation():
    run = Run(seed=50012).advance(900)
    colony = next(c for c in range(8) if run.state.colonies[c].status == "alive" and run.state.colonies[c].memory.foreign
                  and any(r.nest_node_slot >= 0 for r in run.state.colonies[c].memory.foreign))
    own = [c for c in run.state.cohorts if c.colony_id == colony]
    obs = build_observation(run.state.colonies[colony], own, run.state.tick, run.config, decision_index=1, ticks_since_last_decision=30)
    base = action_masks(obs)
    assert base.attack_pairs
    # stale sightings stop being targets, a found nest stays one (the nest is a standing fact, not a sighting)
    stale = replace(obs, sighting_ints=tuple((n, c, ATTACK_SIGHTING_AGE + 1) for n, c, _a in obs.sighting_ints))
    nests = {(slot, c) for c, slot in enumerate(obs.foreign_nest_slots) if slot >= 0}
    assert set(attack_candidates(stale)) == nests & set(attack_candidates(stale)) and set(attack_candidates(stale)) <= nests
    # no known nest and no sighting at all -> nothing to attack
    blind = replace(obs, sighting_ints=(), foreign_nest_slots=tuple(-1 for _ in obs.foreign_nest_slots))
    assert action_masks(blind).attack_pairs == ()
    # no known path to the nest -> not a target
    unreachable = replace(obs, node_ints=tuple((k, v, a, -1) for k, v, a, _p in obs.node_ints))
    assert action_masks(unreachable).attack_pairs == () and not any(action_masks(unreachable).forage_target)
    # a visited node is a scouting target only once its information is 90 ticks old
    visited_slot = next(s for s, (k, v, a, p) in enumerate(obs.node_ints) if v == 1 and s != 0 and p >= 0)
    for age, allowed in ((89, False), (90, True)):
        nodes = list(obs.node_ints)
        k, v, _a, p = nodes[visited_slot]
        nodes[visited_slot] = (k, v, age, p)
        assert action_masks(replace(obs, node_ints=tuple(nodes))).scout_target[visited_slot] is allowed
    # no idle soldiers -> no GUARD / ATTACK; no idle workers -> no scouts
    no_soldiers = action_masks(replace(obs, facts=replace(obs.facts, idle_soldiers_home=0)))
    assert all(row[1] is False and row[2] is False and row[0] and row[3] for row in no_soldiers.squad_mission)
    no_workers = action_masks(replace(obs, facts=replace(obs.facts, idle_workers_home=0)))
    assert no_workers.scout_n == (True, False, False, False, False)
    no_patches = action_masks(replace(obs, node_ints=tuple((0 if k == 1 else k, v, a, p) for k, v, a, p in obs.node_ints)))
    assert no_patches.forage_frac == (True,) + (False,) * 8


def test_action_validation_rejects_every_kind_of_illegal_action():
    obs, masks = first_observation(colony=0)
    good = decode(rand_latents(1), masks)
    validate_action(good, masks)
    slot = next(i for i, ok in enumerate(masks.forage_target) if ok)
    illegal = [
        replace(good, egg_level=4), replace(good, soldier_share_q=-1), replace(good, workers_flee=1),
        replace(good, forage_frac_q=3, forage_targets=(), forage_shares_q=()),
        replace(good, forage_frac_q=3, forage_targets=(slot,), forage_shares_q=(15,)),
        replace(good, forage_frac_q=3, forage_targets=(0,), forage_shares_q=(16,)),  # slot 0 is the own nest
        replace(good, scout_n=1, scout_targets=()), replace(good, scout_n=0, scout_targets=(slot,)),
        replace(good, squads=(SquadActionE1("attack", 2, 5, 0), *good.squads[1:])),  # nobody met: no colony slot 0 exists
        replace(good, squads=(SquadActionE1("withdraw", 2, -1, -1), *good.squads[1:])),
        replace(good, squads=(SquadActionE1("guard", 0, 0, -1), *good.squads[1:])),
        replace(good, version="ActionE1/v2"),
    ]
    for action in illegal:
        with pytest.raises(ActionError):
            validate_action(action, masks)


# ------------------------------------------------------------------------------------------------ decoder


def test_any_finite_latents_decode_to_a_valid_action_on_real_masks():
    run = Run(seed=50012).advance(1200)
    checked = 0
    for colony in range(8):
        if run.state.colonies[colony].status != "alive":
            continue
        own = [c for c in run.state.cohorts if c.colony_id == colony]
        obs = build_observation(run.state.colonies[colony], own, run.state.tick, run.config, decision_index=2, ticks_since_last_decision=30)
        masks = action_masks(obs)
        for seed in range(25):
            action = decode(rand_latents(seed * 7 + colony, scale=(0.5, 6.0, 1e30)[seed % 3]), masks)
            validate_action(action, masks)
            assert sum(action.forage_shares_q) in (0, 16) and all(s >= 1 for s in action.forage_shares_q)
            checked += 1
        for constant in (0.0, 1.0, -1.0, 3.0e38, -3.0e38):
            validate_action(decode([constant] * LATENT_SIZE, masks), masks)
    assert checked >= 100


def test_decoder_is_deterministic_and_total_on_extreme_masks():
    obs, masks = first_observation(colony=3)
    latents = rand_latents(99)
    assert decode(latents, masks) == decode(list(latents), masks)
    empty = replace(masks, forage_target=(False,) * len(masks.forage_target), scout_target=(False,) * len(masks.scout_target),
                    guard_target=(False,) * len(masks.guard_target), attack_pairs=(), forage_frac=(True,) + (False,) * 8,
                    scout_n=(True,) + (False,) * 4, soldiers_idle_home=0, workers_idle_home=0,
                    squad_mission=tuple((True, False, False, True) for _ in range(3)), squad_size=tuple((True, False, False, False, False) for _ in range(3)))
    for seed in range(20):
        action = decode(rand_latents(seed), empty)
        validate_action(action, empty)
        assert action.forage_targets == () and action.scout_targets == () and all(s.mission in ("none", "withdraw") for s in action.squads)
        assert action.forage_frac_q == 0 and action.scout_n == 0


def test_single_option_branches_ignore_their_logits():
    obs, masks = first_observation(colony=3)
    only_one_egg = replace(masks, egg_level=(False, False, True, False))
    a = decode([-5.0] * LATENT_SIZE, only_one_egg)
    b = decode([5.0] * LATENT_SIZE, only_one_egg)
    assert a.egg_level == b.egg_level == 2
    one_patch = [False] * len(masks.forage_target)
    keep = next(i for i, ok in enumerate(masks.forage_target) if ok)
    one_patch[keep] = True
    forced = replace(masks, forage_target=tuple(one_patch))
    z = [0.0] * LATENT_SIZE
    off, _ = LAYOUT["forage_frac"]
    z[off + 8] = 9.0
    action = decode(z, forced)
    assert action.forage_frac_q == 8 and action.forage_targets == (keep,) and action.forage_shares_q == (16,)


def test_masked_options_are_never_chosen_whatever_the_logits():
    obs, masks = first_observation(colony=1)
    legal = [i for i, ok in enumerate(masks.forage_target) if ok]
    few = [False] * len(masks.forage_target)
    few[legal[0]] = True
    narrowed = replace(masks, forage_target=tuple(few))
    z = [0.0] * LATENT_SIZE
    off, size = LAYOUT["forage_ptr"]
    for slot in legal[1:]:
        z[off + slot] = 1e6  # the policy "wants" a forbidden slot
    foff, _ = LAYOUT["forage_frac"]
    z[foff + 5] = 3.0
    action = decode(z, narrowed)
    assert action.forage_targets == (legal[0],)


@pytest.mark.parametrize("bad", [
    [float("nan")] * LATENT_SIZE, [float("inf")] * LATENT_SIZE, [-float("inf")] * LATENT_SIZE,
    [0.0] * (LATENT_SIZE - 1), [0.0] * (LATENT_SIZE + 1), [1e39] * LATENT_SIZE, ["x"] * LATENT_SIZE, [True] * LATENT_SIZE, None,
])
def test_non_finite_or_malformed_latents_are_a_policy_output_error(bad):
    obs, masks = first_observation(colony=0)
    with pytest.raises(PolicyOutputError):
        decode(bad, masks)


def test_one_nan_hidden_inside_valid_latents_is_caught():
    obs, masks = first_observation(colony=0)
    z = list(rand_latents(4))
    z[LAYOUT["flee"][0]] = math.nan
    with pytest.raises(PolicyOutputError):
        check_latents(z)


def test_squads_share_the_idle_soldiers_in_order_and_never_oversubscribe():
    run = Run(seed=50012).advance(900)
    colony = next(c for c in range(8) if run.state.colonies[c].status == "alive" and run.state.colonies[c].memory.foreign)
    own = [c for c in run.state.cohorts if c.colony_id == colony]
    obs = build_observation(run.state.colonies[colony], own, run.state.tick, run.config, decision_index=1, ticks_since_last_decision=30)
    masks = replace(action_masks(obs), soldiers_total=4, soldiers_idle_home=4)
    z = [0.0] * LATENT_SIZE
    for squad in range(3):
        z[LAYOUT[f"mission{squad}"][0] + 1] = 5.0  # everybody wants GUARD
        z[LAYOUT[f"size{squad}"][0] + 4] = 5.0     # of all soldiers
    action = decode(z, masks)
    assert [s.mission for s in action.squads] == ["guard", "none", "none"]  # the first squad takes all four, nobody is left
    validate_action(action, masks)
    z2 = list(z)
    for squad in range(3):
        z2[LAYOUT[f"size{squad}"][0] + 4] = 0.0
        z2[LAYOUT[f"size{squad}"][0] + 1] = 5.0   # a quarter each = 1 of 4 soldiers
    action2 = decode(z2, masks)
    assert [s.mission for s in action2.squads] == ["guard", "guard", "guard"] and all(s.size_q == 1 for s in action2.squads)


# ------------------------------------------------------------------------------------------------ policies and the controller


def test_random_valid_goes_through_latents_and_decoder_and_is_independent_per_colony():
    config = default_config(8, "moderate")
    spec = uniform_spec(8, "random_valid", 11)
    policies = build_policies(spec, config)
    assert all(isinstance(p, RandomValidPolicy) for p in policies)
    obs, masks = first_observation(colony=0)
    streams = {tuple(p.act(obs, masks)[:6]) for p in policies}
    assert len(streams) == 8  # eight private streams, eight different draws
    again = build_policies(spec, config)
    assert all(a.act(obs, masks) == b.act(obs, masks) for a, b in zip(build_policies(spec, config), again))
    other_seed = build_policies(uniform_spec(8, "random_valid", 12), config)
    assert other_seed[0].act(obs, masks) != build_policies(spec, config)[0].act(obs, masks)


def test_policy_spec_round_trips_and_is_checked():
    spec = PolicyRunSpec(7, (PolicySpec("random_valid", ""), PolicySpec("heuristic", "raider")) + (PolicySpec("heuristic", "peaceful"),) * 6)
    assert spec_from_canonical(spec_to_canonical(spec)) == spec
    broken = spec_to_canonical(spec)
    broken["policies"][1]["stream"] = "policy:heuristic:5"
    with pytest.raises(CanonicalError):
        spec_from_canonical(broken)
    with pytest.raises(CanonicalError):
        spec_from_canonical({"schema": "x"})


def test_same_inputs_same_hashes_and_policy_seed_matters():
    config = default_config(8, "moderate")
    spec = uniform_spec(8, "random_valid", 5)
    a = run_policies(config, 50004, spec, 600)
    b = run_policies(config, 50004, spec, 600)
    assert (a.state_hash, a.events_hash, a.decisions_hash) == (b.state_hash, b.events_hash, b.decisions_hash)
    c = run_policies(config, 50004, uniform_spec(8, "random_valid", 6), 600)
    assert c.decisions_hash != a.decisions_hash and c.state_hash != a.state_hash
    d = run_policies(config, 50005, spec, 600)
    assert d.state_hash != a.state_hash


@pytest.mark.parametrize("colonies", [6, 7, 8])
def test_six_seven_and_eight_colonies_run_with_both_policy_kinds(colonies):
    config = default_config(colonies, "moderate")
    for kind in ("random_valid", "heuristic"):
        result = run_policies(config, 50020 + colonies, uniform_spec(colonies, kind, 3, "explorer" if kind == "heuristic" else ""), 450)
        assert result.state.tick > 0 and len(result.state.colonies) == colonies
        assert result.decisions.count >= colonies  # the first planned decision involves every colony
        assert all(len(set(r["colony"] for r in result.decisions.records if r["tick"] == 0)) == colonies for _ in [0])


def test_random_bots_are_not_required_to_survive_but_the_world_stays_valid():
    config = default_config(8, "moderate")
    result = run_policies(config, 50011, uniform_spec(8, "random_valid", 9), 1500)
    assert result.state.tick > 0  # step_world_tick validated every invariant on the way, including food conservation


def test_all_colonies_decide_on_one_snapshot_and_do_not_see_each_others_actions():
    config = default_config(8, "moderate")
    snapshot_a = build_world(config, 50012)
    snapshot_b = copy.deepcopy(snapshot_a)
    routes = RouteTable(snapshot_a.world_map)
    rec_a = [Recorder(p) for p in explorer_policies(config)]
    rec_b = [Recorder(p) for p in explorer_policies(config, attackers=(0, 4))]
    # colony 0 and 4 behave differently in B, yet every other colony decides on the same observation and answers identically in both worlds
    for colony in (0, 4):
        rec_b[colony].inner = RandomValidPolicy(new_stream(1, f"policy:random_valid:{colony}"))
    PolicyController(rec_a)(snapshot_a.state, config, snapshot_a.world_map, routes)
    PolicyController(rec_b)(snapshot_b.state, config, snapshot_b.world_map, routes)
    for colony in range(8):
        if colony in (0, 4):
            continue
        assert rec_a[colony].seen[0][1] == rec_b[colony].seen[0][1]
        assert rec_a[colony].seen[0][2] == rec_b[colony].seen[0][2]
    for colony in (1, 2, 3, 5, 6, 7):
        own_a = sorted((c.caste, c.mission.kind, c.mission.target_node, c.count) for c in snapshot_a.state.cohorts if c.colony_id == colony)
        own_b = sorted((c.caste, c.mission.kind, c.mission.target_node, c.count) for c in snapshot_b.state.cohorts if c.colony_id == colony)
        assert own_a == own_b  # one colony's decision moved nothing of another colony (cohort ids come from a shared counter, so they are not compared)


def test_decision_order_cannot_change_the_result():
    config = default_config(8, "moderate")
    world = build_world(config, 50012)
    forward = copy.deepcopy(world)
    backward = copy.deepcopy(world)
    routes = RouteTable(world.world_map)
    PolicyController(explorer_policies(config))(forward.state, config, forward.world_map, routes)

    class Reversed(PolicyController):
        def __call__(self, state, config, world_map, routes):  # same decisions, applied through a reversed colony order
            from formicmind.eco import apply as apply_module

            original = apply_module.apply_action
            order: list = []

            def deferred(state, cfg, wm, cid, action, sink=None):
                order.append((cid, action, sink))

            import formicmind.eco.policy_run as run_module
            run_module.apply_action = deferred
            try:
                super().__call__(state, config, world_map, routes)
            finally:
                run_module.apply_action = original
            for cid, action, sink in reversed(order):
                original(state, config, world_map, cid, action, sink)

    Reversed(explorer_policies(config))(backward.state, config, backward.world_map, routes)
    def shape(state):
        return sorted((c.colony_id, c.caste, c.count, c.mission.kind, c.mission.target_node) for c in state.cohorts)

    assert shape(forward.state) == shape(backward.state)


def test_a_failing_policy_changes_nothing_and_a_failing_apply_restores_the_world():
    config = default_config(8, "moderate")
    world = build_world(config, 50012)
    routes = RouteTable(world.world_map)
    before = state_hash(world.state)

    class Exploding(ObservationHeuristicPolicy):
        def act(self, obs, masks):
            raise RuntimeError("policy crashed")

    policies = explorer_policies(config)
    policies[5] = Exploding(config, "peaceful")
    with pytest.raises(RuntimeError):
        PolicyController(policies)(world.state, config, world.world_map, routes)
    assert state_hash(world.state) == before  # colonies 0..4 had already decided: none of it was applied

    class Nan(RandomValidPolicy):
        def act(self, obs, masks):
            return (float("nan"),) * LATENT_SIZE

    policies = explorer_policies(config)
    policies[2] = Nan(new_stream(1, "policy:random_valid:2"))
    with pytest.raises(PolicyOutputError):
        PolicyController(policies)(world.state, config, world.world_map, routes)
    assert state_hash(world.state) == before

    import formicmind.eco.policy_run as run_module
    calls = {"n": 0}
    real = run_module.apply_action

    def flaky(state, cfg, wm, cid, action, sink=None):
        calls["n"] += 1
        real(state, cfg, wm, cid, action, sink)
        if calls["n"] == 3:
            raise RuntimeError("the order API refused")

    run_module.apply_action = flaky
    try:
        outbox = routes.outbox
        outbox.clear()
        with pytest.raises(PolicyApplyError):
            PolicyController(explorer_policies(config))(world.state, config, world.world_map, routes)
    finally:
        run_module.apply_action = real
    assert state_hash(world.state) == before and routes.outbox == []  # two colonies were already applied: rolled back, events dropped


def test_an_illegal_action_from_a_scripted_policy_is_a_technical_error():
    config = default_config(8, "moderate")
    world = build_world(config, 50012)
    routes = RouteTable(world.world_map)

    class Forger(ObservationHeuristicPolicy):
        def act(self, obs, masks):
            return replace(super().act(obs, masks), squads=(SquadActionE1("attack", 4, 3, 0), *(SquadActionE1("none", 0, -1, -1),) * 2))

    policies = explorer_policies(config)
    policies[0] = Forger(config, "peaceful")
    before = state_hash(world.state)
    with pytest.raises(ActionError):
        PolicyController(policies)(world.state, config, world.world_map, routes)
    assert state_hash(world.state) == before


def test_orphans_decide_nothing_and_keep_their_standing_foraging():
    run = Run(seed=50012).advance(300)
    state = run.state
    victim = next(c for c in range(8) if state.colonies[c].status == "alive")
    colony = state.colonies[victim]
    plan = (list(colony.orders.forage_targets), colony.orders.forage_frac_q)
    assert plan[0], "the colony had a foraging plan"
    tick = state.tick
    colony.queen_hp, colony.status = 0, "orphan"
    colony.orders.egg_level = colony.orders.soldier_share_q = 0
    state.outcomes[victim] = ColonyOutcome(True, "queen_death", tick)
    recorders = [Recorder(p) for p in explorer_policies(run.config)]
    run.controller = PolicyController(recorders)
    run.advance(120)
    assert recorders[victim].seen == []  # no observation was built for the orphan, no decision asked
    assert all(recorders[c].seen for c in range(8) if c != victim and state.colonies[c].status == "alive")
    assert state.colonies[victim].orders.forage_targets == plan[0]
    assert all(r["colony"] != victim for r in run.controller.log.records)


def test_alert_decisions_follow_the_combat_cadence_and_are_recorded():
    run = Run(seed=50012, attackers=(3,)).advance(900)
    kinds = [r["kind"] for r in run.controller.log.records]
    assert "alert" in kinds and "planned" in kinds
    for record in run.controller.log.records:
        if record["kind"] == "alert":
            assert record["tick"] % run.config.alert_interval == 0 and record["tick"] % run.config.decision_k != 0
    assert any(e.kind == "combat" for e in run.events.events), "the scripted attacker really fought through ActionE1"
    dispatches = [e for e in run.events.events if e.kind == "dispatch" and e.a == 3]
    assert dispatches and all(e.colony in (3,) for e in dispatches)


def test_squad_orders_are_not_resent_while_the_squad_is_out():
    from formicmind.eco.apply import apply_action
    from formicmind.eco.model import CohortE1, Location, Mission

    run = Run(seed=50012, attackers=(3,)).advance(361)
    state = run.state
    sent = [e for e in run.events.events if e.kind == "dispatch" and e.a == 3 and e.colony == 3]
    assert sent and sent[0].tick == 360
    record = next(d for d in run.controller.log.records if d["colony"] == 3 and d["tick"] == 360)
    squad = record["action"]["squads"][0]
    action = replace(decode(rand_latents(0), action_masks(first_observation(3)[0])), squads=(SquadActionE1("attack", 4, squad["node_slot"], squad["colony_slot"]), *(SquadActionE1("none", 0, -1, -1),) * 2),
                     scout_n=0, scout_targets=(), forage_frac_q=0, forage_targets=(), forage_shares_q=())
    home = run.world.world_map.nests[3]
    state.cohorts.append(CohortE1(state.next_cohort_id, 3, "soldier", 3, 90, Location("node", home, -1, -1, 0, 0), Mission("home", -1, -1, "idle", 0), 0))
    state.next_cohort_id += 1
    out = sorted((c.cohort_id, c.count, c.mission.kind, c.mission.target_node) for c in state.cohorts if c.colony_id == 3 and c.mission.kind == "attack")
    assert out, "the first squad is on its way"
    events: list = []
    before = sorted((c.cohort_id, c.count, c.mission.kind, c.mission.target_node) for c in state.cohorts)
    apply_action(state, run.config, run.world.world_map, 3, action, events)
    assert events == [] and sorted((c.cohort_id, c.count, c.mission.kind, c.mission.target_node) for c in state.cohorts) == before
    # once nobody is out any more, the same order is sent again
    for c in state.cohorts:
        if c.colony_id == 3 and c.mission.kind == "attack":
            c.mission = Mission("home", -1, -1, "idle", 0)
            c.loc = Location("node", home, -1, -1, 0, 0)
    apply_action(state, run.config, run.world.world_map, 3, action, events)
    assert [e.kind for e in events] == ["dispatch"]


# ------------------------------------------------------------------------------------------------ replay, sidecar and checkpoint


def test_policy_replay_is_readable_verifiable_and_bound_to_its_decisions():
    config = default_config(8, "moderate")
    spec = PolicyRunSpec(4, (PolicySpec("random_valid", ""), *(PolicySpec("heuristic", "explorer"),) * 6, PolicySpec("heuristic", "raider")))
    replay, decisions = export_policy_replay(config, 50012, spec, 420, 40)
    assert (replay, decisions) == export_policy_replay(config, 50012, spec, 420, 40)
    document = read_replay(replay)
    assert document.policy == spec and document.header["fixture"] == "policy" and document.header["roles"]["7"] == "heuristic:raider"
    assert document.result["decision_count"] == decisions.count(b"\n")
    verify_replay(replay, decisions)
    # the sidecar and the replay are bound: edited records, a dropped line, another file all fail
    lines = decisions.split(b"\n")
    with pytest.raises(ReplayFormatError):
        verify_replay(replay, b"\n".join(lines[1:]))
    with pytest.raises(ReplayFormatError):
        verify_replay(replay, decisions.replace(b'"egg_level":3', b'"egg_level":0', 1) if b'"egg_level":3' in decisions else decisions[:-2] + b"\n")
    # the events in a policy replay are the engine's: dispatches show scouting orders given through ActionE1
    assert any(e.kind == "dispatch" for e in document.events)


def test_policy_replay_rejects_a_forged_policy_header():
    config = default_config(8, "moderate")
    replay, _ = export_policy_replay(config, 50012, uniform_spec(8, "heuristic", 1, "peaceful"), 120, 40)
    forged = replay.replace(b'"policy_seed":"1"', b'"policy_seed":"2"', 1)
    assert forged != replay
    with pytest.raises(ReplayFormatError):
        read_replay(forged)  # the integrity chain breaks


def test_checkpoint_and_resume_reproduce_the_continuous_run():
    config = default_config(8, "moderate")
    spec = PolicyRunSpec(8, (PolicySpec("random_valid", ""),) * 4 + (PolicySpec("heuristic", "explorer"),) * 4)

    def fresh():
        world = build_world(config, 50012)
        return world, RouteTable(world.world_map), PolicyController(build_policies(spec, config)), EventLog()

    def go(world, routes, controller, events, ticks):
        for _ in range(ticks):
            record = step_world_tick(world.state, world.run, config, world.world_map, controller, routes=routes, validate_input=world.state.tick == 0)
            events.extend(record.events)

    whole = fresh()
    go(*whole, 600)
    first = fresh()
    go(*first, 300)
    saved_state, saved_run, saved_controller = copy.deepcopy(first[0].state), copy.deepcopy(first[0].run), first[2].snapshot()
    resumed_world = copy.deepcopy(first[0])
    resumed_world.state, resumed_world.run = saved_state, saved_run
    resumed_controller = PolicyController(build_policies(spec, config))
    resumed_controller.restore(copy.deepcopy(saved_controller))
    resumed_events = EventLog(list(first[3].events), first[3].head)
    go(resumed_world, RouteTable(resumed_world.world_map), resumed_controller, resumed_events, 300)
    assert state_hash(resumed_world.state) == state_hash(whole[0].state)
    assert checkpoint_hash(resumed_world.state, resumed_world.run) == checkpoint_hash(whole[0].state, whole[0].run)
    assert resumed_events.head == whole[3].head
    assert resumed_controller.log.head == whole[2].log.head and resumed_controller.log.count == whole[2].log.count


def test_decision_records_hold_no_world_identifier_and_name_actions_by_slot():
    result = run_policies(default_config(8, "moderate"), 50012, uniform_spec(8, "heuristic", 3, "explorer"), 600)
    for record in result.decisions.records:
        assert set(record) == {"tick", "colony", "kind", "index", "obs", "action", "action_hash", "chain"}
        for squad in record["action"]["squads"]:
            assert set(squad) == {"mission", "size_q", "node_slot", "colony_slot"}
    nodes = len(result.state.colonies[0].memory.nodes)
    assert max((s for r in result.decisions.records for s in r["action"]["forage_targets"]), default=0) < 128
    assert nodes < 128


def test_performance_profile_of_an_eight_colony_world():
    import time
    config = default_config(8, "moderate")
    started = time.perf_counter()
    result = run_policies(config, 50012, uniform_spec(8, "heuristic", 3, "explorer"), 600)
    elapsed = time.perf_counter() - started
    assert elapsed < 60 and result.decisions.count > 100  # generous bound: the point is that observation + masks + decoding are cheap
    assert math.isfinite(elapsed)


# ------------------------------------------------------------------------------------------------ CLI and the published example


def test_policy_run_cli_writes_a_verifiable_replay_and_decisions(tmp_path):
    from formicmind.eco.cli import main

    out, decisions = tmp_path / "run.jsonl", tmp_path / "run.decisions.jsonl"
    argv = ["policy-run", "--seed", "50012", "--policy", "heuristic:explorer", "--policy-for", "2=random_valid", "--policy-for", "5=heuristic:aggressor",
            "--policy-seed", "9", "--ticks", "300", "--out", str(out), "--decisions", str(decisions), "--verify"]
    assert main(argv) == 0
    first = (out.read_bytes(), decisions.read_bytes())
    assert main(argv) == 0 and (out.read_bytes(), decisions.read_bytes()) == first  # reproducible bytes
    document = verify_replay(first[0], first[1])
    assert [p.kind for p in document.policy.policies][2] == "random_valid" and document.policy.policies[5].variant == "aggressor"
    assert main(["replay-verify", str(out), "--decisions", str(decisions)]) == 0
    assert main(["replay-verify", str(out), "--read-only"]) == 0
    for bad in (["--policy", "oracle"], ["--policy", "heuristic:genius"], ["--policy-for", "9=random_valid"], ["--policy-for", "1"], ["--policy-seed", "-1"],
                ["--policy-for", "1=random_valid", "--policy-for", "1=heuristic"]):
        assert main(["policy-run", "--ticks", "10", "--out", str(tmp_path / "x.jsonl"), *bad]) == 3


def test_the_committed_viewer_example_is_a_byte_exact_run_of_the_engine():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    replay = (root / "viewer" / "public" / "samples" / "eco_policy_seed50012.jsonl").read_bytes()
    document = verify_replay(replay)  # strict read, then a full re-execution from the header
    assert document.policy is not None and document.header["seed"] == "50012" and document.result["decision_count"] > 500
    assert document.config.n_colonies == 8 and document.config.scenario == "moderate"
    assert any(e.kind == "combat" for e in document.events) and any(e.kind == "dispatch" and e.a == 3 for e in document.events)


def test_the_committed_world_report_matches_a_fresh_run():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "docs" / "ecosystem_examples" / "for48_policy_runs.json").read_text(encoding="utf-8"))
    assert [w["scenario"] for w in report["worlds"]].count("moderate") >= 2 and len(report["worlds"]) >= 3
    assert all(w["reproduced"] and w["colonies"] == 8 for w in report["worlds"])
    world = report["worlds"][2]  # the one with fights: recompute it
    spec = spec_from_canonical(world["policy"])
    result = run_policies(default_config(8, world["scenario"]), int(world["map_seed"]), spec, world["ticks"])
    assert {"state": result.state_hash, "events": result.events_hash, "decisions": result.decisions_hash} == world["hashes"]
