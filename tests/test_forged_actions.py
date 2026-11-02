"""Review FOR-11 P1: a hand-built ValidatedActionV2 must not bypass validation; all-or-nothing policy phase."""

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
from formicmind.actions import (
    ActionMaskV2,
    ActionV2,
    InvalidAction,
    ValidatedActionV2,
    get_action_mask,
    validate_action,
)
from formicmind.events import events_hash
from formicmind.orders import apply_joint_orders
from formicmind.rng import policy_stream_id
from formicmind.snapshot import begin_policy_phase
from formicmind.tick import PolicyController, ScriptedController, step_tick

from test_tick import advance, random_policy

Q = 65535
HOME_Q = (Q, 0, 0, 0, 0, 0)
T = True
PERMISSIVE_MASK = ActionMaskV2((T,) * 3, (T,) * 6, (T,) * 7, ((T, T),) * 7)


class Forger:
    """Controller that wraps an arbitrary object in ValidatedActionV2 without validate_action."""

    def __init__(self, action):
        self.action = action

    def act(self, tick, observation, run):
        return ValidatedActionV2(self.action)


class ValidatedAgainstPermissiveMask:
    """Validated for real, but against a mask that is not this side's real mask."""

    def __init__(self, action):
        self.action = action

    def act(self, tick, observation, run):
        return validate_action(observation, PERMISSIVE_MASK, self.action)


FORGED = {
    "lay_order=3": ActionV2(3, HOME_Q, 0, False),
    "lay_order=bool": ActionV2(True, HOME_Q, 0, False),
    "lay_order=-1": ActionV2(-1, HOME_Q, 0, False),
    "army_choice=7": ActionV2(0, HOME_Q, 7, False),
    "avoid_center=int": ActionV2(0, HOME_Q, 1, 1),
    "support=4": ActionV2(0, (16384, 16384, 16384, 16383, 0, 0), 0, False),
    "sum=65534": ActionV2(0, (65534, 0, 0, 0, 0, 0), 0, False),
    "shares=list": ActionV2(0, [65535, 0, 0, 0, 0, 0], 0, False),
    "shares=negative": ActionV2(0, (65536, -1, 0, 0, 0, 0), 0, False),
    "shares=float": ActionV2(0, (65535.0, 0, 0, 0, 0, 0), 0, False),
    "garrison+avoid": ActionV2(0, HOME_Q, 0, True),            # masked out by the real mask
    "defend(C)+avoid": ActionV2(0, HOME_Q, 3, True),           # masked out by the real mask
    "not an ActionV2": "junk",
    "None": None,
    "plain tuple": (0, HOME_Q, 0, False),
}


def mid_game():
    """A run stopped right before policy tick 30: units on the move, non-zero windows."""
    ids = [policy_stream_id("alpha", 0), policy_stream_id("beta", 0)]
    state, run = reset(default_config(), 5, ids)
    honest = (PolicyController(random_policy, ids[0]), PolicyController(random_policy, ids[1]))
    advance(state, run, honest, 1)                      # tick 0: random orders, units start moving
    advance(state, run, (ScriptedController({}), ScriptedController({})), 29)
    assert state.tick == 30 and [w.window_ticks for w in run.windows] == [30, 30]
    return state, run, ids


def fingerprint(state, run):
    return (
        state_hash(state), checkpoint_hash(state, run), state.tick, list(run.windows),
        [c.intel.clone() for c in state.colonies],
        [(u.node, u.from_node, u.to_node, u.arrival_tick, u.target_node)
         for c in state.colonies for u in c.units],
        [(c.lay_order, tuple(c.forage_q), c.army_choice, c.avoid_center) for c in state.colonies],
    )


def assert_untouched(state, run, before):
    assert fingerprint(state, run) == before
    assert [w.window_ticks for w in run.windows] == [30, 30]               # windows not reset


@pytest.mark.parametrize("forged_side", [0, 1])
@pytest.mark.parametrize("name", sorted(FORGED))
def test_forged_validated_action_is_rejected_and_nothing_is_applied(name, forged_side):
    state, run, ids = mid_game()
    honest = PolicyController(random_policy, ids[1 - forged_side])        # consumes its RNG before the failure
    ctrl = [None, None]
    ctrl[forged_side], ctrl[1 - forged_side] = Forger(FORGED[name]), honest
    before = fingerprint(state, run)
    with pytest.raises(InvalidAction):
        step_tick(state, run, tuple(ctrl), movement_only=True)
    assert_untouched(state, run, before)


@pytest.mark.parametrize("forged_side", [0, 1])
def test_action_validated_against_a_foreign_mask_is_revalidated_with_the_real_mask(forged_side):
    state, run, ids = mid_game()
    ctrl = [None, None]
    ctrl[forged_side] = ValidatedAgainstPermissiveMask(ActionV2(0, HOME_Q, 0, True))  # garrison + avoid
    ctrl[1 - forged_side] = ScriptedController({})
    before = fingerprint(state, run)
    with pytest.raises(InvalidAction):
        step_tick(state, run, tuple(ctrl), movement_only=True)
    assert_untouched(state, run, before)


def test_a_lay_order_that_the_real_mask_forbids_cannot_be_smuggled_in():
    state, run, ids = mid_game()
    state.colonies[0].queen_hp = 0                                        # real lay mask is (True, False, False)
    obs = build_observation(begin_policy_phase(state.clone(), tuple(run.windows)), 0)
    assert get_action_mask(obs).lay == (True, False, False)
    before = fingerprint(state, run)
    ctrl = (ValidatedAgainstPermissiveMask(ActionV2(1, HOME_Q, 0, False)), ScriptedController({}))
    with pytest.raises(InvalidAction):
        step_tick(state, run, ctrl, movement_only=True)
    assert_untouched(state, run, before)


def test_honest_controllers_are_unaffected_by_the_revalidation():
    state, run, ids = mid_game()
    ctrl = (PolicyController(random_policy, ids[0]), PolicyController(random_policy, ids[1]))
    result = step_tick(state, run, ctrl, movement_only=True)
    assert state.tick == 31 and [e.kind for e in result.events][:2] == ["order", "order"]
    assert [w.window_ticks for w in run.windows] == [1, 1]


class Boom(RuntimeError):
    pass


def test_failure_after_orders_were_applied_rolls_the_whole_policy_phase_back(monkeypatch):
    import formicmind.tick as tick_module
    state, run, ids = mid_game()
    ctrl = (PolicyController(random_policy, ids[0]), PolicyController(random_policy, ids[1]))
    colonies = list(state.colonies)
    units = [(u, u.node, u.target_node) for c in state.colonies for u in c.units]
    before = fingerprint(state, run)

    def exploding_dispatch(*args, **kwargs):
        raise Boom("dispatch failed after orders were committed")

    monkeypatch.setattr(tick_module, "dispatch_available", exploding_dispatch)
    with pytest.raises(Boom):
        step_tick(state, run, ctrl, movement_only=True)
    assert_untouched(state, run, before)
    # rollback is in place: the very same objects hold the restored values
    assert all(a is b for a, b in zip(state.colonies, colonies))
    current = [u for c in state.colonies for u in c.units]
    assert all(u is v and u.node == n and u.target_node == t for (u, n, t), v in zip(units, current))


def test_failure_in_the_middle_of_applying_orders_rolls_back_partial_commits(monkeypatch):
    import formicmind.tick as tick_module
    state, run, ids = mid_game()
    ctrl = (PolicyController(random_policy, ids[0]), PolicyController(random_policy, ids[1]))
    before = fingerprint(state, run)
    real = tick_module.apply_joint_orders

    def half_applied(state_, actions, phase=0):
        real(state_, actions, phase)                  # everything is committed ...
        state_.colonies[0].food += 999                # ... plus a stray mutation ...
        raise Boom("... and then it fails")

    monkeypatch.setattr(tick_module, "apply_joint_orders", half_applied)
    with pytest.raises(Boom):
        step_tick(state, run, ctrl, movement_only=True)
    assert_untouched(state, run, before)


def test_a_failed_policy_phase_leaves_no_events_and_a_retry_equals_a_run_that_never_failed(monkeypatch):
    import formicmind.tick as tick_module
    state, run, ids = mid_game()
    ctrl = (PolicyController(random_policy, ids[0]), PolicyController(random_policy, ids[1]))
    twin_state, twin_run = checkpoint_from_bytes(checkpoint_to_bytes(state, run), state.config)
    original = tick_module.dispatch_available

    def fail_once(*args, **kwargs):
        raise Boom("once")

    monkeypatch.setattr(tick_module, "dispatch_available", fail_once)
    with pytest.raises(Boom):
        step_tick(state, run, ctrl, movement_only=True)
    monkeypatch.setattr(tick_module, "dispatch_available", original)
    again = step_tick(state, run, ctrl, movement_only=True)               # retry from the restored state
    clean = step_tick(twin_state, twin_run, ctrl, movement_only=True)     # never failed
    assert events_hash(again.events) == events_hash(clean.events)
    assert checkpoint_hash(state, run) == checkpoint_hash(twin_state, twin_run)


@pytest.mark.parametrize("name", sorted(n for n in FORGED if n != "defend(C)+avoid"))
def test_apply_joint_orders_itself_refuses_malformed_wrapped_actions_without_touching_state(name):
    """Defence in depth: the structural contract is re-checked even without an observation.

    ``defend(C)+avoid_center`` is structurally fine (only the mask forbids it), so it is
    covered by the engine-level tests above, not here.
    """
    state, run, ids = mid_game()
    good = ValidatedActionV2(ActionV2(0, HOME_Q, 0, False))
    forged = ValidatedActionV2(FORGED[name])
    before = (state_hash(state), checkpoint_hash(state, run))
    for pair in ((forged, good), (good, forged)):
        with pytest.raises(InvalidAction):
            apply_joint_orders(state, pair)
        assert (state_hash(state), checkpoint_hash(state, run)) == before
