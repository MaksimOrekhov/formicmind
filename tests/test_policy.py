"""The policy boundary: (observation, mask, policy_rng) -> PolicyLogitsV2, one decode path."""

import pytest

from formicmind import default_config, reset
from formicmind.actions import ActionMaskV2, ValidatedActionV2, get_action_mask
from formicmind.decoder import PolicyLogitsV2, PolicyOutputError, decode_action_logits
from formicmind.observation import ObservationV2, build_observation
from formicmind.policy import decide
from formicmind.rng import Pcg32, policy_stream_id
from formicmind.snapshot import begin_policy_phase

from helpers import busy_state, fresh_state, mirror_state


def constant_policy(observation, mask, policy_rng):
    return PolicyLogitsV2([1, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0.5])


def test_decide_runs_observation_mask_policy_decoder_validator():
    obs = build_observation(begin_policy_phase(fresh_state(0)), 0)
    rng = Pcg32.seeded(1, 2)
    result = decide(constant_policy, obs, rng)
    assert isinstance(result, ValidatedActionV2)
    assert result.action == decode_action_logits(constant_policy(obs, get_action_mask(obs), rng),
                                                 get_action_mask(obs))
    assert result.action.lay_order == 0 and result.action.army_choice == 3
    assert result.action.avoid_center is False  # defend(C): avoid_center is masked out
    q = result.action.forage_share_q   # logits: N1=2, N2=1 are the only positive ones -> support 2
    assert q[0] == 0 and q[1] > q[2] > 0 and sum(q) == 65535


def test_policy_receives_exactly_observation_mask_and_policy_rng():
    seen = []

    def spy(*args, **kwargs):
        seen.append((args, kwargs))
        return PolicyLogitsV2([0.0] * 17)

    obs = build_observation(begin_policy_phase(fresh_state(0)), 0)
    rng = Pcg32.seeded(5, 6)
    decide(spy, obs, rng)
    (args, kwargs), = seen
    assert kwargs == {} and len(args) == 3
    assert isinstance(args[0], ObservationV2) and isinstance(args[1], ActionMaskV2) and args[2] is rng


def test_malformed_policy_output_is_a_technical_error_not_a_fallback():
    obs = build_observation(begin_policy_phase(fresh_state(0)), 0)
    rng = Pcg32.seeded(1, 1)
    for bad in (lambda o, m, r: None, lambda o, m, r: [0.0] * 16,
                lambda o, m, r: [float("nan")] * 17, lambda o, m, r: "logits"):
        with pytest.raises(PolicyOutputError):
            decide(bad, obs, rng)


def test_plain_sequences_are_accepted_as_logits():
    obs = build_observation(begin_policy_phase(fresh_state(0)), 0)
    assert decide(lambda o, m, r: [0.0] * 17, obs, Pcg32.seeded(1, 1)).action.army_choice == 0


def random_policy(observation, mask, policy_rng):
    return PolicyLogitsV2([policy_rng.next_u32() / 2**31 - 1 for _ in range(17)])


def test_policy_rng_is_role_keyed_so_seat_swaps_keep_each_policys_stream():
    _, run = reset(default_config(), 20261102, [policy_stream_id("alpha", 0), policy_stream_id("beta", 0)])
    state = busy_state()
    snapshot = begin_policy_phase(state)
    swapped_snapshot = begin_policy_phase(mirror_state(state))

    def play(snap, seats):
        r = run.clone()
        return {name: decide(random_policy, build_observation(snap, side),
                             r.stream(policy_stream_id(name, 0))).action
                for name, side in seats.items()}

    straight = play(snapshot, {"alpha": 0, "beta": 1})
    swapped = play(swapped_snapshot, {"alpha": 1, "beta": 0})
    # same situation from the mirrored seat + same role streams -> the same actions
    assert straight == swapped


def test_policy_is_stateless_between_calls():
    obs = build_observation(begin_policy_phase(busy_state()), 0)
    a = decide(random_policy, obs, Pcg32.seeded(7, 9)).action
    decide(random_policy, obs, Pcg32.seeded(8, 9))
    assert decide(random_policy, obs, Pcg32.seeded(7, 9)).action == a
