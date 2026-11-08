"""FOR-42 phase 1 + rollout: honest rewards, truncation handling, hidden-information boundary, RNG independence."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from formicmind.rl.maps import (  # noqa: E402
    HELDOUT_NAMED_MAPS, TRAIN_NAMED_MAPS, map_config, map_split,
)
from formicmind.rl.network import ActorCritic, actor_state_to_bytes  # noqa: E402
from formicmind.rl.parallel import execute_chunk  # noqa: E402
from formicmind.rl.reward import shaping_potential, step_rewards, terminal_reward  # noqa: E402
from formicmind.rl.rollout import decode_trajectory  # noqa: E402


def _refs(seed: int = 0, hidden=(16, 16)):
    torch.manual_seed(seed)
    net = ActorCritic(hidden)
    return {"current": {"bytes": actor_state_to_bytes(net.actor_state_dict())}}, net


def _spec(seed=11, map_id="classic", second="rl", max_ticks=None, shaping=0.0, record_b=True):
    a = {"kind": "rl", "ref": "current", "stream": ("rl-current", 0), "record": True}
    if second == "rl":
        b = {"kind": "rl", "ref": "current", "stream": ("rl-current", 1), "record": record_b}
    else:
        b = {"kind": "baseline", "ref": second, "replicate": 0}
    return {"match_id": 0, "seed": seed, "map_id": map_id, "seats": [a, b], "max_ticks": max_ticks,
            "shaping_coef": shaping, "gamma": 0.99, "tag": {}}


def test_terminal_reward_mapping_and_zero_sum():
    assert [terminal_reward(s) for s in (1.0, 0.5, 0.0)] == [1.0, 0.0, -1.0]
    with pytest.raises(ValueError):
        terminal_reward(0.7)
    refs, _ = _refs()
    results = execute_chunk(([_spec(seed=s) for s in range(20, 26)], refs))
    for r in results:
        assert not r["truncated"]
        rewards = [decode_trajectory(seat["trajectory"])["reward"][-1].item() for seat in r["seats"]]
        assert sum(rewards) == pytest.approx(0.0)  # a mirror self-play episode is zero-sum
        assert r["seats"][0]["trajectory"]["terminal_reward"] == -r["seats"][1]["trajectory"]["terminal_reward"]


def test_without_shaping_only_the_last_decision_is_rewarded():
    refs, _ = _refs()
    result = execute_chunk(([_spec(seed=5)], refs))[0]
    for seat in result["seats"]:
        reward = decode_trajectory(seat["trajectory"])["reward"]
        assert reward[:-1].abs().sum() == 0
        assert reward[-1].item() in (-1.0, 0.0, 1.0)


def test_truncated_episode_gets_no_terminal_reward_and_a_bootstrap_observation():
    refs, _ = _refs()
    result = execute_chunk(([_spec(seed=5, second="turtle:v2", max_ticks=60)], refs))[0]
    assert result["truncated"] and result["winner"] is None and result["seats"][0]["score"] is None
    traj = result["seats"][0]["trajectory"]
    assert traj["truncated"] and traj["terminal_reward"] is None
    assert len(traj["bootstrap_obs"]) == 223 * 4  # observation of the cutoff state, for V(final)
    assert decode_trajectory(traj)["reward"].abs().sum() == 0  # never turned into a loss or a draw
    assert traj["steps"] == 2  # decisions at ticks 0 and 30; the cutoff state (tick 60) is only bootstrapped


def test_shaping_telescopes_and_is_bounded():
    gamma, coef = 0.99, 0.1
    rng = torch.Generator().manual_seed(0)
    potentials = (torch.rand(100, generator=rng) * coef).tolist()
    shaped = step_rewards(potentials, terminal=None, gamma=gamma, final_potential=0.0)
    # sum_t gamma^t (r'_t) = -Phi_0 (potential-based shaping telescopes); undiscounted sum is bounded by 2 * coef
    discounted = sum(gamma ** t * r for t, r in enumerate(shaped))
    assert discounted == pytest.approx(-potentials[0], abs=1e-9)
    assert abs(sum(shaped)) <= 2 * coef
    won = step_rewards(potentials, terminal=1.0, gamma=gamma)
    assert won[-1] - shaped[-1] == pytest.approx(1.0)


def test_shaping_potential_uses_only_own_public_features_and_is_bounded():
    from formicmind.observation import OBS_V2_LENGTH
    values = [0.0] * OBS_V2_LENGTH
    assert shaping_potential(values, 0.1) == 0.0
    assert shaping_potential([0.9] * OBS_V2_LENGTH, 0.0) == 0.0
    assert 0.0 <= shaping_potential([0.999999] * OBS_V2_LENGTH, 0.1) < 0.1


def test_idle_policy_cannot_farm_reward_a_timeout_draw_is_worth_zero():
    # idle vs idle: nobody dies, nothing is rewarded; the draw by timeout is a 0 outcome
    from formicmind.actions import ActionV2
    from formicmind.match import run_episode
    from formicmind.rules import default_config
    from formicmind.tick import ScriptedController
    idle = ActionV2(0, (65535, 0, 0, 0, 0, 0), 0, False)
    full = run_episode(default_config(), 1, ScriptedController({0: idle}), ScriptedController({0: idle}))
    assert full.score == (0.5, 0.5)  # both colonies starve together: a draw, worth exactly 0
    assert [terminal_reward(x) for x in full.score] == [0.0, 0.0]
    cut = run_episode(default_config(), 1, ScriptedController({0: idle}), ScriptedController({0: idle}), max_ticks=30)
    assert cut.truncated and cut.score is None  # a harness cutoff is not an outcome and has no reward
    assert shaping_potential([0.5] * 223, 0.05) * 2 < 0.1  # shaping alone can never rival the outcome


def test_policy_input_is_exactly_the_legal_observation():
    refs, net = _refs()
    from formicmind.rl.network import actor_state_from_bytes, build_actor
    from formicmind.rl.rollout import run_match
    actor = build_actor(actor_state_from_bytes(refs["current"]["bytes"]))
    seen = []
    original = actor.policy

    def spy(x):
        seen.append(x.clone())
        return original(x)

    actor.policy = spy  # type: ignore[method-assign]
    result = run_match(_spec(seed=3, second="turtle:v2", max_ticks=60), {"current": actor})
    assert seen and all(t.shape == (1, 223) for t in seen)
    recorded = decode_trajectory(result["seats"][0]["trajectory"])["obs"]
    assert torch.equal(torch.cat(seen), recorded)  # the network saw nothing but the recorded ObservationV2


def test_recorded_actions_match_what_the_engine_executed():
    from formicmind.actions import mask_from_flat
    from formicmind.decoder import PolicyLogitsV2, decode_action_logits
    from formicmind.rl.distribution import latent_to_logits
    refs, _ = _refs()
    result = execute_chunk(([_spec(seed=9)], refs))[0]
    for seat in result["seats"]:
        d = decode_trajectory(seat["trajectory"])
        for i in range(d["obs"].shape[0]):
            mask = mask_from_flat(tuple(bool(x) for x in d["mask"][i].tolist()))
            action = decode_action_logits(PolicyLogitsV2(latent_to_logits(d["ints"][i].tolist(), d["z"][i].tolist())), mask)
            assert list(action.forage_share_q) == d["executed"][i].tolist()
            assert [action.lay_order, action.army_choice, int(action.avoid_center)] == d["executed_choices"][i].tolist()
        assert torch.isfinite(d["logp"]).all() and (d["logp"] <= 0).all()


def test_policy_rng_streams_are_independent_of_seats_and_reproducible():
    refs, _ = _refs()
    one = execute_chunk(([_spec(seed=5)], refs))[0]
    again = execute_chunk(([_spec(seed=5)], refs))[0]
    other_seed = execute_chunk(([_spec(seed=6)], refs))[0]
    t = lambda r, i: r["seats"][i]["trajectory"]  # noqa: E731
    assert t(one, 0)["ints"] == t(again, 0)["ints"] and t(one, 0)["logp"] == t(again, 0)["logp"]  # same seed -> same match
    assert t(one, 0)["ints"] != t(other_seed, 0)["ints"]  # new seed -> new policy samples
    # identical weights in mirror seats still behave differently: two distinct streams, not one shared sequence
    assert t(one, 0)["ints"] != t(one, 1)["ints"]
    assert one["seats"][0]["stats"]["decisions"] > 0


def test_maps_are_valid_mirror_topologies_and_splits_are_disjoint():
    assert not set(TRAIN_NAMED_MAPS) & set(HELDOUT_NAMED_MAPS)
    for map_id in (*TRAIN_NAMED_MAPS, *HELDOUT_NAMED_MAPS, "gen:0", "gen:17", "gen:1000000", "gen:1000042"):
        config = map_config(map_id)
        assert config.balance == map_config("classic").balance  # only topology moves, never the rules or balance
        assert map_split(map_id) in ("train", "heldout")
    assert map_config("gen:7") == map_config("gen:7")
    assert len({tuple(map_config(f"gen:{i}").topology.edges) for i in range(30)}) > 5  # genuinely diverse
    assert map_split("gen:5") == "train" and map_split("gen:1000005") == "heldout"
    with pytest.raises(ValueError):
        map_config("gen:-1")
