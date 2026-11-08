"""FOR-49: the stochastic policy over ActionE1 - shapes, masks, normalisation, pointer choice without replacement, exact log-prob of the same
sample, the shared decoder reproducing the sample, and isolation of colonies and of hidden facts."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch", reason="the learning stack needs PyTorch (installed in Windows CI)")

from test_eco_rl_helpers import peaky_model, recorded_decisions
from formicmind.eco.action import ActionMasksE1, validate_action
from formicmind.eco.decoder import decode
from formicmind.eco.rl.distribution import (
    I_EGG, I_FLEE, I_FRAC, I_FT, I_K, I_SCOUT_N, I_SCOUT_T, I_SOLDIER, I_SQUAD, I_ST, SAMPLE_INTS, Sample, evaluate, sample_rows,
)
from formicmind.eco.rl.features import cat, select, step_batch
from formicmind.eco.rl.network import ActorCriticE1
from formicmind.eco.rl.rollout import MirrorRuntime
from formicmind.rng import new_stream, policy_stream_id


def _rngs(n, seed=1, offset=0):
    return [new_stream(seed, policy_stream_id("learned", offset + i)) for i in range(n)]


def _forward(model, items):
    batch = cat([step_batch(o, m) for o, m in items])
    hidden = model.config.hidden
    with torch.no_grad():
        heads, value, _, _ = model.step(batch, torch.zeros(batch.size, hidden), torch.zeros(batch.size, hidden))
    return batch, heads, value


def _check_action_matches(action, row, masks):
    assert (action.egg_level, action.soldier_share_q, action.workers_flee, action.forage_frac_q) == (row[I_EGG], row[I_SOLDIER], bool(row[I_FLEE]), row[I_FRAC])
    k = row[I_K] if row[I_FRAC] > 0 else 0
    assert action.forage_targets == tuple(row[I_FT:I_FT + k]) and len(action.forage_shares_q) == k
    assert action.scout_n == row[I_SCOUT_N]
    t = row[I_SCOUT_T] if row[I_SCOUT_N] > 0 else 0
    assert action.scout_targets == tuple(row[I_ST:I_ST + t])
    for s, squad in enumerate(action.squads):
        mission, size, guard, attack = row[I_SQUAD + 4 * s:I_SQUAD + 4 * s + 4]
        assert squad.mission == ("none", "guard", "attack", "withdraw")[mission]
        if mission in (1, 2):
            assert squad.size_q == size + 1
        if mission == 1:
            assert squad.node_slot == guard
        if mission == 2:
            assert (squad.node_slot, squad.colony_slot) == masks.attack_pairs[attack]


# ------------------------------------------------------------------------------------------------ sampling, decoder and masks


def test_the_shared_decoder_reproduces_every_sampled_choice_and_every_action_is_legal():
    decisions = recorded_decisions()
    seen = {"forage": 0, "scout": 0, "guard": 0, "attack": 0, "two_scouts": 0, "withdraw": 0, "k3": 0}
    for model_seed, greedy in ((1, False), (2, False), (3, True)):
        model = peaky_model(model_seed)
        for start in range(0, len(decisions), 64):
            items = decisions[start:start + 64]
            for repeat in range(3):
                batch, heads, _ = _forward(model, items)
                sample, latents = sample_rows(heads, batch, _rngs(len(items), seed=7 + repeat, offset=start), greedy)
                for (obs, masks), latent, row in zip(items, latents, sample.ints.tolist()):
                    action = decode(latent, masks)
                    validate_action(action, masks)
                    _check_action_matches(action, row, masks)
                    seen["forage"] += action.forage_frac_q > 0
                    seen["k3"] += len(action.forage_targets) == 3
                    seen["scout"] += action.scout_n > 0
                    seen["two_scouts"] += len(action.scout_targets) == 2
                    seen["guard"] += any(s.mission == "guard" for s in action.squads)
                    seen["attack"] += any(s.mission == "attack" for s in action.squads)
                    seen["withdraw"] += any(s.mission == "withdraw" for s in action.squads)
    assert all(count > 0 for count in seen.values()), seen        # the test really walked every branch of the decoder


def test_the_log_prob_of_a_sample_is_the_same_number_by_two_independent_routes():
    decisions = recorded_decisions()
    model = peaky_model(5, scale=8.0)
    worst = 0.0
    for start in range(0, len(decisions), 80):
        items = decisions[start:start + 80]
        batch, heads, _ = _forward(model, items)
        py: list[float] = []
        sample, _ = sample_rows(heads, batch, _rngs(len(items), offset=start), False, py)
        ev = evaluate(heads, batch, sample)
        worst = max(worst, float((ev.log_prob - torch.tensor(py)).abs().max()))
        assert torch.isfinite(ev.log_prob).all() and torch.isfinite(ev.entropy_discrete).all() and torch.isfinite(ev.entropy_z).all()
        assert (ev.entropy_discrete >= -1e-5).all()
    assert worst < 1e-3


def test_masked_options_get_probability_zero_and_forced_branches_cost_nothing():
    obs, masks = recorded_decisions()[0]
    forced = replace(
        masks, egg_level=(True, False, False, False), soldier_share=(True, False, False, False, False),
        forage_frac=(True,) + (False,) * 8, scout_n=(True,) + (False,) * 4, workers_flee=(True, False),
        forage_target=(False,) * len(masks.forage_target), scout_target=(False,) * len(masks.scout_target),
        guard_target=(False,) * len(masks.guard_target), attack_pairs=(), soldiers_idle_home=0, soldiers_total=0,
        squad_mission=((True, False, False, False),) * 3)
    batch = step_batch(obs, forced)
    model = peaky_model(9)
    with torch.no_grad():
        heads, _, _, _ = model.step(batch, torch.zeros(1, 128), torch.zeros(1, 128))
        for greedy in (False, True):
            sample, latents = sample_rows(heads, batch, _rngs(1), greedy)
            ev = evaluate(heads, batch, sample)
            assert ev.log_prob.item() == 0.0 and ev.entropy_discrete.item() == 0.0 and ev.entropy_z.item() == 0.0
            action = decode(latents[0], forced)
            assert (action.egg_level, action.forage_frac_q, action.scout_n) == (0, 0, 0) and all(s.mission == "none" for s in action.squads)
    # a masked option can never be drawn, however confident the network is about it
    only_two = replace(masks, egg_level=(False, True, False, True))
    batch2 = step_batch(obs, only_two)
    with torch.no_grad():
        heads2, _, _, _ = model.step(batch2, torch.zeros(1, 128), torch.zeros(1, 128))
    drawn = {int(sample_rows(heads2, batch2, _rngs(1, seed=s), False)[0].ints[0, I_EGG]) for s in range(300)}
    assert drawn <= {1, 3} and len(drawn) == 2


def _enumerated_sample(egg=0, frac=0, k=0, targets=(), z=(0.0, 0.0, 0.0)):
    row = [-1] * SAMPLE_INTS
    row[I_EGG], row[I_SOLDIER], row[I_FLEE], row[I_FRAC], row[I_K] = egg, 0, 0, frac, k
    for j, node in enumerate(targets):
        row[I_FT + j] = node
    row[I_SCOUT_N] = 0
    for s in range(3):
        row[I_SQUAD + 4 * s] = 0
    mission_allowed = torch.tensor([[[True, False, False, False]] * 3])
    return Sample(torch.tensor([row]), torch.tensor([list(z)], dtype=torch.float32), mission_allowed)


def test_choices_are_normalised_and_pointers_are_chosen_without_replacement():
    obs, masks = recorded_decisions()[0]
    legal = [False] * len(masks.forage_target)
    legal[1] = legal[2] = legal[3] = True
    restricted = replace(
        masks, egg_level=(True, True, False, False), soldier_share=(True,) + (False,) * 4, forage_frac=(True,) + (False,) * 7 + (True,),
        scout_n=(True,) + (False,) * 4, workers_flee=(True, False), forage_target=tuple(legal),
        scout_target=(False,) * len(masks.scout_target), guard_target=(False,) * len(masks.guard_target), attack_pairs=(),
        soldiers_idle_home=0, soldiers_total=0, squad_mission=((True, False, False, False),) * 3)
    batch = step_batch(obs, restricted)
    model = peaky_model(11, scale=3.0)
    with torch.no_grad():
        heads, _, _, _ = model.step(batch, torch.zeros(1, 128), torch.zeros(1, 128))
        total, outcomes = 0.0, 0
        for egg in (0, 1):
            ev = evaluate(heads, batch, _enumerated_sample(egg=egg, frac=0))
            total += math.exp(ev.log_prob.item())
            outcomes += 1
            for k in (1, 2, 3):
                perms = [()]
                for _ in range(k):
                    perms = [p + (n,) for p in perms for n in (1, 2, 3) if n not in p]
                for perm in perms:
                    ev = evaluate(heads, batch, _enumerated_sample(egg=egg, frac=8, k=k, targets=perm))
                    total += math.exp((ev.log_prob - ev.log_prob_z).item())      # the real-valued share logits integrate to 1 by themselves
                    outcomes += 1
        assert outcomes == 2 * (1 + 3 + 6 + 6)
        assert total == pytest.approx(1.0, abs=1e-5)
        # sampled pointer choices are distinct legal slots
        for seed in range(200):
            sample, _ = sample_rows(heads, batch, _rngs(1, seed=seed), False)
            row = sample.ints[0].tolist()
            if row[I_FRAC] > 0:
                chosen = row[I_FT:I_FT + row[I_K]]
                assert len(set(chosen)) == len(chosen) and all(legal[n] for n in chosen)


def test_the_networks_outputs_are_finite_for_every_block_shape_and_a_rare_empty_block():
    decisions = recorded_decisions()
    model = ActorCriticE1()
    first = decisions[0]          # the very first observation: no foreign colonies, sightings or events
    last = decisions[-1]
    rich = max(decisions, key=lambda d: len(d[0].sighting_features) + len(d[0].foreign_features) + len(d[0].event_features))
    for items in ([first], [last], [rich], [first, last, rich], list(decisions[:200])):
        batch, heads, value = _forward(model, items)
        batch.validate()
        assert torch.isfinite(value).all() and all(torch.isfinite(t).all() for t in heads.cat.values())
        assert torch.isfinite(heads.mu).all() and torch.isfinite(heads.forage).all() and torch.isfinite(heads.attack).all()
        assert heads.forage.shape[0] == len(items) and heads.guard.shape[0] == 3 and heads.attack.shape[:2] == (3, len(items))
    assert len(first[0].foreign_features) == 0 and len(rich[0].foreign_features) > 0


def test_a_batch_can_be_cut_into_steps_and_put_back_together_losslessly():
    items = list(recorded_decisions()[:40])
    batch = cat([step_batch(o, m) for o, m in items])
    ids = torch.tensor([7, 3, 3, 39, 0])
    part = select(batch, ids)
    part.validate()
    direct = cat([step_batch(*items[i]) for i in ids.tolist()])
    assert set(part.t) == set(direct.t)
    for name in part.t:
        assert torch.equal(part.t[name], direct.t[name]), name


# ------------------------------------------------------------------------------------------------ isolation


def test_one_colonys_decision_does_not_depend_on_which_other_colonies_decide_with_it():
    decisions = recorded_decisions()
    model = peaky_model(13, scale=4.0)
    items = list(decisions[100:108])
    solo = items[2]
    batch_all, heads_all, value_all = _forward(model, items)
    batch_one, heads_one, value_one = _forward(model, [solo])
    assert torch.allclose(heads_all.cat["frac"][2], heads_one.cat["frac"][0], atol=1e-5)
    assert torch.allclose(heads_all.mu[2], heads_one.mu[0], atol=1e-5) and torch.allclose(value_all[2], value_one[0], atol=1e-5)
    streams = _rngs(8, seed=99)
    streams[2] = new_stream(3, policy_stream_id("learned", 2))
    s_all, _ = sample_rows(heads_all, batch_all, streams, False)
    s_one, _ = sample_rows(heads_one, batch_one, [new_stream(3, policy_stream_id("learned", 2))], False)
    assert s_all.ints[2].tolist() == s_one.ints[0].tolist()


def test_every_colony_has_its_own_memory_and_its_own_random_stream():
    decisions = recorded_decisions()
    model = peaky_model(17, scale=4.0)
    obs, masks = decisions[50]
    runtime = MirrorRuntime(model, 3, policy_seed=5)
    assert float(runtime.h_actor.abs().sum()) == 0.0 and float(runtime.h_critic.abs().sum()) == 0.0         # a fresh episode starts from zero memory
    runtime.act_batch([(0, obs, masks)])                                                                    # only colony 0 decides (an alert, say)
    assert float(runtime.h_actor[0].abs().sum()) > 0 and float(runtime.h_actor[1:].abs().sum()) == 0.0
    assert float(runtime.h_critic[0].abs().sum()) > 0 and float(runtime.h_critic[1:].abs().sum()) == 0.0
    # the same observation to three colonies: same network, same memory (zero) for 1 and 2, but private streams -> different draws
    draws = {tuple(runtime.act_batch([(c, obs, masks)])[0]) for c in (1, 2)}
    assert len(draws) == 2
    again = MirrorRuntime(model, 3, policy_seed=5)
    assert again.act_batch([(1, obs, masks)])[0] == MirrorRuntime(model, 3, policy_seed=5).act_batch([(1, obs, masks)])[0]   # reproducible per seed
    assert MirrorRuntime(model, 3, policy_seed=5).act_batch([(1, obs, masks)])[0] != MirrorRuntime(model, 3, policy_seed=6).act_batch([(1, obs, masks)])[0]


def test_the_network_input_is_only_an_observation_and_its_masks():
    import inspect

    assert list(inspect.signature(step_batch).parameters) == ["obs", "masks"]
    obs, masks = recorded_decisions()[10]
    twin = type(obs)(**{f: getattr(obs, f) for f in obs.__dataclass_fields__})        # an equal observation built independently
    batch_a, batch_b = step_batch(obs, masks), step_batch(twin, masks)
    assert all(torch.equal(batch_a.t[k], batch_b.t[k]) for k in batch_a.t)
    assert set(batch_a.t) == {"g", "smask", "squad_base", "units", "n_cnt", "n_feat", "n_kind", "n_legal", "e_cnt", "e_feat", "e_slot", "f_cnt", "f_feat",
                              "f_nest", "c_cnt", "c_feat", "c_slot", "s_cnt", "s_feat", "s_int", "v_cnt", "v_feat", "v_int", "p_cnt", "p_slot"}
    assert isinstance(masks, ActionMasksE1)
