"""FOR-42: GAE, PPO-Clip mathematics and a toy problem with a known answer - all before any real match is played."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from formicmind.decoder import PolicyLogitsV2, decode_action_logits  # noqa: E402
from formicmind.rl.distribution import evaluate_latent, latent_to_logits, sample_latent  # noqa: E402
from formicmind.rl.network import ActorCritic  # noqa: E402
from formicmind.rl.ppo import PPOConfig, gae_padded, ppo_update  # noqa: E402
from formicmind.rng import Pcg32  # noqa: E402
from formicmind.actions import mask_from_flat  # noqa: E402
from rl_helpers import make_mask  # noqa: E402


def reference_gae(rewards, values, last_value, gamma, lam):
    n = len(rewards)
    adv = [0.0] * n
    running = 0.0
    for t in reversed(range(n)):
        nxt = values[t + 1] if t + 1 < n else last_value
        delta = rewards[t] + gamma * nxt - values[t]
        running = delta + gamma * lam * running
        adv[t] = running
    return adv


def test_gae_matches_reference_with_terminal_and_truncation():
    gamma, lam = 0.9, 0.8
    episodes = [
        ([0.0, 0.0, 1.0], [0.2, 0.4, 0.6], 0.0),  # terminal: bootstrap 0
        ([0.1, -0.2], [0.3, 0.1], 0.7),  # truncated: bootstrap from V(final obs)
        ([-1.0], [0.5], 0.0),
    ]
    horizon = max(len(e[0]) for e in episodes)
    rewards = torch.zeros(len(episodes), horizon)
    values = torch.zeros(len(episodes), horizon)
    for i, (r, v, _) in enumerate(episodes):
        rewards[i, :len(r)] = torch.tensor(r)
        values[i, :len(v)] = torch.tensor(v)
    lengths = torch.tensor([len(e[0]) for e in episodes])
    last = torch.tensor([e[2] for e in episodes])
    adv = gae_padded(rewards, values, lengths, last, gamma, lam)
    for i, (r, v, boot) in enumerate(episodes):
        expected = reference_gae(r, v, boot, gamma, lam)
        assert adv[i, :len(r)].tolist() == pytest.approx(expected, abs=1e-6)
        assert adv[i, len(r):].abs().sum() == 0  # padding never leaks


def test_truncation_bootstraps_but_terminal_does_not():
    rewards = torch.zeros(2, 1)
    values = torch.zeros(2, 1)
    adv = gae_padded(rewards, values, torch.tensor([1, 1]), torch.tensor([0.0, 0.8]), 0.5, 0.9)
    assert adv[0, 0].item() == 0.0  # terminal state: no value after it
    assert adv[1, 0].item() == pytest.approx(0.4)  # truncated: gamma * V(final)


def test_ppo_first_ratio_is_identity_and_clipping_blocks_gradient():
    torch.manual_seed(0)
    net = ActorCritic((16, 16))
    flat = make_mask()
    rng = Pcg32.seeded(1, 2)
    obs = torch.rand(64, 223)
    heads = net.policy(obs).detach()
    ints, zs = [], []
    for row in heads:
        i, z = sample_latent(row.tolist(), net.log_std.tolist(), flat, rng)
        ints.append(i)
        zs.append(z)
    mask = torch.tensor([flat] * 64)
    ints_t, z_t = torch.tensor(ints), torch.tensor(zs)
    ev = evaluate_latent(heads, net.log_std.detach(), mask, ints_t, z_t)
    # identical weights -> ratio == 1 to float precision
    again = evaluate_latent(net.policy(obs), net.log_std, mask, ints_t, z_t)
    assert (again.log_prob - ev.log_prob).abs().max().item() < 1e-5

    # a ratio far above 1 + clip with positive advantage must produce zero policy gradient
    net.zero_grad()
    log_ratio = again.log_prob - (ev.log_prob - 1.0)  # ratio = e ~ 2.72 > 1.2
    ratio = torch.exp(log_ratio)
    adv = torch.ones(64)
    surrogate = torch.minimum(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv)
    (-surrogate.mean()).backward()
    assert all(p.grad is None or p.grad.abs().max().item() == 0.0 for p in net.actor.parameters())
    assert all(p.grad is None or p.grad.abs().max().item() == 0.0 for p in net.policy_head.parameters())


def test_update_reports_finite_losses_and_identity_deviation():
    torch.manual_seed(1)
    net = ActorCritic((16, 16))
    flat = make_mask()
    rng = Pcg32.seeded(3, 4)
    obs = torch.rand(256, 223)
    with torch.no_grad():
        heads = net.policy(obs)
    ints, zs, logps = [], [], []
    for row in heads:
        i, z = sample_latent(row.tolist(), net.log_std.tolist(), flat, rng)
        ints.append(i)
        zs.append(z)
    mask = torch.tensor([flat] * 256)
    old = evaluate_latent(heads, net.log_std.detach(), mask, torch.tensor(ints), torch.tensor(zs)).log_prob
    batch = {"obs": obs, "mask": mask, "ints": torch.tensor(ints), "z": torch.tensor(zs), "old_logp": old,
             "adv": torch.randn(256), "ret": torch.randn(256)}
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in net.parameters()]
    stats = ppo_update(net, optimizer, batch, PPOConfig(minibatch=64, epochs=2, target_kl=0.0),
                       torch.Generator().manual_seed(0))
    assert stats["identity_ratio_max_dev"] < 1e-4
    for key in ("loss", "policy_loss", "value_loss", "entropy_discrete", "approx_kl", "grad_norm"):
        assert math.isfinite(stats[key]), key
    assert any((a != b.detach()).any() for a, b in zip(before, net.parameters()))


def test_non_finite_batch_is_rejected_not_learned_from():
    net = ActorCritic((8, 8))
    flat = make_mask()
    obs = torch.rand(8, 223)
    batch = {"obs": obs, "mask": torch.tensor([flat] * 8), "ints": torch.zeros(8, 7, dtype=torch.long),
             "z": torch.zeros(8, 3), "old_logp": torch.zeros(8), "adv": torch.full((8,), float("nan")),
             "ret": torch.zeros(8)}
    with pytest.raises(FloatingPointError):
        ppo_update(net, torch.optim.Adam(net.parameters(), lr=1e-3), batch, PPOConfig(minibatch=8, epochs=1),
                   torch.Generator().manual_seed(0))


def _bandit_batch(net, obs, flat, rng, reward_fn, n):
    mask_obj = mask_from_flat(tuple(flat))
    with torch.no_grad():
        heads = net.policy(obs)
    ints, zs, rewards = [], [], []
    for row in heads:
        i, z = sample_latent(row.tolist(), net.log_std.tolist(), flat, rng)
        action = decode_action_logits(PolicyLogitsV2(latent_to_logits(i, z)), mask_obj)
        ints.append(i)
        zs.append(z)
        rewards.append(reward_fn(action))
    mask = torch.tensor([flat] * n)
    ints_t, z_t = torch.tensor(ints), torch.tensor(zs)
    old = evaluate_latent(heads, net.log_std.detach(), mask, ints_t, z_t).log_prob
    reward = torch.tensor(rewards)
    with torch.no_grad():
        value = net.value(obs)
    adv = reward - value  # one-step episodes: advantage = r - V
    return {"obs": obs, "mask": mask, "ints": ints_t, "z": z_t, "old_logp": old, "adv": adv, "ret": reward}, reward.mean().item()


@pytest.mark.parametrize("kind", ["lay_and_army", "forage_share"])
def test_ppo_solves_a_contextual_bandit_with_a_known_answer(kind):
    """The executed (decoded) action is rewarded; the policy must learn it from rewards alone."""
    torch.manual_seed(5)
    net = ActorCritic((32, 32))
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-3, eps=1e-5)
    flat = make_mask()
    rng = Pcg32.seeded(8, 9)
    obs = torch.rand(512, 223) * 0.2  # fixed contexts; the answer does not depend on them

    if kind == "lay_and_army":
        def reward_fn(action):  # lay a soldier AND raid the enemy home: reward 1, otherwise 0
            return float(action.lay_order == 2 and action.army_choice == 6)
    else:
        def reward_fn(action):  # put the weight of the forage on the centre target
            return action.forage_share_q[3] / 65535.0

    first = None
    for step in range(60):
        batch, mean_reward = _bandit_batch(net, obs, flat, rng, reward_fn, 512)
        first = mean_reward if first is None else first
        ppo_update(net, optimizer, batch, PPOConfig(minibatch=128, epochs=4, ent_coef=0.0, ent_coef_share=0.0,
                                                    target_kl=0.0, lr=3e-3), torch.Generator().manual_seed(step))
    _, final = _bandit_batch(net, obs, flat, rng, reward_fn, 512)
    if kind == "lay_and_army":
        assert first < 0.1 and final > 0.8
    else:
        assert final > first + 0.35 and final > 0.6
