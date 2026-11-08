"""FOR-49: rollout accounting on the real engine, PPO (identity of the log-prob, a known task, no leakage between colonies), model files, the
learned policy behind PolicyE1 in a verified replay, and the train / eval commands."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="the learning stack needs PyTorch (installed in Windows CI)")

from test_eco_rl_helpers import peaky_model, recorded_decisions
from formicmind.eco import default_config
from formicmind.eco.cli import main
from formicmind.eco.engine import RouteTable, step_world_tick
from formicmind.eco.initial import build_world
from formicmind.eco.policy import PolicyRunSpec, PolicySpec
from formicmind.eco.policy_run import PolicyController
from formicmind.eco.replay_file import ReplayFormatError, export_policy_replay, read_replay, verify_replay
from formicmind.eco.rl.checkpoint import MANIFEST, ModelError, WEIGHTS, load_model, save_model
from formicmind.eco.rl.distribution import I_EGG, evaluate, sample_rows
from formicmind.eco.rl.features import cat, step_batch
from formicmind.eco.rl.learned import LearnedColonyPolicy, register_model, variant_for
from formicmind.eco.rl.network import ActorCriticE1
from formicmind.eco.rl.ppo import PPOConfig, TrainingSet, merge_episodes, ppo_update
from formicmind.eco.rl.rollout import MirrorRuntime, SlotPolicy, collect_episode
from formicmind.eco.viability import REWARD_VERSION, interval_discount
from formicmind.rng import new_stream, policy_stream_id

CONFIG = default_config(8, "moderate")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def untrained():
    torch.manual_seed(0)
    model = ActorCriticE1()
    model.eval()
    return model


# ------------------------------------------------------------------------------------------------ rollout


def test_a_rollout_closes_every_interval_with_the_tick_trace_and_ends_each_colony_once(untrained):
    episode = collect_episode(CONFIG, 200000, untrained, 3, 700, reward_scale=10.0)
    slots = episode.slot.tolist()
    assert sorted(set(slots)) == list(range(8)) and episode.batch.size == len(slots)
    ticks_of = [int(t) for t in episode.batch["g"][:, 0] * 6000 + 0.5]       # global feature 0 is tick / horizon
    for colony, row in zip(range(8), episode.metrics):
        steps = [i for i, s in enumerate(slots) if s == colony]
        deltas = [int(episode.delta[i]) for i in steps]
        terminated = [bool(episode.terminated[i]) for i in steps]
        assert all(d >= 1 for d in deltas) and ticks_of[steps[0]] == 0
        starts = [ticks_of[i] for i in steps]
        assert starts == sorted(set(starts))                                              # chronological, never two decisions in one tick
        assert all(a + d == b for a, d, b in zip(starts, deltas, starts[1:]))             # intervals tile the lifetime without gaps
        end = (row["death_tick"] + 1) if row["death_tick"] is not None else episode.ticks
        assert sum(deltas) == end
        assert terminated == [False] * (len(steps) - 1) + [row["death_tick"] is not None]     # the terminal flag is on the last interval, once
        assert float(episode.reward[steps].sum()) == pytest.approx(row["reward"], abs=1e-9)  # the penalty is inside, once; no hidden terms
    assert int(episode.terminated.sum()) == sum(1 for r in episode.metrics if not r["survived"]) > 0


def test_a_truncated_trajectory_bootstraps_with_the_critic_and_a_terminal_one_does_not(untrained):
    truncated = collect_episode(CONFIG, 200001, untrained, 3, 40, reward_scale=10.0)
    assert not truncated.terminated.any() and set(truncated.bootstrap) == set(range(8))
    for colony in range(8):
        last = max(i for i, s in enumerate(truncated.slot.tolist()) if s == colony)
        delta = int(truncated.delta[last])
        reward_disc_scaled = float(truncated.advantage[last]) + float(truncated.value[last]) - interval_discount(delta) * truncated.bootstrap[colony]
        assert 0.0 < reward_disc_scaled < 10.0 * delta / 6000                              # r_disc * scale: positive viability, nothing else
    dead = collect_episode(CONFIG, 200000, untrained, 3, 700, reward_scale=10.0)
    for colony, row in enumerate(dead.metrics):
        assert (colony in dead.bootstrap) == row["survived"]
        if not row["survived"]:
            last = max(i for i, s in enumerate(dead.slot.tolist()) if s == colony)
            assert bool(dead.terminated[last])
            target = float(dead.advantage[last]) + float(dead.value[last])               # = r_disc * scale: no value of any state after death
            assert -10.01 < target < -9.0                                                 # the death penalty (1.0 * scale), paid in this interval


def test_the_rollout_policy_and_the_replay_policy_decide_alike(untrained):
    """Training samples through the batched mirror runtime, the replay through one PolicyE1 object per colony: same weights, same streams, same world
    must give the same decisions (the two forward paths differ only in floating-point noise)."""
    model = peaky_model(21, scale=2.0)

    def run(controller_factory):
        world = build_world(CONFIG, 50012)
        routes = RouteTable(world.world_map)
        controller = controller_factory()
        for _ in range(61):                      # a random network starves its colonies within ~100 ticks: three planned decisions each
            if world.state.world_outcome.finished:
                break
            step_world_tick(world.state, world.run, CONFIG, world.world_map, controller, routes=routes, validate_input=world.state.tick == 0)
        return controller.log.head, controller.log.count

    def batched():
        runtime = MirrorRuntime(model, 8, 11)
        return PolicyController([SlotPolicy(runtime, c) for c in range(8)], batcher=runtime)

    def per_colony():
        return PolicyController([LearnedColonyPolicy(model, new_stream(11, policy_stream_id("learned", c)), greedy=False, name="x") for c in range(8)])

    assert run(batched) == run(per_colony) and run(batched)[1] >= 16


# ------------------------------------------------------------------------------------------------ PPO


def test_the_first_ppo_pass_reproduces_the_rollout_log_probs_and_a_wrong_one_is_caught(untrained):
    model = ActorCriticE1()
    model.load_state_dict(untrained.state_dict())
    episodes = [collect_episode(CONFIG, 200010 + i, model, 5 + i, 360, reward_scale=10.0) for i in range(2)]
    data = merge_episodes(episodes)
    assert data.steps == sum(e.batch.size for e in episodes) and len(data.sequences) == 16
    before = [p.detach().clone() for p in model.parameters()]
    stats = ppo_update(model, torch.optim.Adam(model.parameters(), lr=3e-4), data, PPOConfig(seqs_per_minibatch=6), torch.Generator().manual_seed(0))
    assert stats["identity_ratio_max_dev"] < 1e-4                        # the stored sample has the stored probability: the recurrent unroll is exact
    assert all(map(lambda v: v == v and abs(v) < 1e9, stats.values()))
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    corrupted = TrainingSet(data.batch, data.sample, data.old_logp + 0.5, data.advantage, data.ret, data.sequences, data.steps)
    wrong = ppo_update(model, torch.optim.Adam(model.parameters(), lr=0.0), corrupted, PPOConfig(seqs_per_minibatch=6), torch.Generator().manual_seed(0))
    assert wrong["identity_ratio_max_dev"] > 0.4                         # a log-prob bug shows up here instead of passing for learning


def test_accumulating_gradients_over_chunks_equals_one_big_minibatch(untrained):
    episodes = [collect_episode(CONFIG, 200020, untrained, 5, 200, reward_scale=10.0)]
    data = merge_episodes(episodes)
    results = []
    for chunk in (1, 8):
        model = ActorCriticE1()
        model.load_state_dict(untrained.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        ppo_update(model, optimizer, data, PPOConfig(epochs=1, seqs_per_minibatch=8, seqs_per_chunk=chunk, target_kl=0.0),
                   torch.Generator().manual_seed(0))
        results.append([p.detach().clone() for p in model.parameters()])
    assert all(torch.allclose(a, b, atol=1e-5, rtol=1e-4) for a, b in zip(*results))


def _bandit_probability_of_best(sign: float, iterations: int = 25) -> float:
    obs, masks = recorded_decisions()[0]
    torch.manual_seed(0)
    model = ActorCriticE1()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(0)
    batch = cat([step_batch(obs, masks)] * 64)
    zero = torch.zeros(64, model.config.hidden)
    for it in range(iterations):
        model.eval()
        with torch.no_grad():
            heads, _, _, _ = model.step(batch, zero, zero)
            sample, _ = sample_rows(heads, batch, [new_stream(it, policy_stream_id("learned", i)) for i in range(64)], False)
            old = evaluate(heads, batch, sample).log_prob
        reward = (sample.ints[:, I_EGG] == 2).float() * sign
        data = TrainingSet(batch, sample, old, reward - reward.mean(), reward, [torch.tensor([i]) for i in range(64)], 64)
        ppo_update(model, optimizer, data, PPOConfig(epochs=4, seqs_per_minibatch=64, target_kl=0.0, normalize_advantages=True),
                   generator)
    with torch.no_grad():
        heads, _, _, _ = model.step(cat([step_batch(obs, masks)]), zero[:1], zero[:1])
    return float(torch.softmax(heads.cat["egg"][0], -1)[2])


def test_ppo_finds_the_best_choice_of_a_known_task_and_the_sign_of_the_gradient_is_right():
    towards = _bandit_probability_of_best(+1.0)
    away = _bandit_probability_of_best(-1.0)
    assert towards > 0.6 and away < 0.1, (towards, away)       # uniform start: 0.25


# ------------------------------------------------------------------------------------------------ model files and the learned PolicyE1


def test_a_model_survives_a_save_load_roundtrip_and_a_tampered_one_is_refused(tmp_path):
    model = peaky_model(31, scale=3.0)
    manifest = save_model(tmp_path / "m", model, training={"iterations": 0})
    loaded, again = load_model(tmp_path / "m")
    assert again["weights_sha256"] == manifest["weights_sha256"] and again["observation_version"] == "ObservationE1/v2"
    assert manifest["reward_version"] == again["reward_version"] == REWARD_VERSION == 3        # a new model says which reward it was trained on
    obs, masks = recorded_decisions()[5]
    batch = cat([step_batch(obs, masks)])
    zero = torch.zeros(1, model.config.hidden)
    with torch.no_grad():
        a, va, _, _ = model.step(batch, zero, zero)
        b, vb, _, _ = loaded.step(batch, zero, zero)
    assert torch.equal(a.cat["frac"], b.cat["frac"]) and torch.equal(va, vb)
    shutil.copytree(tmp_path / "m", tmp_path / "bad")
    with open(tmp_path / "bad" / WEIGHTS, "r+b") as handle:
        handle.seek(100)
        handle.write(b"\x00\x01\x02")
    with pytest.raises(ModelError, match="SHA-256"):
        load_model(tmp_path / "bad")
    shutil.copytree(tmp_path / "m", tmp_path / "old")
    data = json.loads((tmp_path / "old" / MANIFEST).read_text(encoding="utf-8"))
    data["observation_version"] = "ObservationE1/v1"
    (tmp_path / "old" / MANIFEST).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ModelError, match="observation_version"):
        load_model(tmp_path / "old")
    with pytest.raises(ModelError):
        load_model(tmp_path / "missing")


def test_a_learned_policy_replay_is_exported_deterministically_and_verified_by_re_execution(tmp_path):
    model = peaky_model(41, scale=4.0)
    manifest = save_model(tmp_path / "m", model)
    register_model(tmp_path / "m")
    spec = PolicyRunSpec(9, tuple(PolicySpec("learned", variant_for(manifest, "sample")) for _ in range(8)))
    replay, decisions = export_policy_replay(CONFIG, 50012, spec, 300, 50)
    again, decisions_again = export_policy_replay(CONFIG, 50012, spec, 300, 50)
    assert replay == again and decisions == decisions_again
    document = verify_replay(replay, decisions)
    assert document.header["policy"]["policies"][0]["kind"] == "learned" and document.result["decision_count"] >= 8 * 3
    greedy = PolicyRunSpec(9, tuple(PolicySpec("learned", variant_for(manifest, "greedy")) for _ in range(8)))
    other, _ = export_policy_replay(CONFIG, 50012, greedy, 300, 50)
    assert other != replay and read_replay(other).policy is not None
    forged = replay.replace(manifest["weights_sha256"][:16].encode(), b"0" * 16)
    with pytest.raises((ReplayFormatError, ValueError)):
        verify_replay(forged)


def test_an_unregistered_model_is_named_in_the_error():
    from formicmind.eco.policy import build_policies

    import formicmind.eco.rl.learned  # noqa: F401  (registers the 'learned' kind)

    spec = PolicyRunSpec(1, tuple(PolicySpec("learned", "sample-ffffffffffffffff") for _ in range(8)))
    with pytest.raises(ValueError, match="ffffffffffffffff"):
        build_policies(spec, CONFIG)


# ------------------------------------------------------------------------------------------------ commands


def test_train_and_eval_commands_run_end_to_end_on_real_worlds(tmp_path, capsys):
    out = tmp_path / "run"
    code = main(["train", "--out", str(out), "--iterations", "2", "--worlds", "2", "--ticks", "120", "--eval-worlds", "1", "--minibatch-seqs", "8",
                 "--seed", "3"])
    assert code == 0
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["iteration"] for r in rows] == [1, 2] and all(r["ppo_identity_ratio_max_dev"] < 1e-4 for r in rows)
    report = json.loads((out / "eval_final.json").read_text(encoding="utf-8"))
    assert {"learned", "learned_greedy", "untrained_init", "random_valid", "heuristic:peaceful", "heuristic:explorer"} <= set(report)
    manifest = json.loads((out / "final" / MANIFEST).read_text(encoding="utf-8"))
    code = main(["eval", "--model", str(out / "final"), "--worlds", "1", "--ticks", "120", "--no-baselines", "--out", str(tmp_path / "e.json"),
                 "--replay-out", str(tmp_path / "r.jsonl"), "--replay-ticks", "120", "--decisions", str(tmp_path / "d.jsonl"), "--verify"])
    assert code == 0 and json.loads((tmp_path / "e.json").read_text(encoding="utf-8"))["model"]["weights_sha256"] == manifest["weights_sha256"]
    capsys.readouterr()
    code = main(["replay-verify", str(tmp_path / "r.jsonl"), "--model", str(out / "final"), "--decisions", str(tmp_path / "d.jsonl")])
    assert code == 0 and "byte-exact re-execution" in capsys.readouterr().out
    bare = subprocess.run([sys.executable, "-m", "formicmind.eco", "replay-verify", str(tmp_path / "r.jsonl")], capture_output=True, text=True,
                          cwd=str(ROOT), env={**os.environ, "PYTHONPATH": "src"})
    assert bare.returncode == 3 and "model" in bare.stderr          # a fresh process without the model cannot re-execute the replay, and says why


# ------------------------------------------------------------------------------------------------ the committed pilot


PILOT = ROOT / "docs" / "ecosystem_examples" / "for49_pilot"
LEARNED_SAMPLE = ROOT / "viewer" / "public" / "samples" / "eco_learned_ppo_seed90000.jsonl"


def test_the_committed_pilot_model_replays_the_committed_viewer_sample_byte_for_byte():
    model, manifest = register_model(PILOT / "model")
    assert manifest["training"]["iterations"] == 120 and manifest["training"]["config"]["worlds_per_iteration"] == 16
    data = LEARNED_SAMPLE.read_bytes()
    document = verify_replay(data)                                       # strict read + full re-execution of the 6000-tick world with the model
    assert document.header["seed"] == "90000" and document.header["ticks_requested"] == 6000
    assert {p.variant for p in document.policy.policies} == {variant_for(manifest, "sample")}
    assert document.result["decision_count"] == 1600 and [o["status"] for o in document.result["colony_outcomes"]] == ["alive"] * 8


def test_the_first_pilot_model_is_marked_as_trained_on_reward_v1():
    """Its manifest predates the field (the model files are kept byte for byte); the loader reads a missing field as the first reward."""
    raw = json.loads((PILOT / "model" / MANIFEST).read_text(encoding="utf-8"))
    assert "reward_version" not in raw
    _, manifest = load_model(PILOT / "model")
    assert manifest["reward_version"] == 1 != REWARD_VERSION


def test_the_committed_pilot_numbers_say_what_the_report_says():
    """The report claims: the network learned (far above untrained and random), but not to play well - it survives with nobody left and the
    scripted peaceful baseline earns more. If these numbers change, docs/ecosystem_v1_training.md must change with them."""
    report = json.loads((PILOT / "eval_report.json").read_text(encoding="utf-8"))
    final = json.loads((PILOT / "eval_final.json").read_text(encoding="utf-8"))
    learned, peaceful = report["learned"], report["heuristic:peaceful"]
    assert learned["reward"] > 0.4 > 0 > report["random_valid"]["reward"] and final["untrained_init"]["reward"] < -0.99
    assert learned["survival"] == 1.0 and learned["hollow"] == 1.0 and learned["workers_end"] == 0.0     # alive on paper: queen, hoard, no adults
    assert peaceful["reward"] > learned["reward"] and peaceful["workers_end"] > 15 and peaceful["hollow"] == 0.0
    assert report["learned"]["reward"] == pytest.approx(final["learned"]["reward"], abs=1e-12)           # the same worlds scored twice agree
    rows = [json.loads(line) for line in (PILOT / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 120 and max(r["ppo_identity_ratio_max_dev"] for r in rows) < 1e-4
    assert rows[0]["reward"] < -0.99 and rows[-1]["reward"] > 0.4


# ------------------------------------------------------------------------------------------------ the committed reward-v2 pilot (stopped at iteration 89)


PILOT2 = ROOT / "docs" / "ecosystem_examples" / "for49_pilot_reward_v2"


def test_the_committed_reward_v2_model_replays_the_committed_replay_byte_for_byte():
    model, manifest = register_model(PILOT2 / "model")
    assert manifest["reward_version"] == 2 and manifest["training"]["iterations"] == 89 and manifest["training"]["stopped_by_owner"] is True
    document = verify_replay((PILOT2 / "replay_seed90000.jsonl").read_bytes())
    assert document.header["seed"] == "90000" and {p.variant for p in document.policy.policies} == {variant_for(manifest, "sample")}
    assert [o["status"] for o in document.result["colony_outcomes"]] == ["alive"] * 8


def test_the_committed_reward_v2_numbers_say_what_the_report_says():
    """The report claims: reward v2 closed the paid-for-nothing path but the network still ends with a queen and no adults (no eggs), far below
    the peaceful baseline. If these numbers change, docs/ecosystem_v1_training.md must change with them."""
    final = json.loads((PILOT2 / "eval_final.json").read_text(encoding="utf-8"))
    learned, peaceful, pilot1 = final["learned"], final["heuristic:peaceful"], final["pilot1_reward_v1_model"]
    assert final["reward_version"] == 2 and final["seeds"] == list(range(90000, 90008)) and learned["colonies"] == 64
    assert learned["hollow"] == 1.0 and learned["workers_end"] == 0.0 and learned["eggs_laid"] == 0.0 and learned["brood_end"] == 0.0
    assert 0.1 < learned["reward"] < 0.2 and learned["reward_v1"] > 0.4 and peaceful["reward"] > 0.5 and peaceful["hollow"] == 0.0
    assert peaceful["eggs_laid"] > 100 and peaceful["workers_mean"] > 15 and final["untrained_init"]["reward"] < -0.99
    assert abs(pilot1["reward"] - learned["reward"]) < 0.02              # no better than the model of pilot 1 re-scored by the new formula
    assert learned["discover"] == learned["contact"] == learned["dispatch_attack"] == learned["combat"] == 0.0
    rows = [json.loads(line) for line in (PILOT2 / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 89 and max(r["ppo_identity_ratio_max_dev"] for r in rows) < 1e-4 and rows[0]["reward"] < -0.99 and rows[-1]["reward"] > 0.1
    assert json.loads((PILOT2 / "config.json").read_text(encoding="utf-8"))["reward_version"] == 2
    analysis = json.loads((PILOT2 / "analysis.json").read_text(encoding="utf-8"))
    rescored = analysis["functional_extinction_terminal_rescoring"]
    assert rescored["learned_v2"]["fired"] == 64 and rescored["heuristic:peaceful"]["fired"] == 0
