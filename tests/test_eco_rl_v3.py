"""FOR-49 reward v3 in the real rollout and the evaluation (needs torch): the agent episode of a functionally extinct colony ends once, the network is not
run for it afterwards, the world is not touched, and the evaluation reports physical and functional survival apart.

The committed reward-v2 model is the fixture: it never lays an egg and its colonies run out of adults when the starting workers die (tick 2001).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from formicmind.eco import default_config
from formicmind.eco.rl.checkpoint import MANIFEST, load_model
from formicmind.eco.rl.evaluate import _controller, _weights
from formicmind.eco.rl.play import colony_metrics, play_world, summarize
from formicmind.eco.rl.rollout import MirrorRuntime, collect_episode
from formicmind.eco.viability import DEATH_PENALTY, Q, REWARD_VERSION

ROOT = Path(__file__).resolve().parents[1]
PILOT2 = ROOT / "docs" / "ecosystem_examples" / "for49_pilot_reward_v2" / "model"
H = 6000
EXTINCT_AT = 2001          # the starting workers reach their lifespan; the death is recorded in the tick, the episode ends at tick + 1


@pytest.fixture(scope="module")
def model():
    loaded, manifest = load_model(PILOT2)
    assert manifest["reward_version"] == 2 != REWARD_VERSION        # an older model, versioned as such
    return loaded


@pytest.fixture(scope="module")
def recorded(model):
    """collect_episode on the held-out map 90000 with every call of the network spied on."""
    calls = []
    real = MirrorRuntime.act_batch

    def spy(self, items):
        out = real(self, items)
        if items:
            calls.append({"slots": [cid for cid, _, _ in items], "ticks": {obs.facts.tick for _, obs, _ in items}, "h_actor": self.h_actor.clone(),
                          "h_critic": self.h_critic.clone(), "runtime": self})
        return out

    MirrorRuntime.act_batch = spy
    try:
        episode = collect_episode(default_config(8, "moderate"), 90000, model, 1, 2300, reward_scale=1.0)
    finally:
        MirrorRuntime.act_batch = real
    return episode, calls


def test_every_colony_ends_its_episode_once_with_the_penalty_and_the_reason(recorded):
    episode, _ = recorded
    assert episode.ticks == 2300                                                     # the world kept running after the agents' episodes were over
    assert all(row["terminal_reason"] == "functional_extinction" and row["death_tick"] == EXTINCT_AT - 1 for row in episode.metrics)
    assert all(row["survived"] and row["hollow"] and not row["functional_survived"] for row in episode.metrics)        # physically alive, functionally gone
    for slot in range(8):
        mask = episode.slot == slot
        assert bool(episode.terminated[mask][-1]) and int(episode.terminated[mask].sum()) == 1            # one terminal step per trajectory, the last one
        assert int(episode.delta[mask].sum()) == EXTINCT_AT                                                # intervals tile [0, death_tick)
        row = episode.metrics[slot]
        # undiscounted rewards of the trajectory = viability sum / (Q H) - one penalty
        assert float(episode.reward[mask].sum()) == pytest.approx(row["reward"], abs=1e-9)
        assert row["reward"] < -DEATH_PENALTY + 0.2 and row["reward"] > -DEATH_PENALTY


def test_the_network_is_not_run_for_a_colony_after_its_episode_and_its_memory_stays_put(recorded):
    _, calls = recorded
    assert calls and max(max(c["ticks"]) for c in calls) < EXTINCT_AT            # no decision at or after the terminal tick
    last = calls[-1]
    runtime = last["runtime"]
    assert torch.equal(runtime.h_actor, last["h_actor"]) and torch.equal(runtime.h_critic, last["h_critic"])      # no recurrent update since
    assert runtime.count == sum(len(c["slots"]) for c in calls)                                                    # no step recorded beyond the calls
    assert runtime.open == {} or all(runtime.meta[i]["terminated"] for i in runtime.open.values())               # nothing left open and running


def test_the_world_of_a_terminated_colony_is_physically_untouched(recorded, model):
    config = default_config(8, "moderate")
    episode, _ = recorded
    # a natural evaluation world of the same model: the policy keeps playing; nobody is retired, only the score shows the terminal
    job = {"kind": "learned-sample", "weights": _weights(model), "network": model.config.to_dict()}
    result = play_world(config, 90000, _controller(job, config), 2300)
    rows = colony_metrics(result)
    assert all(r["terminal_reason"] == "functional_extinction" and r["death_tick"] == EXTINCT_AT - 1 and r["survived"] and not r["functional_survived"]
               for r in rows)
    assert all(s.status == "alive" and s.queen_hp == config.economy.queen_hp for s in result.state.colonies)       # no queen loss, no status change
    assert all(s.food > 0 for s in result.state.colonies)                                                           # the store is still there
    summary = summarize([rows])
    assert summary["physical_survival"] == 1.0 and summary["functional_survival"] == 0.0 and summary["functional_extinction"] == 1.0
    assert summary["terminal_reasons"] == {"functional_extinction": 8} and summary["survival"] == summary["physical_survival"]
    # the three rewards of the same run are reported apart: v1 pays the hollow colony, v2 does not, v3 ends its episode and pays -1 once
    mean = {k: sum(r[k] for r in rows) / len(rows) for k in ("reward", "reward_v1", "reward_v2")}
    assert mean["reward_v1"] > mean["reward_v2"] > mean["reward"] and mean["reward"] < -0.5
    assert [r["terminal_reason"] for r in episode.metrics] == [r["terminal_reason"] for r in rows]


def test_the_manifest_and_the_report_say_which_reward_they_belong_to(tmp_path):
    from formicmind.eco.rl.checkpoint import save_model
    from formicmind.eco.rl.network import ActorCriticE1

    manifest = save_model(tmp_path / "m", ActorCriticE1(), training={})
    assert manifest["reward_version"] == REWARD_VERSION == 3
    assert json.loads((tmp_path / "m" / MANIFEST).read_text(encoding="utf-8"))["reward_version"] == 3
    assert Q == 65536
