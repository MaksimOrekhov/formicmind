"""FOR-42: snapshot pool, planner, checkpoint/resume, Windows-spawn parity, evaluation and a real (tiny) training run."""

from __future__ import annotations

import json
import shutil

import pytest

torch = pytest.importorskip("torch")

from formicmind.rl.checkpoint import RLCheckpointError, latest_checkpoint, load_checkpoint  # noqa: E402
from formicmind.rl.config import RunConfig  # noqa: E402
from formicmind.rl.evaluate import build_suite, run_evaluation  # noqa: E402
from formicmind.rl.maps import map_split  # noqa: E402
from formicmind.rl.network import ActorCritic, actor_state_to_bytes  # noqa: E402
from formicmind.rl.parallel import MatchExecutor  # noqa: E402
from formicmind.rl.pool import (  # noqa: E402
    EVAL_SEED_BASE, TRAIN_SEED_BASE, PoolError, SnapshotPool, plan_matches, thin_active,
)
from formicmind.rl.train import train  # noqa: E402

TINY = dict(hidden_sizes=(16, 16), matches_per_iteration=8, iterations=2, minibatch=128, epochs=2, workers=0,
            device="cpu", eval_every=0, checkpoint_every=1, snapshot_every=1, chunk_size=4)


def tiny_cfg(**overrides) -> RunConfig:
    return RunConfig(**{**TINY, **overrides})


def test_snapshots_are_immutable_and_verified(tmp_path):
    pool = SnapshotPool(tmp_path / "snaps")
    torch.manual_seed(0)
    net = ActorCritic((8, 8))
    info = pool.add(0, 0, net.actor_state_dict())
    with pytest.raises(PoolError):
        pool.add(0, 5, net.actor_state_dict())  # never overwritten
    assert pool.load_info(0) == info
    with open(info.path, "ab") as handle:
        handle.write(b"x")
    with pytest.raises(PoolError):
        pool.load_info(0)
    with pytest.raises(PoolError):
        pool.load_state(info)


def test_planner_is_deterministic_unique_and_uses_frozen_opponents_only(tmp_path):
    pool = SnapshotPool(tmp_path / "snaps")
    net = ActorCritic((8, 8))
    snaps = [pool.add(i, i, net.actor_state_dict()) for i in range(3)]
    kwargs = dict(p_current=0.5, baseline_sparring=0.0, gen_map_fraction=0.5, shaping_coef=0.0, gamma=0.99, max_ticks=None)
    a = plan_matches(42, 3, 200, snaps, **kwargs)
    assert a == plan_matches(42, 3, 200, snaps, **kwargs)
    assert a != plan_matches(42, 4, 200, snaps, **kwargs)
    seeds = [s["seed"] for s in a] + [s["seed"] for s in plan_matches(42, 4, 200, snaps, **kwargs)]
    assert len(set(seeds)) == len(seeds) and min(seeds) >= TRAIN_SEED_BASE and max(seeds) < EVAL_SEED_BASE
    kinds = {s["tag"]["opponent_kind"] for s in a}
    assert kinds == {"current", "snapshot"}  # no scripted baseline unless sparring is switched on explicitly
    assert all(map_split(s["map_id"]) == "train" for s in a)
    only_current = plan_matches(42, 0, 20, [], **kwargs)
    assert {s["tag"]["opponent_kind"] for s in only_current} == {"current"}
    for spec in a:
        streams = [tuple(seat["stream"]) for seat in spec["seats"]]
        assert len(set(streams)) == 2  # never one shared RNG stream
        recorded = [seat["record"] for seat in spec["seats"]]
        if spec["tag"]["opponent_kind"] == "snapshot":
            assert sum(recorded) == 1  # a frozen opponent never produces training data
    sparring = plan_matches(1, 0, 100, snaps, **{**kwargs, "baseline_sparring": 0.3})
    assert "baseline" in {s["tag"]["opponent_kind"] for s in sparring}


def test_thin_active_keeps_first_and_newest():
    class S:
        def __init__(self, i):
            self.snapshot_id = i

    infos = [S(i) for i in range(20)]
    thinned = thin_active(infos, 5)
    assert len(thinned) == 5 and thinned[0].snapshot_id == 0 and thinned[-1].snapshot_id == 19


def test_evaluation_suite_is_fixed_disjoint_from_training_and_uses_heldout_maps():
    quick = build_suite("quick", {"init": {}})
    assert quick == build_suite("quick", {"init": {}})
    assert len({s["seed"] for s in quick}) == len(quick)
    assert all(s["seed"] >= EVAL_SEED_BASE for s in quick)
    assert {s["tag"]["map_split"] for s in quick} <= {"indist", "heldout"}
    assert {s["tag"]["learner_seat"] for s in quick} == {0, 1}
    with pytest.raises(ValueError):
        build_suite("nope")


def test_spawn_workers_reproduce_the_inline_results_exactly(tmp_path):
    torch.manual_seed(3)
    net = ActorCritic((16, 16))
    refs = {"current": {"bytes": actor_state_to_bytes(net.actor_state_dict())}}
    specs = plan_matches(7, 0, 6, [], p_current=1.0, baseline_sparring=0.0, gen_map_fraction=0.5,
                         shaping_coef=0.05, gamma=0.99, max_ticks=None)

    def strip(results):
        return [{k: v for k, v in r.items() if k not in ("wall_seconds", "sim_seconds")}
                | {"seats": [{k: v for k, v in seat.items() if k != "stats"} for seat in r["seats"]]} for r in results]

    inline = MatchExecutor(0, 3).run(specs, refs)
    with MatchExecutor(2, 3) as executor:
        spawned = executor.run(specs, refs)
    assert strip(inline) == strip(spawned)  # bytes of every trajectory, hashes of outcomes, order
    assert len({r["seed"] for r in spawned}) == len(spawned)


def test_a_real_tiny_run_changes_weights_from_rewards_and_resume_is_exact(tmp_path):
    straight = tmp_path / "straight"
    summary = train(tiny_cfg(iterations=3), straight, log=lambda _m: None)
    assert summary["actor_changed"] is True
    lines = [json.loads(x) for x in (straight / "metrics.jsonl").read_text().splitlines()]
    train_lines = [x for x in lines if x["kind"] == "train"]
    assert [x["iteration"] for x in train_lines] == [1, 2, 3]
    for x in train_lines:
        ppo = x["ppo"]
        assert all(map(lambda v: v == v and abs(v) != float("inf"), ppo.values()))  # finite
        assert x["param_delta_l2"] > 0
        assert ppo["identity_ratio_max_dev"] < 1e-3
        assert x["batch"]["samples"] > 0

    split = tmp_path / "split"
    train(tiny_cfg(iterations=2), split, log=lambda _m: None)
    summary_b = train(tiny_cfg(iterations=3), split, resume=True, log=lambda _m: None)
    final_a, _, _ = load_checkpoint(latest_checkpoint(straight))
    final_b, manifest_b, _ = load_checkpoint(latest_checkpoint(split))
    assert manifest_b["iteration"] == 3
    for (name, a), (_, b) in zip(final_a.state_dict().items(), final_b.state_dict().items()):
        assert torch.equal(a, b), name  # resumed training is bit-identical to uninterrupted training
    assert summary_b["counters"]["matches"] == summary["counters"]["matches"] == 24
    # metrics were not duplicated by the resume
    resumed_lines = [json.loads(x) for x in (split / "metrics.jsonl").read_text().splitlines()]
    assert [x["iteration"] for x in resumed_lines if x["kind"] == "train"] == [1, 2, 3]

    with pytest.raises(RuntimeError):
        train(tiny_cfg(iterations=3), straight, log=lambda _m: None)  # refuses to overwrite a finished run
    with pytest.raises(RuntimeError):
        train(tiny_cfg(iterations=4, lr=1e-3), split, resume=True, log=lambda _m: None)  # learning-relevant change


def test_corrupted_checkpoint_is_rejected(tmp_path):
    run = tmp_path / "run"
    train(tiny_cfg(iterations=1), run, log=lambda _m: None)
    directory = latest_checkpoint(run)
    load_checkpoint(directory)
    broken = tmp_path / "broken"
    shutil.copytree(directory, broken)
    with open(broken / "weights.pt", "ab") as handle:
        handle.write(b"\0")
    with pytest.raises(RLCheckpointError):
        load_checkpoint(broken)
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["contracts"]["decoder"] = "other"
    other = tmp_path / "other"
    shutil.copytree(directory, other)
    (other / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RLCheckpointError):
        load_checkpoint(other)


def test_tiny_evaluation_runs_and_aggregates(tmp_path):
    torch.manual_seed(0)
    net = ActorCritic((16, 16))
    with MatchExecutor(0, 4) as executor:
        report = run_evaluation(executor, {"bytes": actor_state_to_bytes(net.actor_state_dict())}, None, "tiny")
    assert report["matches"] == len(build_suite("tiny")) == 4  # 2 baselines x 1 map x 2 seats
    assert report["truncated_matches"] == 0
    for opponent, modes in report["results"].items():
        entry = modes["sample"]["all"]
        assert entry["n"] == entry["wins"] + entry["draws"] + entry["losses"] == 2
        assert abs(sum(entry["lay_fraction"]) - 1.0) < 1e-9
