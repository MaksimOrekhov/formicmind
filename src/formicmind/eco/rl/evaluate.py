"""Honest evaluation of a trained policy next to scripted baselines on held-out maps (FOR-49).

Every contestant plays the same worlds (same map seeds, 8 colonies each, all colonies the same policy) and is scored by the same code
(``play.colony_metrics``: the true viability reward, survival, size, delivered food, exploration / contact / attack events). The learned policy
runs through exactly the per-colony ``PolicyE1`` objects that the replay uses. Baselines are comparison points, never teachers.
"""

from __future__ import annotations

import io

import torch

from ...rng import new_stream
from ..config import default_config
from ..policy import PolicyRunSpec, PolicySpec, build_policies, stream_id_of
from ..policy_run import PolicyController
from ..viability import REWARD_VERSION
from .learned import LearnedColonyPolicy
from .network import ActorCriticE1, NetworkConfig
from .play import colony_metrics, play_world, summarize

BASELINES = (("random_valid", ""), ("heuristic", "peaceful"), ("heuristic", "explorer"))
EVAL_POLICY_SEED = 1


def _controller(job: dict, config):
    kind = job["kind"]
    if kind.startswith("learned"):
        model = ActorCriticE1(NetworkConfig(**job["network"]))
        model.load_state_dict(torch.load(io.BytesIO(job["weights"]), weights_only=True))
        model.eval()
        greedy = kind == "learned-greedy"
        spec_item = PolicySpec("learned", "greedy" if greedy else "sample")
        policies = [LearnedColonyPolicy(model, new_stream(EVAL_POLICY_SEED, stream_id_of(spec_item, c)), greedy=greedy, name=kind)
                    for c in range(config.n_colonies)]
    else:
        base, _, variant = kind.partition(":")
        spec = PolicyRunSpec(EVAL_POLICY_SEED, tuple(PolicySpec(base, variant) for _ in range(config.n_colonies)))
        policies = build_policies(spec, config)
    return PolicyController(policies)


def _eval_job(job: dict) -> list[dict]:
    torch.set_num_threads(1)
    config = default_config(job["colonies"], job["scenario"])
    result = play_world(config, job["seed"], _controller(job, config), job["ticks"])
    return colony_metrics(result)


def _weights(model: ActorCriticE1) -> bytes:
    buffer = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, buffer)
    return buffer.getvalue()


def evaluation_suite(model: ActorCriticE1, config, pool, *, label: str, baselines: bool, extra: dict | None = None,
                     greedy: bool = False) -> dict:
    """Play the held-out worlds with the learned policy (sampled; also greedy if asked), optionally the scripted baselines and extra models."""
    seeds = [config.eval_seed0 + i for i in range(config.eval_worlds)]
    contestants: list[tuple[str, str, ActorCriticE1 | None]] = [("learned", "learned-sample", model)]
    if greedy:
        contestants.append(("learned_greedy", "learned-greedy", model))
    for name, other in (extra or {}).items():
        contestants.append((name, "learned-sample", other))
    if baselines:
        contestants += [(f"{kind}{':' + variant if variant else ''}", f"{kind}{':' + variant if variant else ''}", None) for kind, variant in BASELINES]
    jobs, owners = [], []
    for name, kind, net in contestants:
        weights = _weights(net) if net is not None else b""
        for seed in seeds:
            jobs.append({"kind": kind, "weights": weights, "network": net.config.to_dict() if net is not None else {}, "colonies": config.colonies,
                         "scenario": config.scenario, "seed": seed, "ticks": config.ticks})
            owners.append(name)
    results = pool.map(_eval_job, jobs)
    report: dict = {"label": label, "reward_version": REWARD_VERSION, "seeds": seeds, "ticks": config.ticks, "colonies": config.colonies, "scenario": config.scenario}
    for name, _, _ in contestants:
        runs = [r for owner, r in zip(owners, results) if owner == name]
        report[name] = summarize(runs)
        report[name]["per_world_reward"] = [sum(row["reward"] for row in run) / len(run) for run in runs]
    return report
