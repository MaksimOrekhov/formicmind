"""On-policy rollout in the real E1 engine (FOR-49).

``MirrorRuntime`` is one network driving every colony of a world ("mirror" mode): the colonies that decide at the same tick are answered by one
forward pass, but each colony has its **own** GRU memories (actor and critic) and its **own** PCG32 stream, and each only ever sees its own
observation. While recording it closes every decision interval ``[t_i, t_{i+1})`` with the per-tick viability trace of ``ViabilityTracker`` and
computes the time-aware GAE of contracts 9.2 / 9.2a per colony: terminal intervals (queen death, collapse, functional extinction) pay the penalty once and do not
bootstrap; the interval cut by the end of the run is truncated and bootstraps with the **learned critic** on the final observation, never with
viability.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import Tensor

from ...rng import new_stream, policy_stream_id
from ..action import action_masks
from ..config import ConfigE1
from ..engine import RouteTable, step_world_tick
from ..events import EventLog
from ..initial import build_world
from ..observation import build_observation
from ..policy_run import PolicyController
from ..viability import GAMMA_30, LAMBDA_30, IntervalStep, ViabilityTracker, gae_intervals, interval_rewards
from .distribution import Sample, cat_samples, evaluate, sample_rows
from .features import Batch, cat, step_batch
from .network import ActorCriticE1
from .play import PlayResult, PlayStats, colony_metrics

POLICY_ID = "learned"


@dataclass(slots=True)
class Episode:
    """One finished rollout of a world: every decision of every colony, with its old log-prob, critic value, advantage and return."""

    batch: Batch
    sample: Sample
    old_logp: Tensor
    value: Tensor
    advantage: Tensor
    ret: Tensor
    slot: Tensor            # colony index of every step (the sequence id inside this episode)
    delta: Tensor           # elapsed world ticks of every interval
    reward: Tensor          # undiscounted interval reward (reports)
    terminated: Tensor      # bool
    bootstrap: dict         # colony -> Value_critic(final observation) of the trajectories cut by the end of the run (truncated, not terminal)
    metrics: list[dict]     # per-colony true metrics of the world (reports only)
    ticks: int
    seed: int
    wall: float


class MirrorRuntime:
    """One network, ``n_slots`` private memories and streams. Implements ``act_batch`` for ``PolicyController``."""

    def __init__(self, model: ActorCriticE1, n_slots: int, policy_seed: int, *, greedy: bool = False, record: bool = False,
                 tracker: ViabilityTracker | None = None, horizon: int = 6000, reward_scale: float = 1.0) -> None:
        self.model, self.greedy, self.record = model, greedy, record
        self.tracker, self.horizon, self.reward_scale = tracker, horizon, reward_scale
        hidden = model.config.hidden
        self.h_actor = torch.zeros(n_slots, hidden)
        self.h_critic = torch.zeros(n_slots, hidden)
        self.rngs = [new_stream(policy_seed, policy_stream_id(POLICY_ID, c)) for c in range(n_slots)]
        self.n_slots = n_slots
        # recording
        self.batches: list[Batch] = []
        self.samples: list[Sample] = []
        self.logps: list[Tensor] = []
        self.values: list[Tensor] = []
        self.meta: list[dict] = []          # per step: slot, tick, delta, reward, reward_disc, terminated
        self.open: dict[int, int] = {}      # slot -> index of its step whose interval is still open
        self.count = 0

    # ---------------------------------------------------------------------------------------------- the PolicyController hook
    def act_batch(self, items: list) -> list[tuple[float, ...]]:
        if not items:
            return []
        slots = [cid for cid, _, _ in items]
        if self.record:
            for cid, obs, _ in items:
                self._close(cid, obs.facts.tick, terminated=False)
        batch = cat([step_batch(obs, masks) for _, obs, masks in items])
        ids = torch.tensor(slots)
        with torch.no_grad():
            heads, value, actor_h, critic_h = self.model.step(batch, self.h_actor[ids], self.h_critic[ids])
            sample, latents = sample_rows(heads, batch, [self.rngs[c] for c in slots], self.greedy)
            logp = evaluate(heads, batch, sample).log_prob if self.record else None
        self.h_actor[ids], self.h_critic[ids] = actor_h, critic_h
        if self.record:
            self.batches.append(batch)
            self.samples.append(sample)
            self.logps.append(logp)
            self.values.append(value)
            for cid, obs, _ in items:
                self.meta.append({"slot": cid, "tick": obs.facts.tick, "delta": 0, "reward": 0.0, "reward_disc": 0.0, "terminated": False})
                self.open[cid] = self.count
                self.count += 1
        return [tuple(latent) for latent in latents]

    def _close(self, slot: int, tick: int, *, terminated: bool, stop: int | None = None) -> None:
        index = self.open.pop(slot, None)
        if index is None:
            return
        meta = self.meta[index]
        end = tick if stop is None else stop
        trace = self.tracker.window(slot, meta["tick"], end)
        reward, reward_disc = interval_rewards(trace, self.horizon, terminated=terminated)
        meta.update(delta=end - meta["tick"], reward=reward, reward_disc=reward_disc, terminated=terminated)

    # ---------------------------------------------------------------------------------------------- end of the run
    def finish(self, state, config: ConfigE1, ticks: int, *, gamma30: float = GAMMA_30, lambda30: float = LAMBDA_30) -> dict:
        """Close the open intervals at the end of the run and compute advantages and returns. Returns tensors for ``Episode``."""
        own: dict[int, list] = {}
        for cohort in state.cohorts:
            own.setdefault(cohort.colony_id, []).append(cohort)
        pending: list[tuple[int, object]] = []
        for slot in sorted(self.open):
            death = self.tracker.death_tick(slot)
            if death is not None:
                self._close(slot, ticks, terminated=True, stop=death)
                continue
            colony = state.colonies[slot]
            last = self.meta[self.open[slot]]
            decisions = sum(1 for m in self.meta if m["slot"] == slot)
            obs = build_observation(colony, own.get(slot, []), state.tick, config, decision_index=decisions,
                                    ticks_since_last_decision=state.tick - last["tick"])
            pending.append((slot, step_batch(obs, action_masks(obs))))
            self._close(slot, ticks, terminated=False, stop=state.tick)
        bootstrap: dict[int, float] = {}
        if pending:
            with torch.no_grad():
                ids = torch.tensor([slot for slot, _ in pending])
                values = self.model.bootstrap_value(cat([b for _, b in pending]), self.h_critic[ids])
            bootstrap = {slot: float(v) for (slot, _), v in zip(pending, values)}
        value = torch.cat(self.values) if self.values else torch.zeros(0)
        advantage = torch.zeros(self.count)
        ret = torch.zeros(self.count)
        by_slot: dict[int, list[int]] = {}
        for i, meta in enumerate(self.meta):
            by_slot.setdefault(meta["slot"], []).append(i)
        for slot, indices in by_slot.items():
            steps = [IntervalStep(self.meta[i]["delta"], self.meta[i]["reward_disc"] * self.reward_scale, self.meta[i]["terminated"],
                                  float(value[i])) for i in indices]
            adv, targets = gae_intervals(steps, bootstrap.get(slot, 0.0), gamma30=gamma30, lambda30=lambda30)
            for i, a, r in zip(indices, adv, targets):
                advantage[i], ret[i] = a, r
        return {"advantage": advantage, "ret": ret, "bootstrap": bootstrap}


class SlotPolicy:
    """The ``PolicyE1`` face of one slot of a ``MirrorRuntime`` (the controller still needs one policy object per colony)."""

    def __init__(self, runtime: MirrorRuntime, slot: int) -> None:
        self.runtime, self.slot, self.name = runtime, slot, "learned"

    def act(self, obs, masks):
        return self.runtime.act_batch([(self.slot, obs, masks)])[0]

    def snapshot(self) -> dict:
        return {}

    def restore(self, data: dict) -> None:
        return None


def collect_episode(config: ConfigE1, seed: int, model: ActorCriticE1, policy_seed: int, ticks: int, *, reward_scale: float = 1.0,
                    gamma30: float = GAMMA_30, lambda30: float = LAMBDA_30) -> Episode:
    """One world of ``config.n_colonies`` mirrored colonies, played for ``ticks`` ticks with the stochastic policy and recorded for PPO."""
    started = time.perf_counter()
    world = build_world(config, seed)
    routes = RouteTable(world.world_map)
    tracker = ViabilityTracker(world.state)
    runtime = MirrorRuntime(model, config.n_colonies, policy_seed, record=True, tracker=tracker, horizon=config.horizon,
                            reward_scale=reward_scale)
    controller = PolicyController([SlotPolicy(runtime, c) for c in range(config.n_colonies)], batcher=runtime)
    events, stats = EventLog(), PlayStats(config.n_colonies)
    model.eval()
    for _ in range(ticks):
        record = step_world_tick(world.state, world.run, config, world.world_map, controller, routes=routes,
                                 validate_input=world.state.tick == 0)
        events.extend(record.events)
        tracker.update(world.state, record)
        stats.tick(world.state, record)
        # A colony whose agent episode has ended (queen death, collapse, or - reward v3 - functional extinction) is retired: from the next decision
        # on the network is not run for it (no memory update, no sample, no log-prob, no reward, no new episode). The world rules do not know:
        # the colony keeps its queen, its store, its status and its standing orders, and the other colonies are stepped as before.
        controller.retired |= tracker.terminated()
        if world.state.world_outcome.finished:
            break
    done = runtime.finish(world.state, config, world.state.tick, gamma30=gamma30, lambda30=lambda30)
    play = PlayResult(config, seed, world.state, tracker, events, stats, world.state.tick, time.perf_counter() - started)
    meta = runtime.meta
    return Episode(
        batch=cat(runtime.batches), sample=cat_samples(runtime.samples), old_logp=torch.cat(runtime.logps), value=torch.cat(runtime.values),
        advantage=done["advantage"], ret=done["ret"], slot=torch.tensor([m["slot"] for m in meta]),
        delta=torch.tensor([m["delta"] for m in meta]), reward=torch.tensor([m["reward"] for m in meta], dtype=torch.float64),
        terminated=torch.tensor([m["terminated"] for m in meta]), bootstrap=done["bootstrap"], metrics=colony_metrics(play), ticks=world.state.tick, seed=seed,
        wall=time.perf_counter() - started)
