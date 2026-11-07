"""Shared builders for the FOR-48 policy tests (observation, masks, decoder, controller, leak audits)."""

from __future__ import annotations

import copy
from dataclasses import replace

from formicmind.eco import default_config
from formicmind.eco.action import SquadActionE1, action_masks
from formicmind.eco.engine import RouteTable, step_world_tick
from formicmind.eco.events import EventLog
from formicmind.eco.initial import build_world
from formicmind.eco.observation import build_observation, observation_bytes
from formicmind.eco.policy import ObservationHeuristicPolicy
from formicmind.eco.policy_run import PolicyController


class AttackerPolicy(ObservationHeuristicPolicy):
    """Scripted test policy: explorer economy, buys soldiers, attacks the first legal target as soon as it has two idle soldiers."""

    def act(self, obs, masks):
        action = super().act(obs, masks)
        squads = list(action.squads)
        if masks.attack_pairs and obs.facts.idle_soldiers_home >= 2 and obs.facts.food >= 20:
            node, colony = masks.attack_pairs[0]
            squads[0] = SquadActionE1("attack", 4, node, colony)
        share = 2 if obs.facts.food >= 60 and obs.facts.workers >= 10 else 0
        return replace(action, soldier_share_q=share, squads=tuple(squads))


class Recorder:
    """Wraps a policy and keeps what it was shown: the observation bytes and the masks of every decision."""

    def __init__(self, inner) -> None:
        self.inner, self.name, self.seen = inner, getattr(inner, "name", "recorder"), []

    def act(self, obs, masks):
        self.seen.append((obs, observation_bytes(obs), masks))
        return self.inner.act(obs, masks)

    def snapshot(self) -> dict:
        return self.inner.snapshot()

    def restore(self, data: dict) -> None:
        self.inner.restore(data)


def explorer_policies(config, attackers=()):
    out = [ObservationHeuristicPolicy(config, "explorer") for _ in range(config.n_colonies)]
    for colony in attackers:
        out[colony] = AttackerPolicy(config, "explorer")
    return out


class Run:
    """A stepwise run (world + controller) so a test can stop at any tick and look at the state."""

    def __init__(self, colonies=8, scenario="moderate", seed=50012, policies=None, attackers=()):
        self.config = default_config(colonies, scenario)
        self.world = build_world(self.config, seed)
        self.routes = RouteTable(self.world.world_map)
        self.controller = PolicyController(policies if policies is not None else explorer_policies(self.config, attackers))
        self.events = EventLog()

    @property
    def state(self):
        return self.world.state

    def advance(self, ticks):
        for _ in range(ticks):
            record = step_world_tick(self.world.state, self.world.run, self.config, self.world.world_map, self.controller,
                                     routes=self.routes, validate_input=self.world.state.tick == 0)
            self.events.extend(record.events)
            if self.world.state.world_outcome.finished:
                break
        return self


def first_observation(colony=0, colonies=8, seed=50012):
    run = Run(colonies=colonies, seed=seed)
    own = [c for c in run.state.cohorts if c.colony_id == colony]
    obs = build_observation(run.state.colonies[colony], own, 0, run.config, decision_index=0, ticks_since_last_decision=0)
    return obs, action_masks(obs)


def decide_once(state, config, world_map, routes, policies):
    """Run one planned decision round of fresh recorders on (a copy of) ``state``; returns the recorders."""
    recorders = [Recorder(p) for p in policies]
    PolicyController(recorders)(state, config, world_map, routes)
    return recorders


def copy_world(run):
    return copy.deepcopy(run.world.state), run.world.world_map
