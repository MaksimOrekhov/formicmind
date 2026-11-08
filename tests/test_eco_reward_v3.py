"""FOR-49 reward v3 (contracts 9.1 / 9.5a): the agent-episode terminal ``functional_extinction``. Torch-free.

A physically alive colony with no adult worker, no adult soldier and no worker brood ends its agent episode (penalty once), whatever its store holds.
This is a definition of the reward and the episode: the tracker never writes to the world, and the physical rules of contracts 9.5 are untouched.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from formicmind.eco import default_config
from formicmind.eco.initial import build_world
from formicmind.eco.model import BroodRecord, CohortE1
from formicmind.eco.policy import PolicyRunSpec, PolicySpec, build_policies
from formicmind.eco.policy_run import PolicyController, run_policies
from formicmind.eco.rl.play import play_world
from formicmind.eco.serialization import state_hash
from formicmind.eco.viability import (
    DEATH_PENALTY, Q, REASON_COLLAPSE, REASON_FUNCTIONAL, REASON_QUEEN_DEATH, ViabilityTracker, functional_extinction, interval_rewards,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ecosystem_reward_ref.py"
H = 6000


@pytest.fixture(scope="module")
def ref():
    spec = importlib.util.spec_from_file_location("ecosystem_reward_ref_v3", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _colony0(workers=0, soldiers=0, worker_brood=0, soldier_brood=0, *, food=600, status="alive", queen_hp=None):
    """A real map whose colony 0 is rewritten to the given adult / brood counts; returns (state, tracker v3 started before the rewrite's tick)."""
    world = build_world(default_config(8, "moderate"), 50012)
    state = world.state
    template = next(c for c in state.cohorts if c.colony_id == 0)
    state.cohorts = [c for c in state.cohorts if c.colony_id != 0]
    for caste, count in (("worker", workers), ("soldier", soldiers)):
        if count:
            state.cohorts.append(CohortE1(10_000 + len(state.cohorts), 0, caste, count, template.hp_total, template.loc, template.mission, 0))
    colony = state.colonies[0]
    colony.brood = ([BroodRecord(20_000 + i, "worker", 0, 99) for i in range(worker_brood)]
                    + [BroodRecord(30_000 + i, "soldier", 0, 99) for i in range(soldier_brood)])
    colony.food, colony.status = food, status
    if queen_hp is not None:
        colony.queen_hp = queen_hp
    return state, ViabilityTracker(state, 3)


def _step(state, tracker, ticks=1):
    for _ in range(ticks):
        record = SimpleNamespace(tick=state.tick, delivered_by_colony={}, upkeep_due_by_colony={})
        state.tick += 1
        tracker.update(state, record)


# ------------------------------------------------------------------------------------------------ the terminal condition


@pytest.mark.parametrize("workers,soldiers,worker_brood,soldier_brood,terminal", [
    (0, 0, 0, 0, True),          # nobody at all, a full store: functional extinction
    (1, 0, 0, 0, False),         # one adult worker
    (0, 1, 0, 0, False),         # one adult soldier
    (0, 0, 1, 0, False),         # worker brood only: a generation is on its way
    (0, 0, 0, 3, True),          # soldier brood alone cannot forage: terminal
    (2, 1, 4, 1, False),
])
def test_the_terminal_fires_exactly_when_there_is_no_adult_and_no_worker_brood(workers, soldiers, worker_brood, soldier_brood, terminal, ref):
    state, tracker = _colony0(workers, soldiers, worker_brood, soldier_brood)
    assert state.colonies[0].food == 600 and state.colonies[0].status == "alive" and state.colonies[0].queen_hp > 0
    _step(state, tracker)
    trace = tracker.traces[0]
    assert functional_extinction(workers, soldiers, worker_brood) is terminal is ref.functional_extinction(workers, soldiers, worker_brood, soldier_brood)
    assert (trace.death_tick is not None) is terminal
    assert trace.reason == (REASON_FUNCTIONAL if terminal else None) and tracker.terminal_reason(0) == trace.reason
    assert (0 in tracker.terminated()) is terminal
    assert (trace.v_q[-1] == 0) is terminal or not terminal          # v_q of the terminal tick is zero; a live colony's need not be
    if terminal:
        assert trace.death_tick == 1 and trace.v_q == [0]
    assert all(tracker.traces[i].death_tick is None for i in range(1, 8))       # the other colonies are untouched


def test_the_terminal_ignores_the_store_and_the_old_rewards_do_not_know_it():
    for food in (0, 80, 600):
        state, tracker = _colony0(food=food)
        _step(state, tracker)
        assert tracker.traces[0].reason == REASON_FUNCTIONAL
    for version in (1, 2):
        state, _ = _colony0()
        old = ViabilityTracker(state, version)
        _step(state, old, 5)
        assert old.traces[0].death_tick is None and len(old.traces[0].v_q) == 5 and old.terminal_reason(0) is None


def test_physical_terminals_keep_their_meaning_and_win_over_the_functional_one():
    state, tracker = _colony0(queen_hp=0, status="orphan")             # queen dead and, besides, nobody left
    _step(state, tracker)
    assert tracker.terminal_reason(0) == REASON_QUEEN_DEATH
    state, tracker = _colony0(status="extinct")                         # collapse (queen alive), nobody left
    _step(state, tracker)
    assert tracker.terminal_reason(0) == REASON_COLLAPSE
    state, tracker = _colony0(workers=3, queen_hp=0, status="orphan")   # queen dies with adults around: not a functional case at all
    _step(state, tracker)
    assert tracker.terminal_reason(0) == REASON_QUEEN_DEATH


def test_one_terminal_one_penalty_and_nothing_is_recorded_afterwards():
    state, tracker = _colony0(workers=1)
    _step(state, tracker, 3)
    assert tracker.traces[0].death_tick is None
    state.cohorts = [c for c in state.cohorts if c.colony_id != 0]       # the last worker dies in the next tick
    _step(state, tracker)
    trace = tracker.traces[0]
    assert trace.death_tick == 4 and trace.reason == REASON_FUNCTIONAL and len(trace.v_q) == 4 and trace.v_q[-1] == 0
    snapshot = (trace.death_tick, list(trace.v_q), trace.reason)
    state.colonies[0].brood = [BroodRecord(77, "worker", 3, 30)]         # the nest recovers physically: the agent episode stays over
    _step(state, tracker, 10)
    assert (trace.death_tick, trace.v_q, trace.reason) == snapshot
    r, r_disc = interval_rewards(trace.v_q[-4:], H, terminated=True)
    assert r == pytest.approx(sum(trace.v_q[-4:]) / (Q * H) - DEATH_PENALTY, abs=1e-12)
    assert r_disc == pytest.approx(interval_rewards(trace.v_q[-4:], H, terminated=False)[1] - DEATH_PENALTY * 0.997 ** (4 / 30), abs=1e-12)
    # the penalty is paid in the terminal interval only: the intervals before it carry none
    assert interval_rewards(trace.v_q[:3], H, terminated=False)[0] == pytest.approx(sum(trace.v_q[:3]) / (Q * H), abs=1e-12)


def test_the_reference_episode_end_agrees_with_the_tracker_on_every_combination(ref):
    for status, queen_hp in (("alive", 240), ("orphan", 0), ("extinct", 240)):
        for workers in (0, 2):
            for soldiers in (0, 1):
                for worker_brood in (0, 1):
                    for soldier_brood in (0, 2):
                        state, tracker = _colony0(workers, soldiers, worker_brood, soldier_brood, status=status, queen_hp=queen_hp)
                        _step(state, tracker)
                        assert tracker.terminal_reason(0) == ref.episode_end(status, queen_hp, workers, soldiers, worker_brood), (
                            status, queen_hp, workers, soldiers, worker_brood, soldier_brood)
                        assert ref.episode_end(status, queen_hp, workers, soldiers, worker_brood, version=2) in (None, "queen_death", "collapse")


# ------------------------------------------------------------------------------------------------ the world is not touched


def _controller(retired=()):
    config = default_config(8, "moderate")
    controller = PolicyController(build_policies(PolicyRunSpec(7, (PolicySpec("heuristic", "peaceful"),) * 8), config))
    controller.retired = set(retired)
    return config, controller


def test_the_tracker_only_watches_the_world_state_hash_is_the_same_with_and_without_it():
    config, _ = _controller()
    spec = PolicyRunSpec(7, (PolicySpec("heuristic", "peaceful"),) * 8)
    plain = run_policies(config, 50012, spec, 600)
    watched = play_world(config, 50012, PolicyController(build_policies(spec, config)), 600)
    assert state_hash(watched.state) == plain.state_hash


def test_a_retired_colony_is_not_asked_anything_and_keeps_its_orders_while_the_others_keep_deciding():
    from formicmind.eco.engine import RouteTable, step_world_tick

    def run(retire_at):
        config, controller = _controller()
        world = build_world(config, 50012)
        routes = RouteTable(world.world_map)
        orders_then = None
        for _ in range(400):
            if world.state.tick == retire_at:
                controller.retired = {0}
                orders_then = copy.deepcopy(world.state.colonies[0].orders)
            step_world_tick(world.state, world.run, config, world.world_map, controller, routes=routes, validate_input=world.state.tick == 0)
        return world.state, controller, orders_then

    free_state, free, _ = run(None)
    held_state, held, orders_then = run(150)
    assert free.decisions[0] > 10 and 4 <= held.decisions[0] < free.decisions[0] and held.last_decision[0] < 150
    assert held.decisions[1:] == free.decisions[1:] and all(n > 10 for n in held.decisions[1:])        # the other colonies decide as before
    assert not any(record["colony"] == 0 and record["tick"] >= 150 for record in held.log.records)    # no decision record after the retirement
    assert held_state.colonies[0].orders == orders_then                                                  # standing orders: exactly what it had
    # no decisions means no new dispatches: the idle colony may starve in the world - that is its physical consequence, not a rule of the reward -
    # while the other colonies' physical state is exactly what it was
    assert [(c.food, c.status, c.queen_hp) for c in held_state.colonies[1:]] == [(c.food, c.status, c.queen_hp) for c in free_state.colonies[1:]]
    assert [replace(c, cohort_id=0) for c in held_state.cohorts if c.colony_id != 0] == [replace(c, cohort_id=0) for c in free_state.cohorts if c.colony_id != 0]
    assert held.snapshot()["retired"] == [0] and "retired" not in free.snapshot()
    restored = _controller()[1]
    restored.restore(copy.deepcopy(held.snapshot()))
    assert restored.retired == {0}
