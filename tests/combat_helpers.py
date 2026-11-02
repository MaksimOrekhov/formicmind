"""Builders for the FOR-13 combat / terminal tests: hand-made arenas, no scenario is simulated to reach them."""

import dataclasses

from formicmind import default_config, reset
from formicmind.actions import ActionV2
from formicmind.model import NOWHERE, Unit
from formicmind.rules import SOLDIER, WORKER
from formicmind.tick import ScriptedController, step_tick

Q = 65535
HOME_Q = (Q, 0, 0, 0, 0, 0)
A, N1, N2, C, N4, N5, B = range(7)
BIG_FOOD = 100_000


def act(**kw):
    base = dict(lay_order=0, forage_share_q=HOME_Q, army_choice=0, avoid_center=False)
    base.update(kw)
    return ActionV2(**base)


def cfg(t_max=None, **balance):
    base = default_config()
    out = dataclasses.replace(base, balance=dataclasses.replace(base.balance, **balance))
    return dataclasses.replace(out, t_max=t_max) if t_max is not None else out


def arena(tick=0, config=None, seed=1, streams=(), food=BIG_FOOD):
    """An empty world: no units at all, plenty of food (so nobody starves or collapses)."""
    state, run = reset(config or default_config(), seed, streams)
    for colony in state.colonies:
        colony.units.clear()
        colony.food = food
    state.tick = tick
    return state, run


def add(state, side, caste, count, node, hp=None, target=None):
    """Append ``count`` new units (fresh ascending ids) standing in ``node``; returns them."""
    bal = state.config.balance
    full = bal.worker_hp if caste == WORKER else bal.soldier_hp
    units = []
    for _ in range(count):
        unit = Unit(unit_id=state.allocate_unit_id(), caste=caste, hp=full if hp is None else hp, node=node,
                    from_node=NOWHERE, to_node=NOWHERE, arrival_tick=NOWHERE,
                    target_node=node if target is None else target)
        state.colonies[side].units.append(unit)
        units.append(unit)
    return units


def soldiers(state, side, count, node, **kw):
    return add(state, side, SOLDIER, count, node, **kw)


def workers(state, side, count, node, **kw):
    return add(state, side, WORKER, count, node, **kw)


def ctrl(a=None, b=None):
    return ScriptedController(a or {}), ScriptedController(b or {})


def step(state, run, controllers=None, check=True):
    return step_tick(state, run, controllers or ctrl(), check_invariants=check)


def of_kind(events, kind):
    return [e for e in events if e.kind == kind]


def hp_of(state, side, caste=None):
    return sorted((u.unit_id, u.hp) for u in state.colonies[side].units if caste is None or u.caste == caste)
