"""Shared builders for the FOR-12 economy tests."""

import dataclasses

from formicmind import default_config, reset
from formicmind.actions import ActionV2
from formicmind.tick import ScriptedController, step_tick

Q = 65535
HOME_Q = (Q, 0, 0, 0, 0, 0)
FORAGE_N1_N2 = (0, 43690, 21845, 0, 0, 0)     # 8 workers on local N1, 4 on local N2 (exact for 12 workers)


def act(**kw):
    base = dict(lay_order=0, forage_share_q=HOME_Q, army_choice=0, avoid_center=False)
    base.update(kw)
    return ActionV2(**base)


def cfg(**balance):
    base = default_config()
    return dataclasses.replace(base, balance=dataclasses.replace(base.balance, **balance))


def new_run(a=None, b=None, config=None, seed=1, streams=()):
    state, run = reset(config or default_config(), seed, streams)
    return state, run, (ScriptedController(a or {}), ScriptedController(b or {}))


def tick(state, run, ctrl, n=1, check=True):
    events = []
    for _ in range(n):
        events.extend(step_tick(state, run, ctrl, partial_economy=True, check_invariants=check).events)
    return events


def bare(state, side, workers=0, soldiers=0):
    """Keep only the first ``workers`` workers and ``soldiers`` soldiers of a side."""
    colony = state.colonies[side]
    ws = [u for u in colony.units if u.caste == 0][:workers]
    ss = [u for u in colony.units if u.caste == 1][:soldiers]
    colony.units[:] = sorted(ws + ss, key=lambda u: u.unit_id)


def of_kind(events, kind):
    return [e for e in events if e.kind == kind]
