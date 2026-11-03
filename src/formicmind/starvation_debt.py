"""FOR-25 study harness: chronic deficit and starvation-debt repayment variants (experimental, isolated).

Nothing here changes the L0 rules. The candidate rules A1a/A1b/A2/A3a/A3b and a byte-for-byte copy of the
current rule (A0) live only in this module and are enabled only inside ``with use_variant(...)`` by swapping
``formicmind.tick.upkeep``; ``economy.py`` and the defaults are untouched and the swap is always undone.
The frozen protocol is ``docs/starvation_debt_protocol.md``.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import itertools
import json
import inspect
import math
import platform
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from . import economy as economy_module
from . import tick as tick_module
from .canonical import canonical_dumps
from .economy import _add_window, _adults, starvation_victim_key
from .evaluation import controller_pair
from .events import Event, Phase, StarvationEvent, UpkeepEvent, events_hash
from .gate1_causal import _MirrorNormalizer
from .gate1_context import _build_configs, _policy_descriptor
from .intel import home_node
from .ledger import total_food
from .model import NOWHERE, SimState, Unit
from .reset import reset
from .rules import SIDES, SOLDIER, WORKER
from .runstate import RunState
from .serialization import checkpoint_hash, config_hash, state_hash
from .tick import PolicyController, step_tick
from .versions import (
    ACTION_VERSION, BALANCE_VERSION, DECODER_VERSION, OBSERVATION_VERSION, RNG_VERSION,
    RULES_VERSION, SERIALIZATION_VERSION, STREAM_DERIVATION_VERSION,
)

PROTOCOL_ID = "formicmind.starvation-debt/v1"
PROTOCOL_PATH = Path("docs/starvation_debt_protocol.md")
ERRATUM_PATH = Path("docs/starvation_debt_deviation.md")
PROTOCOL_SHA256 = "bd7b67545183f1e5ee42ccdd3d01be303d4bcedd0805c2b16a8945a29e65e2c0"
ERRATUM_SHA256 = "8329951f58d5e2ce481d2ac0722fb831f81957d8985958dfc36efb7152186086"
BASE_COMMIT = "c2a887cfdfc2e8ef0b3697543582d53aa246f4f1"
BASE_ENGINE_FINGERPRINT = "5fd459d1881a2316ef10f120b6d913706c77591ab22d198d9f7f1b972ef195d8"
SCHEMA_MANIFEST = "formicmind.starvation-debt-manifest/v1"
SCHEMA_ROW = "formicmind.starvation-debt-match/v1"
SCHEMA_SUMMARY = "formicmind.starvation-debt-summary/v1"
SCHEMA_DEMO = "formicmind.starvation-debt-demo/v1"
SCHEMA_REPRO = "formicmind.starvation-debt-repro/v1"
OUT_RELATIVE = Path("results/starvation_debt")
HARNESS_FILE = "starvation_debt.py"

VARIANTS = ("A0", "A1a", "A1b", "A2", "A3a", "A3b")
A3_STREAK = {"A3a": 3, "A3b": 10}
MAPS = ("R0-T0", "R0-T1", "R0-T2", "R1-T0", "R1-T1", "R1-T2")
POLICIES = ("random", "turtle", "rush", "balanced", "raider")
PAIRS = tuple(itertools.combinations(POLICIES, 2))
SEEDS = tuple(range(20270301, 20270307))
SEATS = (False, True)  # swapped?  first policy on A, then first policy on B
SEEDS_SHA256 = "c9a6b6a2f18f2b90ddf4b5e2ce65da411887d9afdc126608defbf20965cc1597"
CONFIG_SHA256 = {
    "R0-T0": "3f6aaa36a0d3d15888fa42f142df1151376794e62b553ab8e5f7659c9794145d",
    "R0-T1": "78e9c6979a2fb3973ae9a275f79ad5b62cb4b6ed3893a8f8469e4cc599e5fb34",
    "R0-T2": "b9e43b2aec34f303f371a9f9e5a2bcd8b5c5d894e608a3ca9e71386339f632f1",
    "R1-T0": "407bb3907099533adfe10b7c300a480b45b5d604dc019693b1ac99f47502303f",
    "R1-T1": "3bf8dabf6011ed88275c3fcfec3875c4da46ef849e7da7b0ac17bb2cd0e9e605",
    "R1-T2": "1b6f1f999246711919fb4dee9238e5fed31c5baac0dca779ea76c47afe167ceb",
}
POLICY_PARAMETER_SHA256 = {
    "random": "254037fbbe5f089fc1f909405ebcc3567573682553d3715d9bab8cf292b07f2c",
    "turtle": "6c9fa1c676a553294c26ffb4f3372953f93ecec73b5d8669f99a451a05e0ee4e",
    "rush": "cfea106a71257d8ebe79cfa6ac090b1927fd3010ca905516e2e41512460ea496",
    "balanced": "03c6760da2b2fc42108133c8b81a91b70e3c086327fbabedf9d3f30eb7dcdd0f",
    "raider": "6d4a23b0071c6ee308b8149850debe488e83d79c7fda50f19ee5053310c39b19",
}
CHECKPOINT_TICKS = (0, 100, 300, 600, 900, 1500, 2100, 2700)
WINDOW = 300
CHRONIC_DEFICIT_TICKS = 30
ZOMBIE_ADULT_DELTA = 2
EPISODE_GAP = 30
RECOVERY_HORIZON = 300
RECOVERY_SPAN = 100
MIN_ZOMBIE_BLOCKS, MIN_ZOMBIE_PAIRS, MIN_ZOMBIE_MAPS = 6, 2, 2
MIN_EXPOSURE, MIN_SUPPORT = 12, 6
REPRO_CASE = {"map": "R0-T0", "first": "turtle", "second": "balanced", "seed": 20270201, "swapped": False}
ARCHIVE_BASELINES = Path("results/gate1_stalling/baselines.jsonl")
EXPECTED_ROWS = len(VARIANTS) * len(MAPS) * len(PAIRS) * len(SEEDS) * len(SEATS)  # 4320
MAIN_BUDGET = EXPECTED_ROWS * 3000
ANCHOR_BUDGET = len(MAPS) * len(PAIRS) * len(SEEDS) * len(SEATS) * 3000
PROTECTED_FILES = (
    "docs/gate1_stalling_protocol.md", "docs/gate1_stalling_findings.md", "docs/gate1_feedback_protocol.md",
    "docs/gate1_context_protocol.md", "docs/gate1_causal_protocol.md", "docs/economy_pilot_protocol.md",
    "results/gate1_stalling/baselines.jsonl", "results/gate1_context/matches.jsonl",
    "results/economy_pilot/runs.json",
)
_ROOT = Path(__file__).resolve().parents[2]


class StarvationDebtError(ValueError):
    """Invalid fixed-study input, integrity failure or strict verification mismatch."""


# --- hashing helpers --------------------------------------------------------------------------------

def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _lf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def _file_sha(path: Path) -> str:
    return _sha(_lf(path.read_bytes()))


class _Digest:
    """Incremental SHA-256 of a canonical JSON array."""

    def __init__(self) -> None:
        self.hash = hashlib.sha256(b"[")
        self.count = 0

    def add(self, value: Any) -> None:
        if self.count:
            self.hash.update(b",")
        self.hash.update(canonical_dumps(value))
        self.count += 1

    def hexdigest(self) -> str:
        result = self.hash.copy()
        result.update(b"]")
        return result.hexdigest()


def source_fingerprint(*, exclude_harness: bool = False) -> str:
    package = _ROOT / "src" / "formicmind"
    entries = [
        {"path": path.relative_to(_ROOT).as_posix(), "sha256": _sha(_lf(path.read_bytes()))}
        for path in sorted(package.rglob("*.py"))
        if not (exclude_harness and path.name == HARNESS_FILE)
    ]
    return _sha(canonical_dumps(entries))


def protocol_sha256() -> str:
    return _file_sha(_ROOT / PROTOCOL_PATH)


# --- candidate rules --------------------------------------------------------------------------------

@dataclass(slots=True)
class VariantContext:
    """Per-match experimental state that is deliberately *not* part of the game state."""

    variant: str
    streak: list[int] = field(default_factory=lambda: [0, 0])
    calls: int = 0


def variant_upkeep(ctx: VariantContext, state: SimState, run: RunState, phase: int = Phase.UPKEEP
                   ) -> list[UpkeepEvent | StarvationEvent]:
    """The upkeep phase of ``economy.upkeep`` with the debt rule of ``ctx.variant`` (A0 is a verbatim copy)."""
    variant = ctx.variant
    if variant not in VARIANTS:
        raise StarvationDebtError(f"unknown variant {variant!r}")
    ctx.calls += 1
    tick = state.tick
    bal = state.config.balance
    cost = bal.starvation_unit_cost
    upkeep_events: list[UpkeepEvent] = []
    deaths: list[StarvationEvent] = []
    for side in SIDES:
        colony = state.colonies[side]
        workers, soldiers = _adults(colony, WORKER), _adults(colony, SOLDIER)
        remainder_before = colony.upkeep_remainder10
        due, remainder_after = divmod(
            remainder_before + bal.worker_upkeep_tenths * workers
            + bal.soldier_upkeep_tenths * soldiers, 10)
        food_before = colony.food
        paid = min(due, food_before)
        missing = due - paid
        colony.food = food_before - paid
        colony.upkeep_remainder10 = remainder_after
        food_after_pay = colony.food
        debt_before = debt = colony.starvation_debt
        reset_flag = False
        victim_count = 0
        if missing:
            ctx.streak[side] = 0
            debt += missing
            home = home_node(state, side)
            order = sorted(colony.units, key=lambda u: starvation_victim_key(home, u))
            while debt >= cost and order:
                victim = order.pop(0)
                before_pay = debt
                debt -= cost
                colony.units.remove(victim)
                deaths.append(StarvationEvent(
                    tick=tick, phase=int(phase), side=side, seq=victim_count,
                    unit_id=victim.unit_id, caste=victim.caste, hp=victim.hp, node=victim.node,
                    to_node=victim.to_node, debt_before=before_pay, debt_after=debt))
                victim_count += 1
        else:
            if due > 0 and variant in A3_STREAK:
                ctx.streak[side] += 1
            if debt:
                if variant == "A0":
                    clear = due > 0
                elif variant == "A1a":
                    clear = due > 0 and food_after_pay >= 1
                elif variant == "A1b":
                    clear = due > 0 and food_after_pay >= due
                elif variant in A3_STREAK:
                    clear = due > 0 and ctx.streak[side] >= A3_STREAK[variant]
                    if clear:
                        ctx.streak[side] = 0
                else:  # A2: surplus food repays debt; the food is destroyed
                    clear = False
                    repay = min(debt, colony.food)
                    colony.food -= repay
                    debt -= repay
                if clear:
                    debt, reset_flag = 0, True
        colony.starvation_debt = debt
        _add_window(run, side, upkeep_paid=paid, debt_delta=debt - debt_before)
        upkeep_events.append(UpkeepEvent(
            tick=tick, phase=int(phase), side=side, adult_workers=workers, adult_soldiers=soldiers,
            remainder_before=remainder_before, remainder_after=remainder_after, due=due,
            paid=paid, missing=missing, food_before=food_before, food_after=food_after_pay,
            debt_before=debt_before, debt_after=debt, deaths=victim_count, debt_reset=reset_flag))
    return [*upkeep_events, *deaths]


@contextlib.contextmanager
def use_variant(variant: str) -> Iterator[VariantContext]:
    """Swap ``tick.upkeep`` for a candidate rule for the duration of the block; always restores it."""
    if variant not in VARIANTS:
        raise StarvationDebtError(f"unknown variant {variant!r}")
    if tick_module.upkeep is not economy_module.upkeep:
        raise StarvationDebtError("tick.upkeep is already replaced; variants cannot be nested")
    ctx = VariantContext(variant)
    tick_module.upkeep = functools.partial(variant_upkeep, ctx)
    try:
        yield ctx
    finally:
        tick_module.upkeep = economy_module.upkeep


# --- stage M: minimal scenarios and edge cases ------------------------------------------------------

MINIMAL_SCENARIOS = {
    "M1": {"income": (11, 11, 14), "ticks": 600, "note": "alternating shortfall"},
    "M2": {"income": (12, 12, 12), "ticks": 600, "note": "constant shortfall"},
    "M3": {"income": (13, 13, 13), "ticks": 600, "note": "sufficient income"},
    "M4": {"income": (11, 11, 11), "ticks": 600, "note": "deficit without a fully paid tick (erratum E1)"},
}
MINIMAL_WORKERS = 32


def _toy_state(workers: int, soldiers: int, *, food: int, debt: int, rem: int, cost: int | None = None
               ) -> tuple[SimState, RunState]:
    from dataclasses import replace
    from .rules import default_config
    config = default_config()
    if cost is not None:
        config = replace(config, balance=replace(config.balance, starvation_unit_cost=cost))
    state, run = reset(config, 1, ())
    colony = state.colonies[0]
    home = home_node(state, 0)
    units = [Unit(1000 + i, WORKER, config.balance.worker_hp, home, NOWHERE, NOWHERE, NOWHERE, home)
             for i in range(workers)]
    units += [Unit(2000 + i, SOLDIER, config.balance.soldier_hp, home, NOWHERE, NOWHERE, NOWHERE, home)
              for i in range(soldiers)]
    colony.units[:] = units
    colony.food, colony.starvation_debt, colony.upkeep_remainder10 = food, debt, rem
    return state, run


def _side0(events: list[Any]) -> tuple[UpkeepEvent, list[StarvationEvent]]:
    up = next(e for e in events if e.kind == "upkeep" and e.side == 0)
    return up, [e for e in events if e.kind == "starvation" and e.side == 0]


def run_single_tick(variant: str | None, *, workers: int, soldiers: int = 0, food: int = 0, debt: int = 0,
                    rem: int = 0, streak: int = 0, cost: int | None = None) -> dict[str, Any]:
    """One real upkeep call of side A; ``variant=None`` calls the unchanged ``economy.upkeep``."""
    state, run = _toy_state(workers, soldiers, food=food, debt=debt, rem=rem, cost=cost)
    state.tick = 5
    if variant is None:
        events = economy_module.upkeep(state, run)
    else:
        ctx = VariantContext(variant, [streak, 0])
        events = variant_upkeep(ctx, state, run)
    up, dead = _side0(events)
    colony = state.colonies[0]
    return {
        "due": up.due, "paid": up.paid, "missing": up.missing, "food_after": colony.food,
        "debt_before": up.debt_before, "debt_after": colony.starvation_debt, "deaths": len(dead),
        "debt_reset": up.debt_reset, "adults_left": len(colony.units), "remainder_after": colony.upkeep_remainder10,
    }


EDGE_CASES = {
    "due_zero_keeps_debt": dict(workers=1, food=0, debt=3, rem=0),
    "zero_food_charges_debt": dict(workers=3, food=0, debt=0, rem=0),
    "debt_4_no_death": dict(workers=10, food=0, debt=0, rem=0),
    "debt_5_one_death": dict(workers=12, food=0, debt=0, rem=2),
    "debt_9_one_death_leaves_4": dict(workers=12, food=0, debt=4, rem=2),
    "debt_14_two_deaths_leaves_4": dict(workers=25, food=0, debt=4, rem=0),
    "debt_68_many_deaths": dict(workers=12, soldiers=4, food=0, debt=60, rem=0),
    "last_adult_keeps_residual_debt": dict(workers=1, food=0, debt=100, rem=9),
    "exact_payment_with_debt": dict(workers=12, food=5, debt=3, rem=2),
    "one_unit_surplus_with_debt": dict(workers=12, food=6, debt=3, rem=2),
    "large_surplus_with_debt": dict(workers=12, food=40, debt=3, rem=2),
    "surplus_exceeds_debt": dict(workers=12, food=40, debt=30, rem=2),
}


def run_minimal(variant: str, scenario: str) -> dict[str, Any]:
    """M1-M3: a fixed income table feeds the real upkeep rule; A0 is checked against ``economy.upkeep``."""
    spec = MINIMAL_SCENARIOS[scenario]
    income = spec["income"]
    state, run = _toy_state(MINIMAL_WORKERS, 0, food=0, debt=0, rem=0)
    prod_state, prod_run = _toy_state(MINIMAL_WORKERS, 0, food=0, debt=0, rem=0)
    ctx = VariantContext(variant)
    colony, prod = state.colonies[0], prod_state.colonies[0]
    totals = Counter()
    digest = _Digest()
    table = []
    max_debt = 0
    starvation_events = []
    for t in range(spec["ticks"]):
        state.tick = prod_state.tick = t
        colony.food += income[t % len(income)]
        prod.food += income[t % len(income)]
        events = variant_upkeep(ctx, state, run)
        up, dead = _side0(events)
        if variant == "A0":
            prod_events = economy_module.upkeep(prod_state, prod_run)
            if [e.to_canonical() for e in prod_events] != [e.to_canonical() for e in events]:
                raise StarvationDebtError(f"A0 differs from economy.upkeep at tick {t} of {scenario}")
        forgiven = up.debt_before if up.debt_reset else 0
        repaid = (up.debt_before - up.debt_after) if (not up.missing and not up.debt_reset) else 0
        totals.update({"due": up.due, "paid": up.paid, "missing": up.missing, "forgiven": forgiven,
                       "repaid": repaid, "deaths": len(dead), "resets": int(up.debt_reset),
                       "deficit_ticks": int(up.missing > 0)})
        max_debt = max(max_debt, up.debt_after, up.debt_before + up.missing)
        starvation_events.extend(
            {"tick": t, "unit_id": e.unit_id, "debt_before": e.debt_before, "debt_after": e.debt_after}
            for e in dead)
        row = [t, up.due, up.paid, up.missing, up.food_before, up.food_after, colony.food, up.debt_before,
               up.debt_after, len(dead), int(up.debt_reset)]
        digest.add(row)
        if t < 12:
            table.append(row)
    identity = (colony.starvation_debt ==
                totals["missing"] - state.config.balance.starvation_unit_cost * totals["deaths"]
                - totals["forgiven"] - totals["repaid"])
    return {
        "scenario": scenario, "variant": variant, "note": spec["note"], "income_cycle": list(income),
        "ticks": spec["ticks"], "initial_workers": MINIMAL_WORKERS, "totals": dict(sorted(totals.items())),
        "adults_end": len(colony.units), "debt_end": colony.starvation_debt, "food_end": colony.food,
        "max_debt_seen": max_debt, "debt_identity_holds": identity,
        "a0_equals_production_upkeep": True if variant == "A0" else "not_applicable",
        "tick_table_sha256": digest.hexdigest(),
        "first_ticks_columns": ["tick", "due", "paid", "missing", "food_before", "food_after_pay",
                                "food_end", "debt_before", "debt_after", "deaths", "debt_reset"],
        "first_ticks": table, "first_starvation_events": starvation_events[:10],
    }


def build_minimal_demo() -> dict[str, Any]:
    scenarios = {s: {v: run_minimal(v, s) for v in VARIANTS} for s in MINIMAL_SCENARIOS}
    edges = {}
    for name, setup in EDGE_CASES.items():
        production = run_single_tick(None, **setup)
        edges[name] = {"setup": setup, "production": production,
                       "variants": {v: run_single_tick(v, **setup) for v in VARIANTS}}
        if edges[name]["variants"]["A0"] != production:
            raise StarvationDebtError(f"A0 differs from economy.upkeep on edge case {name}")
    m1, m4 = scenarios["M1"]["A0"]["totals"], scenarios["M4"]["A0"]["totals"]
    r1 = m1["deaths"] == 0 and m1["missing"] > 0 and m1["forgiven"] == m1["missing"] and m4["deaths"] > 0  # erratum E1
    return {"schema": SCHEMA_DEMO, "r1_mechanism_reproduced": r1, "scenarios": scenarios, "edge_cases": edges,
            "tenths_rounding": _tenths_rounding()}


def _tenths_rounding() -> dict[str, Any]:
    """One worker for ten ticks with ample food: Σdue is 4 and the remainder returns to 0."""
    state, run = _toy_state(1, 0, food=100, debt=0, rem=0)
    dues = []
    for t in range(10):
        state.tick = t
        up, _ = _side0(economy_module.upkeep(state, run))
        dues.append(up.due)
    return {"dues": dues, "sum": sum(dues), "remainder_end": state.colonies[0].upkeep_remainder10}


# --- per-match tracking -----------------------------------------------------------------------------

class _SideTrace:
    """Per-tick series of one colony (world side)."""

    def __init__(self) -> None:
        self.due: list[int] = []
        self.paid: list[int] = []
        self.missing: list[int] = []
        self.forgiven: list[int] = []
        self.repaid: list[int] = []
        self.deaths: list[int] = []
        self.deaths_by_caste = [0, 0]
        self.exact_pay = 0
        self.surplus_pay = 0
        self.due_zero = 0
        self.lay_ok = 0
        self.lay_rejected = 0
        self.hatch = 0
        self.adults: list[int] = []  # adults[t] = adults after t completed ticks
        self.lay_rejected_ticks: list[int] = []
        self.death_ticks_last: list[int] = []
        self.queen_damage_ticks: list[int] = []
        self.checkpoints: dict[int, list[int]] = {}
        self.max_debt = 0


CHECKPOINT_COLUMNS = ["workers", "soldiers", "brood_workers", "brood_soldiers", "food", "debt",
                      "remainder10", "queen_hp"]


def _snapshot(state: SimState, side: int) -> list[int]:
    c = state.colonies[side]
    return [
        sum(1 for u in c.units if u.caste == WORKER), sum(1 for u in c.units if u.caste == SOLDIER),
        sum(1 for b in c.brood if b.caste == WORKER), sum(1 for b in c.brood if b.caste == SOLDIER),
        c.food, c.starvation_debt, c.upkeep_remainder10, c.queen_hp,
    ]


def _flags_for_windows(trace: _SideTrace, completed: int, queen_alive: bool) -> tuple[list[list[int]], dict[str, int]]:
    """Aligned 300-tick windows fully inside the match and the three protocol flags."""
    windows, flags = [], {"chronic": 0, "forgiving": 0, "zombie": 0}
    for k in range(10):
        lo, hi = k * WINDOW, (k + 1) * WINDOW
        if hi > completed or hi > len(trace.missing):
            break
        miss = trace.missing[lo:hi]
        deficit = sum(1 for m in miss if m > 0)
        missing, forgiven = sum(miss), sum(trace.forgiven[lo:hi])
        repaid, deaths = sum(trace.repaid[lo:hi]), sum(trace.deaths[lo:hi])
        a0, a1 = trace.adults[lo], trace.adults[hi]
        chronic = deficit >= CHRONIC_DEFICIT_TICKS
        forgiving = chronic and forgiven >= 1 and 2 * forgiven >= missing
        # the queen is alive at the end of every window that precedes a queen-death terminal
        zombie = forgiving and abs(a1 - a0) <= ZOMBIE_ADULT_DELTA and (queen_alive or hi < completed)
        flags["chronic"] += int(chronic)
        flags["forgiving"] += int(forgiving)
        flags["zombie"] += int(zombie)
        windows.append([k, deficit, missing, forgiven, repaid, deaths, a0, a1, int(chronic), int(forgiving), int(zombie)])
    return windows, flags


WINDOW_COLUMNS = ["k", "deficit_ticks", "missing", "forgiven", "repaid", "deaths", "adults_start", "adults_end",
                  "chronic", "forgiving", "zombie"]


def _episodes(trace: _SideTrace, completed: int) -> list[list[int]]:
    """Starvation-death episodes (gap <= 30) with the two recovery observations (protocol section 7)."""
    ticks = [t for t, d in enumerate(trace.deaths) if d]
    chains: list[list[int]] = []
    for t in ticks:
        if chains and t - chains[-1][-1] <= EPISODE_GAP:
            chains[-1].append(t)
        else:
            chains.append([t])
    missing_flag = [1 if m else 0 for m in trace.missing]
    n = len(missing_flag)
    prefix = [0]
    for flag in missing_flag:
        prefix.append(prefix[-1] + flag)
    result = []
    for chain in chains:
        first, last = chain[0], chain[-1]
        before = trace.adults[first]
        observable = last + RECOVERY_HORIZON <= completed
        rec_econ = rec_pop = None
        if observable:
            rec_econ = any(
                s + RECOVERY_SPAN <= n and prefix[s + RECOVERY_SPAN] - prefix[s] == 0
                for s in range(last + 1, last + RECOVERY_HORIZON + 1))
            rec_pop = any(trace.adults[t] >= before for t in range(last + 1, last + RECOVERY_HORIZON + 1)
                          if t < len(trace.adults))
        result.append([first, last, before, len(chain), int(observable),
                       -1 if rec_econ is None else int(rec_econ), -1 if rec_pop is None else int(rec_pop)])
    return result


EPISODE_COLUMNS = ["first_death_tick", "last_death_tick", "adults_before", "death_ticks", "observable",
                   "recovered_economy", "recovered_population"]


@dataclass(slots=True)
class _RunOutput:
    row: dict[str, Any]
    traces: tuple[_SideTrace, _SideTrace]
    calls: int


def _outcome_object(outcome: Any, state: SimState, first_side: int) -> dict[str, Any] | None:
    if outcome is None:
        return None
    score = outcome.score
    return {
        "reason": outcome.reason.value, "winner": outcome.winner.value, "terminal_tick": outcome.terminal_tick,
        "state_tick": state.tick, "tick_completed": outcome.tick_completed,
        "first_score_x2": int(score[first_side] * 2), "second_score_x2": int(score[1 - first_side] * 2),
    }


def _play_match(config: Any, first: str, second: str, seed: int, swapped: bool, *, normalize: bool,
                detail: bool = False) -> _RunOutput:
    """One complete match through the real ``step_tick`` (patched or not, depending on the caller)."""
    pair = controller_pair(None, a=first, b=second, swapped=swapped)
    controllers = pair.controllers
    if any(not isinstance(c, PolicyController) for c in controllers):
        raise StarvationDebtError("the schedule must use only PolicyController")
    state, run = reset(config, seed, tuple(c.stream_id for c in controllers))
    first_side = 1 if swapped else 0
    traces = (_SideTrace(), _SideTrace())
    normalizer = _MirrorNormalizer(first_side) if normalize else None
    trajectory = _Digest()
    events_trace = _Digest()
    combat_ticks: list[int] = []
    calls = 0
    food = total_food(state)
    detail_log: dict[str, Any] = {"starvation": [], "resets": [], "per100": {}}

    def checkpoint(tick: int) -> None:
        for side in SIDES:
            traces[side].checkpoints[tick] = _snapshot(state, side)

    for side in SIDES:
        traces[side].adults.append(len(state.colonies[side].units))
    checkpoint(0)
    while run.outcome is None and state.tick < config.t_max:
        t = state.tick
        if normalizer is not None:
            normalizer._register_entities(state)
            if t % 10 == 0:
                trajectory.add({"tick": t, "state": normalizer.state_object(state)})
        result = step_tick(state, run, controllers)
        calls += 1
        events = list(result.events)
        events_trace.add({"tick": t, "events_sha256": events_hash(events)})
        regen = lay_cost = paid_total = repaid_total = 0
        per_side_up: dict[int, UpkeepEvent] = {}
        for e in events:
            kind = e.kind
            if kind == "upkeep":
                per_side_up[e.side] = e
            elif kind == "starvation":
                traces[e.side].deaths_by_caste[e.caste] += 1
                if detail and len(detail_log["starvation"]) < 20:
                    detail_log["starvation"].append({
                        "tick": e.tick, "side": e.side, "caste": e.caste, "unit_id": e.unit_id,
                        "debt_before": e.debt_before, "debt_after": e.debt_after})
            elif kind == "regenerate":
                regen += e.added
            elif kind == "lay":
                tr = traces[e.side]
                if e.laid:
                    tr.lay_ok += 1
                    lay_cost += e.total_cost
                else:
                    tr.lay_rejected += 1
                    tr.lay_rejected_ticks.append(t)
            elif kind == "hatch":
                traces[e.side].hatch += 1
            elif kind == "combat":
                combat_ticks.append(t)
            elif kind == "queen_damage":
                traces[e.side].queen_damage_ticks.append(t)
        for side in SIDES:
            up = per_side_up.get(side)
            tr = traces[side]
            if up is None:
                # a queen-death terminal in Combat ends the tick before the economy phases (not completed)
                if per_side_up or run.outcome is None or state.tick == t + 1:
                    raise StarvationDebtError(f"tick {t}: no upkeep event for side {side}")
                continue
            forgiven = up.debt_before if up.debt_reset else 0
            repaid = (up.debt_before - up.debt_after) if (not up.missing and not up.debt_reset) else 0
            tr.due.append(up.due)
            tr.paid.append(up.paid)
            tr.missing.append(up.missing)
            tr.forgiven.append(forgiven)
            tr.repaid.append(repaid)
            tr.deaths.append(up.deaths)
            tr.max_debt = max(tr.max_debt, up.debt_before + up.missing, up.debt_after)
            if up.due == 0:
                tr.due_zero += 1
            elif not up.missing:
                if up.food_after == 0:
                    tr.exact_pay += 1
                else:
                    tr.surplus_pay += 1
            paid_total += up.paid
            repaid_total += repaid
            if detail:
                bucket = detail_log["per100"].setdefault((side, t // 100), Counter())
                bucket.update({"due": up.due, "paid": up.paid, "missing": up.missing,
                               "deficit_ticks": int(up.missing > 0), "resets": int(up.debt_reset),
                               "forgiven": forgiven, "repaid": repaid, "deaths": up.deaths})
                if up.debt_reset and len(detail_log["resets"]) < 20:
                    detail_log["resets"].append({
                        "tick": t, "side": side, "due": up.due, "paid": up.paid,
                        "food_before": up.food_before, "debt_before": up.debt_before})
        new_food = total_food(state)
        if new_food != food + regen - paid_total - repaid_total - lay_cost:
            raise StarvationDebtError(f"food ledger broken at tick {t}")
        food = new_food
        completed = state.tick == t + 1
        if completed:
            for side in SIDES:
                traces[side].adults.append(len(state.colonies[side].units))
            if run.outcome is None and state.tick in CHECKPOINT_TICKS:
                checkpoint(state.tick)
        if normalizer is not None:
            trajectory.add({"tick": t, "events": sorted(
                (canonical_dumps(normalizer.event_object(e)).decode("ascii") for e in events))})
    outcome = run.outcome
    completed_ticks = state.tick
    if normalizer is not None:
        trajectory.add({"final": normalizer.state_object(state)})
    for side in SIDES:
        traces[side].checkpoints.setdefault(-1, _snapshot(state, side))  # terminal snapshot
    names = (first, second)
    sides: dict[str, Any] = {}
    for rel, name in enumerate(names):
        world = first_side if rel == 0 else 1 - first_side
        tr = traces[world]
        queen_alive = state.colonies[world].queen_hp > 0
        windows, flags = _flags_for_windows(tr, completed_ticks, queen_alive)
        episodes = _episodes(tr, completed_ticks)
        missing_sum, deaths_sum = sum(tr.missing), sum(tr.deaths)
        forgiven_sum, repaid_sum = sum(tr.forgiven), sum(tr.repaid)
        debt_end = state.colonies[world].starvation_debt
        identity = debt_end == missing_sum - config.balance.starvation_unit_cost * deaths_sum - forgiven_sum - repaid_sum
        if not identity:
            raise StarvationDebtError(f"debt identity broken for {name} (side {world})")
        terminal_tick = outcome.terminal_tick if outcome is not None else completed_ticks
        lo = terminal_tick - 599
        sides["first" if rel == 0 else "second"] = {
            "policy": name, "world_side": "AB"[world],
            "due": sum(tr.due), "paid": sum(tr.paid), "missing": missing_sum,
            "deficit_ticks": sum(1 for m in tr.missing if m), "reset_ticks": sum(1 for f in tr.forgiven if f),
            "forgiven": forgiven_sum, "repaid": repaid_sum, "starvation_deaths": deaths_sum,
            "starvation_deaths_by_caste": {"worker": tr.deaths_by_caste[WORKER], "soldier": tr.deaths_by_caste[SOLDIER]},
            "max_debt": tr.max_debt, "debt_end": debt_end, "debt_identity_holds": identity,
            "due_zero_ticks": tr.due_zero, "exact_pay_ticks": tr.exact_pay, "surplus_pay_ticks": tr.surplus_pay,
            "lay_ok": tr.lay_ok, "lay_rejected": tr.lay_rejected, "hatch": tr.hatch,
            "queen_alive": queen_alive, "food_end": state.colonies[world].food,
            "adults_end": len(state.colonies[world].units), "adults_best": max(tr.adults),
            "checkpoints": {str(k): v for k, v in sorted(tr.checkpoints.items()) if k >= 0},
            "terminal_snapshot": tr.checkpoints[-1],
            "windows": windows, "flags": flags, "episodes": episodes,
            "last600": {
                "starvation_deaths": sum(d for t, d in enumerate(tr.deaths) if t >= lo),
                "lay_rejected": sum(1 for t in tr.lay_rejected_ticks if t >= lo),
                "queen_damage_events": sum(1 for t in tr.queen_damage_ticks if t >= lo),
            },
        }
    terminal_tick = outcome.terminal_tick if outcome is not None else completed_ticks
    row = {
        "status": "complete" if outcome is not None else "censored_incomplete",
        "terminal": _outcome_object(outcome, state, first_side),
        "ticks_executed": calls,
        "sides": sides,
        "last600_combat_events": sum(1 for t in combat_ticks if t >= terminal_tick - 599),
        "events_trace_sha256": events_trace.hexdigest(),
        "trajectory_sha256": trajectory.hexdigest() if normalizer is not None else "not_measured",
        "final_state_sha256": state_hash(state),
        "final_checkpoint_sha256": checkpoint_hash(state, run),
    }
    if detail:
        row["_detail"] = detail_log
    return _RunOutput(row, traces, calls)


@functools.lru_cache(maxsize=1)
def _configs() -> dict[str, Any]:
    configs = _build_configs()
    for name, expected in CONFIG_SHA256.items():
        if config_hash(configs[name]) != expected:
            raise StarvationDebtError(f"config hash drift for {name}")
    return configs


def row_id(variant: str, map_id: str, first: str, second: str, seed: int, swapped: bool) -> str:
    return f"{variant}|{map_id}|{first}-{second}|{seed}|{'first_B' if swapped else 'first_A'}"


def execute_row(variant: str, map_id: str, first: str, second: str, seed: int, swapped: bool,
                *, anchor: bool | None = None) -> dict[str, Any]:
    """One schedule row of one arm; A0 rows are checked against the unchanged engine."""
    config = _configs()[map_id]
    with use_variant(variant):
        out = _play_match(config, first, second, seed, swapped, normalize=True)
    row = out.row
    row.update({
        "schema": SCHEMA_ROW, "row_id": row_id(variant, map_id, first, second, seed, swapped),
        "variant": variant, "map_id": map_id, "matchup": f"{first}-{second}", "policies": [first, second],
        "seed": seed, "seat": "first_B" if swapped else "first_A", "config_sha256": config_hash(config),
        "stream_ids": [c.stream_id for c in controller_pair(None, a=first, b=second, swapped=swapped).controllers],
    })
    row["anchor_calls"] = 0
    do_anchor = variant == "A0" if anchor is None else anchor
    if do_anchor:
        plain = _play_match(config, first, second, seed, swapped, normalize=False)
        same = (
            plain.row["terminal"] == row["terminal"] and plain.calls == out.calls
            and plain.row["final_state_sha256"] == row["final_state_sha256"]
            and plain.row["final_checkpoint_sha256"] == row["final_checkpoint_sha256"]
            and plain.row["events_trace_sha256"] == row["events_trace_sha256"])
        if not same:
            raise StarvationDebtError(f"A0 differs from the unchanged engine at {row['row_id']}")
        row["anchor_calls"] = plain.calls
        row["anchor_matches_unchanged_engine"] = True
    return row


def schedule(variants: Sequence[str] = VARIANTS, maps: Sequence[str] = MAPS,
             pairs: Sequence[tuple[str, str]] = PAIRS, seeds: Sequence[int] = SEEDS
             ) -> list[tuple[str, str, str, str, int, bool]]:
    return [(v, m, a, b, s, w) for v in variants for m in maps for (a, b) in pairs for s in seeds for w in SEATS]


def _job(args: tuple[Any, ...]) -> dict[str, Any]:
    return execute_row(*args)


def run_schedule(rows: Sequence[tuple[Any, ...]], jobs: int = 1) -> list[dict[str, Any]]:
    if jobs <= 1:
        return [execute_row(*r) for r in rows]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(_job, rows, chunksize=4))


# --- block / cluster analysis -----------------------------------------------------------------------

def _group_blocks(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Distinct paired blocks of one arm: both seat rows of a (map, matchup, seed), grouped by trajectory."""
    by_key: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    for r in rows:
        by_key.setdefault((r["map_id"], r["matchup"], r["seed"]), {})[r["seat"]] = r
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for (map_id, matchup, seed), seats in sorted(by_key.items()):
        if set(seats) != {"first_A", "first_B"}:
            raise StarvationDebtError(f"incomplete seat pair for {map_id} {matchup} {seed}")
        digest = _sha(canonical_dumps([seats["first_A"]["trajectory_sha256"], seats["first_B"]["trajectory_sha256"]]))
        g = groups.setdefault((map_id, matchup, digest), {
            "map_id": map_id, "matchup": matchup, "block_sha256": digest, "seeds": [], "rows": []})
        g["seeds"].append(seed)
        g["rows"].extend([seats["first_A"], seats["first_B"]])
    return list(groups.values())


def _representative(block: dict[str, Any]) -> list[dict[str, Any]]:
    seed = min(block["seeds"])
    return [r for r in block["rows"] if r["seed"] == seed]


def _side_items(rows: Sequence[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for r in rows:
        yield r["sides"]["first"]
        yield r["sides"]["second"]


def _tally(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    reasons = Counter(r["terminal"]["reason"] if r["terminal"] else "incomplete" for r in rows)
    sums = Counter()
    for s in _side_items(rows):
        sums.update({k: s[k] for k in ("due", "paid", "missing", "deficit_ticks", "reset_ticks", "forgiven",
                                       "repaid", "starvation_deaths", "lay_ok", "lay_rejected", "hatch")})
        sums.update({"chronic_windows": s["flags"]["chronic"], "forgiving_windows": s["flags"]["forgiving"],
                     "zombie_windows": s["flags"]["zombie"]})
        for ep in s["episodes"]:
            sums["episodes"] += 1
            if ep[4]:
                sums["episodes_observable"] += 1
                sums["recovered_economy"] += int(ep[5] == 1)
                sums["recovered_population"] += int(ep[6] == 1)
    durations = [r["terminal"]["terminal_tick"] for r in rows if r["terminal"]]
    return {
        "rows": len(rows), "reasons": dict(sorted(reasons.items())),
        "winners": dict(sorted(Counter(r["terminal"]["winner"] for r in rows if r["terminal"]).items())),
        "sums": dict(sorted(sums.items())),
        "duration_total_ticks": sum(durations), "duration_max": max(durations) if durations else 0,
        "no_combat_last600": sum(1 for r in rows if r["terminal"] and r["last600_combat_events"] == 0),
        "adults_end_total": sum(s["adults_end"] for s in _side_items(rows)),
    }


def _arm_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    blocks = _group_blocks(rows)
    rep_rows = [r for b in blocks for r in _representative(b)]
    return {
        "raw": _tally(rows), "distinct": _tally(rep_rows),
        "distinct_blocks": len(blocks),
        "distinct_match_trajectories": len({(r["map_id"], r["matchup"], r["trajectory_sha256"]) for r in rows}),
        "collapse_blocks": sum(1 for b in blocks if any(
            r["terminal"] and r["terminal"]["reason"] == "economic_collapse" for r in _representative(b))),
        "blocks_with_deficit": sum(1 for b in blocks if any(s["missing"] > 0 for s in _side_items(_representative(b)))),
    }


def _zombie_count(rows: Sequence[dict[str, Any]]) -> int:
    return sum(s["flags"]["zombie"] for s in _side_items(rows))


def _a0_blocks(rows_a0: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks = _group_blocks(rows_a0)
    for b in blocks:
        rep = _representative(b)
        b["deficit"] = any(s["missing"] > 0 for s in _side_items(rep))
        b["zombie_windows"] = _zombie_count(rep)
        b["zombie"] = b["zombie_windows"] > 0
    return blocks


def _compare(variant: str, a0_blocks: Sequence[dict[str, Any]], rows_v: Sequence[dict[str, Any]],
             rows_a0: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_id_v = {(r["map_id"], r["matchup"], r["seed"], r["seat"]): r for r in rows_v}
    by_id_0 = {(r["map_id"], r["matchup"], r["seed"], r["seat"]): r for r in rows_a0}
    differs = {k: by_id_v[k]["events_trace_sha256"] != by_id_0[k]["events_trace_sha256"] for k in by_id_0}
    false_triggers = [
        k for k, r in by_id_0.items()
        if all(s["missing"] == 0 for s in _side_items([r])) and differs[k]]
    blocks_v = _group_blocks(rows_v)
    zombie_blocks = [b for b in a0_blocks if b["zombie"]]
    support = remaining = 0
    heterogeneous = 0
    transitions: Counter[str] = Counter()
    for b in a0_blocks:
        keys = {(r["map_id"], r["matchup"], r["seed"], r["seat"]) for r in b["rows"]}
        if len({by_id_v[k]["events_trace_sha256"] for k in keys if k[3] == "first_A"}) > 1 or \
                len({by_id_v[k]["events_trace_sha256"] for k in keys if k[3] == "first_B"}) > 1:
            heterogeneous += 1
        for r in _representative(b):
            k = (r["map_id"], r["matchup"], r["seed"], r["seat"])
            transitions[f"{r['terminal']['reason']}->{by_id_v[k]['terminal']['reason']}"] += 1
        if b["zombie"]:
            keys = {(r["map_id"], r["matchup"], r["seed"], r["seat"]) for r in b["rows"]}
            support += int(any(differs[k] for k in keys))
            remaining += int(any(by_id_v[k]["sides"][s]["flags"]["zombie"] for k in keys for s in ("first", "second")))
    rep_v = [r for b in blocks_v for r in _representative(b)]
    z_v = _zombie_count(rep_v)
    c_v = sum(1 for b in blocks_v if any(r["terminal"]["reason"] == "economic_collapse" for r in _representative(b)))
    z_0 = sum(b["zombie_windows"] for b in a0_blocks)
    c_0 = sum(1 for b in a0_blocks if any(
        r["terminal"]["reason"] == "economic_collapse" for r in _representative(b)))
    return {
        "variant": variant,
        "rows_trajectory_differs_from_a0": sum(differs.values()), "rows": len(differs),
        "a0_blocks_with_zombie": len(zombie_blocks), "support_S": support, "remaining_R": remaining,
        "zombie_windows_Z_variant": z_v, "zombie_windows_Z_a0": z_0,
        "collapse_blocks_C_variant": c_v, "collapse_blocks_C_a0": c_0,
        "false_trigger_rows": len(false_triggers), "heterogeneous_a0_clusters": heterogeneous,
        "outcome_transitions_over_a0_clusters": dict(sorted(transitions.items())),
        "r3_efficacy": bool(support >= MIN_SUPPORT and remaining <= support // 2 and z_v <= z_0 // 2),
        "r4_safety": bool(not false_triggers and c_v - c_0 <= max(3, math.ceil(0.25 * c_0))),
    }


def decide_verdict(a0_blocks: Sequence[dict[str, Any]], comparisons: dict[str, dict[str, Any]],
                   *, r1: bool, integrity_ok: bool) -> dict[str, Any]:
    """Protocol section 9, applied mechanically."""
    exposure = sum(1 for b in a0_blocks if b["deficit"])
    zombie = [b for b in a0_blocks if b["zombie"]]
    r2 = (len(zombie) >= MIN_ZOMBIE_BLOCKS and len({b["matchup"] for b in zombie}) >= MIN_ZOMBIE_PAIRS
          and len({b["map_id"] for b in zombie}) >= MIN_ZOMBIE_MAPS)
    qualifying = sorted(v for v, c in comparisons.items() if c["r3_efficacy"] and c["r4_safety"])
    supported = [v for v, c in comparisons.items() if c["support_S"] >= MIN_SUPPORT]
    reasons = []
    if not integrity_ok:
        reasons.append("S1: integrity failure")
    if exposure < MIN_EXPOSURE:
        reasons.append(f"S2: exposure E={exposure} < {MIN_EXPOSURE}")
    if r2 and not supported:
        reasons.append("S3: R2 holds but no variant has support S(V) >= 6")
    if reasons:
        verdict = "НЕДОСТАТОЧНО ДАННЫХ"
    elif not r2:
        verdict = "ОСТАВИТЬ"
        reasons.append("R2 not met: the mechanism is not material for Baselines v1 on these maps")
    elif not qualifying:
        verdict = "ОСТАВИТЬ"
        reasons.append("R2 met, but no variant satisfies R3 and R4")
    else:
        verdict = "ИЗМЕНИТЬ"
        reasons.append("R2 met and variants " + ", ".join(qualifying) + " satisfy R3 and R4")
    return {
        "verdict": verdict, "reasons": reasons, "r1_mechanism_reproduced": r1, "r2_material_in_engine": r2,
        "exposure_E": exposure, "a0_zombie_blocks": len(zombie),
        "a0_zombie_matchups": sorted({b["matchup"] for b in zombie}),
        "a0_zombie_maps": sorted({b["map_id"] for b in zombie}),
        "qualifying_variants": qualifying, "supported_variants": sorted(supported),
    }


def build_summary(rows: Sequence[dict[str, Any]], demo: dict[str, Any], *, calls: dict[str, int]) -> dict[str, Any]:
    present = [v for v in VARIANTS if any(r["variant"] == v for r in rows)]
    arms = {v: [r for r in rows if r["variant"] == v] for v in present}
    if "A0" not in arms:
        raise StarvationDebtError("the reference arm A0 is missing")
    a0_blocks = _a0_blocks(arms["A0"])
    comparisons = {v: _compare(v, a0_blocks, arms[v], arms["A0"]) for v in present if v != "A0"}
    integrity_ok = all(
        r["status"] == "complete" and all(s["debt_identity_holds"] for s in _side_items([r])) for r in rows
    ) and all(r.get("anchor_matches_unchanged_engine") for r in arms["A0"]) and not any(
        c["false_trigger_rows"] for c in comparisons.values())
    verdict = decide_verdict(a0_blocks, comparisons, r1=demo["r1_mechanism_reproduced"], integrity_ok=integrity_ok)
    by_cell = {}
    for b in a0_blocks:
        if b["zombie"] or b["deficit"]:
            by_cell[f"{b['map_id']}|{b['matchup']}"] = {
                "deficit": b["deficit"], "zombie_windows": b["zombie_windows"],
                "seeds_in_block": len(b["seeds"]), "block_sha256": b["block_sha256"]}
    return {
        "schema": SCHEMA_SUMMARY, "protocol_id": PROTOCOL_ID,
        "rows": len(rows), "calls": calls,
        "arms": {v: _arm_summary(arms[v]) for v in present},
        "a0_blocks": {"distinct": len(a0_blocks), "with_deficit": sum(1 for b in a0_blocks if b["deficit"]),
                      "with_zombie": sum(1 for b in a0_blocks if b["zombie"]),
                      "cells_with_deficit_or_zombie": dict(sorted(by_cell.items()))},
        "comparisons": comparisons, "integrity_ok": integrity_ok, "verdict": verdict,
    }


# --- stage R: unchanged-engine reproduction -----------------------------------------------------------

def build_reproduction() -> dict[str, Any]:
    case = REPRO_CASE
    config = _configs()[case["map"]]
    out = _play_match(config, case["first"], case["second"], case["seed"], case["swapped"], normalize=True, detail=True)
    row = out.row
    detail = row.pop("_detail")
    first_world = 1 if case["swapped"] else 0
    per100 = {}
    for (side, idx), counter in sorted(detail["per100"].items()):
        name = "first" if side == first_world else "second"
        per100.setdefault(name, []).append({"window": idx, **dict(sorted(counter.items()))})
    archive = {"status": "not_measured"}
    path = _ROOT / ARCHIVE_BASELINES
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            a = json.loads(line)
            if (a["map_id"], a["matchup"], a["seed"], a["seat_assignment"]["A"]) == (
                    case["map"], f"{case['first']}-{case['second']}", case["seed"], case["first"]):
                archive = {
                    "status": "found", "archive_sha256": _file_sha(path),
                    "terminal_tick": a["terminal"]["terminal_tick"], "reason": a["terminal"]["reason"],
                    "final_state_sha256": a["final_state_sha256"], "events_trace_sha256": a["events_trace_sha256"],
                    "starvation_events": a["event_counts"].get("starvation"),
                    "last600_starvation_deaths": a["last_600_completed_ticks"]["starvation_deaths"],
                    "last600_lay_food_rejections": a["last_600_completed_ticks"]["lay_food_rejections"],
                    "reproduced_final_state": a["final_state_sha256"] == row["final_state_sha256"],
                    "reproduced_events_trace": a["events_trace_sha256"] == row["events_trace_sha256"],
                    "reproduced_terminal": (a["terminal"]["terminal_tick"], a["terminal"]["reason"]) ==
                    (row["terminal"]["terminal_tick"], row["terminal"]["reason"]),
                }
    zombie_sides = [k for k in ("first", "second") if row["sides"][k]["flags"]["zombie"] > 0]
    return {
        "schema": SCHEMA_REPRO, "case": case, "unchanged_engine": True, "match": row,
        "per100_windows_by_policy": per100, "first_starvation_events": detail["starvation"],
        "first_debt_resets": detail["resets"], "archive_comparison": archive,
        "example_confirmed": bool(zombie_sides), "zombie_sides": zombie_sides,
        "example_rule": "confirmed iff a side has a zombie_window (protocol section 6)",
    }


# --- artifacts ---------------------------------------------------------------------------------------

def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _jsonl(rows: Sequence[dict[str, Any]]) -> bytes:
    return b"".join(canonical_dumps(r) + b"\n" for r in rows)


def _json_bytes(obj: Any) -> bytes:
    return canonical_dumps(obj) + b"\n"


def _versions() -> dict[str, str]:
    return {
        "rules_version": RULES_VERSION, "balance_version": BALANCE_VERSION, "observation": OBSERVATION_VERSION,
        "action": ACTION_VERSION, "decoder": DECODER_VERSION, "rng": RNG_VERSION,
        "serialization": SERIALIZATION_VERSION, "stream_derivation": STREAM_DERIVATION_VERSION,
    }


def _protected_hashes() -> dict[str, str]:
    return {p: _file_sha(_ROOT / p) for p in PROTECTED_FILES if (_ROOT / p).exists()}


def build_manifest(artifacts: dict[str, str], calls: dict[str, int], *, source_commit: str,
                   rows: int = EXPECTED_ROWS, source_tree_dirty: bool | None = None) -> dict[str, Any]:
    manifest = {
        "schema": SCHEMA_MANIFEST, "protocol_id": PROTOCOL_ID, "protocol_sha256": protocol_sha256(),
        "erratum_sha256": _file_sha(_ROOT / ERRATUM_PATH),
        "base_commit": BASE_COMMIT, "base_engine_fingerprint": BASE_ENGINE_FINGERPRINT,
        "engine_fingerprint_excluding_harness": source_fingerprint(exclude_harness=True),
        "package_fingerprint": source_fingerprint(), "source_commit": source_commit,
        "configs": CONFIG_SHA256, "policy_parameter_sha256": POLICY_PARAMETER_SHA256,
        "seeds": list(SEEDS), "seeds_sha256": SEEDS_SHA256, "variants": list(VARIANTS),
        "window_columns": WINDOW_COLUMNS, "episode_columns": EPISODE_COLUMNS,
        "checkpoint_columns": CHECKPOINT_COLUMNS, "checkpoint_ticks": list(CHECKPOINT_TICKS),
        "pairs": [list(p) for p in PAIRS], "maps": list(MAPS), "seats": ["first_A", "first_B"],
        "expected_rows": rows, "calls": calls,
        "budget": {"main": MAIN_BUDGET, "anchor": ANCHOR_BUDGET},
        "versions": _versions(), "artifacts": artifacts, "protected_file_sha256": _protected_hashes(),
        "commands": {"run": "python -m formicmind.starvation_debt run --out <NEW_DIR> --jobs N",
                     "verify": "python -m formicmind.starvation_debt verify --input results/starvation_debt --jobs N"},
        "python": platform.python_version(), "platform": platform.platform(),
    }
    if source_tree_dirty is not None:  # only new runs record it; the published manifest predates the field
        manifest["source_tree_dirty"] = source_tree_dirty
    return manifest


def _check_static_inputs() -> None:
    """The historical environment the published data belong to; regeneration is meaningful only inside it."""
    if source_fingerprint(exclude_harness=True) != BASE_ENGINE_FINGERPRINT:
        raise StarvationDebtError(f"engine sources differ from the frozen base fingerprint; {HISTORICAL_HINT}")
    if _versions() != FROZEN_VERSIONS:
        raise StarvationDebtError(f"engine versions differ from the frozen versions; {HISTORICAL_HINT}")
    if tick_module.upkeep is not economy_module.upkeep:
        raise StarvationDebtError("tick.upkeep is replaced outside use_variant")
    if protocol_sha256() != PROTOCOL_SHA256 or _file_sha(_ROOT / ERRATUM_PATH) != ERRATUM_SHA256:
        raise StarvationDebtError("the frozen protocol or its erratum changed")
    if _protected_hashes() != FROZEN_PROTECTED_FILE_SHA256:
        raise StarvationDebtError("a protected FOR-20..24 file differs from its frozen hash")
    if hashlib.sha256(canonical_dumps(list(SEEDS))).hexdigest() != SEEDS_SHA256:
        raise StarvationDebtError("seed list drifted")
    for name in POLICIES:
        if _policy_descriptor(name)["parameters_sha256"] != POLICY_PARAMETER_SHA256[name]:
            raise StarvationDebtError(f"policy parameter drift for {name}")
    _configs()


def generate(jobs: int = 1, rows_schedule: Sequence[tuple[Any, ...]] | None = None) -> dict[str, bytes | Any]:
    """Regenerate every result artifact from the engine (used by ``run`` and, strictly, by ``verify``)."""
    _check_static_inputs()
    sched = list(schedule()) if rows_schedule is None else list(rows_schedule)
    demo = build_minimal_demo()
    repro = build_reproduction()
    rows = run_schedule(sched, jobs)
    calls = {"main": sum(r["ticks_executed"] for r in rows), "anchor": sum(r["anchor_calls"] for r in rows)}
    if calls["main"] > MAIN_BUDGET or calls["anchor"] > ANCHOR_BUDGET:
        raise StarvationDebtError("step_tick budget exceeded")
    summary = build_summary(rows, demo, calls=calls)
    if tick_module.upkeep is not economy_module.upkeep:
        raise StarvationDebtError("tick.upkeep was not restored")
    return {"rows": rows, "demo": demo, "repro": repro, "summary": summary, "calls": calls}


def _artifact_bytes(result: dict[str, Any]) -> dict[str, bytes]:
    return {
        "matches.jsonl": _jsonl(result["rows"]), "minimal_demo.json": _json_bytes(result["demo"]),
        "reproduction.json": _json_bytes(result["repro"]), "summary.json": _json_bytes(result["summary"]),
    }


def run(out: str | Path, jobs: int = 1, *, _schedule: Sequence[tuple[Any, ...]] | None = None) -> dict[str, Any]:
    """Write the artifacts into a new directory. ``_schedule`` exists only for the unit tests."""
    out_dir = Path(out)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise StarvationDebtError(f"output directory {out_dir} is not empty")
    result = generate(jobs, _schedule)
    if _schedule is None and len(result["rows"]) != EXPECTED_ROWS:
        raise StarvationDebtError("the full schedule must contain exactly 4320 rows")
    files = _artifact_bytes(result)
    manifest = build_manifest({k: _sha(v) for k, v in files.items()}, result["calls"],
                              source_commit=_git("rev-parse", "HEAD"), rows=len(result["rows"]),
                              source_tree_dirty=bool(_git("status", "--porcelain", "--", "src")))
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (out_dir / name).write_bytes(data)
    (out_dir / "manifest.json").write_bytes(_json_bytes(manifest))
    return manifest


# --- strict verification --------------------------------------------------------------------------------

VERIFICATION_SCHEMA = "formicmind.starvation-debt-verification/v1"
# Provenance of the *historical* generation run: environment-dependent or tied to the commit that produced the data.
# They are never required to equal the verifier's own environment, HEAD or source tree, and they are never rewritten.
PROVENANCE_KEYS = frozenset({"source_commit", "package_fingerprint", "source_tree_dirty", "python", "platform",
                             "commands"})
# Checked against the regenerated data instead of a recomputed constant.
DERIVED_KEYS = frozenset({"artifacts", "calls"})

# The result-defining part of the harness (rules, runner, metrics, summary, serialisation and the frozen constants)
# is pinned by hash of the version that generated the published data (source commit 4c2cbeb...). Verifier-only
# changes (manifest checks, provenance, CLI) may evolve; any change to the pinned part is a deliberate update.
CORE_FUNCTIONS = (
    "variant_upkeep", "use_variant", "_toy_state", "_side0", "run_single_tick", "run_minimal", "_tenths_rounding",
    "build_minimal_demo", "_snapshot", "_flags_for_windows", "_episodes", "_outcome_object", "_play_match",
    "_configs", "row_id", "execute_row", "schedule", "_group_blocks", "_representative", "_side_items", "_tally",
    "_arm_summary", "_zombie_count", "_a0_blocks", "_compare", "decide_verdict", "build_summary",
    "build_reproduction", "_jsonl", "_json_bytes", "generate", "_artifact_bytes", "run_schedule", "_job",
    "VariantContext", "_SideTrace",
)
CORE_CONSTANTS = (
    "PROTOCOL_ID", "BASE_COMMIT", "BASE_ENGINE_FINGERPRINT", "VARIANTS", "A3_STREAK", "MAPS", "POLICIES", "PAIRS",
    "SEEDS", "SEATS", "SEEDS_SHA256", "CONFIG_SHA256", "POLICY_PARAMETER_SHA256", "CHECKPOINT_TICKS", "WINDOW",
    "CHRONIC_DEFICIT_TICKS", "ZOMBIE_ADULT_DELTA", "EPISODE_GAP", "RECOVERY_HORIZON", "RECOVERY_SPAN",
    "MIN_ZOMBIE_BLOCKS", "MIN_ZOMBIE_PAIRS", "MIN_ZOMBIE_MAPS", "MIN_EXPOSURE", "MIN_SUPPORT", "REPRO_CASE",
    "EXPECTED_ROWS", "MAIN_BUDGET", "ANCHOR_BUDGET", "MINIMAL_SCENARIOS", "MINIMAL_WORKERS", "EDGE_CASES",
    "WINDOW_COLUMNS", "EPISODE_COLUMNS", "CHECKPOINT_COLUMNS",
)
GENERATOR_CORE_SHA256 = "332bd3d78c9430920f38b2b6384bf11a09a4264cd13ffc5c456d3ec888b603b1"
# The harness that generated the published data (commit 4c2cbeb...) and its package fingerprint, for documentation.
PUBLISHED_GENERATOR_COMMIT = "4c2cbeb4a947865914118441cbe48a9148600df7"
PUBLISHED_GENERATOR_FINGERPRINT = "f8263c407a0509cece3206e139d5d8a6e7c800c21ee33500226f64dd6c2c2ca1"

# Deterministic manifest fields are compared with these frozen literals, never with the current tree, so that a
# manifest can be checked on any checkout (including one whose engine, versions or documents have moved on).
FROZEN_VERSIONS = {
    "rules_version": "l0-proto-FOR1-v0.2.2", "balance_version": "candidate-FOR1-v0.2.1-unvalidated",
    "observation": "FOR-2-v0.2+errata-v0.2.1", "action": "ActionV2", "decoder": "FOR-2-decoder-v0.2.1",
    "rng": "pcg32-xsh-rr-64-32-v1", "serialization": "formicmind-canonical-json-v1",
    "stream_derivation": "sha256-lenprefixed-v1",
}
FROZEN_PROTECTED_FILE_SHA256 = {
    "docs/economy_pilot_protocol.md": "57f33e613599542d89f8a5a6ea8c5d0a33f29022564b056fb2c0a93bd87d93d0",
    "docs/gate1_causal_protocol.md": "a87cf2f399375c7841ae10f0fb34ee14b2f75c62f0b70400ad04f40684a39c22",
    "docs/gate1_context_protocol.md": "b0eb1685a2d0e024e915f42d7833e6f27fc9be4add4c26ceac921b77f07edff3",
    "docs/gate1_feedback_protocol.md": "98b166ad5fd31451990f7aaf49a7fc23c34d5ba10a4193a37d5d992d244eca4b",
    "docs/gate1_stalling_findings.md": "005da6338d901fb949d68165069532ed786715f015b24d1e7152292270d8f505",
    "docs/gate1_stalling_protocol.md": "0d54603c90526f7d51705f5fcf7c70290d854324be5eb73297932810ec1f3b5d",
    "results/economy_pilot/runs.json": "f761f0dbdccebd48c13599638a43c456a2f66fae8891b7efbbb14e30e146ae11",
    "results/gate1_context/matches.jsonl": "cb895bfe85d2aa641c30a05ee300f35e63e9b7b90e580e5f8f1a59c20859b7b4",
    "results/gate1_stalling/baselines.jsonl": "ebd37e04504101c24e7744a1a461e18c38c67c2873f2b9b255270e657a3c4ba6",
}
HISTORICAL_HINT = ("full historical verification needs an isolated checkout of the pinned FOR-25 commit (see "
                   "docs/starvation_debt_verification.md and the historical-for25 CI job); "
                   "`verify --manifest-only` works on any checkout")


def experiment_core_sha256() -> str:
    """Hash of the result-defining sources and constants of this harness (CRLF-insensitive)."""
    module = sys.modules[__name__]
    code = [[name, _sha(inspect.getsource(getattr(module, name)).replace("\r\n", "\n").encode("utf-8"))]
            for name in CORE_FUNCTIONS]
    constants = [[name, repr(getattr(module, name))] for name in CORE_CONSTANTS]
    return _sha(canonical_dumps({"code": code, "constants": constants}))


def fingerprint_at_commit(commit: str) -> str:
    """The package fingerprint of ``src/formicmind`` as committed at ``commit`` (same algorithm as the tree one)."""
    names = [n for n in _git("ls-tree", "-r", "--name-only", commit, "src/formicmind").splitlines()
             if n.endswith(".py")]
    entries = []
    for name in sorted(names, key=lambda n: Path(n).parts):
        blob = subprocess.run(["git", "show", f"{commit}:{name}"], cwd=_ROOT, check=True, capture_output=True).stdout
        entries.append({"path": name, "sha256": _sha(_lf(blob))})
    return _sha(canonical_dumps(entries))


def _commit_available(commit: str) -> bool:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        return False
    return subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=_ROOT,
                          capture_output=True).returncode == 0


def check_generator_binding(manifest: dict[str, Any]) -> str:
    """Relate the historical ``package_fingerprint`` to ``source_commit`` without touching either value.

    ``bound``: the commit is available and its sources hash to the recorded fingerprint. A mismatch is an error.
    ``not_checked_dirty_worktree``: the run declared uncommitted sources, so no commit can match.
    ``commit_unavailable``: the commit is not in this clone (for example a shallow CI checkout); not authenticated.
    """
    commit, fingerprint = manifest.get("source_commit"), manifest.get("package_fingerprint")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise StarvationDebtError("manifest source_commit is not a lowercase commit SHA")
    if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise StarvationDebtError("manifest package_fingerprint is not a SHA-256")
    if manifest.get("source_tree_dirty") is True:
        return "not_checked_dirty_worktree"
    if not _commit_available(commit):
        return "commit_unavailable"
    if fingerprint_at_commit(commit) != fingerprint:
        raise StarvationDebtError("manifest package_fingerprint does not match the sources at source_commit")
    return "bound"


def expected_frozen_manifest(rows: int, source_commit: str = "") -> dict[str, Any]:
    """The manifest the frozen study must carry. Tree-derived values are replaced by frozen literals, so the result
    is independent of the current checkout (engine, versions, documents); provenance and derived keys are placeholders."""
    expected = build_manifest({}, {}, source_commit=source_commit, rows=rows)
    expected.update({
        "protocol_sha256": PROTOCOL_SHA256, "erratum_sha256": ERRATUM_SHA256,
        "engine_fingerprint_excluding_harness": BASE_ENGINE_FINGERPRINT, "versions": dict(FROZEN_VERSIONS),
        "protected_file_sha256": dict(FROZEN_PROTECTED_FILE_SHA256),
    })
    return expected


def check_manifest_fields(manifest: dict[str, Any], rows: int) -> list[str]:
    """Every deterministic manifest field must equal its frozen value (no missing or extra keys)."""
    expected = expected_frozen_manifest(rows, str(manifest.get("source_commit", "")))
    skip = PROVENANCE_KEYS | DERIVED_KEYS
    got = {k: v for k, v in manifest.items() if k not in skip}
    want = {k: v for k, v in expected.items() if k not in skip}
    missing, extra = sorted(set(want) - set(got)), sorted(set(got) - set(want))
    different = sorted(k for k in want if k in got and canonical_dumps(got[k]) != canonical_dumps(want[k]))
    if missing or extra or different:
        raise StarvationDebtError(
            f"manifest deterministic fields differ (missing {missing}, unexpected {extra}, different {different})")
    return sorted(want)


def verify(input_path: str | Path, jobs: int = 1, *, _schedule: Sequence[tuple[Any, ...]] | None = None,
           record: str | Path | None = None, manifest_only: bool = False) -> dict[str, Any]:
    """Strict verification: check every deterministic manifest field, regenerate everything with the engine, compare.

    Compatibility: the verifier need not be the harness version that generated the data. It accepts a manifest whose
    historical provenance fields (``PROVENANCE_KEYS``) differ from the current environment, tree and HEAD, provided
    (a) all other manifest fields equal freshly computed values, (b) the result-defining core of this harness equals
    the pinned core of the generator, (c) regeneration reproduces every artifact byte for byte, and (d) the recorded
    fingerprint is consistent with ``source_commit`` whenever that commit is available.

    ``manifest_only=True`` performs only the checkout-independent part (manifest fields, provenance binding, artifact
    hashes, harness core) and says so in its result; it does not prove reproducibility. The full mode regenerates
    every artifact and therefore refuses, explicitly, outside the historical environment (``_check_static_inputs``).
    """
    root = Path(input_path)
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StarvationDebtError(f"cannot read manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise StarvationDebtError("manifest is not an object")
    if experiment_core_sha256() != GENERATOR_CORE_SHA256:
        raise StarvationDebtError("the result-defining harness core differs from the pinned generator core")
    rows_expected = EXPECTED_ROWS if _schedule is None else len(_schedule)
    checked = check_manifest_fields(manifest, rows_expected)  # before any (slow) regeneration
    if manifest.get("expected_rows") != rows_expected:
        raise StarvationDebtError("row count differs from the protocol")
    binding = check_generator_binding(manifest)
    expected_names = {"matches.jsonl", "minimal_demo.json", "reproduction.json", "summary.json"}
    if not isinstance(manifest.get("artifacts"), dict) or set(manifest["artifacts"]) != expected_names:
        raise StarvationDebtError("manifest artifact list differs from the protocol")
    if not isinstance(manifest.get("calls"), dict):
        raise StarvationDebtError("manifest calls are missing")
    for name, digest in manifest["artifacts"].items():
        if _sha(_lf((root / name).read_bytes())) != digest:  # a Windows checkout may have converted LF to CRLF
            raise StarvationDebtError(f"artifact {name} does not match its manifest hash")
    if manifest_only:
        outcome = {
            "ok": True, "mode": "manifest_only", "rows": manifest["expected_rows"], "verification_calls": None,
            "generator_binding": binding, "deterministic_fields_checked": checked,
            "reproducibility": "not_verified_in_this_mode",
        }
        if record is not None:
            Path(record).write_bytes(_json_bytes(_verification_record(manifest, outcome)))
        return outcome
    _check_static_inputs()
    result = generate(jobs, _schedule)
    files = _artifact_bytes(result)
    for name, data in files.items():
        if _lf((root / name).read_bytes()) != data:
            raise StarvationDebtError(f"regenerated {name} differs from the saved artifact")
    if manifest["calls"] != result["calls"]:
        raise StarvationDebtError("recorded call counts differ")
    outcome = {
        "ok": True, "mode": "full_regeneration", "rows": len(result["rows"]),
        "verification_calls": result["calls"], "generator_binding": binding,
        "deterministic_fields_checked": checked,
    }
    if record is not None:
        Path(record).write_bytes(_json_bytes(_verification_record(manifest, outcome)))
    return outcome


def _verification_record(manifest: dict[str, Any], outcome: dict[str, Any]) -> dict[str, Any]:
    """Who verified what: the generator's real provenance next to the (different) verifier identity."""
    return {
        "schema": VERIFICATION_SCHEMA, "result": "ok", "mode": outcome["mode"], "rows": outcome["rows"],
        "verification_calls": outcome["verification_calls"],
        "manifest_sha256": _sha(canonical_dumps(manifest) + b"\n"),
        "artifact_sha256": manifest["artifacts"],
        "generator": {
            "source_commit": manifest["source_commit"], "package_fingerprint": manifest["package_fingerprint"],
            "core_sha256": GENERATOR_CORE_SHA256, "binding_to_source_commit": outcome["generator_binding"],
            "recorded_in": "manifest.json (unchanged since generation)",
        },
        "verifier": {
            "package_fingerprint": source_fingerprint(), "core_sha256": experiment_core_sha256(),
            "engine_fingerprint_excluding_harness": source_fingerprint(exclude_harness=True),
            "same_package_as_generator": source_fingerprint() == manifest["package_fingerprint"],
            "engine_matches_frozen_base": source_fingerprint(exclude_harness=True) == BASE_ENGINE_FINGERPRINT,
        },
        "deterministic_fields_checked": outcome["deterministic_fields_checked"],
        "provenance_fields_not_compared": sorted(PROVENANCE_KEYS),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="formicmind.starvation_debt")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--out" if name == "run" else "--input", required=True, dest="path")
        p.add_argument("--jobs", type=int, default=1)
        if name == "verify":
            p.add_argument("--record", default=None, help="write a canonical verification record to this path")
            p.add_argument("--manifest-only", action="store_true", dest="manifest_only",
                           help="check manifest fields, provenance and artifact hashes only (any checkout)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            manifest = run(args.path, args.jobs)
            print(json.dumps({"ok": True, "rows": manifest["expected_rows"], "calls": manifest["calls"]}))
        else:
            print(json.dumps(verify(args.path, args.jobs, record=args.record, manifest_only=args.manifest_only)))
    except StarvationDebtError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
