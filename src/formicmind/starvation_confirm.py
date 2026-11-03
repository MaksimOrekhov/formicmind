"""FOR-35 study harness: independent confirmation of the starvation-debt rule (experimental, isolated).

Nothing here changes the L0 rules. The candidate rules (A1a, A1b, A2, A3b) and a copy of the current rule (A0)
live only in this module and are enabled only inside ``with use_variant(...)`` by swapping ``formicmind.tick.upkeep``;
``economy.py`` and the defaults are untouched and the swap is always undone. The module is written independently of the
FOR-25 harness ``starvation_debt.py`` and does not import it. Frozen documents: ``docs/starvation_confirm_protocol.md``
and the errata ``docs/starvation_confirm_erratum_*.md``.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import itertools
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from . import economy as economy_module
from . import tick as tick_module
from .canonical import canonical_dumps
from .economy import _add_window, _adults, starvation_victim_key
from .evaluation import controller_pair
from .events import Event, Phase, StarvationEvent, UpkeepEvent, events_hash
from .gate1_causal import _MirrorNormalizer
from .gate1_context import _build_configs
from .intel import home_node
from .ledger import total_food
from .model import NOWHERE, SimState, Unit
from .reset import reset
from .rules import LAY_NONE, SIDES, SOLDIER, WORKER
from .runstate import RunState
from .serialization import checkpoint_hash, config_hash, state_hash
from .tick import PolicyController, step_tick
from .baselines import baseline_spec
from .versions import (
    ACTION_VERSION, BALANCE_VERSION, DECODER_VERSION, OBSERVATION_VERSION, RNG_VERSION,
    RULES_VERSION, SERIALIZATION_VERSION, STREAM_DERIVATION_VERSION,
)

PROTOCOL_ID = "formicmind.starvation-confirm/v1"
PROTOCOL_PATH = Path("docs/starvation_confirm_protocol.md")
ERRATA = {  # number -> (path, SHA-256 with CRLF normalised to LF)
    1: (Path("docs/starvation_confirm_erratum_1.md"),
        "a8782cb3e099933032c5843661530b3b274abffa96718d961cfb7f82634e387b"),
    2: (Path("docs/starvation_confirm_erratum_2.md"),
        "314a0b2fbdc237344a51328605ca8706167584edd5ebfcd8f7e741424fb654fd"),
}
PROTOCOL_SHA256 = "445df9d9b499abee5124cbf467b10e0a82626d871129289c75041145265deeda"
PROTOCOL_COMMIT = "012bfe618fa33ff9958cd4ac982c30fd3d88e71c"
BASE_COMMIT = "0bb12ad0978f0bd7c72fa67066f3f8af53f449cb"
FOR25_MERGE_COMMIT = "720e2d7cd3465afbc74e1fe7a1849f1c5b56f334"
BASE_ENGINE_FINGERPRINT = "ff8b73e4a0a2cbd5f7638e3b56bc590a76dc427c36ff19aac11337dd02ca0611"
HARNESS_FILES = ("starvation_debt.py", "starvation_confirm.py")  # excluded from the engine fingerprint
SCHEMA_MANIFEST = "formicmind.starvation-confirm-manifest/v1"
SCHEMA_ROW = "formicmind.starvation-confirm-match/v1"
SCHEMA_SUMMARY = "formicmind.starvation-confirm-summary/v1"
SCHEMA_DEMO = "formicmind.starvation-confirm-demo/v1"
SCHEMA_ANCHORS = "formicmind.starvation-confirm-anchors/v1"
OUT_RELATIVE = Path("results/starvation_confirm")

VARIANTS = ("A0", "A1a", "A1b", "A2", "A3b")
CANDIDATES = VARIANTS[1:]
A3_STREAK = {"A3b": 10}
MAPS = ("R0-T0", "R0-T1", "R0-T2", "R1-T0", "R1-T1", "R1-T2")
POLICIES = ("random", "turtle", "rush", "balanced", "raider")
POLICY_VERSION = "v2"
PAIRS = tuple(itertools.combinations(POLICIES, 2))
SEEDS = tuple(range(20270501, 20270509))
SEATS = (False, True)  # swapped?  first policy on A, then first policy on B
SEEDS_SHA256 = "fd0105d4e2a1c7c8038e1576b4a54cb4f62bed8caa53be0862e2a54a334c5f99"
SEATS_SHA256 = "12e5f2013089bb9cd19561e998b89853deb9eb1b4e8e53dcf1bf4d75d389924b"
CONFIG_SHA256 = {
    "R0-T0": "3f6aaa36a0d3d15888fa42f142df1151376794e62b553ab8e5f7659c9794145d",
    "R0-T1": "78e9c6979a2fb3973ae9a275f79ad5b62cb4b6ed3893a8f8469e4cc599e5fb34",
    "R0-T2": "b9e43b2aec34f303f371a9f9e5a2bcd8b5c5d894e608a3ca9e71386339f632f1",
    "R1-T0": "407bb3907099533adfe10b7c300a480b45b5d604dc019693b1ac99f47502303f",
    "R1-T1": "3bf8dabf6011ed88275c3fcfec3875c4da46ef849e7da7b0ac17bb2cd0e9e605",
    "R1-T2": "1b6f1f999246711919fb4dee9238e5fed31c5baac0dca779ea76c47afe167ceb",
}
POLICY_PARAMETER_SHA256 = {
    "random": "81f737be30b6e6a4e88d678c4924c0872835e4b2532bfbdc9a90b3c7fe5b016d",
    "turtle": "89475d3a782da8c5991ea69d9bc536a3733555f6b21ddcd1ae2a56c35ad40f8c",
    "rush": "ca4674d7c4ab7a5c9ab710a4e0f3a87556780de80995af3369a082615f6ee751",
    "balanced": "82b77f86dd0faa6ae7609ad52790b483462d4652817c55855947aa3afdc882a2",
    "raider": "2ed300ef8187697ac05e3ebf9f2647db18e1f72f1fa10a0bf947dda785575be3",
}
PROFILE_TICKS = (0, 100, 300, 600, 900, 1500, 2100, 2700)
TIMELINE_ROWS = (("R0-T0", "turtle", "raider", 20270501, False), ("R0-T0", "turtle", "balanced", 20270501, False))
T_MAX = 3000
WINDOW = 300
N_WINDOWS = 10
POP_DELTA = 2          # |A_end - A_start| <= 2 and the new-peak margin
CHURN_MIN = 3          # hatches >= 3 and deaths >= 3 in one window
DEFICIT_TICKS = 30     # >= 30 ticks with missing > 0 in a window
PERSISTENT_WINDOWS = 5
EPISODE_GAP = 30
RECOVERY_HORIZON = 300
RECOVERY_SPAN = 100
EXPECTED_ROWS = len(VARIANTS) * len(MAPS) * len(PAIRS) * len(SEEDS) * len(SEATS)  # 4800
EXPECTED_ANCHORS = len(MAPS) * len(PAIRS) * len(SEEDS) * len(SEATS)  # 960
MAIN_BUDGET = EXPECTED_ROWS * T_MAX
ANCHOR_BUDGET = EXPECTED_ANCHORS * T_MAX
MIN_EXPOSURE, MIN_EXPOSURE_PAIRS, MIN_EXPOSURE_MAPS, MIN_EXPOSURE_CELLS = 6, 2, 2, 3
MIN_A0_CLUSTERS = 120
_ROOT = Path(__file__).resolve().parents[2]
STUDY_FILE_PATTERNS = (
    r"^docs/starvation_confirm_.*", r"^results/starvation_confirm.*", r"^src/formicmind/starvation_confirm\.py$",
    r"^tests/test_starvation_confirm.*", r"^scripts/starvation_confirm_.*",
)
REFERENCE_FILES = ("README.md", "docs/reliability.md")  # pointers only; not protected


class StarvationConfirmError(ValueError):
    """Invalid fixed-study input, integrity failure or strict verification mismatch."""


# --- hashing helpers --------------------------------------------------------------------------------

def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _lf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def _file_sha(path: Path) -> str:
    return _sha(_lf(path.read_bytes()))


def _canon_sha(value: Any) -> str:
    return _sha(canonical_dumps(value))


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


def source_fingerprint(*, exclude_harness: bool = False, root: Path | None = None) -> str:
    base = root or _ROOT
    entries = [
        {"path": path.relative_to(base).as_posix(), "sha256": _sha(_lf(path.read_bytes()))}
        for path in sorted((base / "src" / "formicmind").rglob("*.py"))
        if not (exclude_harness and path.name in HARNESS_FILES)
    ]
    return _canon_sha(entries)


def _is_study_file(rel: str) -> bool:
    return any(re.match(p, rel) for p in STUDY_FILE_PATTERNS)


# --- candidate rules --------------------------------------------------------------------------------

def debt_rule(variant: str, *, due: int, food: int, debt: int, streak: int) -> tuple[int, int, bool, int]:
    """Rule applied on a tick with ``missing == 0``.

    ``food`` is the stock left after paying the upkeep. Returns ``(debt, food, reset, streak)``. ``reset`` is true
    when the variant erased the debt (the food is untouched); A2 repays from the stock instead and never resets.
    """
    if variant not in VARIANTS:
        raise StarvationConfirmError(f"unknown variant {variant!r}")
    if due > 0 and variant in A3_STREAK:
        streak += 1
    if not debt:
        return debt, food, False, streak
    if variant == "A0":
        clear = due > 0
    elif variant == "A1a":
        clear = due > 0 and food >= 1
    elif variant == "A1b":
        clear = due > 0 and food >= due
    elif variant in A3_STREAK:
        clear = due > 0 and streak >= A3_STREAK[variant]
        if clear:
            streak = 0
    else:  # A2: leftover food repays the debt and is destroyed (also when nothing was due)
        repay = min(debt, food)
        return debt - repay, food - repay, False, streak
    return (0, food, True, streak) if clear else (debt, food, False, streak)


@dataclass(slots=True)
class VariantContext:
    """Per-match experimental state that is deliberately *not* part of the game state."""

    variant: str
    streak: list[int] = field(default_factory=lambda: [0, 0])
    calls: int = 0


def variant_upkeep(ctx: VariantContext, state: SimState, run: RunState, phase: int = Phase.UPKEEP
                   ) -> list[UpkeepEvent | StarvationEvent]:
    """``economy.upkeep`` with the debt rule of ``ctx.variant`` applied after a fully paid tick."""
    ctx.calls += 1
    tick = state.tick
    bal = state.config.balance
    cost = bal.starvation_unit_cost
    upkeeps: list[UpkeepEvent] = []
    deaths: list[StarvationEvent] = []
    for side in SIDES:
        colony = state.colonies[side]
        workers, soldiers = _adults(colony, WORKER), _adults(colony, SOLDIER)
        rem_before = colony.upkeep_remainder10
        due, rem_after = divmod(rem_before + bal.worker_upkeep_tenths * workers
                                + bal.soldier_upkeep_tenths * soldiers, 10)
        food_before = colony.food
        paid = min(due, food_before)
        missing = due - paid
        colony.food = food_before - paid
        colony.upkeep_remainder10 = rem_after
        food_after_pay = colony.food
        debt_before = debt = colony.starvation_debt
        reset_flag = False
        victims = 0
        if missing:
            ctx.streak[side] = 0
            debt += missing
            home = home_node(state, side)
            order = sorted(colony.units, key=lambda u: starvation_victim_key(home, u))
            while debt >= cost and order:
                victim = order.pop(0)
                before = debt
                debt -= cost
                colony.units.remove(victim)
                deaths.append(StarvationEvent(
                    tick=tick, phase=int(phase), side=side, seq=victims, unit_id=victim.unit_id,
                    caste=victim.caste, hp=victim.hp, node=victim.node, to_node=victim.to_node,
                    debt_before=before, debt_after=debt))
                victims += 1
        else:
            debt, colony.food, reset_flag, ctx.streak[side] = debt_rule(
                ctx.variant, due=due, food=colony.food, debt=debt, streak=ctx.streak[side])
        colony.starvation_debt = debt
        _add_window(run, side, upkeep_paid=paid, debt_delta=debt - debt_before)
        upkeeps.append(UpkeepEvent(
            tick=tick, phase=int(phase), side=side, adult_workers=workers, adult_soldiers=soldiers,
            remainder_before=rem_before, remainder_after=rem_after, due=due, paid=paid, missing=missing,
            food_before=food_before, food_after=food_after_pay, debt_before=debt_before, debt_after=debt,
            deaths=victims, debt_reset=reset_flag))
    return [*upkeeps, *deaths]


@contextlib.contextmanager
def use_variant(variant: str) -> Iterator[VariantContext]:
    """Swap ``tick.upkeep`` for a candidate rule for the duration of the block; always restores it."""
    if variant not in VARIANTS:
        raise StarvationConfirmError(f"unknown variant {variant!r}")
    if tick_module.upkeep is not economy_module.upkeep:
        raise StarvationConfirmError("tick.upkeep is already replaced; variants cannot be nested")
    ctx = VariantContext(variant)
    tick_module.upkeep = functools.partial(variant_upkeep, ctx)
    try:
        yield ctx
    finally:
        tick_module.upkeep = economy_module.upkeep


# --- stage M: minimal scenarios and edge cases ------------------------------------------------------

MINIMAL_SCENARIOS = {
    "M1": {"income": (11, 11, 14), "ticks": 600},
    "M2": {"income": (12, 12, 12), "ticks": 600},
    "M3": {"income": (13, 13, 13), "ticks": 600},
    "M4": {"income": (11, 11, 11), "ticks": 600},
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


def _side0(events: Sequence[Any]) -> tuple[UpkeepEvent, list[StarvationEvent]]:
    up = next(e for e in events if e.kind == "upkeep" and e.side == 0)
    return up, [e for e in events if e.kind == "starvation" and e.side == 0]


def single_tick(variant: str | None, *, workers: int, soldiers: int = 0, food: int = 0, debt: int = 0,
                rem: int = 0, streak: int = 0, cost: int | None = None) -> dict[str, Any]:
    """One real upkeep call of side A; ``variant=None`` calls the unchanged ``economy.upkeep``."""
    state, run = _toy_state(workers, soldiers, food=food, debt=debt, rem=rem, cost=cost)
    state.tick = 5
    if variant is None:
        events = economy_module.upkeep(state, run)
    else:
        events = variant_upkeep(VariantContext(variant, [streak, 0]), state, run)
    up, dead = _side0(events)
    colony = state.colonies[0]
    return {"due": up.due, "paid": up.paid, "missing": up.missing, "food_after": colony.food,
            "debt_before": up.debt_before, "debt_after": colony.starvation_debt, "deaths": len(dead),
            "debt_reset": up.debt_reset, "adults_left": len(colony.units), "remainder_after": colony.upkeep_remainder10}


EDGE_CASES: dict[str, dict[str, Any]] = {
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
    """M1-M4: a fixed income table feeds the real upkeep rule; A0 is checked against ``economy.upkeep``."""
    spec = MINIMAL_SCENARIOS[scenario]
    income = spec["income"]
    state, run = _toy_state(MINIMAL_WORKERS, 0, food=0, debt=0, rem=0)
    ref_state, ref_run = _toy_state(MINIMAL_WORKERS, 0, food=0, debt=0, rem=0)
    ctx = VariantContext(variant)
    colony, ref = state.colonies[0], ref_state.colonies[0]
    totals: Counter[str] = Counter()
    digest = _Digest()
    table: list[list[int]] = []
    max_debt = 0
    for t in range(spec["ticks"]):
        state.tick = ref_state.tick = t
        colony.food += income[t % len(income)]
        ref.food += income[t % len(income)]
        events = variant_upkeep(ctx, state, run)
        up, dead = _side0(events)
        if variant == "A0":
            ref_events = economy_module.upkeep(ref_state, ref_run)
            if [e.to_canonical() for e in ref_events] != [e.to_canonical() for e in events]:
                raise StarvationConfirmError(f"A0 differs from economy.upkeep at tick {t} of {scenario}")
        forgiven = up.debt_before if up.debt_reset else 0
        repaid = (up.debt_before - up.debt_after) if (not up.missing and not up.debt_reset) else 0
        totals.update({"due": up.due, "paid": up.paid, "missing": up.missing, "forgiven": forgiven,
                       "repaid": repaid, "deaths": len(dead), "deficit_ticks": int(up.missing > 0)})
        max_debt = max(max_debt, up.debt_after, up.debt_before + up.missing)
        digest.add([t, up.due, up.paid, up.missing, up.food_before, up.food_after, colony.food,
                    up.debt_before, up.debt_after, len(dead), int(up.debt_reset)])
        if t < 12:
            table.append([t, up.due, up.paid, up.missing, up.food_after, up.debt_before, up.debt_after,
                          len(dead), int(up.debt_reset)])
    identity = colony.starvation_debt == (totals["missing"] - state.config.balance.starvation_unit_cost
                                         * totals["deaths"] - totals["forgiven"] - totals["repaid"])
    return {"scenario": scenario, "variant": variant, "income_cycle": list(income), "ticks": spec["ticks"],
            "workers_start": MINIMAL_WORKERS, "totals": {k: totals[k] for k in sorted(totals)},
            "adults_end": len(colony.units), "debt_end": colony.starvation_debt, "food_end": colony.food,
            "max_debt": max_debt, "debt_identity_holds": identity, "tick_table_sha256": digest.hexdigest(),
            "first_ticks_columns": ["tick", "due", "paid", "missing", "food_after_pay", "debt_before",
                                    "debt_after", "deaths", "debt_reset"],
            "first_ticks": table}


def build_minimal_demo() -> dict[str, Any]:
    scenarios = {s: {v: run_minimal(v, s) for v in VARIANTS} for s in MINIMAL_SCENARIOS}
    edges: dict[str, Any] = {}
    for name, setup in EDGE_CASES.items():
        production = single_tick(None, **setup)
        edges[name] = {"setup": setup, "production": production,
                       "variants": {v: single_tick(v, **setup) for v in VARIANTS}}
        if edges[name]["variants"]["A0"] != production:
            raise StarvationConfirmError(f"A0 differs from economy.upkeep on edge case {name}")
    m1 = scenarios["M1"]["A0"]["totals"]
    k1 = m1["deaths"] == 0 and m1["missing"] > 0 and m1["forgiven"] == m1["missing"]
    k3 = {}
    for v in CANDIDATES:
        t = scenarios["M1"][v]["totals"]
        k3[v] = t["forgiven"] == 0 and (t["deaths"] > 0 or t["repaid"] > 0)
    return {"schema": SCHEMA_DEMO, "k1_mechanism_reproduced": k1, "k3_mechanism_removed": k3,
            "scenarios": scenarios, "edge_cases": edges}


# --- windows and the primary stagnation measure ------------------------------------------------------

SIDE_WINDOW_FIELDS = ("a_start", "a_end", "n_start", "n_end", "n_max", "peak_before", "hatches", "starvation_deaths",
                      "combat_deaths", "deficit_ticks", "missing", "forgiven", "repaid", "pop_flat", "churn",
                      "new_peak", "stalled", "deficit", "zombie", "queen_end")
MATCH_WINDOW_FIELDS = ("conflict", "stagnant", "stagnant_deficit", "stagnant_sat")


@dataclass(slots=True)
class SideSeries:
    """Per-colony series. ``adults``/``brood``/``queen`` are indexed by completed ticks ``m`` (0..n_c);
    the other lists are indexed by executed tick ``t``."""

    adults: list[int] = field(default_factory=list)
    brood: list[int] = field(default_factory=list)
    queen: list[int] = field(default_factory=list)
    due: list[int] = field(default_factory=list)
    paid: list[int] = field(default_factory=list)
    missing: list[int] = field(default_factory=list)
    forgiven: list[int] = field(default_factory=list)
    repaid: list[int] = field(default_factory=list)
    starvation_deaths: list[int] = field(default_factory=list)
    combat_deaths: list[int] = field(default_factory=list)
    hatches: list[int] = field(default_factory=list)


def analyze_side_windows(s: SideSeries, completed: int) -> list[list[int] | None]:
    """Window facts of one colony; ``None`` for windows not fully completed (never zero)."""
    n = [a + b for a, b in zip(s.adults, s.brood)]
    prefix: list[int] = []
    best = -1
    for value in n:
        best = max(best, value)
        prefix.append(best)
    out: list[list[int] | None] = []
    for k in range(N_WINDOWS):
        lo, hi = WINDOW * k, WINDOW * k + WINDOW
        if hi > completed:
            out.append(None)
            continue
        a0, a1, n0, n1 = s.adults[lo], s.adults[hi], n[lo], n[hi]
        nmax, peak = max(n[lo + 1:hi + 1]), prefix[lo]
        hatches, dst, dcb = sum(s.hatches[lo:hi]), sum(s.starvation_deaths[lo:hi]), sum(s.combat_deaths[lo:hi])
        deficit_ticks = sum(1 for m in s.missing[lo:hi] if m > 0)
        missing, forgiven, repaid = sum(s.missing[lo:hi]), sum(s.forgiven[lo:hi]), sum(s.repaid[lo:hi])
        pop_flat = abs(a1 - a0) <= POP_DELTA
        churn = hatches >= CHURN_MIN and dst + dcb >= CHURN_MIN
        new_peak = nmax > peak + POP_DELTA
        queen_end = bool(s.queen[hi])
        stalled = queen_end and not new_peak and (pop_flat or churn)
        deficit = deficit_ticks >= DEFICIT_TICKS
        zombie = deficit and forgiven >= 1 and 2 * forgiven >= missing and pop_flat and queen_end
        out.append([a0, a1, n0, n1, nmax, peak, hatches, dst, dcb, deficit_ticks, missing, forgiven, repaid,
                    int(pop_flat), int(churn), int(new_peak), int(stalled), int(deficit), int(zombie), int(queen_end)])
    return out


def analyze_match_windows(first: list[list[int] | None], second: list[list[int] | None],
                          conflict: Sequence[int], completed: int) -> list[list[int] | None]:
    """Match-level windows from the two colonies' window facts. The primary stagnation flag reads neither
    ``forgiven`` nor ``debt_reset`` nor the debt: only population, combat and the deficit-tick count."""
    idx = {name: i for i, name in enumerate(SIDE_WINDOW_FIELDS)}
    out: list[list[int] | None] = []
    for k in range(N_WINDOWS):
        a, b = first[k], second[k]
        if a is None or b is None:
            out.append(None)
            continue
        lo = WINDOW * k
        clash = int(any(conflict[lo:lo + WINDOW]))
        stagnant = int(not clash and a[idx["stalled"]] and b[idx["stalled"]]
                       and a[idx["queen_end"]] and b[idx["queen_end"]])
        deficit = a[idx["deficit"]] or b[idx["deficit"]]
        out.append([clash, stagnant, int(stagnant and deficit), int(stagnant and not deficit)])
    return out


def derive_stagnation(windows: Sequence[Sequence[int] | None], first: Sequence[Sequence[int] | None],
                      second: Sequence[Sequence[int] | None]) -> dict[str, Any]:
    """Row-level stagnation counts and flags (also used to re-derive them from stored windows)."""
    wi = {name: i for i, name in enumerate(MATCH_WINDOW_FIELDS)}
    si = {name: i for i, name in enumerate(SIDE_WINDOW_FIELDS)}
    stagnant = sum(w[wi["stagnant"]] for w in windows if w is not None)
    stag_def = sum(w[wi["stagnant_deficit"]] for w in windows if w is not None)
    stag_sat = sum(w[wi["stagnant_sat"]] for w in windows if w is not None)
    sd_missing = sd_forgiven = 0
    for k, w in enumerate(windows):
        if w is not None and w[wi["stagnant_deficit"]]:
            for side in (first, second):
                sd_missing += side[k][si["missing"]]
                sd_forgiven += side[k][si["forgiven"]]
    persistent_def = stag_def >= PERSISTENT_WINDOWS
    return {"stagnant": stagnant, "stagnant_deficit": stag_def, "stagnant_sat": stag_sat,
            "sd_missing": sd_missing, "sd_forgiven": sd_forgiven,
            "persistent": stagnant >= PERSISTENT_WINDOWS, "persistent_deficit": persistent_def,
            "forgiveness_supported": persistent_def and 2 * sd_forgiven >= sd_missing,
            "zombie_first": sum(w[si["zombie"]] for w in first if w is not None),
            "zombie_second": sum(w[si["zombie"]] for w in second if w is not None)}


def analyze_episodes(s: SideSeries, completed: int) -> dict[str, int]:
    """Starvation-death episodes (gap <= 30 ticks) and their recovery within 300 ticks (censored otherwise)."""
    ticks = [t for t, d in enumerate(s.starvation_deaths) if d and t < completed]
    episodes: list[tuple[int, int]] = []
    for t in ticks:
        if episodes and t - episodes[-1][1] <= EPISODE_GAP:
            episodes[-1] = (episodes[-1][0], t)
        else:
            episodes.append((t, t))
    observable = recovered_economy = recovered_population = 0
    for first_tick, last_tick in episodes:
        if last_tick + RECOVERY_HORIZON > completed:
            continue
        observable += 1
        before = s.adults[first_tick]
        if any(start + RECOVERY_SPAN <= completed and not any(s.missing[start:start + RECOVERY_SPAN])
               for start in range(last_tick + 1, last_tick + RECOVERY_HORIZON + 1)):
            recovered_economy += 1
        if any(s.adults[t + 1] >= before
               for t in range(last_tick + 1, last_tick + RECOVERY_HORIZON + 1) if t + 1 <= completed):
            recovered_population += 1
    return {"episodes": len(episodes), "observable": observable, "recovered_economy": recovered_economy,
            "recovered_population": recovered_population}


# --- one match --------------------------------------------------------------------------------------

def _snapshot(state: SimState, side: int) -> list[int]:
    colony = state.colonies[side]
    workers = sum(1 for u in colony.units if u.caste == WORKER)
    return [workers, len(colony.units) - workers, len(colony.brood), colony.food, colony.starvation_debt,
            colony.upkeep_remainder10, int(colony.queen_hp > 0)]


@functools.lru_cache(maxsize=1)
def _configs() -> dict[str, Any]:
    configs = _build_configs()
    for name, expected in CONFIG_SHA256.items():
        if config_hash(configs[name]) != expected:
            raise StarvationConfirmError(f"config hash drift for {name}")
    return configs


def row_id(variant: str, map_id: str, first: str, second: str, seed: int, swapped: bool) -> str:
    return f"{variant}|{map_id}|{first}-{second}|{seed}|{'first_B' if swapped else 'first_A'}"


def _terminal_object(outcome: Any, state: SimState, first_side: int) -> dict[str, Any]:
    score = outcome.score
    return {"reason": outcome.reason.value, "winner": outcome.winner.value, "terminal_tick": outcome.terminal_tick,
            "state_tick": state.tick, "tick_completed": outcome.tick_completed,
            "sides": [int(s) for s in outcome.sides],
            "first_score_x2": int(score[first_side] * 2), "second_score_x2": int(score[1 - first_side] * 2)}


def _play(config: Any, first: str, second: str, seed: int, swapped: bool, *, full: bool,
          reference: Sequence[str] | None = None) -> dict[str, Any]:
    """One complete match through the real ``step_tick`` (patched or not, as decided by the caller).

    ``reference`` is the per-tick event-hash list of the A0 run of the same match; it only serves to report the first
    tick at which this run's event stream differs (-1 when it never does)."""
    pair = controller_pair(None, a=f"{first}:{POLICY_VERSION}", b=f"{second}:{POLICY_VERSION}", swapped=swapped)
    controllers = pair.controllers
    if any(not isinstance(c, PolicyController) for c in controllers):
        raise StarvationConfirmError("the schedule must use only PolicyController")
    state, run = reset(config, seed, tuple(c.stream_id for c in controllers))
    first_side = 1 if swapped else 0
    series = (SideSeries(), SideSeries())
    normalizer = _MirrorNormalizer(first_side) if full else None
    trajectory, trace = _Digest(), _Digest()
    tick_hashes: list[str] = []
    first_divergence = -1
    conflict: list[int] = []
    profiles: dict[int, list[list[int]]] = {}
    extra = [Counter(), Counter()]  # per world side: harvest, egg_spend, lay_ok, ... event counters
    army: list[Counter[int]] = [Counter(), Counter()]
    calls = 0
    food = total_food(state)

    def mark(m: int) -> None:
        for side in SIDES:
            colony = state.colonies[side]
            series[side].adults.append(len(colony.units))
            series[side].brood.append(len(colony.brood))
            series[side].queen.append(int(colony.queen_hp > 0))
        if m in PROFILE_TICKS:
            profiles[m] = [_snapshot(state, 0), _snapshot(state, 1)]

    if full:
        mark(0)
    while run.outcome is None and state.tick < config.t_max:
        t = state.tick
        if normalizer is not None:
            normalizer._register_entities(state)
            if t % 10 == 0:
                trajectory.add({"tick": t, "state": normalizer.state_object(state)})
        result = step_tick(state, run, controllers)
        calls += 1
        events = list(result.events)
        tick_hash = events_hash(events)
        trace.add({"tick": t, "events_sha256": tick_hash})
        if full:
            tick_hashes.append(tick_hash)
            if reference is not None and first_divergence < 0 and (t >= len(reference) or reference[t] != tick_hash):
                first_divergence = t
        if full:
            tick_hatch, tick_dst, tick_dcb = [0, 0], [0, 0], [0, 0]
            ups: dict[int, UpkeepEvent] = {}
            regen = lay_cost = paid_total = repaid_total = 0
            clash = 0
            for e in events:
                kind = e.kind
                if kind == "upkeep":
                    ups[e.side] = e
                elif kind == "starvation":
                    tick_dst[e.side] += 1
                elif kind == "hatch":
                    tick_hatch[e.side] += 1
                elif kind == "death":
                    tick_dcb[e.side] += 1
                elif kind == "regenerate":
                    regen += e.added
                elif kind == "harvest":
                    extra[e.side]["harvest"] += e.harvested
                elif kind == "lay":
                    if e.laid:
                        extra[e.side]["lay_ok"] += 1
                        extra[e.side]["egg_spend"] += e.total_cost
                        lay_cost += e.total_cost
                    else:
                        extra[e.side]["lay_rejected"] += 1
                elif kind == "order":
                    extra[e.side]["orders"] += 1
                    extra[e.side]["lay_none_orders"] += int(e.lay_order == LAY_NONE)
                    army[e.side][e.army_choice] += 1
                elif kind == "combat":
                    clash = 1
                    extra[e.attacker_side]["combat_attacks"] += 1
                    extra[e.defender_side]["combat_defences"] += 1
                    extra[e.attacker_side]["damage_dealt"] += e.applied
                elif kind == "queen_damage":
                    clash = 1
                    extra[e.side]["queen_damage_events"] += 1
                    extra[e.side]["queen_damage"] += e.damage
            conflict.append(clash)
            completed_tick = state.tick == t + 1
            for side in SIDES:
                s, up = series[side], ups.get(side)
                if up is None:
                    # a queen-death terminal in Combat cuts the tick before the economy phases
                    if ups or run.outcome is None or completed_tick:
                        raise StarvationConfirmError(f"tick {t}: no upkeep event for side {side}")
                    for lst in (s.due, s.paid, s.missing, s.forgiven, s.repaid):
                        lst.append(0)
                else:
                    forgiven = up.debt_before if up.debt_reset else 0
                    repaid = (up.debt_before - up.debt_after) if (not up.missing and not up.debt_reset) else 0
                    s.due.append(up.due)
                    s.paid.append(up.paid)
                    s.missing.append(up.missing)
                    s.forgiven.append(forgiven)
                    s.repaid.append(repaid)
                    paid_total += up.paid
                    repaid_total += repaid
                    c = extra[side]
                    c["due_zero_ticks"] += int(up.due == 0)
                    c["exact_pay_ticks"] += int(up.due > 0 and not up.missing and up.food_after == 0)
                    c["max_debt"] = max(c["max_debt"], up.debt_before + up.missing, up.debt_after)
                s.starvation_deaths.append(tick_dst[side])
                s.combat_deaths.append(tick_dcb[side])
                s.hatches.append(tick_hatch[side])
            new_food = total_food(state)
            if new_food != food + regen - paid_total - repaid_total - lay_cost:
                raise StarvationConfirmError(f"food ledger broken at tick {t}")
            food = new_food
            if completed_tick:
                mark(state.tick)
        if normalizer is not None:
            trajectory.add({"tick": t, "events": sorted(
                canonical_dumps(normalizer.event_object(e)).decode("ascii") for e in events)})
    outcome = run.outcome
    if outcome is None:
        raise StarvationConfirmError("match ended without a terminal outcome")
    if normalizer is not None:
        trajectory.add({"final": normalizer.state_object(state)})
    if reference is not None and first_divergence < 0 and len(reference) != len(tick_hashes):
        first_divergence = min(len(reference), len(tick_hashes))
    return {"tick_hashes": tick_hashes, "first_divergence": first_divergence, "state": state, "run": run, "outcome": outcome, "calls": calls, "first_side": first_side,
            "series": series, "conflict": conflict, "profiles": profiles, "extra": extra, "army": army,
            "events_trace": trace.hexdigest(),
            "trajectory": trajectory.hexdigest() if normalizer is not None else "not_measured",
            "config": config}


def _anchor_object(play: dict[str, Any]) -> dict[str, Any]:
    state, outcome = play["state"], play["outcome"]
    return {"reason": outcome.reason.value, "winner": outcome.winner.value, "terminal_tick": outcome.terminal_tick,
            "state_tick": state.tick, "calls": play["calls"], "events_trace_sha256": play["events_trace"],
            "final_state_sha256": state_hash(state), "final_checkpoint_sha256": checkpoint_hash(state, play["run"])}


def _row_from_play(variant: str, map_id: str, first: str, second: str, seed: int, swapped: bool,
                   play: dict[str, Any]) -> dict[str, Any]:
    state, outcome, series, config = play["state"], play["outcome"], play["series"], play["config"]
    first_side = play["first_side"]
    completed = state.tick
    for s in series:
        if len(s.adults) != completed + 1:
            raise StarvationConfirmError("completed-tick series length mismatch")
    side_windows = [analyze_side_windows(series[w], completed) for w in SIDES]
    first_w, second_w = side_windows[first_side], side_windows[1 - first_side]
    windows = analyze_match_windows(first_w, second_w, play["conflict"], completed)
    stag = derive_stagnation(windows, first_w, second_w)
    terminal = _terminal_object(outcome, state, first_side)
    cost = config.balance.starvation_unit_cost
    sides: dict[str, Any] = {}
    for rel, name in enumerate((first, second)):
        world = first_side if rel == 0 else 1 - first_side
        s, c, colony = series[world], play["extra"][world], state.colonies[world]
        missing, deaths = sum(s.missing), sum(s.starvation_deaths)
        forgiven, repaid = sum(s.forgiven), sum(s.repaid)
        identity = colony.starvation_debt == missing - cost * deaths - forgiven - repaid
        if not identity:
            raise StarvationConfirmError(f"debt identity broken for {name} (side {world})")
        profile = {str(m): play["profiles"][m][world] for m in PROFILE_TICKS if m in play["profiles"]}
        sides["first" if rel == 0 else "second"] = {
            "policy": name, "world_side": "AB"[world], "queen_alive": colony.queen_hp > 0,
            "totals": {
                "due": sum(s.due), "paid": sum(s.paid), "missing": missing,
                "deficit_ticks": sum(1 for m in s.missing if m), "reset_ticks": sum(1 for f in s.forgiven if f),
                "forgiven": forgiven, "repaid": repaid, "starvation_deaths": deaths,
                "combat_deaths": sum(s.combat_deaths), "max_debt": c["max_debt"], "debt_end": colony.starvation_debt,
                "due_zero_ticks": c["due_zero_ticks"], "exact_pay_ticks": c["exact_pay_ticks"],
                "harvest": c["harvest"], "lay_ok": c["lay_ok"], "lay_rejected": c["lay_rejected"],
                "egg_spend": c["egg_spend"], "hatches": sum(s.hatches), "orders": c["orders"],
                "lay_none_orders": c["lay_none_orders"], "raid_orders": play["army"][world][6],
                "combat_attacks": c["combat_attacks"], "combat_defences": c["combat_defences"],
                "damage_dealt": c["damage_dealt"], "queen_damage_events": c["queen_damage_events"],
                "queen_damage": c["queen_damage"], "food_end": colony.food, "adults_end": len(colony.units),
                "brood_end": len(colony.brood), "adults_best": max(s.adults)},
            "army_choices": [play["army"][world][i] for i in range(7)],
            "profile": profile, "terminal_profile": _snapshot(state, world),
            "episodes": analyze_episodes(s, completed),
        }
    # windows are stored by policy order (first, second) rather than by world side
    sides["first"]["windows"], sides["second"]["windows"] = first_w, second_w
    return {
        "schema": SCHEMA_ROW, "row_id": row_id(variant, map_id, first, second, seed, swapped),
        "variant": variant, "map_id": map_id, "first": first, "second": second, "seed": seed,
        "seat": "first_B" if swapped else "first_A", "config_sha256": config_hash(config),
        "status": "complete", "terminal": terminal, "completed_ticks": completed, "calls": play["calls"],
        "sides": sides, "windows": windows, "stagnation": stag,
        "first_divergence_tick": play["first_divergence"],
        "conflict_ticks": sum(play["conflict"]),
        "events_trace_sha256": play["events_trace"], "trajectory_sha256": play["trajectory"],
        "final_state_sha256": state_hash(state), "final_checkpoint_sha256": checkpoint_hash(state, play["run"]),
    }


def anchor_matches(row: Mapping[str, Any], anchor: Mapping[str, Any]) -> bool:
    t = row["terminal"]
    return (anchor["reason"] == t["reason"] and anchor["winner"] == t["winner"]
            and anchor["terminal_tick"] == t["terminal_tick"] and anchor["state_tick"] == t["state_tick"]
            and anchor["calls"] == row["calls"]
            and anchor["events_trace_sha256"] == row["events_trace_sha256"]
            and anchor["final_state_sha256"] == row["final_state_sha256"]
            and anchor["final_checkpoint_sha256"] == row["final_checkpoint_sha256"])


def row_indicators(row: Mapping[str, Any]) -> dict[str, int]:
    """Per-row indicators of erratum E1 (tl, col); chg is computed against the A0 row."""
    t = row["terminal"]
    turtle_world = None
    for rel in ("first", "second"):
        if row["sides"][rel]["policy"] == "turtle":
            turtle_world = "AB".index(row["sides"][rel]["world_side"])
    tl = int(turtle_world is not None and t["reason"] in ("queen_death", "economic_collapse")
             and turtle_world in t["sides"])
    return {"tl": tl, "col": int(t["reason"] == "economic_collapse")}


# --- A0 clusters, aggregation (erratum E1), resolution classes ---------------------------------------

RowKey = tuple  # (map_id, first, second, seed, seat)
SEAT_NAMES = ("first_A", "first_B")
CLASS_ORDER = ("ADJUSTED_ACTIVE", "ADJUSTED_STATIC", "QUEEN_DEATH", "COLLAPSED", "PERSISTS")  # best -> worst


def row_key(row: Mapping[str, Any]) -> RowKey:
    return (row["map_id"], row["first"], row["second"], row["seed"], row["seat"])


@dataclass(frozen=True, slots=True)
class Cluster:
    """A0 cluster: seeds of one (map, pair) whose composite A0 block hash coincides (erratum E1.2)."""

    cluster_id: str
    map_id: str
    pair: tuple[str, str]
    seeds: tuple[int, ...]
    block_sha256: str

    def keys(self) -> list[RowKey]:
        return [(self.map_id, self.pair[0], self.pair[1], seed, seat) for seed in self.seeds for seat in SEAT_NAMES]


def build_clusters(a0_rows: Iterable[Mapping[str, Any]]) -> list[Cluster]:
    """Clusters are built from A0 rows only; the trajectory hashes of any other variant never enter."""
    seats: dict[tuple, dict[str, str]] = {}
    for row in a0_rows:
        if row["variant"] != "A0":
            raise StarvationConfirmError("clusters are defined by A0 rows only")
        seats.setdefault((row["map_id"], row["first"], row["second"], row["seed"]), {})[row["seat"]] = \
            row["trajectory_sha256"]
    groups: dict[tuple, list[int]] = {}
    for (map_id, first, second, seed), by_seat in sorted(seats.items()):
        if set(by_seat) != set(SEAT_NAMES):
            raise StarvationConfirmError(f"incomplete seat pair for {map_id} {first}-{second} {seed}")
        digest = _canon_sha([by_seat["first_A"], by_seat["first_B"]])
        groups.setdefault((map_id, first, second, digest), []).append(seed)
    order = {m: i for i, m in enumerate(MAPS)}
    clusters = [Cluster(f"{m}|{a}-{b}|{min(s)}", m, (a, b), tuple(sorted(s)), d)
                for (m, a, b, d), s in groups.items()]
    return sorted(clusters, key=lambda c: (order[c.map_id], PAIRS.index(c.pair), c.seeds[0]))


def indicator_table(rows: Iterable[Mapping[str, Any]], a0_by_key: Mapping[RowKey, Mapping[str, Any]]
                    ) -> dict[RowKey, dict[str, int]]:
    """tl, col, chg per schedule row (chg compares (reason, winner) with the A0 row of the same key)."""
    table: dict[RowKey, dict[str, int]] = {}
    for row in rows:
        key = row_key(row)
        ind = row_indicators(row)
        base = a0_by_key[key]["terminal"]
        t = row["terminal"]
        ind["chg"] = int((t["reason"], t["winner"]) != (base["reason"], base["winner"]))
        table[key] = ind
    return table


def cluster_indicator(cluster: Cluster, table: Mapping[RowKey, Mapping[str, int]], name: str) -> int:
    """Worst case over every member row of the cluster (erratum E1.4)."""
    return max(table[key][name] for key in cluster.keys())


def cluster_is_heterogeneous(cluster: Cluster, table: Mapping[RowKey, Mapping[str, int]], name: str) -> bool:
    return len({table[key][name] for key in cluster.keys()}) > 1


def ceil_frac(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def e1_counts(clusters: Sequence[Cluster], table_v: Mapping[RowKey, Mapping[str, int]],
              table_a0: Mapping[RowKey, Mapping[str, int]], exposed_ids: Iterable[str]) -> dict[str, Any]:
    """K5/K6 quantities on the common A0 clusters (erratum E1.5). Depends only on per-row indicators."""
    exposed = set(exposed_ids)
    turtle = [c for c in clusters if "turtle" in c.pair]
    non = [c for c in clusters if c.cluster_id not in exposed]
    out: dict[str, Any] = {"n_clusters": len(clusters), "n_turtle_clusters": len(turtle),
                           "n_exposed": len(exposed), "n_non_exposed": len(non)}
    for label, group, name in (("t_loss", turtle, "tl"), ("collapses", list(clusters), "col")):
        v = [cluster_indicator(c, table_v, name) for c in group]
        a = [cluster_indicator(c, table_a0, name) for c in group]
        out[label] = sum(v)
        out[label + "_a0"] = sum(a)
        out[label + "_worse"] = sum(1 for x, y in zip(v, a) if x > y)
        out[label + "_better"] = sum(1 for x, y in zip(v, a) if x < y)
        out[label + "_heterogeneous"] = [c.cluster_id for c in group if cluster_is_heterogeneous(c, table_v, name)]
    changed = [cluster_indicator(c, table_v, "chg") for c in non]
    out["x_changed"] = sum(changed)
    out["x_heterogeneous"] = [c.cluster_id for c in non if cluster_is_heterogeneous(c, table_v, "chg")]
    return out


def k5_ok(counts: Mapping[str, Any]) -> bool:
    base = counts["t_loss_a0"]
    return counts["t_loss"] - base <= max(3, ceil_frac(base, 4))


def k6_ok(counts: Mapping[str, Any]) -> bool:
    base = counts["collapses_a0"]
    return (counts["collapses"] - base <= max(3, ceil_frac(base, 4))
            and counts["x_changed"] <= max(3, ceil_frac(counts["n_non_exposed"], 10)))


def resolution_class(row0: Mapping[str, Any], row_v: Mapping[str, Any]) -> str:
    z0, z = row0["stagnation"]["stagnant_deficit"], row_v["stagnation"]["stagnant_deficit"]
    if z >= ceil_frac(z0, 2):
        return "PERSISTS"
    reason = row_v["terminal"]["reason"]
    if reason == "economic_collapse":
        return "COLLAPSED"
    if reason == "queen_death":
        return "QUEEN_DEATH"
    return "ADJUSTED_STATIC" if row_v["stagnation"]["stagnant"] >= PERSISTENT_WINDOWS else "ADJUSTED_ACTIVE"


def cluster_resolution(cluster: Cluster, a0_by_key: Mapping[RowKey, Mapping[str, Any]],
                       v_by_key: Mapping[RowKey, Mapping[str, Any]]) -> str:
    """Worst class over the exposure rows of the cluster (A0 rows persistently stagnant with deficit and
    supported by forgiveness), across members and seats."""
    classes = [resolution_class(a0_by_key[k], v_by_key[k]) for k in cluster.keys()
               if a0_by_key[k]["stagnation"]["forgiveness_supported"]]
    if not classes:
        raise StarvationConfirmError(f"{cluster.cluster_id} has no exposure rows")
    return max(classes, key=CLASS_ORDER.index)


def exposed_clusters(clusters: Sequence[Cluster], a0_by_key: Mapping[RowKey, Mapping[str, Any]]) -> list[Cluster]:
    return [c for c in clusters if any(a0_by_key[k]["stagnation"]["forgiveness_supported"] for k in c.keys())]


def decide(*, integrity_ok: bool, k1: bool, exposure: int, spread_ok: bool, n_clusters: int,
           passing: Mapping[str, bool], collateral: Mapping[str, tuple[int, int]]) -> dict[str, Any]:
    """Protocol section 10 verdict (checked in order). ``collateral[v] = (X, T_loss increase)`` for tie-breaks."""
    if not integrity_ok or not k1:
        return {"verdict": "INSUFFICIENT_DATA", "reason": "K0 integrity or K1 mechanism not satisfied", "candidate": None}
    if n_clusters < MIN_A0_CLUSTERS:
        return {"verdict": "INSUFFICIENT_DATA", "reason": "fewer than 120 distinct A0 blocks", "candidate": None}
    if 0 < exposure < MIN_EXPOSURE:
        return {"verdict": "INSUFFICIENT_DATA", "reason": "0 < E < 6", "candidate": None}
    if exposure >= MIN_EXPOSURE and not spread_ok:
        return {"verdict": "INSUFFICIENT_DATA", "reason": "E >= 6 but K2 distribution not satisfied", "candidate": None}
    winners = [v for v in CANDIDATES if passing.get(v)]
    if exposure == 0:
        return {"verdict": "KEEP_A0", "reason": "E = 0", "candidate": None}
    if not winners:
        return {"verdict": "KEEP_A0", "reason": "no candidate satisfies K3-K7", "candidate": None}
    simple = {"A1b": 0, "A2": 1, "A3b": 2, "A1a": 3}
    best = min(winners, key=lambda v: (collateral[v][0], collateral[v][1], simple[v]))
    return {"verdict": "NEW_RULES_VERSION", "reason": "at least one candidate satisfies K3-K7", "candidate": best,
            "passing": winners, "weak": exposure in (6, 7)}


# --- summary (preregistered analyses) ---------------------------------------------------------------

def _frac(num: int, den: int) -> dict[str, int]:
    return {"num": num, "den": den}


def _sum_sides(rows: Iterable[Mapping[str, Any]], field_name: str) -> int:
    return sum(r["sides"][rel]["totals"][field_name] for r in rows for rel in ("first", "second"))


def _variant_totals(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reasons = Counter(r["terminal"]["reason"] for r in rows)
    winners = Counter(r["terminal"]["winner"] for r in rows)
    out: dict[str, Any] = {
        "rows_raw": len(rows), "terminal_reasons_raw_rows": {k: reasons[k] for k in sorted(reasons)},
        "winners_raw_rows": {k: winners[k] for k in sorted(winners)},
        "step_tick_calls": sum(r["calls"] for r in rows),
        "stagnant_windows_raw": sum(r["stagnation"]["stagnant"] for r in rows),
        "stagnant_deficit_windows_raw": sum(r["stagnation"]["stagnant_deficit"] for r in rows),
        "stagnant_sat_windows_raw": sum(r["stagnation"]["stagnant_sat"] for r in rows),
        "persistent_rows_raw": sum(int(r["stagnation"]["persistent"]) for r in rows),
        "persistent_deficit_rows_raw": sum(int(r["stagnation"]["persistent_deficit"]) for r in rows),
        "forgiveness_supported_rows_raw": sum(int(r["stagnation"]["forgiveness_supported"]) for r in rows),
        "zombie_windows_raw": sum(r["stagnation"]["zombie_first"] + r["stagnation"]["zombie_second"] for r in rows),
        "rows_with_deficit_raw": sum(int(any(r["sides"][rel]["totals"]["deficit_ticks"] for rel in ("first", "second")))
                                     for r in rows),
        "conflict_ticks_raw": sum(r["conflict_ticks"] for r in rows),
    }
    for name in ("due", "paid", "missing", "forgiven", "repaid", "starvation_deaths", "combat_deaths", "harvest",
                 "lay_ok", "lay_rejected", "egg_spend", "hatches", "orders", "lay_none_orders", "raid_orders",
                 "deficit_ticks", "exact_pay_ticks", "queen_damage_events"):
        out[name + "_raw"] = _sum_sides(rows, name)
    episodes = Counter()
    for r in rows:
        for rel in ("first", "second"):
            episodes.update(r["sides"][rel]["episodes"])
    out["episodes_raw"] = {k: episodes[k] for k in sorted(episodes)}
    return out


def _distinct_blocks(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    seats: dict[tuple, dict[str, Mapping[str, Any]]] = {}
    for r in rows:
        seats.setdefault((r["map_id"], r["first"], r["second"], r["seed"]), {})[r["seat"]] = r
    blocks: dict[tuple, list[Mapping[str, Any]]] = {}
    for (m, a, b, _seed), by_seat in sorted(seats.items()):
        digest = _canon_sha([by_seat["first_A"]["trajectory_sha256"], by_seat["first_B"]["trajectory_sha256"]])
        blocks.setdefault((m, a, b, digest), []).extend(by_seat.values())
    return {"distinct_blocks": len(blocks),
            "distinct_blocks_persistent": sum(1 for rs in blocks.values() if any(r["stagnation"]["persistent"] for r in rs)),
            "distinct_blocks_persistent_deficit": sum(
                1 for rs in blocks.values() if any(r["stagnation"]["persistent_deficit"] for r in rs)),
            "distinct_blocks_timeout": sum(1 for rs in blocks.values()
                                           if all(r["terminal"]["reason"] == "timeout" for r in rs)),
            "distinct_blocks_collapse": sum(1 for rs in blocks.values()
                                            if any(r["terminal"]["reason"] == "economic_collapse" for r in rs)),
            "distinct_blocks_queen_death": sum(1 for rs in blocks.values()
                                               if any(r["terminal"]["reason"] == "queen_death" for r in rs))}


def _turtle_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in PAIRS:
        if "turtle" not in pair:
            continue
        sub = [r for r in rows if (r["first"], r["second"]) == pair]
        agg: Counter[str] = Counter()
        for r in sub:
            rel = "first" if r["first"] == "turtle" else "second"
            side = r["sides"][rel]
            ind = row_indicators(r)
            agg.update({"rows": 1, "queen_alive_end": int(side["queen_alive"]), "turtle_lost": ind["tl"],
                        "adults_end": side["totals"]["adults_end"], "brood_end": side["totals"]["brood_end"],
                        "starvation_deaths": side["totals"]["starvation_deaths"],
                        "lay_rejected": side["totals"]["lay_rejected"], "missing": side["totals"]["missing"],
                        "forgiven": side["totals"]["forgiven"],
                        "score_x2": r["terminal"]["first_score_x2" if rel == "first" else "second_score_x2"]})
        out["-".join(pair)] = {k: agg[k] for k in sorted(agg)}
    return out


def build_summary(rows: Sequence[Mapping[str, Any]], demo: Mapping[str, Any],
                  anchors: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """All preregistered analyses recomputed from the stored rows (and nothing else)."""
    by_variant: dict[str, dict[RowKey, Mapping[str, Any]]] = {v: {} for v in VARIANTS}
    for r in rows:
        by_variant[r["variant"]][row_key(r)] = r
    a0 = by_variant["A0"]
    clusters = build_clusters(a0.values())
    exposed = exposed_clusters(clusters, a0)
    exposed_ids = [c.cluster_id for c in exposed]
    E = len(exposed)
    spread = {"pairs": sorted({"-".join(c.pair) for c in exposed}), "maps": sorted({c.map_id for c in exposed}),
              "cells": sorted({f"{c.map_id}|{'-'.join(c.pair)}" for c in exposed})}
    spread_ok = (len(spread["pairs"]) >= MIN_EXPOSURE_PAIRS and len(spread["maps"]) >= MIN_EXPOSURE_MAPS
                 and len(spread["cells"]) >= MIN_EXPOSURE_CELLS)
    n_anchor_ok = sum(1 for a in anchors if a.get("matches") is True)
    k7_violations: list[str] = []
    for v in CANDIDATES:
        for key, row in by_variant[v].items():
            base = a0[key]
            no_deficit = not any(base["sides"][rel]["totals"]["deficit_ticks"] for rel in ("first", "second"))
            if no_deficit and row["trajectory_sha256"] != base["trajectory_sha256"]:
                k7_violations.append(row["row_id"])
    counts_ok = (len(rows) == EXPECTED_ROWS and all(len(by_variant[v]) == EXPECTED_ROWS // len(VARIANTS)
                                                     for v in VARIANTS)
                 and all(r["status"] == "complete" for r in rows))
    integrity = {"rows": len(rows), "rows_expected": EXPECTED_ROWS, "schedule_complete": counts_ok,
                 "anchors": len(anchors), "anchors_expected": EXPECTED_ANCHORS, "anchors_matching": n_anchor_ok,
                 "k7_false_positive_rows": k7_violations}
    integrity["ok"] = bool(counts_ok and len(anchors) == EXPECTED_ANCHORS and n_anchor_ok == EXPECTED_ANCHORS
                           and not k7_violations)
    k1 = bool(demo["k1_mechanism_reproduced"])
    a0_table = indicator_table(a0.values(), a0)
    variants: dict[str, Any] = {}
    passing: dict[str, bool] = {}
    collateral: dict[str, tuple[int, int]] = {}
    for v in VARIANTS:
        vrows = list(by_variant[v].values())
        entry: dict[str, Any] = {"totals": _variant_totals(vrows), "blocks": _distinct_blocks(vrows),
                                 "turtle": _turtle_metrics(vrows)}
        differs = [r for r in vrows if r["trajectory_sha256"] != a0[row_key(r)]["trajectory_sha256"]]
        entry["trajectory_differs_from_a0_rows_raw"] = len(differs)
        entry["first_divergence_ticks_raw"] = sorted(r["first_divergence_tick"] for r in differs)
        table = indicator_table(vrows, a0)
        counts = e1_counts(clusters, table, a0_table, exposed_ids)
        entry["e1"] = counts
        entry["exposure_clusters_total"] = E
        if v != "A0":
            classes = Counter(cluster_resolution(c, a0, by_variant[v]) for c in exposed) if E else Counter()
            viable = classes["ADJUSTED_STATIC"] + classes["ADJUSTED_ACTIVE"]
            k3 = bool(demo["k3_mechanism_removed"][v])
            k4 = E > 0 and viable >= ceil_frac(E, 2)
            k5, k6 = k5_ok(counts), k6_ok(counts)
            k7 = not any(rid.startswith(v + "|") for rid in k7_violations)
            entry["classes"] = {k: classes[k] for k in CLASS_ORDER}
            entry["classes_by_cluster"] = {c.cluster_id: cluster_resolution(c, a0, by_variant[v]) for c in exposed}
            entry["viable"] = viable
            entry["criteria"] = {"K3": k3, "K4": k4, "K5": k5, "K6": k6, "K7": k7}
            passing[v] = bool(E > 0 and k3 and k4 and k5 and k6 and k7)
            collateral[v] = (counts["x_changed"], counts["t_loss"] - counts["t_loss_a0"])
        variants[v] = entry
    tr_cells = []
    for c in clusters:
        if c.pair != ("turtle", "raider"):
            continue
        cell: dict[str, Any] = {"cluster": c.cluster_id, "map": c.map_id, "seeds": len(c.seeds), "variants": {}}
        for v in VARIANTS:
            vr = [by_variant[v][k] for k in c.keys()]
            cell["variants"][v] = {
                "reasons": sorted({r["terminal"]["reason"] for r in vr}),
                "stagnant": max(r["stagnation"]["stagnant"] for r in vr),
                "stagnant_deficit": max(r["stagnation"]["stagnant_deficit"] for r in vr),
                "stagnant_sat": max(r["stagnation"]["stagnant_sat"] for r in vr),
                "persistent": any(r["stagnation"]["persistent"] for r in vr)}
        persistent_a0 = cell["variants"]["A0"]["persistent"]
        cell["a0_persistent"] = persistent_a0
        cell["rule_dependence"] = ("not_applicable" if not persistent_a0 else
                                   "rule_sensitive" if any(not cell["variants"][v]["persistent"] for v in CANDIDATES)
                                   else "rule_independent")
        tr_cells.append(cell)
    tr_timeout = [c for c in tr_cells if c["variants"]["A0"]["reasons"] == ["timeout"]]
    tr_persist = [c for c in tr_timeout if all(c["variants"][v]["reasons"] == ["timeout"] for v in CANDIDATES)]
    a1a_m1 = demo["scenarios"]["M1"]["A1a"]["totals"]["forgiven"]
    comparison = {
        "note": "descriptive; Baselines v2 differ from the Baselines v1 used by FOR-25",
        "c1_a1a_keeps_forgiving_on_m1": {"a1a_forgiven_on_m1": a1a_m1, "reproduced": a1a_m1 > 0},
        "c2_turtle_raider_timeout_persists": {
            "a0_timeout_cells": len(tr_timeout), "timeout_under_all_candidates": len(tr_persist),
            "reproduced": bool(tr_timeout) and 2 * len(tr_persist) >= len(tr_timeout)},
        "c3_turtle_loss_not_lower_than_a0": {
            "reproduced": all(variants[v]["e1"]["t_loss"] >= variants[v]["e1"]["t_loss_a0"] for v in CANDIDATES)},
        "c4_collateral_outcome_changes_up_to_ten_percent": {
            "x_by_variant": {v: variants[v]["e1"]["x_changed"] for v in CANDIDATES},
            "n_non_exposed": variants["A0"]["e1"]["n_non_exposed"],
            "reproduced": all(10 * variants[v]["e1"]["x_changed"] <= variants["A0"]["e1"]["n_non_exposed"]
                              for v in CANDIDATES)},
    }
    decision = decide(integrity_ok=integrity["ok"], k1=k1, exposure=E, spread_ok=spread_ok, n_clusters=len(clusters),
                      passing=passing, collateral=collateral)
    if decision["verdict"] == "NEW_RULES_VERSION":
        decision["weak"] = bool(decision["weak"] or len(spread["cells"]) <= 1)
    exposure_detail = [{"cluster": c.cluster_id, "map": c.map_id, "pair": "-".join(c.pair), "seeds": len(c.seeds),
                        "a0_stagnant_deficit": max(a0[k]["stagnation"]["stagnant_deficit"] for k in c.keys()),
                        "a0_stagnant": max(a0[k]["stagnation"]["stagnant"] for k in c.keys())} for c in exposed]
    return {"schema": SCHEMA_SUMMARY, "protocol_id": PROTOCOL_ID, "preregistered": True,
            "design": {"rows": len(rows), "variants": list(VARIANTS), "a0_clusters": len(clusters),
                       "a0_clusters_turtle": variants["A0"]["e1"]["n_turtle_clusters"],
                       "a0_clusters_non_exposed": variants["A0"]["e1"]["n_non_exposed"],
                       "cluster_members": {c.cluster_id: len(c.seeds) for c in clusters}},
            "integrity": integrity, "stage_m": {"k1": k1, "k3": demo["k3_mechanism_removed"]},
            "exposure": {"E": E, "spread": spread, "spread_ok": spread_ok, "clusters": exposure_detail},
            "variants": variants, "turtle_raider_cells": tr_cells, "for25_comparison": comparison,
            "decision": decision}


# --- schedule, run, manifest --------------------------------------------------------------------------

def schedule_groups() -> list[tuple[str, str, str, int, bool]]:
    """One group = the five arms of one (map, pair, seed, seat); A0 first so it can serve as reference."""
    return [(m, a, b, s, w) for m in MAPS for (a, b) in PAIRS for s in SEEDS for w in SEATS]


def schedule_row_ids() -> list[str]:
    return [row_id(v, m, a, b, s, w) for v in VARIANTS for m in MAPS for (a, b) in PAIRS for s in SEEDS for w in SEATS]


def execute_group(group: tuple[str, str, str, int, bool]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    map_id, first, second, seed, swapped = group
    config = _configs()[map_id]
    rows: list[dict[str, Any]] = []
    reference: list[str] | None = None
    anchor: dict[str, Any] | None = None
    for variant in VARIANTS:
        with use_variant(variant):
            play = _play(config, first, second, seed, swapped, full=True, reference=reference)
        row = _row_from_play(variant, map_id, first, second, seed, swapped, play)
        if variant == "A0":
            reference = play["tick_hashes"]
            if tick_module.upkeep is not economy_module.upkeep:
                raise StarvationConfirmError("tick.upkeep is replaced outside use_variant")
            anchor = {"row_id": row["row_id"], **_anchor_object(_play(config, first, second, seed, swapped, full=False))}
            anchor["matches"] = anchor_matches(row, anchor)
            if not anchor["matches"]:
                raise StarvationConfirmError(f"A0 differs from the unchanged engine: {row['row_id']}")
        rows.append(row)
    assert anchor is not None
    return rows, anchor


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=_ROOT, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def protected_files() -> dict[str, str]:
    listing = subprocess.run(["git", "ls-files", "-z", "src/formicmind", "results", "docs"], cwd=_ROOT, check=True,
                             capture_output=True).stdout.decode("utf-8")
    paths = sorted(p for p in listing.split("\0") if p and not _is_study_file(p) and p not in REFERENCE_FILES)
    return {p: _file_sha(_ROOT / p) for p in paths}


def _versions() -> dict[str, str]:
    return {"rules_version": RULES_VERSION, "balance_version": BALANCE_VERSION,
            "observation_version": OBSERVATION_VERSION, "action_version": ACTION_VERSION,
            "decoder_version": DECODER_VERSION, "rng_version": RNG_VERSION,
            "stream_derivation_version": STREAM_DERIVATION_VERSION, "serialization_version": SERIALIZATION_VERSION}


def _policy_descriptors() -> dict[str, Any]:
    out = {}
    for name in POLICIES:
        spec = baseline_spec(f"{name}:{POLICY_VERSION}")
        out[name] = {"selector": spec.full_id, "policy_id": spec.policy_id, "parameter_sha256": spec.digest(),
                     "stream_id": f"policy:{spec.policy_id}:0"}
    return out


def static_manifest() -> dict[str, Any]:
    """Everything in the manifest that does not depend on the particular run."""
    errata = {str(n): {"path": p.as_posix(), "sha256": _file_sha(_ROOT / p)} for n, (p, _) in sorted(ERRATA.items())}
    return {
        "schema": SCHEMA_MANIFEST, "protocol_id": PROTOCOL_ID, "protocol_path": PROTOCOL_PATH.as_posix(),
        "protocol_sha256": _file_sha(_ROOT / PROTOCOL_PATH), "protocol_commit": PROTOCOL_COMMIT,
        "errata": errata, "base_commit": BASE_COMMIT, "for25_merge_commit": FOR25_MERGE_COMMIT,
        "engine_fingerprint": source_fingerprint(exclude_harness=True),
        "engine_fingerprint_expected": BASE_ENGINE_FINGERPRINT,
        "maps": {m: config_hash(_configs()[m]) for m in MAPS}, "policies": _policy_descriptors(),
        "seeds": list(SEEDS), "seeds_sha256": _canon_sha(list(SEEDS)), "seats_sha256": _canon_sha(list(SEATS)),
        "pairs": ["-".join(p) for p in PAIRS], "variants": list(VARIANTS), "window_ticks": WINDOW,
        "side_window_fields": list(SIDE_WINDOW_FIELDS), "match_window_fields": list(MATCH_WINDOW_FIELDS),
        "versions": _versions(), "t_max": T_MAX,
        "schedule": {"rows": EXPECTED_ROWS, "anchors": EXPECTED_ANCHORS, "row_ids_sha256": _canon_sha(schedule_row_ids())},
        "budget": {"main_step_tick_calls_max": MAIN_BUDGET, "anchor_step_tick_calls_max": ANCHOR_BUDGET},
        "protected_files": protected_files(),
    }


def _git_clean_tracked() -> bool:
    return _git("status", "--porcelain", "--untracked-files=no") == ""


def _untracked_in_study_dirs() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z", "--others", "--exclude-standard", "src", "docs", "tests",
                          "scripts"], cwd=_ROOT, check=True, capture_output=True).stdout.decode("utf-8")
    return [p for p in out.split("\0") if p and not _is_study_file(p)]


def _scan_seed_collisions() -> list[str]:
    """Files outside this study that mention any of the new seeds as a whole number."""
    pattern = re.compile(rb"(?<!\d)(" + b"|".join(str(s).encode() for s in SEEDS) + rb")(?!\d)")
    hits = []
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=_ROOT, check=True, capture_output=True).stdout.decode("utf-8")
    for rel in tracked.split("\0"):
        if not rel or _is_study_file(rel) or rel.startswith("viewer/") or rel.endswith((".png", ".ico", ".svg")):
            continue
        try:
            data = (_ROOT / rel).read_bytes()
        except OSError:
            continue
        if pattern.search(data):
            hits.append(rel)
    return hits


def preflight(*, require_clean: bool = True) -> dict[str, Any]:
    """Conditions of protocol section 1; raises on the first failure."""
    import formicmind
    checks: dict[str, Any] = {}
    toplevel = Path(_git("rev-parse", "--show-toplevel")).resolve()
    module_file = Path(formicmind.__file__).resolve()
    checks["package_from_active_checkout"] = (toplevel == _ROOT.resolve()
                                              and module_file == (_ROOT / "src" / "formicmind" / "__init__.py").resolve())
    head = _git("rev-parse", "HEAD")
    checks["head"] = head
    checks["for25_merged"] = subprocess.run(["git", "merge-base", "--is-ancestor", FOR25_MERGE_COMMIT, "HEAD"],
                                            cwd=_ROOT).returncode == 0
    checks["protocol_commit_ancestor"] = subprocess.run(["git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, "HEAD"],
                                                        cwd=_ROOT).returncode == 0
    checks["protocol_sha256_ok"] = _file_sha(_ROOT / PROTOCOL_PATH) == PROTOCOL_SHA256
    checks["errata_sha256_ok"] = all(_file_sha(_ROOT / p) == h for p, h in ERRATA.values())
    checks["engine_fingerprint_ok"] = source_fingerprint(exclude_harness=True) == BASE_ENGINE_FINGERPRINT
    checks["tracked_tree_clean"] = _git_clean_tracked() if require_clean else True
    checks["no_untracked_source_files"] = not _untracked_in_study_dirs() if require_clean else True
    checks["seeds_sha256_ok"] = _canon_sha(list(SEEDS)) == SEEDS_SHA256
    checks["seats_sha256_ok"] = _canon_sha(list(SEATS)) == SEATS_SHA256
    checks["configs_ok"] = all(config_hash(_configs()[m]) == h for m, h in CONFIG_SHA256.items())
    checks["policies_ok"] = all(d["parameter_sha256"] == POLICY_PARAMETER_SHA256[n]
                                for n, d in _policy_descriptors().items())
    collisions = _scan_seed_collisions()
    checks["seed_collisions"] = collisions
    ok = (all(v for k, v in checks.items() if isinstance(v, bool)) and not collisions)
    checks["ok"] = ok
    if not ok:
        raise StarvationConfirmError("preflight failed: " + json.dumps(checks, sort_keys=True))
    return checks


def _dump_lines(rows: Sequence[Mapping[str, Any]]) -> bytes:
    # json (not canonical_dumps): unmeasured windows are stored as null, never as zero
    return b"".join(json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii") + b"\n"
                    for r in rows)


def _dump_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, indent=1, ensure_ascii=True).encode("ascii") + b"\n"


def _run_groups(workers: int, progress: bool, groups: Sequence[tuple[str, str, str, int, bool]] | None = None
                ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups = list(groups) if groups is not None else schedule_groups()
    rows_by_id: dict[str, dict[str, Any]] = {}
    anchors: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (rows, anchor) in enumerate(pool.map(execute_group, groups, chunksize=2), 1):
            for r in rows:
                rows_by_id[r["row_id"]] = r
            anchors.append(anchor)
            if progress and i % 100 == 0:
                print(f"  {i}/{len(groups)} groups", file=sys.stderr, flush=True)
    ordered = [rows_by_id[row_id(v, m, a, b, seed, swapped)] for v in VARIANTS for (m, a, b, seed, swapped) in groups]
    anchors.sort(key=lambda a: a["row_id"])
    return ordered, anchors


def generate(workers: int, progress: bool = False, groups: Sequence[tuple[str, str, str, int, bool]] | None = None
             ) -> dict[str, Any]:
    """Regenerate every artifact (without the run-time manifest fields). ``groups`` is for tests only."""
    demo = build_minimal_demo()
    rows, anchors = _run_groups(workers, progress, groups)
    summary = build_summary(rows, demo, anchors)
    budget_main, budget_anchor = sum(r["calls"] for r in rows), sum(a["calls"] for a in anchors)
    if budget_main > MAIN_BUDGET or budget_anchor > ANCHOR_BUDGET:
        raise StarvationConfirmError("step_tick budget exceeded")
    return {"rows": rows, "anchors": anchors, "demo": demo, "summary": summary,
            "calls": {"main": budget_main, "anchor": budget_anchor}}


def _artifact_bytes(gen: Mapping[str, Any]) -> dict[str, bytes]:
    return {"matches.jsonl": _dump_lines(gen["rows"]),
            "anchors.json": _dump_json({"schema": SCHEMA_ANCHORS, "anchors": gen["anchors"]}),
            "minimal_demo.json": _dump_json(gen["demo"]),
            "summary.json": _dump_json(gen["summary"])}


def run_study(out: Path, workers: int) -> dict[str, Any]:
    preflight()
    if out.exists() and any(out.iterdir()):
        raise StarvationConfirmError(f"refusing to write into the non-empty directory {out}")
    out.mkdir(parents=True, exist_ok=True)
    try:
        gen = generate(workers, progress=True)
        artifacts = _artifact_bytes(gen)
    except BaseException as exc:  # keep the incomplete attempt separately and stop
        (out / "incomplete_attempt.json").write_bytes(_dump_json({"failure": repr(exc), "head": _git("rev-parse", "HEAD")}))
        raise
    for name, data in artifacts.items():
        (out / name).write_bytes(data)
    manifest = static_manifest()
    manifest.update({
        "source_commit": _git("rev-parse", "HEAD"), "generator_package_fingerprint": source_fingerprint(),
        "artifacts": {name: _sha(_lf(data)) for name, data in sorted(artifacts.items())},
        "step_tick_calls": gen["calls"], "rows_written": len(gen["rows"]), "anchors_written": len(gen["anchors"]),
        "python": platform.python_version(), "platform": platform.platform(),
        "commands": ["python -m formicmind.starvation_confirm preflight",
                     "python -m formicmind.starvation_confirm run --out <NEW_DIR>",
                     "python -m formicmind.starvation_confirm verify --input <DIR>"]})
    (out / "manifest.json").write_bytes(_dump_json(manifest))
    return manifest


def compare_artifacts(directory: Path, manifest: Mapping[str, Any], gen: Mapping[str, Any]) -> list[str]:
    """Differences between the stored artifacts / manifest hashes and a fresh regeneration."""
    problems: list[str] = []
    for name, data in _artifact_bytes(gen).items():
        path = directory / name
        stored = _lf(path.read_bytes()) if path.is_file() else b""
        if stored != _lf(data):
            problems.append(f"artifact differs from regeneration: {name}")
        if manifest.get("artifacts", {}).get(name) != _sha(_lf(data)):
            problems.append(f"manifest hash differs: {name}")
    if manifest.get("step_tick_calls") != gen["calls"]:
        problems.append("step_tick call counters differ")
    return problems


def verify_study(directory: Path, workers: int) -> dict[str, Any]:
    """Strict verification: regenerate everything with the engine and compare byte for byte."""
    preflight(require_clean=False)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    expected = static_manifest()
    problems: list[str] = []
    for key, value in expected.items():
        if key == "protected_files":
            for path, digest in manifest.get(key, {}).items():
                if not (_ROOT / path).is_file() or _file_sha(_ROOT / path) != digest:
                    problems.append(f"protected file changed or removed: {path}")
            if not manifest.get(key):
                problems.append("manifest has no protected_files")
        elif manifest.get(key) != value:
            problems.append(f"manifest field differs: {key}")
    gen = generate(workers)
    problems += compare_artifacts(directory, manifest, gen)
    if problems:
        raise StarvationConfirmError("verification failed: " + "; ".join(problems))
    return {"ok": True, "rows": len(gen["rows"]), "anchors": len(gen["anchors"]), "verification_calls": gen["calls"],
            "generator_fingerprint_matches": manifest.get("generator_package_fingerprint") == source_fingerprint()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m formicmind.starvation_confirm")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    run_p = sub.add_parser("run")
    run_p.add_argument("--out", type=Path, required=True)
    run_p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ver_p = sub.add_parser("verify")
    ver_p.add_argument("--input", type=Path, required=True)
    ver_p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            result = preflight()
        elif args.command == "run":
            result = run_study(args.out, args.workers)
            result = {"ok": True, "rows": result["rows_written"], "step_tick_calls": result["step_tick_calls"]}
        else:
            result = verify_study(args.input, args.workers)
    except StarvationConfirmError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
