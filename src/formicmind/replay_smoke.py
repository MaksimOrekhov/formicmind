"""Replay v1 smoke recordings: ``python -m formicmind.replay_smoke --out DIR`` (or ``--scenario NAME --out FILE.jsonl``).

Records the four public dev matches of ``match_smoke`` (queen win t175, simultaneous queen draw t39, collapse t46,
timeout t2999), a short harness **cutoff** (``cutoff``: the first 45 ticks of the raid, ``outcome=None``) and a match
between two **RNG-consuming** random policies (``rng_policy``, 400 ticks), then reads, seeks and verifies each file.
Functional check only: arbitrary engineering seed (no FOR-4 sealed seeds), not a balance, performance or Gate result.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .decoder import PolicyLogitsV2
from .match_smoke import MATCH_SMOKE_SEED, SCENARIOS
from .replay import ReplayReader, SeatSpec, record_episode, verify
from .rng import policy_stream_id
from .rules import default_config
from .tick import Controller, PolicyController, ScriptedController

CUTOFF_TICKS = 45
RNG_POLICY_TICKS = 400
EXTRA_SCENARIOS = ("cutoff", "rng_policy")


def random_policy(observation, mask, policy_rng):
    """A policy that really consumes its own stream: 17 uniform logits in [-1, 1) per decision."""
    return PolicyLogitsV2([policy_rng.next_u32() / 2**31 - 1 for _ in range(17)])


@dataclass(frozen=True)
class Plan:
    controllers: tuple[Controller, Controller]
    max_ticks: int | None
    seats: tuple[SeatSpec, SeatSpec] | None = None


def plan_for(name: str) -> Plan:
    if name in SCENARIOS:
        sc = SCENARIOS[name]
        return Plan((ScriptedController(sc.script_a), ScriptedController(sc.script_b)), None)
    if name == "cutoff":
        sc = SCENARIOS["raid_win"]
        return Plan((ScriptedController(sc.script_a), ScriptedController(sc.script_b)), CUTOFF_TICKS)
    if name == "rng_policy":
        a, b = policy_stream_id("rnd-a", 0), policy_stream_id("rnd-b", 0)
        return Plan((PolicyController(random_policy, a), PolicyController(random_policy, b)), RNG_POLICY_TICKS)
    raise KeyError(name)


ALL_SCENARIOS = (*SCENARIOS, *EXTRA_SCENARIOS)


def record_scenario(name: str, path: Path, seed: int = MATCH_SMOKE_SEED):
    plan = plan_for(name)
    return record_episode(path, default_config(), seed, *plan.controllers, max_ticks=plan.max_ticks,
                          seats=plan.seats, source=f"replay_smoke:{name}",
                          meta={"label": "FUNCTIONAL CHECK - not a meta/performance/Gate 0 readiness result"})


def main(argv: Sequence[str] | None = None, write: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(prog="python -m formicmind.replay_smoke", description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="a directory (one <scenario>.jsonl each) or one FILE.jsonl")
    parser.add_argument("--scenario", action="append", choices=ALL_SCENARIOS,
                        help="scenario to record (repeatable; default: all)")
    parser.add_argument("--seed", type=int, default=MATCH_SMOKE_SEED, help="engineering seed (never a sealed seed)")
    args = parser.parse_args(argv)
    names = args.scenario or list(ALL_SCENARIOS)
    out = Path(args.out)
    single = out.suffix == ".jsonl"
    if single and len(names) != 1:
        parser.error("--out FILE.jsonl needs exactly one --scenario")
    if not single:
        out.mkdir(parents=True, exist_ok=True)
    write(f"replay smoke recordings, seed {args.seed}, default candidate balance - FUNCTIONAL CHECK, not a "
          "meta/performance/Gate 0 result")
    write(f"{'scenario':<17} {'bytes':>9} {'records':>7} {'ticks':>5}  {'record s':>8} {'verify s':>8}  outcome")
    for name in names:
        path = out if single else out / f"{name}.jsonl"
        t0 = time.perf_counter()
        _, summary = record_scenario(name, path, args.seed)
        t1 = time.perf_counter()
        report = verify(ReplayReader.open(path))
        t2 = time.perf_counter()
        outcome = (f"{report.winner} {report.reason} t={report.terminal_tick}" if report.finished
                   else f"truncated at state.tick {report.state_tick}")
        write(f"{name:<17} {summary.size_bytes:>9} {summary.records:>7} {summary.ticks:>5}  {t1 - t0:>8.2f} "
              f"{t2 - t1:>8.2f}  {outcome}")
        write(f"    events_hash={summary.events_hash}\n    state_hash={summary.final_state_hash}\n"
              f"    checkpoint_hash={summary.final_checkpoint_hash}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
