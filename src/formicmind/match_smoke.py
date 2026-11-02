"""Full-match smoke runs (``reset -> run_episode -> terminal``). **A functional check, nothing else.**

Four scripted matches on the default candidate balance (``candidate-FOR1-v0.2.1-unvalidated``), both sides driven
by ``ScriptedController`` through the one ``step_tick`` (``check_invariants=True``: the food ledger, the unit rules
and the combat damage conservation are re-verified after every tick):

* ``raid_win``        A banks workers, then soldiers, then raids B's home; B sits at home with 4 soldiers.
                      Terminal: B's queen dies in Combat -> winner A.
* ``mutual_raid_draw`` Both send every soldier to the enemy home around the centre (``avoid_center``); both homes
                      are empty of workers, both queens fall in the same Combat phase -> draw.
* ``collapse``        A never forages (all workers stay home) and starves; B forages. Terminal: A's economy
                      collapses at the end of a tick -> winner B.
* ``timeout_draw``    Both play a symmetric economy and never fight; the match ends by ``T_max`` -> draw.

This is **not** a balance result, not a meta/strategy study, not a performance measurement and not a Gate 0/Gate 1
readiness check. The seed is an arbitrary engineering constant, not a FOR-4 sealed seed. Run it with
``python -m formicmind.match_smoke [scenario ...]``.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass

from .actions import ActionV2
from .match import EpisodeResult, run_episode
from .rules import Q_TOTAL, default_config
from .tick import ScriptedController

MATCH_SMOKE_SEED = 20261102  # arbitrary engineering constant, not a FOR-4 sealed seed
LABEL = "FUNCTIONAL CHECK - not a meta/performance/Gate 0 readiness result"

HOME = (Q_TOTAL, 0, 0, 0, 0, 0)
N1_N2 = (0, 43690, 21845, 0, 0, 0)  # 8 workers on local N1, 4 on local N2 (exact for 12 workers)


def _act(lay: int = 0, shares: tuple[int, ...] = N1_N2, army: int = 0, avoid: bool = False) -> ActionV2:
    return ActionV2(lay_order=lay, forage_share_q=shares, army_choice=army, avoid_center=avoid)


@dataclass(frozen=True, slots=True)
class SmokeScenario:
    name: str
    description: str
    script_a: dict[int, ActionV2]
    script_b: dict[int, ActionV2]


SCENARIOS: dict[str, SmokeScenario] = {s.name: s for s in (
    SmokeScenario(
        "raid_win",
        "A: lay workers t=0, soldiers t=60, raid via the centre t=150; B: forage, garrison, lay nothing.",
        {0: _act(lay=1), 60: _act(lay=2), 150: _act(lay=0, army=6)},
        {0: _act()},
    ),
    SmokeScenario(
        "mutual_raid_draw",
        "Both: forage N1/N2, send all soldiers to the enemy home with avoid_center at t=0.",
        {0: _act(army=6, avoid=True)},
        {0: _act(army=6, avoid=True)},
    ),
    SmokeScenario(
        "collapse",
        "A: every worker stays home (no income) and starves; B: forage, garrison.",
        {0: _act(shares=HOME)},
        {0: _act()},
    ),
    SmokeScenario(
        "timeout_draw",
        "Both: lay workers until t=90, then nothing; never leave the economy; ends by T_max.",
        {0: _act(lay=1), 90: _act(lay=0)},
        {0: _act(lay=1), 90: _act(lay=0)},
    ),
)}


@dataclass(frozen=True, slots=True)
class MatchSmokeReport:
    scenario: SmokeScenario
    result: EpisodeResult
    narrative: tuple[str, ...]

    def row(self) -> str:
        r = self.result
        o = r.outcome
        assert o is not None
        return (f"{self.scenario.name:<17} {o.winner.value:<5} {o.reason.value:<18} {o.terminal_tick:>5} "
                f"{r.final_tick:>5}  {r.queen_hp[0]:>3}/{r.queen_hp[1]:<3}  {r.food[0]:>5}/{r.food[1]:<5}  "
                f"{r.workers[0]:>2}/{r.workers[1]:<2} {r.soldiers[0]:>2}/{r.soldiers[1]:<2}")

    def render(self) -> str:
        r = self.result
        lines = [f"== {self.scenario.name}: {self.scenario.description}", *self.narrative,
                 f"   events={len(r.events)} {dict(r.event_counts)}",
                 f"   config_hash:     {r.config_hash}",
                 f"   events_hash:     {r.events_hash}",
                 f"   state_hash:      {r.state_hash}",
                 f"   checkpoint_hash: {r.checkpoint_hash}"]
        return "\n".join(lines)


def narrate(result: EpisodeResult) -> tuple[str, ...]:
    """A few lines that show the combat and the terminal events of a match."""
    side = "AB"
    names = ("A", "N1", "N2", "C", "N4", "N5", "B")
    attacks: dict[tuple[int, int, str], list] = defaultdict(list)
    deaths: dict[tuple[int, int], list[int]] = defaultdict(list)
    queen: dict[int, list] = defaultdict(list)
    for e in result.events:
        if e.kind == "combat":
            attacks[(e.attacker_side, e.node, e.target)].append(e)
        elif e.kind == "death":
            deaths[(e.side, e.caste)].append(e.tick)
        elif e.kind == "queen_damage":
            queen[e.side].append(e)
    out = []
    for (att, node, target), evs in sorted(attacks.items(), key=lambda kv: (kv[1][0].tick, kv[0])):
        out.append(
            f"   combat {side[att]} at {names[node]} -> {target}s: ticks {evs[0].tick}..{evs[-1].tick} "
            f"({len(evs)} phases), soldiers {evs[0].attacker_soldiers}->{evs[-1].attacker_soldiers}, "
            f"dmg/phase {evs[0].damage}{'(home x5/4)' if evs[0].home_bonus else ''}, "
            f"applied {sum(x.applied for x in evs)}, wasted {sum(x.wasted for x in evs)}")
    for (owner, caste), ticks in sorted(deaths.items()):
        out.append(f"   {side[owner]} lost {len(ticks)} {('workers', 'soldiers')[caste]} in combat "
                   f"(t={ticks[0]}..{ticks[-1]})")
    for owner, evs in sorted(queen.items()):
        out.append(f"   queen {side[owner]}: hit t={evs[0].tick}..{evs[-1].tick}, hp "
                   f"{evs[0].queen_hp_before} -> {evs[-1].queen_hp_after}")
    for e in result.events:
        if e.kind == "terminal":
            who = "+".join(side[s] for s in e.sides) or "-"
            out.append(f"   TERMINAL t={e.tick}: {e.reason} sides={who} winner={e.winner} phases_executed="
                       f"{e.phases_executed} tick_completed={e.tick_completed} queen_hp={e.queen_hp}")
    return tuple(out)


def run_match_smoke(name: str, seed: int = MATCH_SMOKE_SEED, check_invariants: bool = True) -> MatchSmokeReport:
    scenario = SCENARIOS[name]
    result = run_episode(
        default_config(), seed, ScriptedController(scenario.script_a), ScriptedController(scenario.script_b),
        check_invariants=check_invariants)
    assert result.outcome is not None and not result.truncated
    return MatchSmokeReport(scenario, result, narrate(result))


def main(argv: list[str] | None = None) -> None:
    names = list(sys.argv[1:] if argv is None else argv) or list(SCENARIOS)
    reports = [run_match_smoke(n) for n in names]
    print(f"match smoke runs, seed {MATCH_SMOKE_SEED}, default candidate balance - {LABEL}\n")
    print("scenario          win   reason             t_end t_fin  queen    food          W     S")
    print("                                                         A/B      A/B           A/B   A/B")
    for report in reports:
        print(report.row())
    print()
    for report in reports:
        print(report.render())
        print()


if __name__ == "__main__":
    main()
