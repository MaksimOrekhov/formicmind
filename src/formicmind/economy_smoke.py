"""Economy smoke run (default 120 ticks). **PARTIAL (NO COMBAT/TERMINAL)** - not a match.

Two scripted policies on the default (candidate, unvalidated) balance; soldiers garrison at home on
both sides, so the missing Combat is never exercised. It prints the economic diagnostics that the
FOR-1 errata F pilot will need later (adults, brood, warehouse and node food, harvest, upkeep,
lay/hatch, starvation) and re-checks the food ledger and the unit rules after every tick.

    A  "boom":    t=0 lay workers, 8 on N1 + 4 on N2; t=60 lay soldiers; t=90 stop laying.
    B  "barracks": t=0 lay soldiers, 8 on N4 + 4 on N5; t=60 lay workers.

Nothing here is a balance calibration or a Gate 0/Gate 1 result. Run it with
``python -m formicmind.economy_smoke``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .actions import ActionV2
from .config import SimConfig
from .events import Event, events_hash
from .ledger import EconomyTotals, total_food
from .reset import reset
from .rules import NODE_NAMES, SIDES, SOLDIER, WORKER, default_config
from .serialization import checkpoint_hash, state_hash
from .tick import ScriptedController, TickMode, step_tick

ECONOMY_SMOKE_SEED = 20261102  # arbitrary engineering constant, not a FOR-4 sealed seed
ECONOMY_SMOKE_TICKS = 120
LABEL = "PARTIAL (NO COMBAT/TERMINAL)"
BUCKET = 10

_Q = 65535
_SHARES = (0, 43690, 21845, 0, 0, 0)  # 8 workers on N1 (local), 4 on N2

A_SCRIPT = {
    0: ActionV2(lay_order=1, forage_share_q=_SHARES, army_choice=0, avoid_center=False),
    60: ActionV2(lay_order=2, forage_share_q=_SHARES, army_choice=0, avoid_center=False),
    90: ActionV2(lay_order=0, forage_share_q=_SHARES, army_choice=0, avoid_center=False),
}
B_SCRIPT = {
    0: ActionV2(lay_order=2, forage_share_q=_SHARES, army_choice=0, avoid_center=False),
    60: ActionV2(lay_order=1, forage_share_q=_SHARES, army_choice=0, avoid_center=False),
}


@dataclass(frozen=True, slots=True)
class EconomySmokeReport:
    ticks: int
    eggs_per_lay: int
    rows: tuple[str, ...]
    totals: tuple[str, ...]
    ledger: str
    events_hash: str
    state_hash: str
    checkpoint_hash: str

    def render(self) -> str:
        head = (f"economy smoke run, {self.ticks} ticks, eggs_per_lay={self.eggs_per_lay} - {LABEL}: "
                "no Combat, no QueenTerminal, no Terminal check (not a match, not a balance result)")
        return "\n".join([
            head, "", *self.rows, "", *self.totals, "", self.ledger, "",
            f"events_hash:     {self.events_hash}",
            f"state_hash:      {self.state_hash}",
            f"checkpoint_hash: {self.checkpoint_hash}",
        ])


def _census(state, side: int) -> tuple[int, int, int, int]:
    c = state.colonies[side]
    return (
        sum(1 for u in c.units if u.caste == WORKER), sum(1 for u in c.units if u.caste == SOLDIER),
        sum(1 for b in c.brood if b.caste == WORKER), sum(1 for b in c.brood if b.caste == SOLDIER),
    )


def run_economy_smoke(
    ticks: int = ECONOMY_SMOKE_TICKS, seed: int = ECONOMY_SMOKE_SEED, eggs_per_lay: int = 1,
    config: SimConfig | None = None,
) -> EconomySmokeReport:
    if config is None:
        base = default_config()
        config = dataclasses.replace(base, balance=dataclasses.replace(base.balance, eggs_per_lay=eggs_per_lay))
    else:
        eggs_per_lay = config.balance.eggs_per_lay
    state, run = reset(config, seed)
    initial_food = total_food(state)
    controllers = (ScriptedController(A_SCRIPT), ScriptedController(B_SCRIPT))
    all_events: list[Event] = []
    rows = [
        f"(row 'tick' = state.tick after the tick; harvest..dead- are sums over the last {BUCKET} ticks)",
        "tick | side  adults W/S  brood W/S  food | harvest upkeep eggs  hatch+ dead- | node food "
        + "/".join(NODE_NAMES[n] for n in (1, 2, 3, 4, 5)),
        "-----+-------------------------------------------------------------------------------------",
    ]
    bucket = EconomyTotals()
    for _ in range(ticks):
        result = step_tick(state, run, controllers, partial_economy=True, check_invariants=True)
        assert result.mode is TickMode.PARTIAL_ECONOMY and not result.physics_complete
        all_events.extend(result.events)
        bucket.add(result.events)
        if state.tick % BUCKET == 0 or state.tick == ticks:
            for side in SIDES:
                w, s, bw, bs = _census(state, side)
                t = bucket.sides[side]
                nodes = "/".join(str(state.node_food[n]) for n in (1, 2, 3, 4, 5))
                rows.append(
                    f"{state.tick:>4} | {'AB'[side]}    {w:>3}/{s:<3}      {bw:>2}/{bs:<2}     "
                    f"{state.colonies[side].food:>4} | {t.harvested:>7} {t.upkeep_paid:>6} {t.egg_spend:>4}  "
                    f"{sum(t.hatched):>5} {sum(t.starved):>5} | {nodes if side == 0 else ''}")
            bucket = EconomyTotals()

    totals = EconomyTotals()
    totals.add(all_events)
    lines = [f"totals over {ticks} ticks (per side):"]
    for side in SIDES:
        t = totals.sides[side]
        w, s, bw, bs = _census(state, side)
        lines.append(
            f"  {'AB'[side]}: harvest={t.harvested} upkeep_paid={t.upkeep_paid} (missing {t.upkeep_missing}) "
            f"egg_spend={t.egg_spend} eggs_laid={t.eggs_laid} lay_blocked_by_food={t.lay_blocked_food} "
            f"hatched W/S={t.hatched[0]}/{t.hatched[1]} starved W/S={t.starved[0]}/{t.starved[1]} "
            f"-> adults W/S={w}/{s}, brood W/S={bw}/{bs}, food={state.colonies[side].food}, "
            f"debt={state.colonies[side].starvation_debt}, rem10={state.colonies[side].upkeep_remainder10}")
    expected = totals.expected_total_food(initial_food)
    ledger = (f"food ledger: nodes+warehouses now {total_food(state)} == start {initial_food} + regen "
              f"{totals.regen_added} - upkeep {sum(s.upkeep_paid for s in totals.sides)} - eggs "
              f"{sum(s.egg_spend for s in totals.sides)} = {expected}: {'OK' if expected == total_food(state) else 'BROKEN'}")
    assert expected == total_food(state)

    return EconomySmokeReport(
        ticks=ticks, eggs_per_lay=eggs_per_lay, rows=tuple(rows), totals=tuple(lines), ledger=ledger,
        events_hash=events_hash(all_events), state_hash=state_hash(state),
        checkpoint_hash=checkpoint_hash(state, run),
    )


def main() -> None:
    print(run_economy_smoke().render())


if __name__ == "__main__":
    main()
