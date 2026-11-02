"""Movement-only smoke run (60-100 ticks). NOT a match, NOT a strategy episode.

Scripted scenario (side B keeps the reset orders: all workers home, garrison):

* t=0   A sends 8 workers to N1 (4 stay home) and all 4 soldiers to defend C.
* t=30  A orders a raid with ``avoid_center`` while its soldiers stand in C: C is the
        current node, so they may leave through it (audit D1b).
* t=60  A repeats the same orders.

There is no combat, economy or hatching in this movement-only mode (the economy smoke is
``formicmind.economy_smoke``): the raiding soldiers reach B and stand there next
to B's own soldiers; each side's soldiers keep the other's units blocked, which is only an
artifact of combat not being implemented. Run it with ``python -m formicmind.smoke``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .actions import ActionV2
from .events import Event, events_hash
from .model import NOWHERE, SimState
from .observation import build_observation
from .reset import reset
from .rules import NODE_NAMES, SOLDIER, WORKER, default_config
from .serialization import checkpoint_hash, state_hash
from .snapshot import begin_policy_phase
from .tick import ScriptedController, step_tick

SMOKE_SEED = 20261102  # arbitrary engineering constant, not a FOR-4 sealed seed
SMOKE_TICKS = 90

_CASTE = {WORKER: "workers", SOLDIER: "soldiers"}
_SIDE = ("A", "B")

A_SCRIPT = {
    0: ActionV2(lay_order=0, forage_share_q=(21845, 43690, 0, 0, 0, 0), army_choice=3, avoid_center=False),
    30: ActionV2(lay_order=0, forage_share_q=(21845, 43690, 0, 0, 0, 0), army_choice=6, avoid_center=True),
    60: ActionV2(lay_order=0, forage_share_q=(21845, 43690, 0, 0, 0, 0), army_choice=6, avoid_center=True),
}


@dataclass(frozen=True, slots=True)
class SmokeReport:
    ticks: int
    lines: tuple[str, ...]  # aggregated, human-readable movement events
    positions: tuple[str, ...]  # final positions per side
    observation_note: str
    event_count: int
    events_hash: str
    state_hash: str
    checkpoint_hash: str

    def render(self) -> str:
        head = f"movement-only smoke run, {self.ticks} ticks (not a match: no combat/economy/hatching)"
        body = ["", "events (aggregated):", *("  " + line for line in self.lines),
                "", "final positions:", *("  " + p for p in self.positions),
                "", self.observation_note,
                "", f"raw events: {self.event_count}",
                f"events_hash:     {self.events_hash}",
                f"state_hash:      {self.state_hash}",
                f"checkpoint_hash: {self.checkpoint_hash}"]
        return "\n".join([head, *body])


def summarize_events(events: list[Event]) -> list[str]:
    """Group depart/arrive events of units that moved together into single lines."""
    groups: Counter = Counter()
    order: list[tuple] = []
    for e in events:
        if e.kind == "depart":
            key = (e.tick, 1, e.side, e.caste, e.from_node, e.to_node, e.arrival_tick, e.phase)
        elif e.kind == "arrive":
            key = (e.tick, 0, e.side, e.caste, e.node, -1, -1, e.phase)
        else:
            continue
        if key not in groups:
            order.append(key)
        groups[key] += 1
    lines = []
    for key in sorted(order):
        tick, is_depart, side, caste, a, b, eta, _phase = key
        n = groups[key]
        if is_depart:
            lines.append(f"t={tick:<3} {_SIDE[side]} sends {n} {_CASTE[caste]} {NODE_NAMES[a]}->{NODE_NAMES[b]} (arrive t={eta})")
        else:
            lines.append(f"t={tick:<3} {_SIDE[side]} {n} {_CASTE[caste]} arrive at {NODE_NAMES[a]}")
    return lines


def _positions(state: SimState) -> list[str]:
    out = []
    for side, colony in enumerate(state.colonies):
        places: Counter = Counter()
        for u in colony.units:
            where = NODE_NAMES[u.node] if u.node != NOWHERE else (
                f"{NODE_NAMES[u.from_node]}->{NODE_NAMES[u.to_node]}")
            places[(where, u.caste)] += 1
        text = ", ".join(f"{n} {_CASTE[c]} at {w}" for (w, c), n in sorted(places.items()))
        out.append(f"{_SIDE[side]}: {text}")
    return out


def run_movement_smoke(ticks: int = SMOKE_TICKS, seed: int = SMOKE_SEED) -> SmokeReport:
    state, run = reset(default_config(), seed)
    controllers = (ScriptedController(A_SCRIPT), ScriptedController({}))
    events: list[Event] = []
    note = ""
    for _ in range(ticks):
        if state.tick == 60:  # what A can see when it decides at t=60 (its soldiers stand in B)
            obs = build_observation(begin_policy_phase(state.clone(), tuple(run.windows)), 0)
            note = (f"A's observation at t=60: enemy home known={obs.block('enemy_known')[6]:g}, "
                    f"current={obs.block('enemy_current')[6]:g}, enemy queen hp ratio="
                    f"{obs.block('enemy_queen_hp_last')[0]:g}, window_ticks/K={obs.block('own_global')[17]:g}")
        events.extend(step_tick(state, run, controllers, movement_only=True).events)
    return SmokeReport(
        ticks=ticks,
        lines=tuple(summarize_events(events)),
        positions=tuple(_positions(state)),
        observation_note=note or "(no policy tick at 60 within this run)",
        event_count=len(events),
        events_hash=events_hash(events),
        state_hash=state_hash(state),
        checkpoint_hash=checkpoint_hash(state, run),
    )


def main() -> None:
    print(run_movement_smoke().render())


if __name__ == "__main__":
    main()
