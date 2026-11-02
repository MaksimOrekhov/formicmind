"""Immutable, side-filtered ``PolicySnapshot``.

One snapshot is frozen from the pre-action state at a Policy tick and used to build
both sides' observations, so neither side's action can influence the other's
observation. Each ``SideView`` holds only what that side may know: its own colony
and its own private intel record (already written by ``update_local_intel``). The
enemy colony, true food of unobserved nodes and all RNG are not copied in at all.

Everything is frozen dataclasses and tuples; the public ``SimConfig`` is frozen too.

``WindowStats`` are per-side counters of the elapsed policy window
(FOR-2 own_global[25..29]). The tick engine does not exist yet, so the caller
supplies them (zeros by default); where the engine accumulates them (and that they
must be part of a future checkpoint) is a decision for the tick PR.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import SimConfig
from .intel import IntelPhase, update_local_intel
from .model import SimState
from .rules import SIDES


@dataclass(frozen=True, slots=True)
class WindowStats:
    harvest: int = 0
    upkeep_paid: int = 0
    egg_spend: int = 0
    starvation_debt_delta: int = 0  # signed
    window_ticks: int = 0

    def __post_init__(self) -> None:
        values = (self.harvest, self.upkeep_paid, self.egg_spend,
                  self.starvation_debt_delta, self.window_ticks)
        if any(type(v) is not int for v in values):
            raise TypeError("WindowStats fields must be ints")
        if min(self.harvest, self.upkeep_paid, self.egg_spend, self.window_ticks) < 0:
            raise ValueError("window counters must be non-negative")


@dataclass(frozen=True, slots=True)
class UnitView:
    unit_id: int
    caste: int
    hp: int
    node: int  # NOWHERE when in transit
    to_node: int  # edge destination when in transit
    arrival_tick: int


@dataclass(frozen=True, slots=True)
class BroodView:
    caste: int
    hatch_tick: int


@dataclass(frozen=True, slots=True)
class NodeIntelView:
    seen_tick: int
    food_last: int
    enemy_workers_last: int
    enemy_soldiers_last: int


@dataclass(frozen=True, slots=True)
class SideView:
    side: int
    food: int
    queen_hp: int
    next_lay_tick: int
    upkeep_remainder10: int
    starvation_debt: int
    lay_order: int
    forage_q: tuple[int, ...]
    army_choice: int
    avoid_center: bool
    units: tuple[UnitView, ...]
    brood: tuple[BroodView, ...]
    intel_nodes: tuple[NodeIntelView, ...]  # by world node id
    enemy_queen_hp_last: int
    enemy_queen_seen_tick: int
    window: WindowStats


@dataclass(frozen=True, slots=True)
class PolicySnapshot:
    config: SimConfig
    tick: int
    views: tuple[SideView, SideView]

    def view(self, side: int) -> SideView:
        return self.views[side]


def freeze_policy_snapshot(
    state: SimState, windows: tuple[WindowStats, WindowStats] | None = None
) -> PolicySnapshot:
    """Freeze both sides' views. Intel must already be updated (``PRE_POLICY``)."""
    from .observation import check_observation_config  # avoid an import cycle

    config = state.config
    check_observation_config(config)
    if state.tick % config.policy_interval_k != 0:
        raise ValueError(f"tick {state.tick} is not a Policy tick (K={config.policy_interval_k})")
    windows = windows if windows is not None else (WindowStats(), WindowStats())
    if len(windows) != len(SIDES) or not all(isinstance(w, WindowStats) for w in windows):
        raise TypeError("windows must be one WindowStats per side")

    views = []
    for side in SIDES:
        c = state.colonies[side]
        views.append(SideView(
            side=side,
            food=c.food,
            queen_hp=c.queen_hp,
            next_lay_tick=c.next_lay_tick,
            upkeep_remainder10=c.upkeep_remainder10,
            starvation_debt=c.starvation_debt,
            lay_order=c.lay_order,
            forage_q=tuple(c.forage_q),
            army_choice=c.army_choice,
            avoid_center=c.avoid_center,
            units=tuple(
                UnitView(u.unit_id, u.caste, u.hp, u.node, u.to_node, u.arrival_tick)
                for u in c.units
            ),
            brood=tuple(BroodView(b.caste, b.hatch_tick) for b in c.brood),
            intel_nodes=tuple(
                NodeIntelView(n.seen_tick, n.food_last, n.enemy_workers_last, n.enemy_soldiers_last)
                for n in c.intel.nodes
            ),
            enemy_queen_hp_last=c.intel.enemy_queen_hp_last,
            enemy_queen_seen_tick=c.intel.enemy_queen_seen_tick,
            window=windows[side],
        ))
    return PolicySnapshot(config=config, tick=state.tick, views=(views[0], views[1]))


def begin_policy_phase(
    state: SimState, windows: tuple[WindowStats, WindowStats] | None = None
) -> PolicySnapshot:
    """``PRE_POLICY`` intel update for both sides, then one shared frozen snapshot.

    The two updates read only game state, never each other's intel, so their order
    does not matter.
    """
    for side in SIDES:
        update_local_intel(state, side, IntelPhase.PRE_POLICY)
    return freeze_policy_snapshot(state, windows)
