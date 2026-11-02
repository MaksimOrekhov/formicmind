"""Fixture states for the agent-boundary tests (no tick physics is simulated).

Seeds are arbitrary engineering constants, never FOR-4 sealed seeds.
"""

from formicmind import default_config, reset
from formicmind.model import NOWHERE, BroodRecord, IntelMemory, NodeIntel, SimState, Unit
from formicmind.rules import MIRROR

A, N1, N2, C, N4, N5, B = range(7)


def fresh_state(tick: int = 0) -> SimState:
    state, _ = reset(default_config(), 20261102)
    state.tick = tick
    return state


def place(unit: Unit, node: int) -> None:
    unit.node, unit.from_node, unit.to_node, unit.arrival_tick = node, NOWHERE, NOWHERE, NOWHERE


def transit(unit: Unit, src: int, dst: int, arrival_tick: int) -> None:
    unit.node, unit.from_node, unit.to_node, unit.arrival_tick = NOWHERE, src, dst, arrival_tick


def unit_of(state: SimState, side: int, unit_id: int) -> Unit:
    return next(u for u in state.colonies[side].units if u.unit_id == unit_id)


def busy_state(tick: int = 60) -> SimState:
    """An asymmetric mid-game-like fixture (positions set by hand, not simulated)."""
    s = fresh_state(tick)
    a, b = s.colonies
    # side A: workers 1-12, soldiers 13-16
    for uid in (1, 2, 3):
        place(unit_of(s, 0, uid), N1)
    place(unit_of(s, 0, 4), C)
    transit(unit_of(s, 0, 5), N1, C, tick + 1)
    transit(unit_of(s, 0, 6), N1, C, tick + 1)
    transit(unit_of(s, 0, 7), N1, C, tick + 2)
    unit_of(s, 0, 7).hp = 11
    for uid in (13, 14):
        place(unit_of(s, 0, uid), N2)
    transit(unit_of(s, 0, 15), A, N2, tick + 2)
    unit_of(s, 0, 15).hp = 17
    a.food, a.queen_hp, a.next_lay_tick = 130, 200, tick + 4
    a.upkeep_remainder10, a.starvation_debt = 6, 3
    a.lay_order, a.forage_q, a.army_choice, a.avoid_center = 1, [10000, 30000, 25535, 0, 0, 0], 3, False
    a.brood = [BroodRecord(33, 0, tick - 10, tick + 10), BroodRecord(34, 1, tick - 15, tick + 15)]
    s.next_unit_id = 35
    # side B: workers 17-28, soldiers 29-32
    for uid in (29, 30):
        place(unit_of(s, 1, uid), C)
    for uid in (17, 18):
        place(unit_of(s, 1, uid), N4)
    transit(unit_of(s, 1, 19), N4, C, tick + 1)
    b.food, b.queen_hp = 40, 90
    b.lay_order, b.forage_q, b.army_choice, b.avoid_center = 2, [0, 0, 0, 0, 65535, 0], 6, True
    s.node_food = [0, 33, 21, 77, 45, 12, 0]
    # an older sighting of N5 by A (made at tick 30)
    a.intel.nodes[N5] = NodeIntel(seen_tick=30, food_last=25, enemy_workers_last=2, enemy_soldiers_last=1)
    return s


def mirror_state(state: SimState) -> SimState:
    """The same situation seen from the other seat: sides swapped, nodes mirrored."""
    mapped = lambda n: n if n == NOWHERE else MIRROR[n]  # noqa: E731
    out = state.clone()
    out.node_food = [state.node_food[MIRROR[i]] for i in range(7)]
    for side in (0, 1):
        src = state.colonies[1 - side].clone()
        src.side = side
        for u in src.units:
            u.node, u.from_node, u.to_node = mapped(u.node), mapped(u.from_node), mapped(u.to_node)
            u.target_node = mapped(u.target_node)
        src.intel = IntelMemory(
            nodes=[src.intel.nodes[MIRROR[i]] for i in range(7)],
            enemy_queen_hp_last=src.intel.enemy_queen_hp_last,
            enemy_queen_seen_tick=src.intel.enemy_queen_seen_tick,
        )
        out.colonies[side] = src
    return out
