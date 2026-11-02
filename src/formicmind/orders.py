"""Applying orders: worker quotas (FOR-2 section 4.2) and unit targets.

``apply_joint_orders`` is atomic: both sides' plans are computed from the unmodified
state first and only then committed; it accepts only ``ValidatedActionV2`` objects and re-checks their
structure before planning (mask legality is checked by the caller with the observation). It sets standing orders and unit ``target_node``;
it does **not** move anybody (dispatch is ``movement.dispatch_available``).

Worker assignment follows FOR-2 section 4.2 literally:

1. ``D_i = N * q_i``, ``base_i = D_i // Q``, ``rem_i = D_i mod Q``; the ``left = N - sum(base)``
   leftover workers go to the targets with the largest ``rem_i`` (ties: category order
   home, N1, N2, C, N4, N5).
2. Concrete workers are ordered by current place - stationed units by node in the side's
   canonical order, then units in transit by next node and ETA - and by ``unit_id`` inside
   a place. Targets are filled in category order. Units in transit keep their edge and ETA;
   only ``target_node`` changes.

Audit item D5: this order does not minimise travel (e.g. workers N2=6, C=6 with new quotas
N1=6, N2=6 cost 48 worker-ticks instead of 12). That is a *rules* question and is pinned by
a test, not optimised here.

All soldiers get the same target ``local_to_world(side, army_choice)`` (0 = home,
1..5 = defend that node, 6 = enemy home). ``avoid_center`` applies to soldiers' routing
only; workers always take the plain shortest path.
"""

from __future__ import annotations

from .actions import InvalidAction, ValidatedActionV2, check_action_structure
from .events import OrderEvent, RetargetEvent, Phase
from .model import NOWHERE, ColonyState, SimState, Unit
from .rules import ARMY_GARRISON_HOME, FORAGE_CATEGORIES, Q_TOTAL, SIDES, SOLDIER, WORKER, local_to_world


def worker_quotas(worker_count: int, shares_q: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Integer worker counts per forage category from quantized shares (largest remainder)."""
    if len(shares_q) != FORAGE_CATEGORIES or sum(shares_q) != Q_TOTAL or min(shares_q) < 0:
        raise ValueError("shares must be 6 non-negative ints summing to Q")
    if worker_count < 0:
        raise ValueError("worker_count must be non-negative")
    demand = [worker_count * q for q in shares_q]
    base = [d // Q_TOTAL for d in demand]
    rem = [d % Q_TOTAL for d in demand]
    left = worker_count - sum(base)
    assert 0 <= left <= sum(1 for r in rem if r > 0)
    for i in sorted(range(FORAGE_CATEGORIES), key=lambda i: (-rem[i], i))[:left]:
        base[i] += 1
    assert sum(base) == worker_count
    return tuple(base)


def _place_key(side: int, unit: Unit) -> tuple[int, int, int, int]:
    # local_to_world is an involution, so it also maps world -> local
    if unit.node != NOWHERE:
        return (0, local_to_world(side, unit.node), 0, unit.unit_id)
    return (1, local_to_world(side, unit.to_node), unit.arrival_tick, unit.unit_id)


def _plan_side(state: SimState, side: int, action_q: tuple[int, ...], army_choice: int
               ) -> tuple[dict[int, int], tuple[int, ...]]:
    colony: ColonyState = state.colonies[side]
    workers = sorted((u for u in colony.units if u.caste == WORKER), key=lambda u: _place_key(side, u))
    quotas = worker_quotas(len(workers), action_q)
    targets: dict[int, int] = {}
    index = 0
    for category, count in enumerate(quotas):
        for unit in workers[index:index + count]:
            targets[unit.unit_id] = local_to_world(side, category)
        index += count
    army_target = local_to_world(side, army_choice)
    for unit in colony.units:
        if unit.caste == SOLDIER:
            targets[unit.unit_id] = army_target
    return targets, quotas


def apply_joint_orders(
    state: SimState, actions: tuple[ValidatedActionV2, ValidatedActionV2],
    phase: int = Phase.POLICY,
) -> list[OrderEvent | RetargetEvent]:
    if not isinstance(actions, tuple) or len(actions) != len(SIDES) or not all(
        isinstance(a, ValidatedActionV2) for a in actions
    ):
        raise TypeError("apply_joint_orders needs one ValidatedActionV2 per side")

    for validated in actions:  # a hand-built wrapper must not smuggle in a malformed action
        check_action_structure(validated.action)
        if validated.action.army_choice == ARMY_GARRISON_HOME and validated.action.avoid_center:
            raise InvalidAction("garrison_home cannot set avoid_center")  # FOR-2 section 4.1

    plans = [
        _plan_side(state, side, actions[side].action.forage_share_q, actions[side].action.army_choice)
        for side in SIDES
    ]
    # --- commit (nothing below can fail on valid input) ---
    tick = state.tick
    orders: list[OrderEvent | RetargetEvent] = []
    retargets: list[RetargetEvent] = []
    for side in SIDES:
        action = actions[side].action
        colony = state.colonies[side]
        colony.lay_order = action.lay_order
        colony.forage_q = list(action.forage_share_q)
        colony.army_choice = action.army_choice
        colony.avoid_center = action.avoid_center
        orders.append(OrderEvent(
            tick=tick, phase=int(phase), side=side, lay_order=action.lay_order,
            forage_share_q=tuple(action.forage_share_q), army_choice=action.army_choice,
            avoid_center=action.avoid_center,
            army_target_node=local_to_world(side, action.army_choice),
            worker_quota=plans[side][1],
        ))
        for unit in colony.units:
            new_target = plans[side][0][unit.unit_id]
            if new_target != unit.target_node:
                retargets.append(RetargetEvent(
                    tick=tick, phase=int(phase), side=side, unit_id=unit.unit_id, caste=unit.caste,
                    old_target=unit.target_node, new_target=new_target))
                unit.target_node = new_target
    retargets.sort(key=lambda e: e.unit_id)
    return [*orders, *retargets]
