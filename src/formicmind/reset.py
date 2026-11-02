"""``reset(config, seed)``: exactly the FOR-1 v0.2.1 initial state, no tick executed."""

from __future__ import annotations

from collections.abc import Iterable

from .config import SimConfig
from .invariants import validate_config, validate_state
from .model import NOWHERE, BroodRecord, ColonyState, IntelMemory, NodeIntel, SimState, Unit
from .rules import (
    ARMY_GARRISON_HOME,
    LAY_NONE,
    NODE_COUNT,
    Q_TOTAL,
    SIDES,
    SOLDIER,
    WORKER,
)
from .runstate import RunState


def _home_node(config: SimConfig, side: int) -> int:
    return next(n.node_id for n in config.topology.nodes if n.home_side == side)


def initial_state(config: SimConfig) -> SimState:
    """Initial game state. Rules v0.2.1 consume no randomness, so it has no seed."""
    validate_config(config)
    bal = config.balance
    state = SimState(
        config=config,
        tick=0,
        node_food=[r.start_food for r in bal.resources],
        colonies=[],
        next_unit_id=1,
    )
    for side in SIDES:
        home = _home_node(config, side)
        units: list[Unit] = []
        for caste, count, hp in (
            (WORKER, bal.start_workers, bal.worker_hp),
            (SOLDIER, bal.start_soldiers, bal.soldier_hp),
        ):
            for _ in range(count):
                units.append(Unit(
                    unit_id=state.allocate_unit_id(), caste=caste, hp=hp, node=home,
                    from_node=NOWHERE, to_node=NOWHERE, arrival_tick=NOWHERE, target_node=home,
                ))
        brood: list[BroodRecord] = []
        state.colonies.append(ColonyState(
            side=side,
            food=bal.start_food,
            queen_hp=bal.queen_hp,
            next_lay_tick=0,
            upkeep_remainder10=0,
            starvation_debt=0,
            lay_order=LAY_NONE,
            forage_q=[Q_TOTAL, 0, 0, 0, 0, 0],  # all workers on "home"
            army_choice=ARMY_GARRISON_HOME,
            avoid_center=False,
            units=units,
            brood=brood,
            intel=IntelMemory(nodes=[NodeIntel() for _ in range(NODE_COUNT)]),
        ))
    validate_state(state)
    return state


def reset(
    config: SimConfig, seed: int, stream_ids: Iterable[str] = ()
) -> tuple[SimState, RunState]:
    """Return ``(SimState, RunState)``.

    ``seed`` is the u64 root seed of the RNG registry; the ``world`` stream is always
    registered, extra stream ids (e.g. ``policy:<policy_id>:<replicate>``) are added
    on request. The game state itself does not depend on the seed.
    """
    return initial_state(config), RunState.new(seed, stream_ids)
