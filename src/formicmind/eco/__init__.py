"""FormicMind Ecosystem v1 (E1): shared multi-colony world. E1-a = map generator, typed schemas, hashes, CLI; E1-b = live economy;
E1-c = exploration with private colony memory, contacts, validated orders, local combat, queen death and orphans, typed events.

Independent of the L0 engine: this package imports only ``formicmind.canonical`` and ``formicmind.rng`` from L0 and nothing in L0
imports it (enforced by tests/test_eco_boundary.py). Design: docs/ecosystem_v1_{rfc,contracts,economy,migration}.md.
"""

from .config import ConfigE1, default_config
from .initial import World, build_world, initial_world_state
from .mapgen import MapGenerationError, generate_map
from .model import MapE1, WorldStateE1
from .engine import RouteTable, TickRecord, step_world_tick
from .events import EventE1, EventLog
from .fixtures import FIXTURE_CHOICES, FIXTURE_NAMES, FixtureController, PeacefulFixture, fixture_controller
from .orders import IllegalOrderError, order_attack, order_guard, order_scout, order_withdraw
from .simulation import SimulationResult, benchmark_simulation, interaction_sweep, run_simulation, sweep_simulations

__all__ = ["ConfigE1", "MapE1", "MapGenerationError", "World", "WorldStateE1", "build_world", "default_config",
           "generate_map", "initial_world_state", "RouteTable", "TickRecord", "step_world_tick",
           "FIXTURE_NAMES", "FIXTURE_CHOICES", "PeacefulFixture", "fixture_controller", "SimulationResult", "run_simulation",
           "sweep_simulations", "benchmark_simulation", "interaction_sweep", "EventE1", "EventLog", "FixtureController",
           "IllegalOrderError", "order_scout", "order_guard", "order_attack", "order_withdraw"]
