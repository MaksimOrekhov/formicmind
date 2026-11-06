"""FormicMind Ecosystem v1 (E1): shared multi-colony world, E1-a = map generator, typed schemas, hashes, CLI.

Independent of the L0 engine: this package imports only ``formicmind.canonical`` and ``formicmind.rng`` from L0 and nothing in L0
imports it (enforced by tests/test_eco_boundary.py). Design: docs/ecosystem_v1_{rfc,contracts,economy,migration}.md.
"""

from .config import ConfigE1, default_config
from .initial import World, build_world, initial_world_state
from .mapgen import MapGenerationError, generate_map
from .model import MapE1, WorldStateE1

__all__ = ["ConfigE1", "MapE1", "MapGenerationError", "World", "WorldStateE1", "build_world", "default_config",
           "generate_map", "initial_world_state"]
