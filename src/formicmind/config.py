"""Typed, immutable configuration: pinned topology plus *candidate* balance.

Topology (nodes, edges) is part of the prototype rules. Everything numeric in
``CandidateBalance`` is a candidate from FOR-1 v0.2.1 and has not been validated
by the economic pilot; do not read it as an approved balance.

Integer units: food and HP in whole units, time in ticks, upkeep in tenths of a
food unit per tick.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NodeSpec:
    node_id: int
    name: str
    home_side: int  # side index owning this node as its nest, -1 if none


@dataclass(frozen=True, slots=True)
class EdgeSpec:
    a: int  # a < b
    b: int
    length: int  # ticks


@dataclass(frozen=True, slots=True)
class Topology:
    nodes: tuple[NodeSpec, ...]
    edges: tuple[EdgeSpec, ...]  # sorted by (a, b)


@dataclass(frozen=True, slots=True)
class NodeResource:
    node_id: int
    start_food: int
    capacity: int
    regen: int
    max_harvest: int


@dataclass(frozen=True, slots=True)
class CandidateBalance:
    resources: tuple[NodeResource, ...]  # one per node, ordered by node_id
    start_food: int
    start_workers: int
    start_soldiers: int
    worker_hp: int
    soldier_hp: int
    queen_hp: int
    worker_egg_cost: int
    soldier_egg_cost: int
    eggs_per_lay: int
    lay_interval_ticks: int
    worker_hatch_delay: int
    soldier_hatch_delay: int
    worker_upkeep_tenths: int
    soldier_upkeep_tenths: int
    starvation_unit_cost: int  # unpaid food units per adult lost
    soldier_damage: int
    home_defense_num: int  # home-defender damage multiplier = num / den
    home_defense_den: int


@dataclass(frozen=True, slots=True)
class SimConfig:
    rules_version: str
    balance_version: str
    topology: Topology
    balance: CandidateBalance
    policy_interval_k: int
    t_max: int
