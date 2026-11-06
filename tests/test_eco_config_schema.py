"""FOR-44: ConfigE1 / MapE1 / WorldStateE1 / checkpoint - strict canonical schemas, hashes, invalid input rejection."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from formicmind.canonical import CanonicalError, canonical_dumps, canonical_loads
from formicmind.eco.codec import SchemaError, from_canonical, to_canonical
from formicmind.eco.config import (
    SCENARIOS, SIZES, ConfigE1, ConfigError, config_from_bytes, config_from_canonical, config_hash, config_to_bytes,
    default_config, patch_parameters, scaled, validate_config,
)
from formicmind.eco.initial import build_world, initial_world_state
from formicmind.eco.model import MapE1, PileState, WorldStateE1
from formicmind.eco.serialization import (
    checkpoint_from_bytes, checkpoint_hash, checkpoint_to_bytes, compute_map_hash, map_from_bytes, map_to_bytes,
    state_from_bytes, state_hash, state_to_bytes,
)
from formicmind.eco.validate import InvariantError, validate_state

SEED = 42


@pytest.fixture(scope="module")
def world():
    return build_world(default_config(8, "moderate"), SEED)


# ------------------------------------------------------------------------------------------------ config


def test_default_configs_follow_the_documents():
    for n in SIZES:
        for scenario in SCENARIOS:
            cfg = default_config(n, scenario)
            m = cfg.map_params
            assert cfg.n_colonies == n and cfg.scenario == scenario
            assert m.n_nodes == 8 * n + 32 and 80 <= m.n_nodes <= 120
            assert cfg.patch_count() == 4 * n and 24 <= cfg.patch_count() <= 40
            assert (cfg.horizon, cfg.decision_k, cfg.alert_interval) == (6000, 30, 10)
    big = default_config(8, "moderate")
    assert (big.map_params.n_nodes, big.patch_count()) == (96, 32)


@pytest.mark.parametrize("scenario, expected", [
    # (capacity, regen, site_cap) per zone: docs/ecosystem_v1_economy.md section 3, sigma = 1.5 / 1.0 / 0.6
    ("abundance", {"near": (150, 8, 15), "border": (180, 9, 18), "far": (240, 12, 24)}),
    ("moderate", {"near": (100, 5, 10), "border": (120, 6, 12), "far": (160, 8, 16)}),
    ("severe", {"near": (60, 3, 6), "border": (72, 4, 7), "far": (96, 5, 10)}),
])
def test_scenario_scaling_reproduces_the_economy_document(scenario, expected):
    cfg = default_config(8, scenario)
    for zone, triple in expected.items():
        assert patch_parameters(cfg, zone) == triple


def test_scenario_scaling_matches_the_budget_calculator_rounding():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "ecosystem_budget.py"
    spec = importlib.util.spec_from_file_location("ecosystem_budget_cmp", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for scenario, sigma in (("abundance", 1.5), ("moderate", 1.0), ("severe", 0.6)):
        sc = module.scenario(scenario, sigma)
        cfg = default_config(8, scenario)
        for cls, zone in zip(sc.classes, ("near", "border", "far")):
            assert (cls.capacity, cls.regen, cls.site_cap) == patch_parameters(cfg, zone), (scenario, zone)
    assert scaled(7, "severe") == 4 and scaled(5, "abundance") == 8 and scaled(1, "severe") == 1


def test_config_roundtrip_and_hash_are_stable_and_sensitive():
    cfg = default_config(8, "moderate")
    assert config_from_bytes(config_to_bytes(cfg)) == cfg
    assert config_hash(cfg) == config_hash(config_from_bytes(config_to_bytes(cfg)))
    others = {
        config_hash(replace(cfg, horizon=5000)), config_hash(default_config(7, "moderate")), config_hash(default_config(8, "severe")),
        config_hash(replace(cfg, economy=replace(cfg.economy, lifespan_worker=1500))),
        config_hash(replace(cfg, options=replace(cfg.options, stop_when_alone=True))),
    }
    assert len(others) == 5 and config_hash(cfg) not in others


def test_golden_config_hash_regression():
    # regression values issued by this implementation (as in L0 docs/foundation.md section 4): authoritative after review
    assert config_hash(default_config(8, "moderate")) == "05a9b71d67fadcbb73595ad5e8d522f8b59e9fe64dc1007275bb3f161439e909"


@pytest.mark.parametrize("mutate", [
    lambda c: replace(c, n_colonies=5), lambda c: replace(c, n_colonies=9), lambda c: replace(c, scenario="mild"),
    lambda c: replace(c, schema="formicmind.config/1"), lambda c: replace(c, eco_rules_version="l0-proto-FOR1-v0.2.2"),
    lambda c: replace(c, horizon=0), lambda c: replace(c, decision_k=0), lambda c: replace(c, alert_interval=31),
    lambda c: replace(c, map_params=replace(c.map_params, n_nodes=79)), lambda c: replace(c, map_params=replace(c.map_params, n_nodes=121)),
    lambda c: replace(c, map_params=replace(c.map_params, n_border=0, n_far=0)),
    lambda c: replace(c, map_params=replace(c.map_params, n_border=20, n_far=20)),
    lambda c: replace(c, map_params=replace(c.map_params, max_edge_ticks=1)),
    lambda c: replace(c, map_params=replace(c.map_params, max_degree=1)),
    lambda c: replace(c, map_params=replace(c.map_params, max_attempts=0)),
    lambda c: replace(c, economy=replace(c.economy, worker_hp=0)),
    lambda c: replace(c, economy=replace(c.economy, narrow_lane_cap=9)),
    lambda c: replace(c, economy=replace(c.economy, start_food=601)),
    lambda c: replace(c, economy=replace(c.economy, start_age_spread_permille=1001)),
    lambda c: replace(c, economy=replace(c.economy, age_bucket=0)),
    lambda c: replace(c, economy=replace(c.economy, lay_interval_ticks=-1)),
    lambda c: replace(c, resources=replace(c.resources, far=replace(c.resources.far, site_cap=1))),
    lambda c: replace(c, resources=replace(c.resources, near=replace(c.resources.near, capacity=0))),
])
def test_invalid_configs_are_rejected(mutate):
    cfg = mutate(default_config(8, "moderate"))
    with pytest.raises(ConfigError):
        validate_config(cfg)
    with pytest.raises((ConfigError, CanonicalError)):
        config_from_canonical(to_canonical(cfg))


def test_default_config_rejects_bad_arguments():
    for args in ((5, "moderate"), (9, "moderate"), (8, "extreme")):
        with pytest.raises(ConfigError):
            default_config(*args)
    with pytest.raises(ConfigError):
        default_config(8, "moderate", n_nodes=60)


def test_config_loader_is_strict():
    data = to_canonical(default_config(8, "moderate"))
    for broken in (
        {**data, "extra": 1}, {k: v for k, v in data.items() if k != "horizon"}, {**data, "horizon": True}, {**data, "horizon": "6000"},
        {**data, "horizon": 1.5}, {**data, "scenario": 7}, {**data, "economy": {**data["economy"], "oops": 1}},
        {**data, "economy": [1, 2]}, {**data, "options": {**data["options"], "stop_when_alone": 0}}, [data], None,
    ):
        with pytest.raises(CanonicalError):
            config_from_canonical(broken)
    with pytest.raises(CanonicalError):
        config_from_bytes(b'{"n_colonies": 8, "n_colonies": 7}')  # duplicate keys
    with pytest.raises(CanonicalError):
        config_from_bytes(b'{ "n_colonies":8}')  # not canonical bytes


# ------------------------------------------------------------------------------------------------ map


def test_map_roundtrip_and_tamper_detection(world):
    cfg, world_map = world.config, world.world_map
    data = map_to_bytes(world_map)
    assert map_from_bytes(data, cfg) == world_map
    assert world_map.map_hash == compute_map_hash(world_map) and len(world_map.map_hash) == 64

    doc = canonical_loads(data)
    tampered = copy.deepcopy(doc)
    tampered["edges"][0]["length"] = 2 if tampered["edges"][0]["length"] != 2 else 3
    with pytest.raises(CanonicalError):  # the hash no longer matches the content
        map_from_bytes(canonical_dumps(tampered), cfg)
    with pytest.raises(CanonicalError):  # a different config must not accept this map
        map_from_bytes(data, default_config(8, "severe"))
    with pytest.raises(CanonicalError):
        map_from_bytes(data, default_config(7, "moderate"))


@pytest.mark.parametrize("edit", [
    lambda d: d["nodes"].reverse(),
    lambda d: d["nests"].__setitem__(0, d["nests"][1]),
    lambda d: d["nodes"][0].__setitem__("x", -1),
    lambda d: d["edges"].append(copy.deepcopy(d["edges"][0])),
    lambda d: d["edges"][0].__setitem__("lane_cap", 5),
    lambda d: d["edges"][0].__setitem__("length", 9),
    lambda d: d.__setitem__("schema", "formicmind.map/1"),
    lambda d: d.__setitem__("map_seed", 42),
    lambda d: d.__setitem__("map_seed", "042"),
    lambda d: d["generation"].__setitem__("algorithm", "other"),
    lambda d: d["generation"].__setitem__("attempts", 0),
    lambda d: d.__setitem__("n_colonies", 7),
    lambda d: d["nodes"][0].__setitem__("kind", "teleporter"),
])
def test_invalid_maps_are_rejected(world, edit):
    doc = copy.deepcopy(canonical_loads(map_to_bytes(world.world_map)))
    edit(doc)
    with pytest.raises(CanonicalError):
        map_from_bytes(canonical_dumps(doc), world.config)


def test_a_patch_with_wrong_parameters_or_missing_patch_spec_is_rejected(world):
    doc = copy.deepcopy(canonical_loads(map_to_bytes(world.world_map)))
    patch = next(n for n in doc["nodes"] if n["patch"])
    patch["patch"][0]["regen"] += 1
    with pytest.raises(CanonicalError):
        map_from_bytes(canonical_dumps(doc), world.config)
    doc = copy.deepcopy(canonical_loads(map_to_bytes(world.world_map)))
    next(n for n in doc["nodes"] if n["patch"])["patch"] = []
    with pytest.raises(CanonicalError):
        map_from_bytes(canonical_dumps(doc), world.config)


def test_u64_seeds_survive_beyond_the_javascript_safe_range():
    cfg = default_config(6, "severe")
    seed = (1 << 64) - 1
    w = build_world(cfg, seed)
    assert w.world_map.map_seed == seed
    assert canonical_loads(map_to_bytes(w.world_map))["map_seed"] == str(seed)
    assert map_from_bytes(map_to_bytes(w.world_map), cfg) == w.world_map


# ------------------------------------------------------------------------------------------------ state


def test_initial_state_matches_the_economy_start(world):
    cfg, state = world.config, world.state
    e = cfg.economy
    assert state.tick == 0 and len(state.colonies) == 8 and state.piles == [] and state.edge_flow == []
    assert [p.stock for p in state.patches] == [n.patch[0].start_stock for n in world.world_map.nodes if n.patch]
    assert sum(p.stock for p in state.patches) == 16 * 100 + 8 * 120 + 8 * 160  # every patch starts full (moderate)
    for col in state.colonies:
        mine = [c for c in state.cohorts if c.colony_id == col.colony_id]
        workers = sum(c.count for c in mine if c.caste == "worker")
        soldiers = sum(c.count for c in mine if c.caste == "soldier")
        assert (workers, soldiers, col.food, col.queen_hp, col.status) == (e.start_workers, e.start_soldiers, e.start_food, e.queen_hp, "alive")
        assert sum(b.count for b in col.ages.workers) == workers and sum(b.count for b in col.ages.soldiers) == soldiers
        assert all(c.loc.kind == "node" and c.loc.node_id == world.world_map.nests[col.colony_id] for c in mine)
        assert col.brood == [] and col.orders.egg_level == 0 and [s.mission for s in col.orders.squads] == ["none"] * 3
    assert [c.cohort_id for c in state.cohorts] == list(range(len(state.cohorts))) and state.next_cohort_id == len(state.cohorts)
    assert state.world_outcome.finished is False and not any(o.finished for o in state.outcomes)


def test_initial_age_ledger_has_no_demographic_cliff(world):
    col = world.state.colonies[0]
    starts = [b.bucket_start for b in col.ages.workers]
    assert starts == sorted(starts) and min(starts) >= -(world.config.economy.lifespan_worker // 2) and max(starts) <= 0
    assert max(b.count for b in col.ages.workers) <= 2  # spread over many buckets, not one cohort born at tick 0


def test_colony_memory_contains_only_the_local_neighbourhood(world):
    """No hidden information: only nodes within the initial radius are visited; foreign nests / colonies / unseen nodes are absent."""
    cfg, world_map = world.config, world.world_map
    from formicmind.eco.graph import adjacency, dijkstra

    adj = adjacency(len(world_map.nodes), [(e.a, e.b, e.length) for e in world_map.edges])
    for col in world.state.colonies:
        nest = world_map.nests[col.colony_id]
        dist = dijkstra(adj, nest)
        mem = col.memory
        assert mem.nodes[0].node_id == nest and mem.nodes[0].visited
        visited = {n.node_id for n in mem.nodes if n.visited}
        assert visited == {v for v, d in enumerate(dist) if d <= cfg.economy.initial_known_radius}
        known = {n.node_id for n in mem.nodes}
        assert all(dist[v] <= cfg.economy.initial_known_radius + max(e.length for e in world_map.edges) for v in known)
        other_nests = set(world_map.nests) - {nest}
        assert not (known & other_nests), "a foreign nest must not be known at tick 0"
        assert mem.foreign == [] and mem.sightings == [] and mem.events == [] and mem.piles_seen == []
        for n in mem.nodes:
            if not n.visited:
                assert (n.kind, n.patch_capacity, n.seen_tick, n.last_stock_tick) == ("unknown", 0, -1, -1)
        # edges are known only next to visited nodes
        slots = {i: n for i, n in enumerate(mem.nodes)}
        assert all(slots[e.slot_a].visited or slots[e.slot_b].visited for e in mem.edges)


def test_state_roundtrip_hash_and_golden(world):
    data = state_to_bytes(world.state)
    again = state_from_bytes(data, world.config, world.world_map)
    assert again == world.state and state_hash(again) == state_hash(world.state)
    assert state_hash(world.state) == "03ac8f74e7d35df5b48a7ce3206171acd726ef5e9889317df7380ab727d82d81"
    assert world.world_map.map_hash == "fce2596c0957b76dedb700165c4170b155aeae039c458c45ee7ddcb71fd6600d"
    other = initial_world_state(world.config, world.world_map)
    assert other == world.state and state_hash(other) == state_hash(world.state)
    mutated = copy.deepcopy(world.state)
    mutated.colonies[3].food -= 1
    assert state_hash(mutated) != state_hash(world.state)


def _mutated(world, edit):
    state = copy.deepcopy(world.state)
    edit(state)
    return state


@pytest.mark.parametrize("edit", [
    lambda s: setattr(s.colonies[0], "food", 601),
    lambda s: setattr(s.colonies[0], "food", -1),
    lambda s: setattr(s.colonies[0], "queen_hp", 241),
    lambda s: setattr(s.colonies[0], "status", "orphan"),  # orphan needs a dead queen
    lambda s: setattr(s.cohorts[0], "count", s.cohorts[0].count + 1),  # ledger no longer matches the cohorts
    lambda s: setattr(s.cohorts[0], "hp_total", 1),
    lambda s: setattr(s.cohorts[1], "load", 3),  # soldiers carry nothing
    lambda s: setattr(s.cohorts[0], "cohort_id", 5),
    lambda s: setattr(s, "next_cohort_id", 0),
    lambda s: setattr(s.patches[0], "stock", 10**6),
    lambda s: s.patches.pop(),
    lambda s: setattr(s, "tick", s.tick + 10**6),
    lambda s: setattr(s, "map_hash", "0" * 64),
    lambda s: setattr(s, "config_hash", "0" * 64),
    lambda s: s.colonies[0].ages.workers.append(s.colonies[0].ages.workers[0]),
    lambda s: setattr(s.colonies[0].ages.workers[0], "bucket_start", -7),
    lambda s: setattr(s.colonies[0].memory.nodes[0], "visited", False),
    lambda s: setattr(s.colonies[0].memory.nodes[-1], "kind", "nest"),  # information about an unvisited node
    lambda s: setattr(s.colonies[0].memory.nodes[-1], "seen_tick", 0),
    lambda s: setattr(s.colonies[0].memory.nodes[1], "node_id", s.colonies[0].memory.nodes[0].node_id),
    lambda s: setattr(s.colonies[0].memory.edges[0], "length", 99),
    lambda s: setattr(s.colonies[0].orders, "egg_level", 4),
    lambda s: setattr(s.colonies[0].orders, "forage_targets", [1]),  # share list missing
    lambda s: s.colonies[0].orders.squads.pop(),
    lambda s: s.outcomes.pop(),
    lambda s: setattr(s.outcomes[0], "finished", True),
    lambda s: setattr(s.world_outcome, "reason", "timeout"),
    lambda s: setattr(s, "piles", [PileState(0, -5)]),
])
def test_invalid_states_are_rejected(world, edit):
    with pytest.raises(InvariantError):
        validate_state(_mutated(world, edit), world.config, world.world_map)


def test_state_loader_rejects_schema_violations(world):
    doc = canonical_loads(state_to_bytes(world.state))
    for edit in (lambda d: d.pop("tick"), lambda d: d.__setitem__("tick", True), lambda d: d.__setitem__("extra", 1),
                 lambda d: d["colonies"][0].__setitem__("status", "dead"),
                 lambda d: d["cohorts"][0]["loc"].__setitem__("kind", "air"),
                 lambda d: d["colonies"][0]["memory"]["nodes"][0].__setitem__("visited", 1),
                 lambda d: d["colonies"].__setitem__(0, None)):
        broken = copy.deepcopy(doc)
        edit(broken)
        with pytest.raises(CanonicalError):
            state_from_bytes(canonical_dumps(broken), world.config, world.world_map)


def test_a_state_of_another_map_is_rejected(world):
    other = build_world(world.config, SEED + 1)
    with pytest.raises(InvariantError):
        validate_state(world.state, world.config, other.world_map)


# ------------------------------------------------------------------------------------------------ checkpoint / run state


def test_checkpoint_roundtrip_binds_state_and_rng(world):
    data = checkpoint_to_bytes(world.config, world.world_map, world.state, world.run)
    cfg, world_map, state, run = checkpoint_from_bytes(data)
    assert (cfg, world_map, state, run) == (world.config, world.world_map, world.state, world.run)
    assert checkpoint_hash(state, run) == checkpoint_hash(world.state, world.run)
    assert [s.stream_id for s in run.streams] == ["map", "world"] and run.root_seed == SEED
    from formicmind.rng import new_stream

    assert run.streams[1].state == new_stream(SEED, "world").state  # "world" is reserved and never consumed in E1-a
    # the state hash ignores the RNG, the checkpoint hash does not
    run2 = copy.deepcopy(run)
    run2.streams[1].state += 2
    assert checkpoint_hash(state, run2) != checkpoint_hash(state, run) and state_hash(state) == state_hash(world.state)


def test_checkpoint_rejects_foreign_or_mixed_parts(world):
    doc = canonical_loads(checkpoint_to_bytes(world.config, world.world_map, world.state, world.run))
    other = build_world(world.config, SEED + 7)
    mixed = copy.deepcopy(doc)
    mixed["map"] = canonical_loads(map_to_bytes(other.world_map))
    for broken in (mixed, {**doc, "schema": "formicmind.checkpoint/3"}, {**doc, "extra": 1}, {k: v for k, v in doc.items() if k != "run"}):
        with pytest.raises(CanonicalError):
            checkpoint_from_bytes(canonical_dumps(broken))
    even = copy.deepcopy(doc)
    even["run"]["streams"][0]["inc"] = "2"
    with pytest.raises(CanonicalError):
        checkpoint_from_bytes(canonical_dumps(even))
    unsorted = copy.deepcopy(doc)
    unsorted["run"]["streams"].reverse()
    with pytest.raises(CanonicalError):
        checkpoint_from_bytes(canonical_dumps(unsorted))


def test_codec_is_strict_about_types(world):
    with pytest.raises(SchemaError):
        from_canonical(MapE1, [])
    with pytest.raises(SchemaError):
        from_canonical(WorldStateE1, {})
    doc = to_canonical(world.world_map)
    assert from_canonical(MapE1, doc) == world.world_map
    big = copy.deepcopy(doc)
    big["nodes"][0]["x"] = 2**60
    with pytest.raises(SchemaError):
        from_canonical(MapE1, big)
    canonical_dumps(doc)  # no float / None / tuple anywhere in the encoded map
    canonical_dumps(to_canonical(world.state))
