"""Initial-knowledge isolation, bootstrap actions and replay compatibility."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from formicmind.canonical import canonical_loads
from formicmind.eco.action import action_masks, validate_action
from formicmind.eco.config import default_config
from formicmind.eco.curriculum import apply_curriculum_masks
from formicmind.eco.engine import RouteTable, step_world_tick
from formicmind.eco.initial import build_world, initial_world_state
from formicmind.eco.knowledge import DEFAULT_KNOWLEDGE, UNKNOWN_FOOD_V1
from formicmind.eco.memory import known_path_ticks, slot_of_node
from formicmind.eco.observation import build_observation
from formicmind.eco.policy import BOUNDED_UNKNOWN_FOOD, ObservationHeuristicPolicy, UnknownFoodScoutPolicy, build_policies, uniform_spec
from formicmind.eco.policy_run import PolicyController
from formicmind.eco.replay_file import export_policy_replay, export_replay, verify_replay
from formicmind.eco.serialization import checkpoint_from_bytes, checkpoint_hash, checkpoint_to_bytes, state_hash

ARTIFACTS = Path(__file__).resolve().parents[1] / "docs" / "ecosystem_examples"


def _observation(state, config, colony=0):
    cohorts = [c for c in state.cohorts if c.colony_id == colony]
    return build_observation(state.colonies[colony], cohorts, state.tick, config, decision_index=0, ticks_since_last_decision=0)


def test_initial_memory_and_tick_zero_scout_with_the_unchanged_starting_economy():
    config = default_config()
    control = build_world(config, 42)
    world = build_world(config, 42, initial_knowledge=UNKNOWN_FOOD_V1)
    assert world.world_map == control.world_map and world.config == control.config
    assert world.run.streams == control.run.streams
    physical_colonies = [replace(colony, memory=original.memory) for colony, original in zip(world.state.colonies, control.state.colonies, strict=True)]
    assert replace(world.state, colonies=physical_colonies) == control.state
    assert state_hash(world.state) != state_hash(control.state)

    for colony in world.state.colonies:
        memory = colony.memory
        nest = world.world_map.nests[colony.colony_id]
        neighbors = {edge.a if edge.b == nest else edge.b for edge in world.world_map.edges if nest in (edge.a, edge.b)}
        assert {n.node_id for n in memory.nodes if n.visited} == {nest}
        assert {n.node_id for n in memory.nodes[1:]} == neighbors and neighbors
        assert all(n.kind == "unknown" and n.last_stock == n.patch_capacity == 0 and n.seen_tick == n.last_stock_tick == -1 for n in memory.nodes[1:])
        assert len(memory.edges) == len(neighbors) and all(e.slot_a == 0 for e in memory.edges)
        assert all(known_path_ticks(memory, nest, target) > 0 for target in neighbors)

        obs = _observation(world.state, config, colony.colony_id)
        masks = apply_curriculum_masks(obs, action_masks(obs), "scout-v1")
        assert obs.facts.workers == obs.facts.idle_workers_home == 10 and obs.facts.food == 80
        assert not any(masks.forage_target) and any(masks.scout_target)
        assert not any(masks.forage_frac[1:])
        scout = UnknownFoodScoutPolicy(config).act(obs, masks)
        assert scout.scout_n == 1 and not scout.forage_targets
        validate_action(scout, masks)
        assert replace(scout, scout_n=0, scout_targets=()) == ObservationHeuristicPolicy(config).act(obs, masks)
        busy = replace(obs, facts=replace(obs.facts, scouts_out=1))
        assert UnknownFoodScoutPolicy(config).act(busy, masks).scout_n == 0


def test_swapping_hidden_patch_and_junction_does_not_change_initial_observation_or_action():
    config = default_config()
    world = build_world(config, 42, initial_knowledge=UNKNOWN_FOOD_V1)
    nodes = list(world.world_map.nodes)
    frontier = {node.node_id for colony in world.state.colonies for node in colony.memory.nodes[1:]}
    patch = next(node for node in nodes if node.node_id in frontier and node.patch)
    junction = next(node for node in nodes if node.node_id in frontier and node.kind == "junction")
    nodes[patch.node_id] = replace(patch, kind="junction", patch=())
    nodes[junction.node_id] = replace(junction, kind="patch", patch=patch.patch)
    changed = initial_world_state(config, replace(world.world_map, nodes=tuple(nodes)), initial_knowledge=UNKNOWN_FOOD_V1)
    for colony in range(config.n_colonies):
        before, after = _observation(world.state, config, colony), _observation(changed, config, colony)
        assert before == after
        policy = UnknownFoodScoutPolicy(config)
        assert policy.act(before, action_masks(before)) == policy.act(after, action_masks(after))


def test_checkpoint_roundtrip_and_default_bytes():
    config = default_config()
    normal = build_world(config, 42)
    explicit = build_world(config, 42, initial_knowledge=DEFAULT_KNOWLEDGE)
    diagnostic = build_world(config, 42, initial_knowledge=UNKNOWN_FOOD_V1)

    def encode(world):
        return checkpoint_to_bytes(world.config, world.world_map, world.state, world.run)

    assert encode(normal) == encode(explicit)
    assert set(canonical_loads(encode(normal))["run"]) == {"root_seed", "streams"}
    assert checkpoint_hash(normal.state, normal.run) == checkpoint_hash(explicit.state, explicit.run)
    data = encode(diagnostic)
    assert canonical_loads(data)["run"]["initial_knowledge"] == UNKNOWN_FOOD_V1
    assert checkpoint_to_bytes(*checkpoint_from_bytes(data)) == data

    source = PolicyController([], initial_knowledge=UNKNOWN_FOOD_V1)
    assert source.snapshot()["initial_knowledge"] == UNKNOWN_FOOD_V1
    assert "initial_knowledge" not in PolicyController([]).snapshot()
    with pytest.raises(ValueError, match="initial_knowledge"):
        PolicyController([]).restore(source.snapshot())


def test_policy_and_idle_replays_bind_initial_knowledge_and_reexecute_byte_for_byte():
    config = default_config()
    spec = replace(uniform_spec(8, "heuristic", 1, BOUNDED_UNKNOWN_FOOD), curriculum="scout-v1")
    with pytest.raises(ValueError, match="requires unknown-food-v1"):
        export_policy_replay(config, 42, spec, 60)
    replay, decisions = export_policy_replay(config, 42, spec, 60, initial_knowledge=UNKNOWN_FOOD_V1)
    document = verify_replay(replay, decisions)
    assert document.initial_knowledge == UNKNOWN_FOOD_V1 and document.policy == spec
    assert any(event.tick == 0 and event.kind == "dispatch" and event.a == event.b == 1 for event in document.events)
    idle = export_replay(config, 42, "idle", ticks=30, initial_knowledge=UNKNOWN_FOOD_V1)
    assert verify_replay(idle).initial_knowledge == UNKNOWN_FOOD_V1
    with pytest.raises(ValueError, match="only idle"):
        export_replay(config, 42, initial_knowledge=UNKNOWN_FOOD_V1)
    normal = export_replay(config, 42, "idle", ticks=30)
    assert "initial_knowledge" not in verify_replay(normal).header


def test_real_scout_arrival_enables_forage_and_delivery_from_the_same_patch():
    config = default_config()
    world = build_world(config, 90000, initial_knowledge=UNKNOWN_FOOD_V1)
    spec = replace(uniform_spec(8, "heuristic", 1, BOUNDED_UNKNOWN_FOOD), curriculum="scout-v1")
    controller = PolicyController(build_policies(spec, config, initial_knowledge=UNKNOWN_FOOD_V1), curriculum=spec.curriculum,
                                  initial_knowledge=UNKNOWN_FOOD_V1)
    routes = RouteTable(world.world_map)
    colony, node = 1, 46
    slot = slot_of_node(world.state.colonies[colony].memory)[node]
    assert not action_masks(_observation(world.state, config, colony)).forage_target[slot]
    records = {}
    for tick in range(44):
        records[tick] = step_world_tick(world.state, world.run, config, world.world_map, controller, routes=routes, track_forage_flows=True)
        if tick == 3:
            assert action_masks(_observation(world.state, config, colony)).forage_target[slot]
    assert (colony, node, 1) in records[3].scout_arrivals
    assert (colony, node, "scout") in records[3].discoveries
    forage_action = next(r["action"] for r in controller.log.records if r["tick"] == 30 and r["colony"] == colony)
    assert slot in forage_action["forage_targets"]
    assert all((colony, node) not in records[tick].forage_flows.harvested for tick in range(33))
    assert records[33].forage_flows.harvested[colony, node] > 0
    assert all((colony, node) not in records[tick].forage_flows.delivered for tick in range(43))
    assert records[43].forage_flows.delivered[colony, node] == 16


def test_saved_negative_gate_matches_the_verified_diagnostic_replay():
    folder = ARTIFACTS / "for54_unknown_food"
    report = json.loads((folder / "preflight.json").read_text(encoding="utf-8"))
    assert report["seeds"] == list(range(90000, 90008)) and report["horizon"] == 6000
    assert not report["gate"]["passed"]
    assert report["controls"]["no_scout"]["summary"]["delivered"] == 0
    scout = report["controls"]["bounded"]["summary"]
    assert scout["colonies_with_tick0_scout"] == 64 and scout["colonies_with_delivery_from_scout_patch"] == 5
    assert scout["functional_survival"] == 0 and scout["scouts_active_peak"] == 1
    replay = verify_replay((folder / f"replay_seed{report['replay_seed']}.jsonl").read_bytes())
    assert replay.chain_head == report["replay_chain_head"]
    assert replay.initial_knowledge == report["initial_knowledge"] == UNKNOWN_FOOD_V1


def test_existing_model_and_diagnostic_manifest_drive_eval_and_replay_separately(tmp_path):
    from formicmind.eco.cli import main
    from formicmind.eco.rl.checkpoint import load_model, save_model
    from formicmind.eco.rl.rollout import collect_episode
    from formicmind.eco.rl.train import TrainConfig

    model, old_manifest = load_model(ARTIFACTS / "for52_scout" / "model")
    assert "initial_knowledge" not in old_manifest and "initial_knowledge" not in TrainConfig().to_dict()
    assert TrainConfig(initial_knowledge=UNKNOWN_FOOD_V1).to_dict()["initial_knowledge"] == UNKNOWN_FOOD_V1
    path = tmp_path / "model"
    saved = save_model(path, model, curriculum="scout-v1", initial_knowledge=UNKNOWN_FOOD_V1)
    assert saved["initial_knowledge"] == UNKNOWN_FOOD_V1
    report, replay = tmp_path / "eval.json", tmp_path / "learned.jsonl"
    assert main(["eval", "--model", str(path), "--worlds", "1", "--ticks", "15", "--no-baselines",
                 "--out", str(report), "--replay-out", str(replay), "--replay-ticks", "15", "--verify"]) == 0
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["initial_knowledge"] == result["model"]["trained_initial_knowledge"] == UNKNOWN_FOOD_V1
    assert result["curriculum"] == result["model"]["trained_curriculum"] == "scout-v1"
    assert verify_replay(replay.read_bytes()).initial_knowledge == UNKNOWN_FOOD_V1
    episode = collect_episode(default_config(), 90000, model, 1, 15, curriculum="scout-v1", initial_knowledge=UNKNOWN_FOOD_V1)
    assert all(row["delivered"] == 0 for row in episode.metrics)
