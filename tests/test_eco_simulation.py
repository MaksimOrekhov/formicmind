from formicmind.eco.config import default_config
from formicmind.eco.cli import build_parser
from formicmind.eco.engine import RouteTable, step_world_tick
from formicmind.eco.fixtures import fixture_controller
from formicmind.eco.initial import build_world
from formicmind.eco.serialization import checkpoint_from_bytes, checkpoint_hash, checkpoint_to_bytes, state_hash
from formicmind.eco.simulation import run_simulation


def test_short_real_map_run_reports_transport_shortage_and_for44_distances():
    result = run_simulation(default_config(8, "moderate"), 42, "peaceful_fair", ticks=100,
                            sample_interval=25, include_map=False)
    report = result.report

    assert report["ticks_completed"] == 100
    assert report["food_ledger"]["difference"] == 0
    assert report["first_signals"]["first_delivery_tick"] is not None
    assert report["first_signals"]["upkeep_shortfall_tick"] is not None
    assert report["first_signals"]["starvation_death_tick"] is not None
    assert report["totals"]["delivered"] > 0
    assert report["map_distance_diagnostics"]["border"]["actual_rank_distances"][0]["mean_ticks_x100"] > 700
    assert report["map_distance_diagnostics"]["far"]["actual_rank_distances"][2]["mean_ticks_x100"] > 1000
    access = report["map_distance_diagnostics"]["economic_access"]
    assert access["radius_ticks"] == 10
    assert access["calculator_assumption_food_per_colony_x1000"] == 24_000
    assert 0 < access["mean_fair_share_food_per_colony_x1000"]
    assert report["map_distance_diagnostics"]["border"]["economic_radius_ticks"] == 10
    assert all(row["tick"] % 25 == 0 for row in report["time_series"][1:])


def test_cli_accepts_short_peaceful_fixture_aliases():
    parser = build_parser()
    fair = parser.parse_args(["run", "--fixture", "fair"])
    greedy = parser.parse_args(["sim-sweep", "--fixtures", "greedy"])
    assert fair.fixture == "fair"
    assert greedy.fixtures == ["greedy"]


def test_tick_3000_serialized_checkpoint_continues_deterministically():
    config = default_config(8, "moderate")
    continuous = build_world(config, 91003)
    routes = RouteTable(continuous.world_map)
    policy = fixture_controller("peaceful_fair")
    for _ in range(3000):
        step_world_tick(continuous.state, continuous.run, config, continuous.world_map, policy,
                        routes=routes, validate_input=continuous.state.tick == 0)

    checkpoint = checkpoint_to_bytes(config, continuous.world_map, continuous.state, continuous.run)
    resumed_config, resumed_map, resumed_state, resumed_run = checkpoint_from_bytes(checkpoint)
    resumed_routes = RouteTable(resumed_map)
    assert continuous.state.tick == resumed_state.tick == 3000
    assert checkpoint_hash(continuous.state, continuous.run) == checkpoint_hash(resumed_state, resumed_run)

    for _ in range(3000):
        step_world_tick(continuous.state, continuous.run, config, continuous.world_map, policy,
                        routes=routes, validate_input=False)
        step_world_tick(resumed_state, resumed_run, resumed_config, resumed_map, policy,
                        routes=resumed_routes, validate_input=False)

    assert state_hash(continuous.state) == state_hash(resumed_state)
    assert checkpoint_hash(continuous.state, continuous.run) == checkpoint_hash(resumed_state, resumed_run)
