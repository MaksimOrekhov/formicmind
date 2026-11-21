"""FOR-52: the curriculum profile ``scout-v1`` (stage 3) - the static mask (nest-v1 minus the scout restriction), the conditional scout branch of the
action distribution, the reporting-only attribution of a first visit to a scout or a forager (no double counting, no effect on any hash), replay and
checkpoint behaviour, and no regression of ``none`` / ``nest-v1``."""

from __future__ import annotations

import copy
import dataclasses
import json
import random
from pathlib import Path

import pytest
import torch

from formicmind.eco import default_config
from formicmind.eco.action import ActionError, action_masks, validate_action
from formicmind.eco.curriculum import (
    CURRICULUM_NEST_V1, CURRICULUM_SCOUT_V1, CurriculumError, apply_curriculum_masks, describe_profile, validate_profile,
)
from formicmind.eco.decoder import LATENT_SIZE, decode
from formicmind.eco.policy import PolicyRunSpec, PolicySpec, build_policies, spec_from_canonical, spec_to_canonical, uniform_spec
from formicmind.eco.policy_run import PolicyController, run_policies
from formicmind.eco.replay_file import export_policy_replay, read_replay, verify_replay
from test_eco_rl_helpers import peaky_model, recorded_decisions

ROOT = Path(__file__).resolve().parents[1]
MASK_FIELDS = [f.name for f in dataclasses.fields(type(recorded_decisions()[0][1]))]


@pytest.fixture(scope="module")
def decisions():
    return recorded_decisions(1500, 50012)


# ------------------------------------------------------------------------------------------------ the static mask


def test_scout_v1_is_a_known_profile_and_describes_itself():
    assert validate_profile("scout-v1") == CURRICULUM_SCOUT_V1
    assert "scout-v1" in describe_profile("scout-v1") and "scouts" in describe_profile("scout-v1")
    for bad in ("scout", "Scout-v1", "nest-scout-v1"):
        with pytest.raises(CurriculumError):
            validate_profile(bad)


def test_scout_v1_changes_only_soldier_share_and_squad_missions_and_leaves_scouting_to_the_observation(decisions):
    changed = {"soldier_share", "squad_mission"}
    offered = 0
    for obs, base in decisions:
        eff = apply_curriculum_masks(obs, base, CURRICULUM_SCOUT_V1)
        assert eff.soldier_share == (base.soldier_share[0], False, False, False, False) and eff.soldier_share[0]
        assert eff.squad_mission == ((True, False, False, False),) * 3
        assert eff.scout_n == base.scout_n and eff.scout_target == base.scout_target              # exactly the observation-derived masks
        for name in MASK_FIELDS:
            if name not in changed:
                assert getattr(eff, name) == getattr(base, name), name
        assert all(e <= b for e, b in zip(eff.soldier_share, base.soldier_share))                # still only switches options off
        assert apply_curriculum_masks(obs, eff, CURRICULUM_SCOUT_V1) == eff
        offered += any(base.scout_n[1:]) and any(eff.scout_target)
    assert offered > 20                                                                         # the recorded worlds do offer scouts (not vacuous)


def test_scout_v1_differs_from_nest_v1_only_in_the_scout_options_and_the_old_profiles_are_unchanged(decisions):
    differing = 0
    for obs, base in decisions:
        nest = apply_curriculum_masks(obs, base, CURRICULUM_NEST_V1)
        scout = apply_curriculum_masks(obs, base, CURRICULUM_SCOUT_V1)
        assert nest.scout_n == (True, False, False, False, False)                                # nest-v1 as before (FOR-50 tests pin the rest)
        assert apply_curriculum_masks(obs, base, "none") is base
        for name in MASK_FIELDS:
            if name != "scout_n":
                assert getattr(nest, name) == getattr(scout, name), name
        differing += nest.scout_n != scout.scout_n
    assert differing > 0


# ------------------------------------------------------------------------------------------------ decoder, validator, distribution


def test_latent_vectors_decode_to_legal_actions_that_may_scout_but_never_order_soldiers_or_squads(decisions):
    rng = random.Random(5)
    scouting = 0
    for obs, base in decisions[::5]:
        eff = apply_curriculum_masks(obs, base, CURRICULUM_SCOUT_V1)
        for _ in range(8):
            action = decode([rng.uniform(-6, 6) for _ in range(LATENT_SIZE)], eff)
            validate_action(action, eff)
            assert action.soldier_share_q == 0 and [s.mission for s in action.squads] == ["none"] * 3
            assert (action.scout_n > 0) == bool(action.scout_targets)
            scouting += action.scout_n > 0
    assert scouting > 10


def test_the_validator_accepts_scouts_under_scout_v1_and_still_refuses_them_under_nest_v1(decisions):
    checked = 0
    for obs, base in decisions:
        if not (base.scout_n[1] and any(base.scout_target)):
            continue
        plain = decode([0.0] * LATENT_SIZE, apply_curriculum_masks(obs, base, CURRICULUM_NEST_V1))
        scouting = dataclasses.replace(plain, scout_n=1, scout_targets=(base.scout_target.index(True),))
        validate_action(scouting, apply_curriculum_masks(obs, base, CURRICULUM_SCOUT_V1))
        with pytest.raises(ActionError):
            validate_action(scouting, apply_curriculum_masks(obs, base, CURRICULUM_NEST_V1))
        with pytest.raises(ActionError):                                                         # a soldier order is refused under scout-v1 too
            validate_action(dataclasses.replace(scouting, soldier_share_q=1), apply_curriculum_masks(obs, base, CURRICULUM_SCOUT_V1))
        checked += 1
    assert checked > 20


def _batch(decisions, profile, step=7):
    from formicmind.eco.rl.features import cat, step_batch

    rows = decisions[::step]
    effective = [apply_curriculum_masks(obs, masks, profile) for obs, masks in rows]
    return cat([step_batch(obs, m) for (obs, _), m in zip(rows, effective)]), effective


def test_the_scout_branch_is_a_real_conditional_distribution_and_log_prob_is_that_of_the_sample(decisions):
    from formicmind.eco.rl.distribution import I_SCOUT_N, I_SOLDIER, I_SQUAD, evaluate, sample_rows
    from formicmind.rng import new_stream

    model = peaky_model(3, 1.0)
    batch, effective = _batch(decisions, CURRICULUM_SCOUT_V1)
    rngs = [new_stream(11, f"scout:{i}") for i in range(batch.size)]
    py_logp: list[float] = []
    with torch.no_grad():
        heads, _v, _a, _c = model.step(batch, torch.zeros(batch.size, model.config.hidden), torch.zeros(batch.size, model.config.hidden))
        sample, latents = sample_rows(heads, batch, rngs, False, py_logp)
        log_prob = evaluate(heads, batch, sample).log_prob
    assert (sample.ints[:, I_SOLDIER] == 0).all() and all((sample.ints[:, I_SQUAD + 4 * s] == 0).all() for s in range(3))
    assert (sample.ints[:, I_SCOUT_N] > 0).any() and (sample.ints[:, I_SCOUT_N] == 0).any()   # both "scout" and "do not" are sampled
    assert torch.allclose(log_prob, torch.tensor(py_logp, dtype=torch.float32), atol=1e-4)
    for latent, masks in zip(latents, effective):
        action = decode(latent, masks)
        validate_action(action, masks)                                                            # sampled targets are legal slots, one per scout at most
        assert len(action.scout_targets) <= max(1, action.scout_n) and (action.scout_n == 0) == (action.scout_targets == ())


def test_single_option_scout_n_has_no_gradient_but_a_free_scout_n_does(decisions):
    """With a single legal option (no idle worker or nothing to scout) the head contributes nothing; where scouts are offered, it learns."""
    from formicmind.eco.rl.distribution import evaluate, sample_rows
    from formicmind.eco.rl.features import SCOUT_N_AT
    from formicmind.rng import new_stream

    model = peaky_model(5, 1.0)
    batch, _ = _batch(decisions, CURRICULUM_SCOUT_V1)
    free = batch["smask"][:, SCOUT_N_AT:SCOUT_N_AT + 5].sum(-1) > 1
    assert free.any() and (~free).any()
    zeros = torch.zeros(batch.size, model.config.hidden)
    with torch.no_grad():
        sample, _ = sample_rows(model.step(batch, zeros, zeros)[0], batch, [new_stream(2, f"x:{i}") for i in range(batch.size)], False)
    heads, *_ = model.step(batch, zeros, zeros)
    heads.cat["scout_n"].retain_grad()
    evaluate(heads, batch, sample).log_prob.sum().backward()
    grad = heads.cat["scout_n"].grad
    assert torch.count_nonzero(grad[free]) > 0 and torch.count_nonzero(grad[~free]) == 0


# ------------------------------------------------------------------------------------------------ physical scouting and honest attribution


def _play(profile, variant, seed=90000, ticks=1500, scenario="moderate"):
    from formicmind.eco.rl.play import play_world

    config = default_config(8, scenario)
    spec = PolicyRunSpec(1, tuple(PolicySpec("heuristic", variant) for _ in range(8)), profile)
    return play_world(config, seed, PolicyController(build_policies(spec, config), curriculum=profile), ticks)


def test_scouts_physically_reach_unknown_nodes_and_the_attribution_adds_up_without_double_counting():
    from formicmind.eco.rl.play import colony_metrics

    result = _play(CURRICULUM_SCOUT_V1, "explorer")
    rows = colony_metrics(result)
    assert sum(r["scout_orders"] for r in rows) > 0 and sum(r["scout_arrivals"] for r in rows) > 0
    assert sum(r["discovered_by_scout"] for r in rows) > 0
    for row in rows:
        # every first visit is credited to exactly one source: the four counts partition the discover events of the colony
        assert row["discovered_by_scout"] + row["discovered_by_forage"] + row["discovered_mixed"] + row["discovered_other"] == row["discover"]
        assert row["scouts_sent"] >= row["scout_orders"] and row["scout_arrivals"] <= row["scouts_sent"]      # an arrival needs a sortie
        assert row["dispatch_attack"] == 0 and row["combat"] == 0 and 0 <= row["scout_share"] <= 1
    # a discovery event of the world really is the first visit of the node by that colony: no node twice
    seen = set()
    for item in result.events.events:
        if item.kind == "discover":
            assert (item.colony, item.node) not in seen
            seen.add((item.colony, item.node))


def test_a_colony_that_never_scouts_has_no_scout_credit_and_the_peaceful_run_is_all_forage_or_other():
    from formicmind.eco.rl.play import colony_metrics

    for row in colony_metrics(_play(CURRICULUM_SCOUT_V1, "peaceful")):
        assert row["scout_orders"] == row["scout_arrivals"] == row["discovered_by_scout"] == row["discovered_mixed"] == 0
        assert row["discovered_by_forage"] + row["discovered_other"] == row["discover"]


def test_the_attribution_is_reporting_only_and_changes_no_state_event_or_hash(monkeypatch):
    """Run the same world twice, once with the engine's diagnostics and once with ``sense`` forbidden to fill them: states and events are equal."""
    import formicmind.eco.engine as engine
    from formicmind.eco.memory import discovery_source
    from formicmind.eco.rl.play import play_world

    assert discovery_source({"scout"}) == "scout" and discovery_source({"forage"}) == "forage"
    assert discovery_source({"scout", "forage", "home"}) == "mixed" and discovery_source({"home"}) == "other" and discovery_source(()) == "other"
    config = default_config(8, "moderate")
    spec = PolicyRunSpec(1, tuple(PolicySpec("heuristic", "explorer") for _ in range(8)), "scout-v1")

    def run():
        return play_world(config, 90000, PolicyController(build_policies(spec, config), curriculum="scout-v1"), 600)

    with_diagnostics = run()
    original = engine.sense
    monkeypatch.setattr(engine, "sense", lambda state, world_map, events, discoveries=None: original(state, world_map, events))
    without = run()
    assert with_diagnostics.state == without.state and with_diagnostics.events.head == without.events.head
    assert with_diagnostics.events.events == without.events.events and with_diagnostics.stats.found["scout"] != without.stats.found["scout"]


def test_diagnostics_are_filled_by_the_engine_but_a_tick_record_is_not_a_policy_input():
    from formicmind.eco.engine import RouteTable, step_world_tick
    from formicmind.eco.initial import build_world
    from formicmind.eco.observation import ObservationE1

    config = default_config(8, "moderate")
    world = build_world(config, 90000)
    spec = PolicyRunSpec(1, tuple(PolicySpec("heuristic", "explorer") for _ in range(8)), "scout-v1")
    controller = PolicyController(build_policies(spec, config), curriculum="scout-v1")
    routes = RouteTable(world.world_map)
    discoveries, arrivals = [], []
    for _ in range(900):
        record = step_world_tick(world.state, world.run, config, world.world_map, controller, routes=routes, validate_input=world.state.tick == 0)
        discoveries += record.discoveries
        arrivals += record.scout_arrivals
        assert len(record.discoveries) == sum(1 for e in record.events if e.kind == "discover")      # one attribution per discover event
    assert any(source == "scout" for _c, _n, source in discoveries) and arrivals
    assert all(0 <= colony < 8 for colony, _node, _count in arrivals)
    assert "discover" not in {f.name for f in dataclasses.fields(ObservationE1)} and not hasattr(ObservationE1, "discoveries")


# ------------------------------------------------------------------------------------------------ nothing else moved


def test_the_old_profiles_and_the_full_action_space_produce_the_same_worlds_as_before_the_diagnostics():
    """peaceful never scouts, so scout-v1 / nest-v1 / none give one world; the profile only marks the decision records."""
    config = default_config(8, "abundance")
    peaceful = uniform_spec(8, "heuristic", 1, "peaceful").policies
    runs = [run_policies(config, 90002, PolicyRunSpec(1, peaceful, profile), ticks=900) for profile in ("none", "nest-v1", "scout-v1")]
    assert len({(r.state_hash, r.events_hash) for r in runs}) == 1
    assert len({r.decisions_hash for r in runs}) == 3
    assert all("curriculum" not in record for record in runs[0].decisions.records)


def test_the_effective_scout_mask_also_governs_the_run_of_a_scouting_policy():
    """Under scout-v1 the explorer scouts (dispatch events), under nest-v1 the very same policy cannot."""
    scouting = run_policies(default_config(8, "moderate"), 90000, PolicyRunSpec(1, tuple(PolicySpec("heuristic", "explorer") for _ in range(8)), "scout-v1"), ticks=900)
    nest = run_policies(default_config(8, "moderate"), 90000, PolicyRunSpec(1, tuple(PolicySpec("heuristic", "explorer") for _ in range(8)), "nest-v1"), ticks=900)
    assert any(e.kind == "dispatch" and e.a == 1 for e in scouting.events.events)
    assert not any(e.kind == "dispatch" for e in nest.events.events)
    assert all(r["action"]["soldier_share_q"] == 0 and [s["mission"] for s in r["action"]["squads"]] == ["none"] * 3 for r in scouting.decisions.records)
    assert any(r["action"]["scout_n"] > 0 for r in scouting.decisions.records) and all(r["curriculum"] == "scout-v1" for r in scouting.decisions.records)
    assert not any(e.kind == "combat" or (e.kind == "dispatch" and e.a != 1) for e in scouting.events.events)      # no war, only scouts


def test_checkpoints_specs_and_replays_carry_the_profile_and_resume_refuses_another():
    config = default_config(8, "moderate")
    explorer = tuple(PolicySpec("heuristic", "explorer") for _ in range(8))
    scout = PolicyController(build_policies(PolicyRunSpec(1, explorer, "scout-v1"), config), curriculum="scout-v1")
    nest = PolicyController(build_policies(PolicyRunSpec(1, explorer, "nest-v1"), config), curriculum="nest-v1")
    snapshot = scout.snapshot()
    assert snapshot["curriculum"] == "scout-v1"
    with pytest.raises(ValueError):
        nest.restore(snapshot)
    scout.restore(snapshot)
    spec = PolicyRunSpec(4, explorer, "scout-v1")
    assert spec_to_canonical(spec)["curriculum"] == "scout-v1" and spec_from_canonical(spec_to_canonical(spec)) == spec
    replay, decisions = export_policy_replay(config, 90000, spec, 360, 120)
    document = read_replay(replay)
    assert document.policy.curriculum == "scout-v1" and document.header["policy"]["curriculum"] == "scout-v1"
    verify_replay(replay, decisions)
    with pytest.raises(Exception):
        read_replay(replay.replace(b'"curriculum":"scout-v1"', b'"curriculum":"nest-v1"', 1))


def test_training_and_evaluation_accept_the_profile_and_record_it(tmp_path, capsys):
    from formicmind.eco.cli import main
    from formicmind.eco.rl.checkpoint import load_model, save_model
    from formicmind.eco.rl.network import ActorCriticE1

    source = tmp_path / "source"
    save_model(source, ActorCriticE1(), training={}, curriculum="nest-v1")
    out = tmp_path / "run"
    code = main(["train", "--out", str(out), "--scenario", "moderate", "--curriculum", "scout-v1", "--init", str(source), "--iterations", "1", "--worlds", "2",
                 "--ticks", "200", "--eval-worlds", "1", "--minibatch-seqs", "8", "--seed", "1"])
    assert code == 0
    assert json.loads((out / "config.json").read_text(encoding="utf-8"))["curriculum"] == "scout-v1"
    manifest = json.loads((out / "final" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["curriculum"] == "scout-v1" and manifest["training"]["config"]["curriculum"] == "scout-v1"
    assert load_model(out / "final")[1]["curriculum"] == "scout-v1"
    final = json.loads((out / "eval_final.json").read_text(encoding="utf-8"))
    assert final["curriculum"] == "scout-v1" and final["heuristic:explorer"]["scout_orders"] >= 0
    assert {"discovered_by_scout", "discovered_by_forage", "share_scouted_new_node", "scout_share"} <= set(final["learned"])
    assert main(["eval", "--model", str(out / "final"), "--scenario", "moderate", "--worlds", "1", "--ticks", "120", "--no-baselines", "--out", str(tmp_path / "e.json"),
                 "--replay-out", str(tmp_path / "r.jsonl"), "--replay-ticks", "120", "--frame-every", "40", "--decisions", str(tmp_path / "d.jsonl"), "--verify"]) == 0
    assert read_replay((tmp_path / "r.jsonl").read_bytes()).policy.curriculum == "scout-v1"
    assert main(["replay-verify", str(tmp_path / "r.jsonl"), "--model", str(out / "final"), "--decisions", str(tmp_path / "d.jsonl")]) == 0
    assert "curriculum=scout-v1" in capsys.readouterr().out
