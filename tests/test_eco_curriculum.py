"""FOR-50: the curriculum action masks (profile ``nest-v1``) - a pure intersection of masks used by sampling, log-prob, decoder, validator and the
orders; ``none`` is byte-for-byte the old behaviour; the profile travels through training, checkpoints, evaluation, replay and decision records."""

from __future__ import annotations

import ast
import dataclasses
import json
import random
from pathlib import Path

import pytest
import torch

from formicmind.canonical import CanonicalError
from formicmind.eco import default_config
from formicmind.eco.action import ActionError, ActionE1, SquadActionE1, action_masks, validate_action
from formicmind.eco.cli import main
from formicmind.eco.curriculum import (
    CURRICULUM_NEST_V1, CURRICULUM_PROFILES, CurriculumError, apply_curriculum_masks, describe_profile, validate_profile,
)
from formicmind.eco.decoder import LATENT_SIZE, decode
from formicmind.eco.policy import PolicyRunSpec, PolicySpec, spec_from_canonical, spec_to_canonical, uniform_spec
from formicmind.eco.policy_run import PolicyController, run_policies
from formicmind.eco.replay_file import export_policy_replay, read_replay, verify_replay
from formicmind.eco.viability import REWARD_VERSION
from test_eco_rl_helpers import peaky_model, recorded_decisions

ROOT = Path(__file__).resolve().parents[1]
MASK_FIELDS = [f.name for f in dataclasses.fields(type(recorded_decisions()[0][1]))]


@pytest.fixture(scope="module")
def decisions():
    return recorded_decisions(1500, 50012)


# ------------------------------------------------------------------------------------------------ the pure mask function


def test_profile_none_returns_the_very_same_masks(decisions):
    for obs, masks in decisions[::7]:
        assert apply_curriculum_masks(obs, masks, "none") is masks
        assert apply_curriculum_masks(obs, masks) is masks


def test_an_unknown_profile_is_refused():
    for bad in ("nest", "Nest-v1", "", None, 1):
        with pytest.raises(CurriculumError):
            validate_profile(bad)
    with pytest.raises(CurriculumError):
        apply_curriculum_masks(None, None, "moderate")      # type: ignore[arg-type]
    assert CURRICULUM_PROFILES == ("none", "nest-v1") and validate_profile("nest-v1") == "nest-v1"
    assert "nest-v1" in describe_profile("nest-v1") and "full" in describe_profile("none")


def test_nest_v1_changes_exactly_the_three_listed_masks_and_only_switches_options_off(decisions):
    changed = {"soldier_share", "scout_n", "squad_mission"}
    seen_soldiers = seen_scout_option = seen_attack_option = 0
    for obs, base in decisions:
        eff = apply_curriculum_masks(obs, base, CURRICULUM_NEST_V1)
        assert eff.soldier_share == (base.soldier_share[0], False, False, False, False) and eff.soldier_share[0]
        assert eff.scout_n == (True, False, False, False, False)
        assert eff.squad_mission == ((True, False, False, False),) * 3
        for name in MASK_FIELDS:
            if name not in changed:
                assert getattr(eff, name) == getattr(base, name), name        # the physical and the free choices are untouched
        # intersection: never a flag on that the base mask had off
        assert all(e <= b for e, b in zip(eff.scout_n, base.scout_n)) and all(e <= b for e, b in zip(eff.soldier_share, base.soldier_share))
        seen_soldiers += any(base.soldier_share[1:])
        seen_scout_option += any(base.scout_n[1:])
        seen_attack_option += any(row[2] for row in base.squad_mission)
        assert apply_curriculum_masks(obs, eff, CURRICULUM_NEST_V1) == eff               # idempotent
    # the recorded worlds really offered the options that the profile removes (the test would be vacuous otherwise)
    assert seen_soldiers > 0 and seen_scout_option > 0 and seen_attack_option > 0


def test_the_free_choices_stay_free(decisions):
    """egg level (including 0), the forage fraction and targets and the flee flag are not touched by the profile."""
    for obs, base in decisions[::5]:
        eff = apply_curriculum_masks(obs, base, CURRICULUM_NEST_V1)
        assert eff.egg_level == (True, True, True, True) and eff.forage_frac == base.forage_frac
        assert eff.forage_target == base.forage_target and eff.workers_flee == (True, True)


def test_the_mask_function_reads_nothing_but_the_masks():
    """Static check: the module imports only the observation-only action/observation modules (no world state, map, engine, rng, replay)."""
    tree = ast.parse((ROOT / "src" / "formicmind" / "eco" / "curriculum.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(("." * node.level) + (node.module or ""))
        elif isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
    assert imported <= {"__future__", "dataclasses", ".action", ".observation"}
    # and it does not use the observation argument at all (the masks already are a function of it)
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "apply_curriculum_masks")
    used = {n.id for n in ast.walk(function) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    assert "obs" not in used


# ------------------------------------------------------------------------------------------------ decoder and validator see the effective masks


def test_every_latent_vector_decodes_to_a_legal_nest_action(decisions):
    rng = random.Random(7)
    for obs, base in decisions[::11]:
        eff = apply_curriculum_masks(obs, base, CURRICULUM_NEST_V1)
        for _ in range(6):
            z = [rng.uniform(-6, 6) for _ in range(LATENT_SIZE)]
            action = decode(z, eff)
            validate_action(action, eff)
            assert action.soldier_share_q == 0 and action.scout_n == 0 and action.scout_targets == ()
            assert [s.mission for s in action.squads] == ["none"] * 3
            assert action.version == "ActionE1/v1"


def test_the_validator_refuses_what_the_profile_removes_even_if_the_base_masks_allow_it(decisions):
    fired = {"soldier": 0, "scout": 0, "attack": 0, "guard": 0, "withdraw": 0}
    for obs, base in decisions:
        eff = apply_curriculum_masks(obs, base, CURRICULUM_NEST_V1)
        plain = decode([0.0] * LATENT_SIZE, eff)
        if any(base.soldier_share[1:]):
            with pytest.raises(ActionError):
                validate_action(dataclasses.replace(plain, soldier_share_q=1), eff)
            validate_action(dataclasses.replace(plain, soldier_share_q=1), base)             # legal without the profile
            fired["soldier"] += 1
        if base.scout_n[1] and any(base.scout_target):
            target = base.scout_target.index(True)
            scouting = dataclasses.replace(plain, scout_n=1, scout_targets=(target,))
            validate_action(scouting, base)
            with pytest.raises(ActionError):
                validate_action(scouting, eff)
            fired["scout"] += 1
        for index, mission in enumerate(("guard", "attack", "withdraw")):
            if mission == "withdraw" or base.squad_mission[0][index + 1]:
                if mission == "guard" and base.squad_mission[0][1]:
                    squad = SquadActionE1("guard", 1, base.guard_target.index(True), -1)
                elif mission == "attack" and base.squad_mission[0][2]:
                    squad = SquadActionE1("attack", 1, *base.attack_pairs[0])
                elif mission == "withdraw":
                    squad = SquadActionE1("withdraw", 0, -1, -1)
                else:
                    continue
                action = dataclasses.replace(plain, squads=(squad, plain.squads[1], plain.squads[2]))
                validate_action(action, base)
                with pytest.raises(ActionError):
                    validate_action(action, eff)
                fired[mission] += 1
    assert all(count > 0 for count in fired.values()), fired


# ------------------------------------------------------------------------------------------------ sampling and log-prob under the same mask


def _batch(decisions, step=9):
    from formicmind.eco.rl.features import cat, step_batch

    rows = decisions[::step]
    effective = [apply_curriculum_masks(obs, masks, CURRICULUM_NEST_V1) for obs, masks in rows]
    return cat([step_batch(obs, m) for (obs, _), m in zip(rows, effective)]), effective


def test_sampling_never_leaves_the_nest_action_space_and_log_prob_is_that_of_the_sample(decisions):
    from formicmind.eco.rl.distribution import I_FLEE, I_SCOUT_N, I_SOLDIER, I_SQUAD, evaluate, sample_rows
    from formicmind.rng import new_stream

    model = peaky_model(3)
    batch, effective = _batch(decisions)
    rngs = [new_stream(11, f"curriculum:{i}") for i in range(batch.size)]
    py_logp: list[float] = []
    with torch.no_grad():
        heads, _value, _a, _c = model.step(batch, torch.zeros(batch.size, model.config.hidden), torch.zeros(batch.size, model.config.hidden))
        sample, latents = sample_rows(heads, batch, rngs, False, py_logp)
        log_prob = evaluate(heads, batch, sample).log_prob
    assert (sample.ints[:, I_SOLDIER] == 0).all() and (sample.ints[:, I_SCOUT_N] == 0).all()
    assert all((sample.ints[:, I_SQUAD + 4 * s] == 0).all() for s in range(3))                  # every squad mission is "none"
    assert not sample.mission_allowed[:, :, 1:].any()
    assert torch.allclose(log_prob, torch.tensor(py_logp, dtype=torch.float32), atol=1e-4)      # the torch log-prob is that of the sampled path
    for latent, masks in zip(latents, effective):                                              # the shared decoder reproduces the sample and validates
        validate_action(decode(latent, masks), masks)
    assert {int(v) for v in sample.ints[:, I_FLEE]} <= {0, 1}


def test_masked_single_option_heads_contribute_log_prob_zero_and_no_gradient(decisions):
    from formicmind.eco.rl.distribution import _cat_term, evaluate, sample_rows
    from formicmind.eco.rl.features import EGG_AT, SCOUT_N_AT, SOLDIER_AT
    from formicmind.rng import new_stream

    # a categorical with one legal option: log-prob 0 and entropy 0, whatever its logit says
    logits = torch.tensor([[3.0, -2.0, 7.0, 1.0]], requires_grad=True)
    allowed = torch.tensor([[True, False, False, False]])
    logp, entropy = _cat_term(logits, allowed, torch.tensor([0]), None, torch.zeros(1))
    logp.sum().backward()
    assert logp.item() == 0.0 and entropy.item() == 0.0 and torch.count_nonzero(logits.grad) == 0

    model = peaky_model(5)
    batch, _ = _batch(decisions)
    smask = batch["smask"]
    assert smask[:, SOLDIER_AT:SOLDIER_AT + 5].sum(-1).eq(1).all() and smask[:, SCOUT_N_AT:SCOUT_N_AT + 5].sum(-1).eq(1).all()
    assert smask[:, EGG_AT:EGG_AT + 4].all()                                                 # the egg head keeps all four options
    assert batch["squad_base"].sum(-1).eq(1).all()                                           # the squad mission head: "none" only
    zeros = torch.zeros(batch.size, model.config.hidden)
    with torch.no_grad():
        heads, *_ = model.step(batch, zeros, zeros)
        sample, _ = sample_rows(heads, batch, [new_stream(2, f"x:{i}") for i in range(batch.size)], False)
        reference = evaluate(heads, batch, sample)
        # scramble the logits of every masked head: nothing moves, because their only legal option has probability one
        for name in ("soldier", "scout_n", "mission0", "mission1", "mission2"):
            heads.cat[name] = heads.cat[name] + 40.0 * torch.randn_like(heads.cat[name])
        scrambled = evaluate(heads, batch, sample)
    assert torch.allclose(reference.log_prob, scrambled.log_prob, atol=1e-5)
    assert torch.allclose(reference.entropy_discrete, scrambled.entropy_discrete, atol=1e-5)


# ------------------------------------------------------------------------------------------------ the controller applies the one effective mask


class _Scout:
    """Test policy: always asks for a scout, a soldier egg and an attack/guard when the base masks would allow them (never the profile's)."""

    name = "greedy-for-everything"

    def act(self, obs, masks):
        from formicmind.eco.policy import ObservationHeuristicPolicy

        action = ObservationHeuristicPolicy(default_config(8, "moderate"), "explorer").act(obs, action_masks(obs))      # decided against the BASE masks
        return dataclasses.replace(action, soldier_share_q=2, scout_n=1, scout_targets=(next((i for i, ok in enumerate(action_masks(obs).scout_target) if ok), 1),))

    def snapshot(self) -> dict:
        return {}

    def restore(self, data: dict) -> None:
        return None


def test_a_policy_that_ignores_the_effective_masks_is_a_technical_error_not_an_override():
    config = default_config(8, "abundance")
    from formicmind.eco.engine import RouteTable, step_world_tick
    from formicmind.eco.initial import build_world

    world = build_world(config, 90000)
    controller = PolicyController([_Scout() for _ in range(8)], curriculum="nest-v1")
    with pytest.raises(ActionError):
        for _ in range(60):
            step_world_tick(world.state, world.run, config, world.world_map, controller, routes=RouteTable(world.world_map), validate_input=world.state.tick == 0)


def test_random_valid_policy_under_the_profile_only_ever_makes_nest_decisions():
    config = default_config(8, "abundance")
    result = run_policies(config, 90001, PolicyRunSpec(3, uniform_spec(8, "random_valid", 3).policies, "nest-v1"), ticks=1500)
    assert result.decisions.count >= 40
    for record in result.decisions.records:
        action = record["action"]
        assert record["curriculum"] == "nest-v1"
        assert action["soldier_share_q"] == 0 and action["scout_n"] == 0 and action["scout_targets"] == []
        assert [s["mission"] for s in action["squads"]] == ["none"] * 3
    assert any(record["action"]["egg_level"] > 0 for record in result.decisions.records)       # the economy is not forced to a single level
    assert len({record["action"]["egg_level"] for record in result.decisions.records}) > 1
    assert sum(1 for e in result.events.events if e.kind in ("dispatch", "combat", "contact")) == 0


def test_the_peaceful_baseline_is_unaffected_by_the_profile_and_none_adds_nothing():
    """peaceful never uses what the profile removes, so the world it produces is identical; only the decision records are marked."""
    config = default_config(8, "abundance")
    plain = run_policies(config, 90002, uniform_spec(8, "heuristic", 1, "peaceful"), ticks=900)
    nest = run_policies(config, 90002, PolicyRunSpec(1, uniform_spec(8, "heuristic", 1, "peaceful").policies, "nest-v1"), ticks=900)
    assert plain.state_hash == nest.state_hash and plain.events_hash == nest.events_hash
    assert plain.decisions_hash != nest.decisions_hash                         # the profile is part of the decision metadata ...
    assert all("curriculum" not in record for record in plain.decisions.records)       # ... and absent (old bytes) without it
    explicit = run_policies(config, 90002, PolicyRunSpec(1, uniform_spec(8, "heuristic", 1, "peaceful").policies, "none"), ticks=900)
    assert (explicit.state_hash, explicit.events_hash, explicit.decisions_hash) == (plain.state_hash, plain.events_hash, plain.decisions_hash)


def test_soldier_eggs_under_the_profile_are_only_the_engines_own_floor_rule():
    """``soldier_share_q = 0`` cannot (and, with the physics untouched, must not) stop the engine rule that a colony with fewer than two soldiers
    (adults + brood) lays a soldier egg anyway (engine.py, ``future_soldiers < 2``). The starting adults have staggered ages, so the first soldier
    dies of old age early and the rule fires. The nest gate therefore counts WORKER eggs and reports soldier eggs apart: a soldier egg is never a
    choice of the policy under ``nest-v1`` and happens only for a colony that has (at most) two soldiers including brood."""
    from formicmind.eco.engine import RouteTable, step_world_tick
    from formicmind.eco.initial import build_world
    from formicmind.eco.policy import build_policies

    config = default_config(8, "abundance")
    world = build_world(config, 90000)
    controller = PolicyController(build_policies(uniform_spec(8, "heuristic", 1, "peaceful"), config), curriculum="nest-v1")
    routes, soldier_eggs, worker_eggs = RouteTable(world.world_map), 0, 0
    for _ in range(600):
        state = world.state
        before = [sum(c.count for c in state.cohorts if c.colony_id == col.colony_id and c.caste == "soldier") + sum(b.caste == "soldier" for b in col.brood)
                  for col in state.colonies]
        record = step_world_tick(state, world.run, config, world.world_map, controller, routes=routes, validate_input=state.tick == 0)
        for col, future in zip(state.colonies, before):
            laid = [b for b in col.brood if b.laid_tick == record.tick]
            if any(b.caste == "soldier" for b in laid):
                soldier_eggs += 1
                assert future <= 2                                      # only a colony at its soldier floor lays one
            worker_eggs += sum(b.caste == "worker" for b in laid)
    assert all(record["action"]["soldier_share_q"] == 0 for record in controller.log.records)
    assert soldier_eggs > 0 and worker_eggs > soldier_eggs                 # the floor rule does fire, and the economy is still mostly workers


def test_a_controller_checkpoint_remembers_the_profile():
    peaceful = uniform_spec(8, "heuristic", 1, "peaceful")
    from formicmind.eco.policy import build_policies

    config = default_config(8, "abundance")
    nest = PolicyController(build_policies(peaceful, config), curriculum="nest-v1")
    plain = PolicyController(build_policies(peaceful, config))
    snapshot = nest.snapshot()
    assert snapshot["curriculum"] == "nest-v1" and "curriculum" not in plain.snapshot()
    with pytest.raises(ValueError):
        plain.restore(snapshot)
    with pytest.raises(ValueError):
        nest.restore(plain.snapshot())
    nest.restore(snapshot)
    with pytest.raises(CurriculumError):
        PolicyController([], curriculum="bogus")


# ------------------------------------------------------------------------------------------------ specification, replay header, strict reader


def test_the_policy_specification_names_the_profile_only_when_there_is_one():
    plain = uniform_spec(8, "random_valid", 4)
    assert "curriculum" not in spec_to_canonical(plain) and spec_from_canonical(spec_to_canonical(plain)) == plain
    nest = PolicyRunSpec(4, plain.policies, "nest-v1")
    data = spec_to_canonical(nest)
    assert data["curriculum"] == "nest-v1" and spec_from_canonical(data) == nest
    for bad in ({**data, "curriculum": "nest"}, {**data, "curriculum": 1}, {**spec_to_canonical(plain), "curriculum": "none"}, {**data, "x": 1}):
        with pytest.raises(CanonicalError):
            spec_from_canonical(bad)


def test_a_nest_replay_names_its_profile_and_a_plain_one_is_unchanged():
    config = default_config(8, "abundance")
    spec = PolicyRunSpec(2, tuple(PolicySpec("heuristic", "peaceful") for _ in range(8)), "nest-v1")
    replay, decisions = export_policy_replay(config, 90003, spec, 300, 100)
    document = read_replay(replay)
    assert document.header["policy"]["curriculum"] == "nest-v1" and document.policy.curriculum == "nest-v1"
    assert all(json.loads(line)["curriculum"] == "nest-v1" for line in decisions.splitlines())
    verify_replay(replay, decisions)
    plain_replay, plain_decisions = export_policy_replay(config, 90003, uniform_spec(8, "heuristic", 2, "peaceful"), 300, 100)
    assert "curriculum" not in read_replay(plain_replay).header["policy"] and b"curriculum" not in plain_decisions and b"curriculum" not in plain_replay
    tampered = replay.replace(b'"curriculum":"nest-v1"', b'"curriculum":"none"', 1)
    assert tampered != replay
    with pytest.raises(Exception):
        read_replay(tampered)                                                      # the integrity chain / strict header refuses a changed profile


# ------------------------------------------------------------------------------------------------ rollout, checkpoint, training, evaluation


def test_a_nest_rollout_records_only_nest_decisions_with_unchanged_reward_v3_and_physics():
    from formicmind.eco.rl.distribution import I_SCOUT_N, I_SOLDIER, I_SQUAD, evaluate
    from formicmind.eco.rl.rollout import collect_episode

    assert REWARD_VERSION == 3
    model = peaky_model(9, 8.0)
    episode = collect_episode(default_config(8, "abundance"), 90000, model, 1, 700, reward_scale=1.0, curriculum="nest-v1")
    ints = episode.sample.ints
    assert episode.batch.size > 20
    assert (ints[:, I_SOLDIER] == 0).all() and (ints[:, I_SCOUT_N] == 0).all() and all((ints[:, I_SQUAD + 4 * s] == 0).all() for s in range(3))
    assert (episode.batch["smask"][:, 4:9].sum(-1) == 1).all() and (episode.batch["squad_base"].sum(-1) == 1).all()      # stored = effective masks
    assert all(row["dispatch_attack"] == 0 and row["contact"] == 0 and row["combat"] == 0 for row in episode.metrics)
    assert all(row["eggs_laid"] == row["worker_eggs_laid"] + row["soldier_eggs_laid"] for row in episode.metrics)
    # PPO would re-evaluate exactly these samples under exactly these masks: the stored old log-prob is reproduced by the same network
    zeros = torch.zeros(episode.batch.size, model.config.hidden)
    with torch.no_grad():
        recomputed = evaluate(model.step(episode.batch, zeros, zeros)[0], episode.batch, episode.sample).log_prob
    assert recomputed.shape == episode.old_logp.shape          # (memory is recurrent: the values themselves are compared by the PPO identity test of FOR-49)


def test_the_manifest_carries_the_profile_and_older_models_count_as_the_full_action_space(tmp_path):
    from formicmind.eco.rl.checkpoint import MANIFEST, load_model, save_model
    from formicmind.eco.rl.network import ActorCriticE1

    manifest = save_model(tmp_path / "nest", ActorCriticE1(), training={}, curriculum="nest-v1")
    assert manifest["curriculum"] == "nest-v1" and load_model(tmp_path / "nest")[1]["curriculum"] == "nest-v1"
    assert json.loads((tmp_path / "nest" / MANIFEST).read_text(encoding="utf-8"))["curriculum"] == "nest-v1"
    assert save_model(tmp_path / "plain", ActorCriticE1())["curriculum"] == "none"
    with pytest.raises(CurriculumError):
        save_model(tmp_path / "bad", ActorCriticE1(), curriculum="free")
    committed = ROOT / "docs" / "ecosystem_examples" / "for49_reward_v3" / "model"
    assert "curriculum" not in json.loads((committed / MANIFEST).read_text(encoding="utf-8"))           # trained before the profile existed
    assert load_model(committed)[1]["curriculum"] == "none"


def test_a_short_nest_training_run_saves_the_profile_everywhere_and_the_model_replays_under_it(tmp_path, capsys):
    out = tmp_path / "run"
    code = main(["train", "--out", str(out), "--scenario", "abundance", "--curriculum", "nest-v1", "--iterations", "2", "--worlds", "2", "--ticks", "150",
                 "--eval-worlds", "1", "--minibatch-seqs", "8", "--seed", "1"])
    assert code == 0
    assert json.loads((out / "config.json").read_text(encoding="utf-8"))["curriculum"] == "nest-v1"
    for folder in ("latest", "final"):
        manifest = json.loads((out / folder / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["curriculum"] == "nest-v1" and manifest["training"]["config"]["curriculum"] == "nest-v1" and manifest["reward_version"] == 3
    final = json.loads((out / "eval_final.json").read_text(encoding="utf-8"))
    assert final["curriculum"] == "nest-v1" and final["learned"]["dispatch_attack"] == 0
    assert {"share_worker_eggs_ge_50", "worker_eggs_laid", "functional_survival", "physical_survival"} <= set(final["learned"])
    assert final["heuristic:peaceful"]["dispatch_attack"] == 0 and "soldier_eggs_laid" in final["heuristic:peaceful"]   # baselines were scored under the same profile
    # eval takes the profile of the model by default and exports a replay that names it; replay-verify re-executes it with the model
    code = main(["eval", "--model", str(out / "final"), "--scenario", "abundance", "--worlds", "1", "--ticks", "120", "--no-baselines",
                 "--out", str(tmp_path / "e.json"), "--replay-out", str(tmp_path / "r.jsonl"), "--replay-ticks", "120", "--frame-every", "40",
                 "--decisions", str(tmp_path / "d.jsonl"), "--verify"])
    assert code == 0
    report = json.loads((tmp_path / "e.json").read_text(encoding="utf-8"))
    assert report["curriculum"] == "nest-v1" and report["model"]["trained_curriculum"] == "nest-v1"
    assert read_replay((tmp_path / "r.jsonl").read_bytes()).policy.curriculum == "nest-v1"
    assert main(["replay-verify", str(tmp_path / "r.jsonl"), "--model", str(out / "final"), "--decisions", str(tmp_path / "d.jsonl")]) == 0
    assert "curriculum=nest-v1" in capsys.readouterr().out
    # the transfer check without retraining: the same model on the full action space (not a gate of stage 1)
    code = main(["eval", "--model", str(out / "final"), "--scenario", "abundance", "--worlds", "1", "--ticks", "120", "--no-baselines", "--curriculum", "none",
                 "--out", str(tmp_path / "t.json")])
    assert code == 0 and json.loads((tmp_path / "t.json").read_text(encoding="utf-8"))["curriculum"] == "none"


def test_a_plain_training_run_is_not_marked(tmp_path):
    out = tmp_path / "plain"
    assert main(["train", "--out", str(out), "--iterations", "1", "--worlds", "1", "--ticks", "90", "--eval-worlds", "0", "--minibatch-seqs", "8"]) == 0
    assert json.loads((out / "config.json").read_text(encoding="utf-8"))["curriculum"] == "none"
    assert json.loads((out / "final" / "manifest.json").read_text(encoding="utf-8"))["curriculum"] == "none"
