"""FOR-48 leak audit: what a colony may know is exactly its own memory, its own units and public rules.

Three layers of evidence:

1. *structure* - the modules that build or use an observation do not import the world, the map or the engine and never name them;
2. *counterfactual worlds* - on real mid-run states, everything a colony cannot observe is changed in a copy (rival colonies, the true patch stocks,
   unexplored map content, outcomes, even a known rival's death) and the colony's observation bytes, masks and answer stay identical;
3. *targets* - an unmet colony has no slot, an old acquaintance does not vanish from the masks because it died unnoticed.
"""

from __future__ import annotations

import ast
import copy
import inspect
from dataclasses import replace
from pathlib import Path

import pytest

from test_eco_policy_helpers import Recorder, Run, explorer_policies
from formicmind.eco import observation as observation_module
from formicmind.eco.action import action_masks
from formicmind.eco.model import ColonyOutcome, PatchState, PileState
from formicmind.eco.observation import build_observation
from formicmind.eco.policy_run import PolicyController

ECO = Path(__file__).resolve().parents[1] / "src" / "formicmind" / "eco"
INTERFACE_MODULES = ("observation.py", "action.py", "decoder.py", "policy.py")
BANNED_IMPORTS = {"WorldStateE1", "MapE1", "engine", "initial", "mapgen", "serialization", "simulation", "fixtures", "roles", "orders", "combat",
                  "cohorts", "apply", "policy_run", "validate"}
BANNED_TEXT = ("world_map", "state.colonies", "WorldStateE1", "MapE1", "routes", "RouteTable", "state.cohorts", "state.patches", "state.piles")


def _imported_names(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[-1])
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name.split(".")[-1] for alias in node.names)
    return names


def _executable_text(path: Path) -> str:
    """The source without docstrings and comments: docstrings may explain the rule, code must not touch what it forbids."""
    text = path.read_text(encoding="utf-8")
    doc_lines: set[int] = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            doc_lines.update(range(node.lineno, node.end_lineno + 1))
    return "\n".join(line.split("#")[0] for number, line in enumerate(text.splitlines(), 1) if number not in doc_lines)


@pytest.mark.parametrize("name", INTERFACE_MODULES)
def test_interface_modules_do_not_import_or_mention_the_world(name):
    path = ECO / name
    assert not (_imported_names(path) & BANNED_IMPORTS), f"{name} imports world-side code"
    executable = _executable_text(path)
    for banned in BANNED_TEXT:
        assert banned not in executable, f"{name} uses {banned}"


def test_action_and_decoder_do_not_even_import_the_state_model():
    for name in ("action.py", "decoder.py", "policy.py"):
        names = _imported_names(ECO / name)
        assert "model" not in names and not (names & {"ColonyStateE1", "CohortE1", "ColonyMemoryE1", "KnownNode"}), name


def test_the_observation_builder_receives_one_colony_its_cohorts_and_public_constants_only():
    parameters = list(inspect.signature(build_observation).parameters)
    assert parameters == ["colony", "cohorts", "tick", "config", "decision_index", "ticks_since_last_decision", "window", "rolling"]
    assert list(inspect.signature(action_masks).parameters) == ["obs"]
    source = inspect.getsource(observation_module)
    assert "colonies[" not in source.replace("# ", "")  # no indexing of other colonies


# ------------------------------------------------------------------------------------------------ counterfactual worlds


class Silent:
    """A rival policy that only keeps the economy running (so the mutated rivals never trip the order API)."""

    name = "silent"

    def __init__(self, inner) -> None:
        self.inner = inner

    def act(self, obs, masks):
        return self.inner.act(obs, masks)

    def snapshot(self) -> dict:
        return {}

    def restore(self, data: dict) -> None:
        return None


def known_nodes(state, colony):
    return {k.node_id for k in state.colonies[colony].memory.nodes}


def hide_rivals_state(state, world_map, colony, *, kill_rivals):
    """A copy of the world where everything ``colony`` cannot observe differs."""
    alt = copy.deepcopy(state)
    for other, row in enumerate(alt.colonies):
        if other == colony:
            continue
        row.food += 777
        row.starvation_debt += 3
        row.brood = []
        row.orders.egg_level = 3
        alt.outcomes[other] = ColonyOutcome(True, "forged", alt.tick)
        if kill_rivals:
            row.queen_hp, row.status = 0, "extinct"
    if kill_rivals:
        alt.cohorts = [c for c in alt.cohorts if c.colony_id == colony]
    else:
        for cohort in alt.cohorts:
            if cohort.colony_id != colony:
                cohort.count += 4
                cohort.hp_total += 4 * 20
                cohort.load += 3
    alt.patches = [PatchState(p.node_id, 0) for p in alt.patches]
    alt.piles = [PileState(n.node_id, 999) for n in world_map.nodes[:3] if n.node_id not in known_nodes(state, colony)] or alt.piles
    alt.edge_flow = []
    alt_map = replace(world_map, nodes=tuple(
        node if node.node_id in known_nodes(state, colony) or node.kind == "nest"
        else replace(node, x=node.x + 1000, y=node.y - 1000, patch=tuple(replace(p, capacity=p.capacity * 3, start_stock=1) for p in node.patch))
        for node in world_map.nodes))
    return alt, alt_map


def decide(run, state, world_map, only=None):
    recorders = [Recorder(Silent(p)) for p in explorer_policies(run.config, attackers=(1, 3))]
    PolicyController(recorders)(state, run.config, world_map, run.routes)
    return recorders


def comparable(recorder):
    return [(blob, masks) for _obs, blob, masks in recorder.seen]


@pytest.mark.parametrize("seed", [50012, 50004])
def test_unobservable_state_never_reaches_observation_masks_or_answers(seed):
    run = Run(seed=seed, attackers=(1, 3))
    checked = met = 0
    for stop in (0, 300, 600, 900, 1200):
        run.advance(stop - run.state.tick)
        if run.state.tick % run.config.decision_k:
            continue
        base_state, base_map = copy.deepcopy(run.state), run.world.world_map
        base = decide(run, base_state, base_map)
        for colony in range(8):
            if run.state.colonies[colony].status != "alive":
                continue
            for kill in (False, True):
                alt_state, alt_map = hide_rivals_state(run.state, run.world.world_map, colony, kill_rivals=kill)
                alt = decide(run, alt_state, alt_map)
                assert comparable(alt[colony]) == comparable(base[colony]), f"tick {run.state.tick} colony {colony} kill={kill}"
                assert alt[colony].seen[0][0] == base[colony].seen[0][0]
                checked += 1
            met += bool(run.state.colonies[colony].memory.foreign)
    assert checked >= 40 and met >= 1


def test_a_known_rival_does_not_vanish_from_the_masks_when_it_dies_unnoticed():
    run = Run(seed=50012).advance(900)
    colony = rival = None
    for c in range(8):
        memory = run.state.colonies[c].memory
        if run.state.colonies[c].status == "alive" and any(r.nest_node_slot >= 0 for r in memory.foreign):
            colony, rival = c, next(r.colony_id for r in memory.foreign if r.nest_node_slot >= 0)
            break
    assert colony is not None, "this world has an acquaintance with a found nest by tick 900"
    base_state = copy.deepcopy(run.state)
    base = decide(run, base_state, run.world.world_map)[colony].seen[0]
    slot = next(i for i, r in enumerate(run.state.colonies[colony].memory.foreign) if r.colony_id == rival)
    assert any(pair[1] == slot for pair in base[2].attack_pairs), "the nest of the rival is a legal attack target"
    # the rival dies without the colony noticing: queen killed, colony extinct, units gone, food turned to nothing
    dead_state = copy.deepcopy(run.state)
    victim = dead_state.colonies[rival]
    victim.queen_hp, victim.status, victim.food = 0, "extinct", 0
    dead_state.cohorts = [c for c in dead_state.cohorts if c.colony_id != rival]
    dead_state.outcomes[rival] = ColonyOutcome(True, "extinct", dead_state.tick)
    after = decide(run, dead_state, run.world.world_map)[colony].seen[0]
    assert after[1] == base[1] and after[2] == base[2]
    assert any(pair[1] == slot for pair in after[2].attack_pairs), "still targetable: the colony cannot know that the rival is gone"


def test_what_a_colony_has_actually_observed_does_change_its_observation():
    """Control: the audit is not vacuous - changing a colony's own memory or own units does change the bytes."""
    run = Run(seed=50012).advance(600)
    colony = 0
    own = [c for c in run.state.cohorts if c.colony_id == colony]
    base = build_observation(run.state.colonies[colony], own, 600, run.config, decision_index=5, ticks_since_last_decision=30)
    from formicmind.eco.observation import observation_bytes

    memory_changed = copy.deepcopy(run.state.colonies[colony])
    memory_changed.memory.nodes[1].danger_q = 200
    assert observation_bytes(build_observation(memory_changed, own, 600, run.config, decision_index=5, ticks_since_last_decision=30)) != observation_bytes(base)
    fewer = [replace(c, count=max(1, c.count - 1), hp_total=max(1, c.hp_total - 20)) for c in own]
    assert observation_bytes(build_observation(run.state.colonies[colony], fewer, 600, run.config, decision_index=5, ticks_since_last_decision=30)) != observation_bytes(base)
    assert observation_bytes(build_observation(run.state.colonies[colony], own, 601, run.config, decision_index=5, ticks_since_last_decision=31)) != observation_bytes(base)


def test_observation_contains_no_world_identifier_scale_and_slot_indices_stay_local():
    run = Run(seed=50012).advance(1200)
    for colony in range(8):
        row = run.state.colonies[colony]
        if row.status != "alive":
            continue
        own = [c for c in run.state.cohorts if c.colony_id == colony]
        obs = build_observation(row, own, run.state.tick, run.config, decision_index=3, ticks_since_last_decision=30)
        n, f = len(obs.node_features), len(obs.foreign_features)
        assert all(0 <= a < n and 0 <= b < n for a, b in obs.edge_slots)
        assert all(-1 <= s < n for s in obs.foreign_nest_slots) and all(-1 <= t < n and 0 <= s < n for s, t in obs.cohort_slots)
        assert all(0 <= ns < n and 0 <= cs < f for ns, cs, _age in obs.sighting_ints)
        assert all(-1 <= ns < n and -1 <= cs < f for _code, ns, cs in obs.event_ints)
        assert f <= run.config.n_colonies - 1
        # a slot order is the order of discovery: the own nest first, nothing sorted by world id
        world_ids = [k.node_id for k in row.memory.nodes]
        assert world_ids[0] == run.world.world_map.nests[colony]
        # N and M of the world, the number of colonies and any hash appear nowhere in the facts
        facts = obs.facts
        assert facts.tick == run.state.tick
        assert not hasattr(facts, "n_colonies") and not hasattr(facts, "nodes_total")


def test_the_audit_catches_a_planted_leak(monkeypatch):
    """Meta-test: if the controller fed a rival's true food into an observation, the counterfactual comparison above would fail."""
    import formicmind.eco.policy_run as policy_run

    run = Run(seed=50012).advance(600)
    colony = 0
    current: dict = {}
    original = policy_run.build_observation

    def leaky(colony_state, cohorts, tick, config, **kwargs):
        obs = original(colony_state, cohorts, tick, config, **kwargs)
        rival = current["state"].colonies[(colony_state.colony_id + 1) % 8]
        return replace(obs, facts=replace(obs.facts, food=obs.facts.food + rival.food))

    monkeypatch.setattr(policy_run, "build_observation", leaky)
    base_state, base_map = copy.deepcopy(run.state), run.world.world_map
    current["state"] = base_state
    base = decide(run, base_state, base_map)
    alt_state, alt_map = hide_rivals_state(run.state, base_map, colony, kill_rivals=False)
    current["state"] = alt_state
    alt = decide(run, alt_state, alt_map)
    assert comparable(alt[colony]) != comparable(base[colony])
