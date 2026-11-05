"""FOR-38: effective independent samples in the FOR-37 dataset and a bounded exploratory diversity pilot.

Nothing here changes the simulator, ``ObservationV2``, the decoder, ``ActionV2`` or Baselines.  The pilot only
composes public ``Controller`` objects (baseline policies, scripted legal ``ActionV2`` values, a perturbed-teacher
executor) around ``run_episode``; labels are always the Balanced v2 logits on the visited state.  Nothing is trained.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..actions import ActionMaskV2, ActionV2, validate_action
from ..baselines import BaselinePolicy, baseline_spec
from ..config import EdgeSpec, SimConfig, Topology
from ..decoder import PolicyLogitsV2, decode_action_logits
from ..invariants import validate_config
from ..match import run_episode
from ..observation import ObservationV2, check_observation_config
from ..rng import policy_stream_id
from ..routing import shortest_path
from ..rules import default_config
from ..serialization import config_hash
from ..tick import PolicyController
from .dataset import DatasetError, _write_atomic
from .quality_dataset import (
    QUALITY_TEACHER,
    _RecordingTeacher,
    _deduplicate_rows,
    _sha256,
    quality_config,
)

AUDIT_SCHEMA = "formicmind.ml-effective-samples/v1"
PILOT_SCHEMA = "formicmind.ml-diversity-pilot/v1"

PILOT_MAX_MATCHES = 400
PILOT_MAX_TICKS = 3000
PILOT_SEED_MAIN = 380_000_001
PILOT_SEEDS_CONTROL = (380_000_002, 380_000_003)
PILOT_SEEDS_RANDOM = tuple(range(380_000_101, 380_000_109))
PILOT_SEEDS_PERTURB = tuple(range(380_000_201, 380_000_209))
PILOT_MIN_NEW_DECISIONS_PER_CONTEXT = 30
EXISTING_MAPS = ("classic", "cross", "dense")
DETERMINISTIC_OPPONENTS = ("turtle:v2", "rush:v2", "raider:v2", "balanced:v2")
SWITCH_SCHEDULES = (("turtle:v2", "rush:v2"), ("rush:v2", "turtle:v2"),
                    ("turtle:v2", "raider:v2"), ("raider:v2", "turtle:v2"))
SWITCH_POINTS = (3, 6, 12, 24)
PERTURB_EPSILONS = (0.10, 0.25)
PERTURB_OPPONENTS = ("turtle:v2", "rush:v2", "raider:v2")
SPLIT_FRACTIONS = (0.6, 0.2)  # train, validation; the remainder is held-out
RECOMMENDATION_RULE = {
    "max_trajectory_duplicate_rate": 0.10,
    "min_new_decision_share": 0.60,
    "min_validation_retention": 0.80,
    "min_independent_contexts_or_trajectories": 8,
}

# (edge edits, edges to remove); all edits are mirror pairs.  Nodes: A=0 N1=1 N2=2 C=3 N4=4 N5=5 B=6.
PILOT_MAP_EDITS: dict[str, tuple[dict[tuple[int, int], int], tuple[tuple[int, int], ...]]] = {
    "fast_wings": ({(0, 1): 1, (4, 6): 1}, ()),
    "slow_center": ({(1, 3): 4, (3, 4): 4, (2, 3): 4, (3, 5): 4}, ()),
    "wide_ring": ({(1, 2): 2, (4, 5): 2}, ()),
    "far_flank": ({(0, 2): 4, (5, 6): 4, (2, 5): 2}, ()),
    "shortcut": ({(0, 3): 4, (3, 6): 4}, ()),
    "chain": ({(1, 4): 4}, ((2, 5),)),
}

OPENING_SCRIPTS: dict[str, ActionV2] = {
    "idle": ActionV2(0, (65535, 0, 0, 0, 0, 0), 0, False),
    "workers": ActionV2(1, (0, 21845, 21845, 21845, 0, 0), 0, False),
    "soldiers": ActionV2(2, (0, 32768, 32767, 0, 0, 0), 0, False),
    "center": ActionV2(1, (0, 0, 0, 65535, 0, 0), 0, False),
}
OPENING_LENGTHS = (2, 5)


# ---------------------------------------------------------------------------------------------------------
# Part 1 — effective independent samples in a frozen v2 dataset
# ---------------------------------------------------------------------------------------------------------

def _quantiles(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {}

    def at(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] * (1 - (position - lower)) + ordered[upper] * (position - lower)

    return {"min": ordered[0], "p10": at(0.10), "median": at(0.50), "p90": at(0.90), "max": ordered[-1]}


def nearest_neighbor_distances(query: Sequence[Sequence[float]], reference: Sequence[Sequence[float]],
                               *, exclude_self: bool = False) -> list[float]:
    """Euclidean distance from every query vector to its nearest reference vector (descriptive coverage)."""
    result = []
    for qi, vector in enumerate(query):
        best = math.inf
        for ri, other in enumerate(reference):
            if exclude_self and qi == ri:
                continue
            d = math.dist(vector, other)
            if d < best:
                best = d
        result.append(best)
    return result


def effective_sample_audit(
    manifest: dict[str, Any],
    examples: Sequence[Any],
    *,
    action_sequences: dict[str, list[Any]] | None = None,
    include_coverage: bool = True,
) -> dict[str, Any]:
    """Count independent units at every level of the FOR-37 hierarchy; works from the frozen manifest + rows."""
    matches = manifest["matches"]
    by_context: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for match in matches:
        by_context[(match["split"], match["context_id"])].append(match)
    retained_rows: Counter = Counter((e.split, e.context_id) for e in examples)
    retained_hashes_by_match: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        retained_hashes_by_match[example.match_id].add(example.decision_sha256)

    pair_sides: dict[tuple[str, int], dict[int, str]] = defaultdict(dict)
    for match in matches:
        pair_sides[(match["context_id"], match["seed"])][match["teacher_side"]] = match["trajectory_sha256"]
    side_collapsed = sum(1 for sides in pair_sides.values() if len(sides) == 2 and sides[0] == sides[1])

    source_decisions = sum(m["source_decision_count"] for m in matches)
    unique_source_decisions = {h for m in matches for h in m["trajectory_decision_sha256s"]}
    unique_traj = {m["trajectory_sha256"] for m in matches}
    retained_traj = {m["trajectory_sha256"] for m in matches if m.get("retained")}

    contexts = []
    for (split, context_id), group in sorted(by_context.items()):
        traj = {m["trajectory_sha256"] for m in group}
        decisions = [h for m in group for h in m["trajectory_decision_sha256s"]]
        seeds = {m["seed"] for m in group}
        contexts.append({
            "split": split,
            "context_id": context_id,
            "matches": len(group),
            "distinct_seeds": len(seeds),
            "unique_source_trajectories": len(traj),
            "matches_per_unique_trajectory": len(group) / len(traj),
            "seed_sensitive": len(traj) > 1,
            "source_decisions": len(decisions),
            "unique_source_decisions": len(set(decisions)),
            "retained_decisions": retained_rows.get((split, context_id), 0),
            "terminal": dict(sorted(Counter(
                f"{m['reason']}@{m['final_tick']}" for m in group).items())),
            "decisions_per_trajectory": sorted({m["source_decision_count"] for m in group}),
            "emptied_by_global_dedup": retained_rows.get((split, context_id), 0) == 0,
        })

    # Why decisions were dropped inside trajectories that survived.
    dropped_positions = Counter()
    dropped_total = 0
    for match in matches:
        if not match.get("trajectory_retained"):
            continue
        keep = retained_hashes_by_match.get(match["match_id"], set())
        for position, decision in enumerate(match["trajectory_decision_sha256s"]):
            if decision not in keep:
                dropped_total += 1
                dropped_positions["position_0_shared_start_state" if position == 0 else "later_position"] += 1
    dedup = manifest["deduplication"]
    split_summary = {}
    for split in ("train", "validation", "heldout"):
        split_matches = [m for m in matches if m["split"] == split]
        raw = sum(m["source_decision_count"] for m in split_matches)
        kept = sum(retained_rows.get((split, c), 0) for (s, c) in by_context if s == split)
        split_summary[split] = {
            "matches": len(split_matches),
            "unique_source_trajectories": len({m["trajectory_sha256"] for m in split_matches}),
            "retained_trajectory_groups": len({m["trajectory_sha256"] for m in split_matches if m.get("retained")}),
            "source_decisions": raw,
            "retained_decisions": kept,
            "decision_retention": kept / raw if raw else None,
            "contexts": len({c for (s, c) in by_context if s == split}),
            "contexts_emptied": sorted(c for (s, c) in by_context if s == split and retained_rows.get((s, c), 0) == 0),
        }
    report: dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "dataset_sha256": manifest["dataset_sha256"],
        "hierarchy": {
            "match_records": len(matches),
            "distinct_seeds": len({m["seed"] for m in matches}),
            "distinct_maps": len({m["map_id"] for m in matches}),
            "distinct_opponents": len({m["opponent_id"] for m in matches}),
            "distinct_contexts": len({m["context_id"] for m in matches}),
            "seed_context_pairs": len(pair_sides),
            "seed_context_pairs_with_identical_trajectory_on_both_sides": side_collapsed,
            "unique_source_trajectories": len(unique_traj),
            "unique_retained_trajectory_groups": len(retained_traj),
            "source_decisions": source_decisions,
            "unique_source_decisions": len(unique_source_decisions),
            "retained_decisions": len(examples),
            "duplicate_trajectories_removed": dedup["duplicate_trajectories_removed"],
            "duplicate_decisions_removed": dedup["duplicate_decisions_removed"],
        },
        "decision_drop_causes": {"total_inside_surviving_trajectories": dropped_total, **dict(dropped_positions)},
        "splits": split_summary,
        "contexts": contexts,
        "deterministic_contexts": sorted(c["context_id"] + "@" + c["split"] for c in contexts if not c["seed_sensitive"]),
        "seed_sensitive_contexts": sorted(c["context_id"] + "@" + c["split"] for c in contexts if c["seed_sensitive"]),
        "interpretation": {
            "rules_consume_randomness": False,
            "seed_influences": "only policy RNG streams (random:v2); the world has no seed (rules v0.2.1)",
        },
    }
    if action_sequences is not None:
        discrete = {tuple((a[0], a[2], a[3]) for a in seq) for seq in action_sequences.values()}
        full = {tuple((a[0], tuple(a[1]), a[2], a[3]) for a in seq) for seq in action_sequences.values()}
        report["teacher_action_sequences"] = {
            "trajectories_resimulated": len(action_sequences),
            "unique_full_action_sequences": len(full),
            "unique_lay_army_avoid_sequences": len(discrete),
        }
    if include_coverage:
        obs = {split: [e.observation for e in examples if e.split == split] for split in ("train", "validation", "heldout")}
        coverage = {
            "train_to_train_leave_one_out": _quantiles(nearest_neighbor_distances(obs["train"], obs["train"], exclude_self=True)),
            "validation_to_train": _quantiles(nearest_neighbor_distances(obs["validation"], obs["train"])),
            "heldout_to_train": _quantiles(nearest_neighbor_distances(obs["heldout"], obs["train"])),
        }
        report["observation_coverage_l2_nearest_neighbor"] = coverage
    return report


def resimulate_unique_trajectories(manifest: dict[str, Any]) -> dict[str, list[Any]]:
    """Replay one match per unique trajectory fingerprint and return its teacher action sequence.

    Also verifies that the replayed decision fingerprints equal the frozen manifest (full pipeline determinism).
    """
    teacher_spec = baseline_spec(QUALITY_TEACHER)
    first_by_traj: dict[str, dict[str, Any]] = {}
    for match in manifest["matches"]:
        first_by_traj.setdefault(match["trajectory_sha256"], match)
    sequences: dict[str, list[Any]] = {}
    for trajectory, match in first_by_traj.items():
        config = quality_config(match["map_id"])
        opponent_spec = baseline_spec(match["opponent_id"])
        rows: list[dict[str, Any]] = []
        teacher = _RecordingTeacher(
            teacher_spec.build(), policy_stream_id(teacher_spec.policy_id, 0), rows,
            seed=match["seed"], teacher_side=match["teacher_side"], match_id=match["match_id"],
            split=match["split"], context_id=match["context_id"], map_id=match["map_id"],
            opponent_id=match["opponent_id"],
        )
        opponent = PolicyController(opponent_spec.build(), policy_stream_id(opponent_spec.policy_id, 1))
        controllers = (teacher, opponent) if match["teacher_side"] == 0 else (opponent, teacher)
        run_episode(config, match["seed"], controllers[0], controllers[1], max_ticks=manifest["max_ticks"])
        ordered = sorted(rows, key=lambda r: r["tick"])
        if _sha256([r["decision_sha256"] for r in ordered]) != trajectory:
            raise DatasetError(f"resimulation of {match['match_id']} does not reproduce its trajectory fingerprint")
        sequences[trajectory] = [
            [r["teacher_action"]["lay_order"], list(r["teacher_action"]["forage_share_q"]),
             r["teacher_action"]["army_choice"], r["teacher_action"]["avoid_center"]] for r in ordered
        ]
    return sequences


def audit_dataset(dataset_path: str | Path, *, resimulate: bool = False, output: str | Path | None = None) -> dict[str, Any]:
    from .quality_dataset import load_quality_dataset

    manifest, examples = load_quality_dataset(dataset_path)
    sequences = resimulate_unique_trajectories(manifest) if resimulate else None
    report = effective_sample_audit(manifest, examples, action_sequences=sequences)
    if output is not None:
        _write_atomic(Path(output), (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    return report


# ---------------------------------------------------------------------------------------------------------
# Part 2 — exploratory pilot
# ---------------------------------------------------------------------------------------------------------

def _assert_routable(config: SimConfig) -> None:
    for source in range(7):
        for destination in range(7):
            shortest_path(config.topology, source, destination, side=0)
            if destination != 3:
                shortest_path(config.topology, source, destination, side=0, avoid_center=True)


def pilot_config(map_id: str) -> SimConfig:
    """FOR-37 maps unchanged, plus the six FOR-38 mirror-valid topologies; balance and rules are never edited."""
    if map_id in EXISTING_MAPS:
        return quality_config(map_id)
    if map_id not in PILOT_MAP_EDITS:
        raise DatasetError(f"unknown FOR-38 pilot map_id: {map_id}")
    base = default_config()
    edges = {(edge.a, edge.b): edge.length for edge in base.topology.edges}
    edits, removed = PILOT_MAP_EDITS[map_id]
    for key in removed:
        edges.pop(key)
    edges.update(edits)
    topology = Topology(nodes=base.topology.nodes,
                        edges=tuple(EdgeSpec(a, b, length) for (a, b), length in sorted(edges.items())))
    config = SimConfig(rules_version=base.rules_version, balance_version=base.balance_version, topology=topology,
                       balance=base.balance, policy_interval_k=base.policy_interval_k, t_max=base.t_max)
    validate_config(config)
    check_observation_config(config)
    _assert_routable(config)
    return config


@dataclass(frozen=True, slots=True)
class PilotMatchSpec:
    arm: str
    map_id: str
    opponent: str  # "turtle:v2" or "turtle:v2>rush:v2@6"
    behavior: str  # "teacher", "eps0.10", "script:workers@2"
    seed: int
    side: int
    seed_sensitive: bool
    control: bool = False

    @property
    def context_id(self) -> str:
        return f"{self.map_id}__{self.opponent}__{self.behavior}"

    @property
    def match_id(self) -> str:
        return f"{self.arm}/{self.context_id}/seed-{self.seed}/side-{self.side}"

    @property
    def unit_id(self) -> str:
        base = f"{self.arm}|{self.context_id}"
        return f"{base}|{self.seed}" if self.seed_sensitive else base


def plan_pilot() -> list[PilotMatchSpec]:
    """The preregistered 275-match plan (docs/ml_for38_protocol.md)."""
    plan: list[PilotMatchSpec] = []
    new_maps = tuple(PILOT_MAP_EDITS)
    for map_id in new_maps:
        for opponent in DETERMINISTIC_OPPONENTS:
            plan.append(PilotMatchSpec("A", map_id, opponent, "teacher", PILOT_SEED_MAIN, 0, False))
    control_map = new_maps[0]
    for seed in PILOT_SEEDS_CONTROL:
        for side in (0, 1):
            plan.append(PilotMatchSpec("A", control_map, "turtle:v2", "teacher", seed, side, False, control=True))
    plan.append(PilotMatchSpec("A", control_map, "turtle:v2", "teacher", PILOT_SEED_MAIN, 1, False, control=True))
    for map_id in EXISTING_MAPS:
        for opponent in ("raider:v2", "balanced:v2"):
            plan.append(PilotMatchSpec("B", map_id, opponent, "teacher", PILOT_SEED_MAIN, 0, False))
    for map_id in EXISTING_MAPS:
        for first, second in SWITCH_SCHEDULES:
            for point in SWITCH_POINTS:
                plan.append(PilotMatchSpec("B", map_id, f"{first}>{second}@{point}", "teacher", PILOT_SEED_MAIN, 0, False))
    for map_id in EXISTING_MAPS:
        for seed in PILOT_SEEDS_RANDOM:
            plan.append(PilotMatchSpec("B", map_id, "random:v2", "teacher", seed, 0, True))
    for epsilon in PERTURB_EPSILONS:
        for opponent in PERTURB_OPPONENTS:
            for map_id in EXISTING_MAPS:
                for seed in PILOT_SEEDS_PERTURB:
                    plan.append(PilotMatchSpec("C", map_id, opponent, f"eps{epsilon:.2f}", seed, 0, True))
    for script in OPENING_SCRIPTS:
        for length in OPENING_LENGTHS:
            for map_id in EXISTING_MAPS:
                plan.append(PilotMatchSpec("D", map_id, "turtle:v2", f"script:{script}@{length}", PILOT_SEED_MAIN, 0, False))
    if len(plan) > PILOT_MAX_MATCHES:
        raise DatasetError(f"pilot plan has {len(plan)} matches, preregistered maximum is {PILOT_MAX_MATCHES}")
    return plan


class _SwitchingPolicy(BaselinePolicy):
    """Follow baseline ``first`` for ``point`` decisions, then baseline ``second`` (public inputs only)."""

    def __init__(self, first: BaselinePolicy, second: BaselinePolicy, point: int) -> None:
        self.first, self.second, self.point, self.calls = first, second, point, 0

    def __call__(self, observation: ObservationV2, mask: ActionMaskV2, policy_rng: Any) -> PolicyLogitsV2:
        policy = self.first if self.calls < self.point else self.second
        self.calls += 1
        return policy(observation, mask, policy_rng)


def _build_opponent(opponent: str) -> tuple[BaselinePolicy, str]:
    if ">" not in opponent:
        spec = baseline_spec(opponent)
        return spec.build(), spec.policy_id
    pair, point = opponent.rsplit("@", 1)
    first_id, second_id = pair.split(">")
    first, second = baseline_spec(first_id), baseline_spec(second_id)
    policy_id = f"switch-{first.policy_id}-to-{second.policy_id}-at{int(point)}"
    return _SwitchingPolicy(first.build(), second.build(), int(point)), policy_id


PERTURB_POLICY_ID = "pilot-perturb-v1"


class _PilotTeacher(_RecordingTeacher):
    """Records Balanced v2 logits on every visited state, but may play a different legal action."""

    def __init__(self, *args: Any, behavior: str, perturb_stream: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.behavior = behavior
        self.perturb_stream = perturb_stream
        self.decision_index = 0
        self.executed_substitutions = 0
        self._random = baseline_spec("random:v2").build()
        self.epsilon = float(behavior[3:]) if behavior.startswith("eps") else None
        self.script = None
        if behavior.startswith("script:"):
            name, length = behavior[len("script:"):].split("@")
            self.script, self.script_length = OPENING_SCRIPTS[name], int(length)

    def _executed_action(self, tick: int, observation: ObservationV2, mask: ActionMaskV2,
                         teacher_action: ActionV2, run: Any) -> ActionV2:
        index = self.decision_index
        self.decision_index += 1
        if self.script is not None and index < self.script_length:
            self.executed_substitutions += 1
            return self.script
        if self.epsilon is not None:
            stream = run.stream(self.perturb_stream)
            if stream.next_u32() / 2**32 < self.epsilon:
                logits = PolicyLogitsV2.coerce(self._random(observation, mask, stream))
                self.executed_substitutions += 1
                return validate_action(observation, mask, decode_action_logits(logits, mask)).action
        return teacher_action


def run_pilot_match(spec: PilotMatchSpec, *, max_ticks: int = PILOT_MAX_TICKS) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = pilot_config(spec.map_id)
    teacher_spec = baseline_spec(QUALITY_TEACHER)
    opponent_policy, opponent_policy_id = _build_opponent(spec.opponent)
    perturb_stream = policy_stream_id(PERTURB_POLICY_ID, 2)
    rows: list[dict[str, Any]] = []
    teacher = _PilotTeacher(
        teacher_spec.build(), policy_stream_id(teacher_spec.policy_id, 0), rows,
        seed=spec.seed, teacher_side=spec.side, match_id=spec.match_id, split="train",
        context_id=spec.context_id, map_id=spec.map_id, opponent_id=spec.opponent,
        behavior=spec.behavior, perturb_stream=perturb_stream,
    )
    opponent = PolicyController(opponent_policy, policy_stream_id(opponent_policy_id, 1))
    controllers = (teacher, opponent) if spec.side == 0 else (opponent, teacher)
    result = run_episode(config, spec.seed, controllers[0], controllers[1], max_ticks=max_ticks,
                         extra_streams=(perturb_stream,))
    record = {
        "match_id": spec.match_id,
        "arm": spec.arm,
        "split": "train",
        "seed": spec.seed,
        "teacher_side": spec.side,
        "context_id": spec.context_id,
        "map_id": spec.map_id,
        "opponent_id": spec.opponent,
        "behavior": spec.behavior,
        "config_sha256": config_hash(config),
        "ticks_executed": result.ticks_executed,
        "final_tick": result.final_tick,
        "truncated": result.truncated,
        "reason": result.reason,
        "winner": result.winner,
        "substituted_decisions": teacher.executed_substitutions,
        "state_sha256": result.state_hash,
    }
    return rows, record


def assign_pilot_splits(units_by_arm: dict[str, Sequence[str]]) -> dict[str, str]:
    """Preregistered rule: hash-sort the units of each arm, 60% train / 20% validation / rest held-out."""
    assignment: dict[str, str] = {}
    for arm, units in units_by_arm.items():
        ordered = sorted(set(units), key=lambda u: hashlib.sha256(f"for38-pilot-split:{u}".encode()).hexdigest())
        n = len(ordered)
        n_train = math.ceil(n * SPLIT_FRACTIONS[0])
        n_validation = math.ceil(n * SPLIT_FRACTIONS[1])
        for index, unit in enumerate(ordered):
            if index < n_train:
                assignment[unit] = "train"
            elif index < n_train + n_validation:
                assignment[unit] = "validation"
            else:
                assignment[unit] = "heldout"
        if n >= 3 and not any(assignment[u] == "validation" for u in ordered):
            assignment[ordered[n_train]] = "validation"
    return assignment


def _arm_metrics(
    arm_rows: list[dict[str, Any]], arm_matches: list[dict[str, Any]], retained_ids: set[tuple[str, int]],
    for37_decisions: set[str], for37_observations: Sequence[Sequence[float]], *, nn_sample: int = 400,
) -> dict[str, Any]:
    matches = len(arm_matches)
    trajectories = {m["trajectory_sha256"] for m in arm_matches}
    decisions = [r["decision_sha256"] for r in arm_rows]
    unique = set(decisions)
    new = unique - for37_decisions
    by_context_new: dict[str, set[str]] = defaultdict(set)
    for row in arm_rows:
        if row["decision_sha256"] in new:
            by_context_new[(row["map_id"], row["opponent_id"], row.get("behavior", "teacher"))].add(row["decision_sha256"])
    independent_contexts = sorted(
        "|".join(key) for key, values in by_context_new.items() if len(values) >= PILOT_MIN_NEW_DECISIONS_PER_CONTEXT
    )
    validation_rows = [r for r in arm_rows if r["split"] == "validation"]
    validation_kept = sum((r["match_id"], r["tick"]) in retained_ids for r in validation_rows)
    new_rows = sorted({r["decision_sha256"]: r["observation"] for r in arm_rows if r["decision_sha256"] in new}.items())
    sample = [obs for _, obs in new_rows[:nn_sample]]
    nn = nearest_neighbor_distances(sample, for37_observations) if sample and for37_observations else []
    reasons = Counter(m["reason"] for m in arm_matches)
    lengths = [m["decisions"] for m in arm_matches]
    trajectory_dup = 1 - len(trajectories) / matches if matches else 0.0
    new_share = len(new) / len(decisions) if decisions else 0.0
    validation_retention = validation_kept / len(validation_rows) if validation_rows else None
    seed_sensitive = bool(arm_matches) and all(m["seed_sensitive"] for m in arm_matches)
    contexts_or_trajectories = len(trajectories) if seed_sensitive else len(independent_contexts)
    checks = {
        "trajectory_duplicate_rate_at_most": trajectory_dup <= RECOMMENDATION_RULE["max_trajectory_duplicate_rate"],
        "new_decision_share_at_least": new_share >= RECOMMENDATION_RULE["min_new_decision_share"],
        "validation_retention_at_least": validation_retention is not None
        and validation_retention >= RECOMMENDATION_RULE["min_validation_retention"],
        "independent_contexts_or_trajectories_at_least":
            contexts_or_trajectories >= RECOMMENDATION_RULE["min_independent_contexts_or_trajectories"],
    }
    return {
        "matches": matches,
        "unique_trajectories": len(trajectories),
        "trajectory_duplicate_rate": trajectory_dup,
        "decisions_before_dedup": len(decisions),
        "unique_decisions": len(unique),
        "decision_duplicate_rate": 1 - len(unique) / len(decisions) if decisions else 0.0,
        "unique_decisions_absent_from_for37": len(new),
        "new_decision_share_of_all_decisions": new_share,
        "independent_contexts_with_30_new_decisions": independent_contexts,
        "independent_context_count": len(independent_contexts),
        "contexts_or_trajectories_for_rule": contexts_or_trajectories,
        "validation_decisions_before_dedup": len(validation_rows),
        "validation_decisions_retained": validation_kept,
        "validation_retention": validation_retention,
        "mean_decisions_per_match": statistics.fmean(lengths) if lengths else 0.0,
        "decisions_per_match_min_max": [min(lengths), max(lengths)] if lengths else None,
        "terminal_reasons": dict(sorted(reasons.items())),
        "nearest_for37_observation_l2": _quantiles(nn),
        "rule_checks": checks,
        "meets_preregistered_rule": all(checks.values()),
    }


def label_yield(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """How many *non-majority* teacher labels a set of retained decisions holds (class balance, post-hoc)."""
    n = len(rows)
    actions = [r["teacher_action"] for r in rows]
    return {
        "decisions": n,
        "lay_not_none": sum(a["lay_order"] != 0 for a in actions),
        "army_not_garrison": sum(a["army_choice"] != 0 for a in actions),
        "army_defend": sum(a["army_choice"] in (1, 2, 3, 4, 5) for a in actions),
        "army_raid": sum(a["army_choice"] == 6 for a in actions),
        "forage_support_2": sum(sum(1 for v in a["forage_share_q"] if v) == 2 for a in actions),
        "distinct_forage_tuples": len({tuple(a["forage_share_q"]) for a in actions}),
        "distinct_lay_army_avoid": len({(a["lay_order"], a["army_choice"], a["avoid_center"]) for a in actions}),
    }


def _source_label(row: dict[str, Any]) -> str:
    if row["arm"] == "B":
        if row["opponent_id"] == "random:v2":
            return "B3_random_seeds"
        return "B2_switching_schedules" if ">" in row["opponent_id"] else "B1_extra_baselines"
    if row["arm"] == "C":
        return f"C_{row['behavior']}"
    return row["arm"]



def run_diversity_pilot(
    dataset_path: str | Path,
    *,
    plan: Sequence[PilotMatchSpec] | None = None,
    max_ticks: int = PILOT_MAX_TICKS,
    rows_output: str | Path | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Run the preregistered exploratory pilot.  The FOR-37 dataset is read only to know which decisions it holds."""
    from .quality_dataset import load_quality_dataset

    plan = list(plan_pilot() if plan is None else plan)
    if len(plan) > PILOT_MAX_MATCHES:
        raise DatasetError(f"pilot plan has {len(plan)} matches, preregistered maximum is {PILOT_MAX_MATCHES}")
    manifest37, examples37 = load_quality_dataset(dataset_path)
    for37_decisions = {h for m in manifest37["matches"] for h in m["trajectory_decision_sha256s"]}
    for37_observations = [e.observation for e in examples37]
    validation37_raw = sum(m["source_decision_count"] for m in manifest37["matches"] if m["split"] == "validation")
    validation37_kept = sum(1 for e in examples37 if e.split == "validation")

    started = time.perf_counter()
    rows_by_match: dict[str, list[dict[str, Any]]] = {}
    match_records: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    for spec in plan:
        rows, record = run_pilot_match(spec, max_ticks=max_ticks)
        if spec.control:
            record["trajectory_sha256"] = _sha256([r["decision_sha256"] for r in sorted(rows, key=lambda r: r["tick"])])
            record["decisions"] = len(rows)
            controls.append(record)
            continue
        record["seed_sensitive"] = spec.seed_sensitive
        record["unit_id"] = spec.unit_id
        rows_by_match[spec.match_id] = rows
        match_records.append(record)
        record["decisions"] = len(rows)
    elapsed = time.perf_counter() - started

    assignment = assign_pilot_splits({
        arm: [m["unit_id"] for m in match_records if m["arm"] == arm] for arm in sorted({m["arm"] for m in match_records})
    })
    for record in match_records:
        record["split"] = assignment[record["unit_id"]]
        for row in rows_by_match[record["match_id"]]:
            row["split"] = record["split"]
            row["behavior"] = record["behavior"]
            row["arm"] = record["arm"]
    dedup_input = [{k: record[k] for k in ("match_id", "split", "seed", "teacher_side", "context_id")}
                   for record in match_records]
    rows_after, dedup_summary = _deduplicate_rows(rows_by_match, dedup_input)
    for record, dm in zip(match_records, dedup_input):
        record["trajectory_sha256"] = dm["trajectory_sha256"]
        record["trajectory_retained"] = dm["trajectory_retained"]
        record["duplicate_of_match_id"] = dm["duplicate_of_match_id"]
        record["retained_decision_count"] = dm["retained_decision_count"]
    retained_ids = {(r["match_id"], r["tick"]) for r in rows_after}
    all_rows = [r for values in rows_by_match.values() for r in values]
    arms = sorted({m["arm"] for m in match_records})
    metrics_by_arm: dict[str, Any] = {}
    for arm in arms:
        arm_rows = [r for r in all_rows if r["arm"] == arm]
        arm_matches = [m for m in match_records if m["arm"] == arm]
        metrics_by_arm[arm] = _arm_metrics(arm_rows, arm_matches, retained_ids, for37_decisions, for37_observations)

    # Sub-groups inside arm B so a failing sub-source is not hidden by a passing one.
    subgroup_defs = {
        "B1_extra_baselines": lambda m: m["arm"] == "B" and ">" not in m["opponent_id"] and m["opponent_id"] != "random:v2",
        "B2_switching_schedules": lambda m: m["arm"] == "B" and ">" in m["opponent_id"],
        "B3_random_seeds": lambda m: m["arm"] == "B" and m["opponent_id"] == "random:v2",
        "C_eps0.10": lambda m: m["arm"] == "C" and m["behavior"] == "eps0.10",
        "C_eps0.25": lambda m: m["arm"] == "C" and m["behavior"] == "eps0.25",
    }
    subgroups: dict[str, Any] = {}
    for name, predicate in subgroup_defs.items():
        group_matches = [m for m in match_records if predicate(m)]
        if not group_matches:
            continue
        ids = {m["match_id"] for m in group_matches}
        subgroups[name] = _arm_metrics([r for r in all_rows if r["match_id"] in ids], group_matches,
                                       retained_ids, for37_decisions, for37_observations)

    def _retention_by_group_split() -> dict[str, Any]:
        """Post-hoc (not preregistered): all seeds of one (arm, context) share a split; compare validation retention."""
        group_of = {m["match_id"]: f"{m['arm']}|{m['context_id']}" for m in match_records}
        assignment_g = assign_pilot_splits({
            arm: [group_of[m["match_id"]] for m in match_records if m["arm"] == arm] for arm in arms
        })
        copies = {mid: [dict(r) for r in rows] for mid, rows in rows_by_match.items()}
        alt_matches = []
        for record in match_records:
            split = assignment_g[group_of[record["match_id"]]]
            alt_matches.append({"match_id": record["match_id"], "split": split, "seed": record["seed"],
                                "teacher_side": record["teacher_side"], "context_id": record["context_id"]})
            for row in copies[record["match_id"]]:
                row["split"] = split
        alt_rows, _ = _deduplicate_rows(copies, alt_matches)
        alt_ids = {(r["match_id"], r["tick"]) for r in alt_rows}
        out: dict[str, Any] = {}
        groups = {arm: {m["match_id"] for m in match_records if m["arm"] == arm} for arm in arms}
        groups.update({name: {m["match_id"] for m in match_records if predicate(m)}
                       for name, predicate in subgroup_defs.items()})
        for name, ids in groups.items():
            if not ids:
                continue
            val_rows = [r for mid in ids for r in copies[mid] if r["split"] == "validation"]
            kept = sum((r["match_id"], r["tick"]) in alt_ids for r in val_rows)
            out[name] = {"validation_decisions_before_dedup": len(val_rows), "validation_decisions_retained": kept,
                         "validation_retention": kept / len(val_rows) if val_rows else None}
        return out

    posthoc_group_split = _retention_by_group_split()
    total_unique = {r["decision_sha256"] for r in all_rows}
    control_by_unit = defaultdict(set)
    main_hash = next((m["trajectory_sha256"] for m in match_records
                      if m["arm"] == "A" and m["map_id"] == next(iter(PILOT_MAP_EDITS)) and m["opponent_id"] == "turtle:v2"), None)
    for control in controls:
        control_by_unit[control["context_id"]].add(control["trajectory_sha256"])
    control_summary = {
        "matches": len(controls),
        "extra_seeds_and_sides_identical_to_main_trajectory": all(
            c["trajectory_sha256"] == main_hash for c in controls) if main_hash else None,
        "distinct_trajectories_among_controls_plus_main": len({c["trajectory_sha256"] for c in controls} | {main_hash}),
    }
    report: dict[str, Any] = {
        "schema": PILOT_SCHEMA,
        "protocol": "docs/ml_for38_protocol.md",
        "classification": "exploratory source-of-states pilot; no model evaluated, nothing trained",
        "dataset_for37_sha256": manifest37["dataset_sha256"],
        "budget": {
            "planned_matches": len(plan), "max_matches": PILOT_MAX_MATCHES, "max_ticks": max_ticks,
            "executed_matches": len(match_records) + len(controls),
            "within_budget": len(match_records) + len(controls) <= PILOT_MAX_MATCHES,
        },
        "rule": RECOMMENDATION_RULE,
        "for37_reference": {
            "validation_decisions_before_dedup": validation37_raw,
            "validation_decisions_retained": validation37_kept,
            "validation_retention": validation37_kept / validation37_raw,
            "duplicate_trajectories_removed": manifest37["deduplication"]["duplicate_trajectories_removed"],
            "matches": manifest37["match_count"],
        },
        "totals": {
            "matches": len(match_records),
            "decisions_before_dedup": len(all_rows),
            "unique_decisions": len(total_unique),
            "unique_decisions_absent_from_for37": len(total_unique - for37_decisions),
            "retained_after_pilot_dedup": len(rows_after),
            "dedup": dedup_summary,
            "split_counts": dict(Counter(r["split"] for r in rows_after)),
        },
        "arms": metrics_by_arm,
        "subgroups": subgroups,
        "controls": control_summary,
        "posthoc_label_yield": {
            "preregistered": False,
            "note": "added after the first run: non-majority teacher labels among the retained decisions of each source",
            "sources": {name: label_yield([r for r in rows_after if _source_label(r) == name])
                        for name in sorted({_source_label(r) for r in rows_after})},
            "all_pilot": label_yield(rows_after),
            "for37_dataset": label_yield([
                {"teacher_action": {"lay_order": e.teacher_action.lay_order,
                                    "forage_share_q": list(e.teacher_action.forage_share_q),
                                    "army_choice": e.teacher_action.army_choice,
                                    "avoid_center": e.teacher_action.avoid_center}} for e in examples37]),
        },
        "posthoc_group_split_validation_retention": {
            "preregistered": False,
            "note": "added after the first run: same matches, but every seed of an (arm, context) goes to one split",
            "groups": posthoc_group_split,
        },
        "matches": [{k: v for k, v in m.items() if k != "state_sha256"} for m in match_records],
        "runtime": {"seconds": elapsed, "python": platform.python_version()},
    }
    if rows_output is not None:
        raw = b"".join(json.dumps(r, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
                       for r in rows_after)
        _write_atomic(Path(rows_output), raw)
        report["rows_output"] = {"file": Path(rows_output).name, "sha256": hashlib.sha256(raw).hexdigest(),
                                 "rows": len(rows_after)}
    if output is not None:
        _write_atomic(Path(output), (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    return report


__all__ = [
    "AUDIT_SCHEMA", "OPENING_SCRIPTS", "label_yield", "PILOT_MAP_EDITS", "PILOT_MAX_MATCHES", "PILOT_SCHEMA", "PilotMatchSpec",
    "assign_pilot_splits", "audit_dataset", "effective_sample_audit", "nearest_neighbor_distances", "pilot_config",
    "plan_pilot", "resimulate_unique_trajectories", "run_diversity_pilot", "run_pilot_match",
]
