"""FOR-40: independent generated maps, rare-class aware v3 dataset, and the pre-registered data gate.

The protocol (docs/ml_for40_protocol.md) is committed before any collection.  Nothing here changes the engine,
``ObservationV2``, the decoder, ``ActionV2`` or Baselines; the v1/v2 readers stay untouched.  Maps differ only in the
public ``Topology``; labels are always Balanced v2 logits on the visited state.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..actions import ActionV2, flatten_mask_v2, get_action_mask, mask_from_flat, validate_action
from ..baselines import baseline_spec
from ..config import EdgeSpec, SimConfig, Topology
from ..decoder import PolicyLogitsV2, decode_action_logits
from ..invariants import validate_config
from ..match import run_episode
from ..observation import OBS_V2_LENGTH, ObservationV2, check_observation_config
from ..rng import policy_stream_id
from ..routing import shortest_path
from ..rules import MIRROR, default_config
from ..serialization import config_hash
from ..tick import PolicyController
from ..versions import ACTION_VERSION, DECODER_VERSION, OBSERVATION_VERSION
from .dataset import DatasetError, _write_atomic
from .diversity import PERTURB_POLICY_ID, PILOT_MAP_EDITS, _build_opponent, _PilotTeacher, pilot_config
from .quality_dataset import (
    QUALITY_TEACHER,
    _canonical_bytes,
    _decision_payload,
    _deduplicate_rows,
    _git_provenance,
    _parse_action,
    _sha256,
    quality_config,
)

FOR40_SCHEMA = "formicmind.behavior-cloning/v3"
FOR40_PROTOCOL = "docs/ml_for40_protocol.md"
FOR40_PROTOCOL_COMMIT = "efafe006baa3dba6a2ee5ec49511210b911cee86"
SPLITS = ("train", "validation", "heldout")
GENERATOR_SEED_START = 400_000_001
MAP_COUNTS = {"train": 24, "validation": 8, "heldout": 8}
MAP_PLAN_SHA256 = "c294b38b777453401c17df660188e46770b9002741081ed22ace587e33f2c920"
SPLIT_SEED_BASE = {"train": 401_000_000, "validation": 402_000_000, "heldout": 403_000_000}
REAL_MATCH_SEEDS = tuple(range(404_000_001, 404_000_005))
TRAIN_SEEDS = (405_000_001, 405_000_002, 405_000_003)
UNTRAINED_SEED = 405_900_001
MAX_TICKS = 3000
MAX_MATCHES = 1500
MAX_WALL_SECONDS = 40 * 60
DISTURBED_CAP = 0.30
MIN_MAP_DISTANCE = 2
EDGE_COUNT_RANGE = (7, 13)
DET_OPPONENTS_TRAIN = ("turtle:v2", "balanced:v2")
DET_OPPONENTS_HELDOUT = ("turtle:v2", "balanced:v2", "rush:v2", "raider:v2")
RANDOM_OPPONENT = "random:v2"
SWITCH_OPPONENTS = ("turtle:v2>balanced:v2@3", "turtle:v2>balanced:v2@12",
                    "balanced:v2>turtle:v2@3", "balanced:v2>turtle:v2@12")
EPSILONS = (0.10, 0.25)
EPS_OPPONENTS = ("turtle:v2", "balanced:v2", "random:v2")
EPS_SEED_COUNT = {"train": 4, "validation": 2}
RANDOM_SEED_COUNT = 8
EXPECTED_MATCHES = {"train": 912, "validation": 208, "heldout": 96}

# Class thresholds of the pre-registered data gate (decisions, maps); ``None`` means no map requirement.
GATE_THRESHOLDS: dict[str, dict[str, tuple[int, int | None]]] = {
    "train": {
        "lay_worker": (150, 10), "lay_soldier": (300, 12), "lay_none": (500, None),
        "army_defend": (200, 8), "army_raid": (100, 8), "army_garrison": (500, None),
        "forage_support_2": (200, 12), "forage_support_3": (500, None),
        "avoid_true_conditional": (100, 6), "avoid_false_conditional": (60, 6),
    },
    "validation": {
        "lay_worker": (30, 3), "lay_soldier": (60, 4), "army_defend": (40, 3), "army_raid": (30, 3),
        "forage_support_2": (40, None), "forage_support_3": (100, None),
        "avoid_true_conditional": (20, None), "avoid_false_conditional": (15, None),
    },
    "heldout": {
        "lay_worker": (30, 3), "lay_soldier": (60, 4), "army_defend": (40, 3), "army_raid": (30, 3),
        "forage_support_2": (40, None), "forage_support_3": (100, None),
        "avoid_true_conditional": (20, None), "avoid_false_conditional": (15, None),
    },
}
GATE_MIN_DECISIONS = {"train": (3000, 300), "validation": (500, 60), "heldout": (500, 50)}
GATE_MIN_MAP_DECISIONS_HELDOUT = 30
GATE_MIN_OPPONENT_DECISIONS_HELDOUT = 40
HELDOUT_OPPONENT_KEYS = ("turtle:v2", "balanced:v2", "random:v2", "rush:v2", "raider:v2")


# ---------------------------------------------------------------------------------------------------------
# Map generator
# ---------------------------------------------------------------------------------------------------------

def _mirror_pair(a: int, b: int) -> tuple[int, int]:
    return tuple(sorted((MIRROR[a], MIRROR[b])))  # type: ignore[return-value]


@lru_cache(maxsize=1)
def edge_orbits() -> tuple[tuple[int, int], ...]:
    """The 11 mirror orbits of non-(A,B) edges; representative = lexicographically smaller pair."""
    representatives = set()
    for a in range(7):
        for b in range(a + 1, 7):
            if (a, b) == (0, 6):
                continue
            representatives.add(min((a, b), _mirror_pair(a, b)))
    return tuple(sorted(representatives))


def candidate_vector(seed: int) -> tuple[int, ...]:
    vector = []
    for index in range(len(edge_orbits())):
        digest = hashlib.sha256(f"for40-map:{seed}:{index}".encode()).digest()
        vector.append(1 + digest[1] % 4 if digest[0] < 128 else 0)
    return tuple(vector)


def vector_edges(vector: tuple[int, ...]) -> dict[tuple[int, int], int]:
    edges: dict[tuple[int, int], int] = {}
    for (a, b), length in zip(edge_orbits(), vector, strict=True):
        if length:
            edges[(a, b)] = length
            edges[_mirror_pair(a, b)] = length
    return edges


def config_from_vector(vector: tuple[int, ...]) -> SimConfig:
    base = default_config()
    topology = Topology(
        nodes=base.topology.nodes,
        edges=tuple(EdgeSpec(a, b, length) for (a, b), length in sorted(vector_edges(vector).items())),
    )
    config = SimConfig(
        rules_version=base.rules_version, balance_version=base.balance_version, topology=topology,
        balance=base.balance, policy_interval_k=base.policy_interval_k, t_max=base.t_max,
    )
    validate_config(config)
    check_observation_config(config)
    for source in range(7):
        for destination in range(7):
            shortest_path(config.topology, source, destination, side=0)
            if destination != 3:
                shortest_path(config.topology, source, destination, side=0, avoid_center=True)
    return config


def vector_of_config(config: SimConfig) -> tuple[int, ...]:
    lengths = {(edge.a, edge.b): edge.length for edge in config.topology.edges}
    return tuple(lengths.get(pair, 0) for pair in edge_orbits())


def vector_distance(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    return sum(x != y for x, y in zip(a, b, strict=True))


def vector_key(vector: tuple[int, ...]) -> str:
    return "-".join(str(value) for value in vector)


def reference_vectors() -> dict[str, tuple[int, ...]]:
    result = {name: vector_of_config(quality_config(name)) for name in ("classic", "cross", "dense")}
    result.update({name: vector_of_config(pilot_config(name)) for name in PILOT_MAP_EDITS})
    return result


@dataclass(frozen=True, slots=True)
class MapSpec:
    seed: int
    vector: tuple[int, ...]
    split: str

    @property
    def map_id(self) -> str:
        return f"g{self.seed}"

    @property
    def key(self) -> str:
        return vector_key(self.vector)


@lru_cache(maxsize=1)
def map_plan() -> tuple[MapSpec, ...]:
    """The 40 accepted maps in acceptance order with their pre-registered split."""
    references = list(reference_vectors().values())
    accepted: list[tuple[int, tuple[int, ...]]] = []
    seed = GENERATOR_SEED_START
    total = sum(MAP_COUNTS.values())
    while len(accepted) < total:
        if seed > GENERATOR_SEED_START + 100_000:
            raise DatasetError("map generator exhausted its candidate budget")
        vector = candidate_vector(seed)
        candidate_seed, seed = seed, seed + 1
        edge_count = len(vector_edges(vector))
        if not EDGE_COUNT_RANGE[0] <= edge_count <= EDGE_COUNT_RANGE[1]:
            continue
        try:
            config_from_vector(vector)
        except Exception:  # noqa: BLE001 - any validation/routing failure rejects the candidate
            continue
        if any(vector_distance(vector, other) < MIN_MAP_DISTANCE
               for other in [*references, *(v for _, v in accepted)]):
            continue
        accepted.append((candidate_seed, vector))
    ranked = sorted(accepted, key=lambda item: hashlib.sha256(
        f"for40-map-split:{vector_key(item[1])}".encode()).hexdigest())
    split_of: dict[int, str] = {}
    for index, (candidate_seed, _) in enumerate(ranked):
        split_of[candidate_seed] = ("train" if index < MAP_COUNTS["train"]
                                    else "validation" if index < MAP_COUNTS["train"] + MAP_COUNTS["validation"]
                                    else "heldout")
    return tuple(MapSpec(s, v, split_of[s]) for s, v in accepted)


def map_plan_sha256() -> str:
    canonical = json.dumps([[m.seed, m.key, m.split] for m in map_plan()], separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def maps_of_split(split: str) -> tuple[MapSpec, ...]:
    return tuple(m for m in map_plan() if m.split == split)


@lru_cache(maxsize=None)
def map_by_id(map_id: str) -> MapSpec:
    for spec in map_plan():
        if spec.map_id == map_id:
            return spec
    raise DatasetError(f"unknown FOR-40 map_id: {map_id}")


def for40_config(map_id: str) -> SimConfig:
    return config_from_vector(map_by_id(map_id).vector)


# ---------------------------------------------------------------------------------------------------------
# Collection plan
# ---------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class For40Match:
    split: str
    map_id: str
    source: str
    opponent: str
    behavior: str  # "teacher" or "eps0.10"
    seed: int

    @property
    def disturbed(self) -> bool:
        return self.source in ("S_eps", "S_switch")

    @property
    def context_id(self) -> str:
        return f"{self.map_id}__{self.source}__{self.opponent}__{self.behavior}"

    @property
    def match_id(self) -> str:
        return f"{self.split}/{self.context_id}/seed-{self.seed}"

    @property
    def seed_context(self) -> str:
        return f"{self.map_id}|{self.source}|{self.seed}"


def plan_for40(maps: tuple[MapSpec, ...] | None = None) -> list[For40Match]:
    """The pre-registered match plan (docs/ml_for40_protocol.md); ``maps`` is only overridden by smoke tests."""
    selected = map_plan() if maps is None else maps
    plan: list[For40Match] = []
    for spec in selected:
        split, base = spec.split, SPLIT_SEED_BASE[spec.split]
        if split == "heldout":
            for opponent in DET_OPPONENTS_HELDOUT:
                plan.append(For40Match(split, spec.map_id, "H_det", opponent, "teacher", base))
            for k in range(RANDOM_SEED_COUNT):
                plan.append(For40Match(split, spec.map_id, "H_random", RANDOM_OPPONENT, "teacher", base + 10 + k))
            continue
        for opponent in DET_OPPONENTS_TRAIN:
            plan.append(For40Match(split, spec.map_id, "P_det", opponent, "teacher", base))
        for k in range(RANDOM_SEED_COUNT):
            plan.append(For40Match(split, spec.map_id, "P_random", RANDOM_OPPONENT, "teacher", base + 10 + k))
        for epsilon, offset in zip(EPSILONS, (100, 200), strict=True):
            for opponent in EPS_OPPONENTS:
                for k in range(EPS_SEED_COUNT[split]):
                    plan.append(For40Match(split, spec.map_id, "S_eps", opponent, f"eps{epsilon:.2f}",
                                           base + offset + k))
        for opponent in SWITCH_OPPONENTS:
            plan.append(For40Match(split, spec.map_id, "S_switch", opponent, "teacher", base))
    return plan


def plan_sha256(plan: list[For40Match]) -> str:
    return _sha256([item.match_id for item in plan])


def run_for40_match(item: For40Match, *, max_ticks: int = MAX_TICKS) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = for40_config(item.map_id)
    teacher_spec = baseline_spec(QUALITY_TEACHER)
    opponent_policy, opponent_policy_id = _build_opponent(item.opponent)
    perturb_stream = policy_stream_id(PERTURB_POLICY_ID, 2)
    rows: list[dict[str, Any]] = []
    teacher = _PilotTeacher(
        teacher_spec.build(), policy_stream_id(teacher_spec.policy_id, 0), rows,
        seed=item.seed, teacher_side=0, match_id=item.match_id, split=item.split,
        context_id=item.context_id, map_id=item.map_id, opponent_id=item.opponent,
        behavior=item.behavior, perturb_stream=perturb_stream,
    )
    opponent = PolicyController(opponent_policy, policy_stream_id(opponent_policy_id, 1))
    result = run_episode(config, item.seed, teacher, opponent, max_ticks=max_ticks, extra_streams=(perturb_stream,))
    for row in rows:
        row["schema"] = FOR40_SCHEMA
        row["source"] = item.source
        row["behavior"] = item.behavior
        row["disturbed"] = item.disturbed
        row["seed_context"] = item.seed_context
    record = {
        "match_id": item.match_id, "split": item.split, "seed": item.seed, "teacher_side": 0,
        "context_id": item.context_id, "map_id": item.map_id, "source": item.source,
        "opponent_id": item.opponent, "behavior": item.behavior, "disturbed": item.disturbed,
        "seed_context": item.seed_context,
        "ticks_executed": result.ticks_executed, "final_tick": result.final_tick, "truncated": result.truncated,
        "winner": result.winner, "reason": result.reason,
        "score": None if result.score is None else list(result.score),
        "config_sha256": result.config_hash, "state_sha256": result.state_hash,
        "substituted_decisions": teacher.executed_substitutions,
    }
    return rows, record


def _disturbed_share(rows: list[dict[str, Any]], split: str) -> float:
    in_split = [r for r in rows if r["split"] == split]
    return sum(r["disturbed"] for r in in_split) / len(in_split) if in_split else 0.0


def _dedup(rows_by_match: dict[str, list[dict[str, Any]]], records: dict[str, dict[str, Any]],
           active: set[str]) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]]]:
    matches = [records[mid] for mid in sorted(active)]
    rows, summary = _deduplicate_rows({mid: rows_by_match[mid] for mid in sorted(active)}, matches)
    return rows, summary, matches


def collect_for40_rows(plan: list[For40Match], *, max_ticks: int = MAX_TICKS,
                       wall_limit: float | None = MAX_WALL_SECONDS) -> dict[str, Any]:
    """Run the plan, apply the disturbed cap deterministically, and return rows + match records."""
    if len(plan) > MAX_MATCHES:
        raise DatasetError(f"plan has {len(plan)} matches, preregistered maximum is {MAX_MATCHES}")
    if len({item.match_id for item in plan}) != len(plan):
        raise DatasetError("plan contains duplicate match ids")
    started = time.perf_counter()
    rows_by_match: dict[str, list[dict[str, Any]]] = {}
    records: dict[str, dict[str, Any]] = {}
    for item in plan:
        rows, record = run_for40_match(item, max_ticks=max_ticks)
        rows_by_match[item.match_id] = rows
        records[item.match_id] = record
        if wall_limit is not None and time.perf_counter() - started > wall_limit:
            raise DatasetError("collection exceeded the preregistered wall-time limit")
    active = set(records)
    removed: list[dict[str, Any]] = []
    while True:
        retained, summary, matches = _dedup(rows_by_match, records, active)
        violating = [s for s in ("train", "validation") if _disturbed_share(retained, s) > DISTURBED_CAP]
        if not violating:
            break
        candidates = sorted(
            (mid for mid in active if records[mid]["disturbed"] and records[mid]["split"] in violating),
            key=lambda mid: hashlib.sha256(mid.encode()).hexdigest(), reverse=True)
        if not candidates:
            raise DatasetError("cannot satisfy the disturbed-decision cap")
        victim = candidates[0]
        active.discard(victim)
        removed.append({"match_id": victim, "split": records[victim]["split"]})
    for mid in set(records) - active:
        records[mid].update({"removed_by_disturbed_cap": True, "trajectory_retained": False, "retained": False,
                             "retained_decision_count": 0})
    for mid in active:
        records[mid]["removed_by_disturbed_cap"] = False
    return {
        "rows": retained, "matches": [records[item.match_id] for item in plan], "dedup": summary,
        "disturbed_cap": {"cap": DISTURBED_CAP, "matches_removed": removed,
                          "share_after": {s: _disturbed_share(retained, s) for s in ("train", "validation")}},
        "wall_seconds": time.perf_counter() - started, "executed_matches": len(plan),
    }


# ---------------------------------------------------------------------------------------------------------
# Class coverage
# ---------------------------------------------------------------------------------------------------------

def army_group(choice: int) -> str:
    return "garrison" if choice == 0 else "raid" if choice == 6 else "defend"


def decision_classes(row: dict[str, Any]) -> list[str]:
    """Class tags of one decision (works on raw rows and on loaded examples via ``example_row``)."""
    action = row["teacher_action"]
    mask = row["mask"]
    tags = [f"lay_{('none', 'worker', 'soldier')[action['lay_order']]}", f"army_{army_group(action['army_choice'])}",
            f"forage_support_{sum(1 for v in action['forage_share_q'] if v)}"]
    army = action["army_choice"]
    if mask[16 + 2 * army] and mask[17 + 2 * army]:
        tags.append("avoid_true_conditional" if action["avoid_center"] else "avoid_false_conditional")
    return tags


def class_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Decisions, distinct maps, trajectories and unique decision hashes per class tag."""
    decisions: Counter = Counter()
    maps: dict[str, set[str]] = defaultdict(set)
    trajectories: dict[str, set[str]] = defaultdict(set)
    unique: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for tag in decision_classes(row):
            decisions[tag] += 1
            maps[tag].add(row["map_id"])
            trajectories[tag].add(row["trajectory_sha256"])
            unique[tag].add(row["decision_sha256"])
    return {tag: {"decisions": decisions[tag], "maps": len(maps[tag]), "trajectories": len(trajectories[tag]),
                  "unique_decisions": len(unique[tag])} for tag in sorted(decisions)}


def example_rows(examples: list["For40Example"]) -> list[dict[str, Any]]:
    return [{
        "split": e.split, "map_id": e.map_id, "source": e.source, "opponent_id": e.opponent_id,
        "trajectory_sha256": e.trajectory_sha256, "decision_sha256": e.decision_sha256,
        "mask": list(flatten_mask_v2(e.mask)), "disturbed": e.disturbed,
        "teacher_action": {"lay_order": e.teacher_action.lay_order,
                           "forage_share_q": list(e.teacher_action.forage_share_q),
                           "army_choice": e.teacher_action.army_choice,
                           "avoid_center": e.teacher_action.avoid_center},
    } for e in examples]


# ---------------------------------------------------------------------------------------------------------
# Dataset writer / loader
# ---------------------------------------------------------------------------------------------------------

def _map_manifest() -> list[dict[str, Any]]:
    result = []
    references = reference_vectors()
    for spec in map_plan():
        config = config_from_vector(spec.vector)
        train_vectors = [m.vector for m in maps_of_split("train") if m.map_id != spec.map_id]
        result.append({
            "map_id": spec.map_id, "generator_seed": spec.seed, "vector": list(spec.vector), "split": spec.split,
            "config_sha256": config_hash(config),
            "edges": [[e.a, e.b, e.length] for e in config.topology.edges],
            "min_distance_to_references": min(vector_distance(spec.vector, v) for v in references.values()),
            "min_distance_to_train_maps": (min(vector_distance(spec.vector, v) for v in train_vectors)
                                           if spec.split != "train" else None),
        })
    return result


def leakage_report(rows: list[dict[str, Any]], matches: list[dict[str, Any]]) -> dict[str, Any]:
    def groups(key: str, source: list[dict[str, Any]]) -> dict[str, set[str]]:
        out: dict[str, set[str]] = defaultdict(set)
        for item in source:
            out[str(item[key])].add(item["split"])
        return out
    active = [m for m in matches if not m.get("removed_by_disturbed_cap")]
    checks = {
        "maps_in_multiple_splits": sorted(k for k, v in groups("map_id", active).items() if len(v) > 1),
        "config_hashes_in_multiple_splits": sorted(k for k, v in groups("config_sha256", active).items() if len(v) > 1),
        "seed_contexts_in_multiple_splits": sorted(k for k, v in groups("seed_context", active).items() if len(v) > 1),
        "decisions_in_multiple_splits": sorted(k for k, v in groups("decision_sha256", rows).items() if len(v) > 1),
        "trajectories_in_multiple_splits": sorted(k for k, v in groups("trajectory_sha256", rows).items() if len(v) > 1),
    }
    return {**{k: len(v) for k, v in checks.items()}, "clean": all(not v for v in checks.values())}


def build_manifest(output: Path, jsonl: bytes, collected: dict[str, Any], plan: list[For40Match]) -> dict[str, Any]:
    rows, matches = collected["rows"], collected["matches"]
    teacher_spec = baseline_spec(QUALITY_TEACHER)
    git_sha, git_dirty = _git_provenance()
    split_counts = {s: sum(r["split"] == s for r in rows) for s in SPLITS}
    funnel = {}
    for split in SPLITS:
        split_matches = [m for m in matches if m["split"] == split]
        active = [m for m in split_matches if not m.get("removed_by_disturbed_cap")]
        funnel[split] = {
            "planned_matches": len(split_matches), "matches_removed_by_disturbed_cap": len(split_matches) - len(active),
            "source_trajectories": len({m["trajectory_sha256"] for m in active}),
            "retained_trajectories": len({m["trajectory_sha256"] for m in active if m["retained"]}),
            "source_decisions": sum(m["source_decision_count"] for m in active),
            "retained_decisions": split_counts[split],
            "unique_decisions": len({r["decision_sha256"] for r in rows if r["split"] == split}),
        }
    by_source = {}
    for split in SPLITS:
        for source in sorted({r["source"] for r in rows if r["split"] == split}):
            by_source[f"{split}/{source}"] = class_coverage([r for r in rows if r["split"] == split and r["source"] == source])
    return {
        "schema": FOR40_SCHEMA,
        "created_by": "formicmind.ml.for40.collect_for40_dataset",
        "protocol": {"path": FOR40_PROTOCOL, "commit": FOR40_PROTOCOL_COMMIT},
        "dataset_file": output.name, "dataset_sha256": hashlib.sha256(jsonl).hexdigest(),
        "example_count": len(rows), "match_count": len(matches),
        "contracts": {"observation": OBSERVATION_VERSION, "observation_length": OBS_V2_LENGTH,
                      "action": ACTION_VERSION, "decoder": DECODER_VERSION},
        "teacher": {"id": teacher_spec.full_id, "policy_id": teacher_spec.policy_id,
                    "parameters": teacher_spec.parameter_object(), "parameter_sha256": teacher_spec.digest()},
        "map_plan": {"sha256": map_plan_sha256(), "maps": _map_manifest()},
        "plan": {"sha256": plan_sha256(plan), "matches": len(plan),
                 "per_split": {s: sum(1 for p in plan if p.split == s) for s in SPLITS}},
        "max_ticks": MAX_TICKS,
        "matches": matches,
        "deduplication": {
            **collected["dedup"], "canonical_split_priority": list(SPLITS),
            "decision_fingerprint": "observation+mask+teacher_logits+teacher_action; excludes episode metadata",
            "trajectory_fingerprint": "ordered decision fingerprints; excludes seed, side, map, opponent, tick",
            "split_example_counts": split_counts,
            "split_unique_trajectory_counts": {s: len({r["trajectory_sha256"] for r in rows if r["split"] == s}) for s in SPLITS},
            "split_unique_decision_counts": {s: len({r["decision_sha256"] for r in rows if r["split"] == s}) for s in SPLITS},
        },
        "disturbed_cap": collected["disturbed_cap"],
        "funnel": funnel,
        "class_coverage": {s: class_coverage([r for r in rows if r["split"] == s]) for s in SPLITS},
        "class_coverage_by_source": by_source,
        "leakage": leakage_report(rows, matches),
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "collection_wall_seconds": collected["wall_seconds"]},
        "source": {"git_sha": git_sha, "git_dirty": git_dirty},
    }


def collect_for40_dataset(output: str | Path, *, plan: list[For40Match] | None = None) -> dict[str, Any]:
    output = Path(output)
    plan = plan_for40() if plan is None else plan
    collected = collect_for40_rows(plan)
    rows = collected["rows"]
    if not rows:
        raise DatasetError("FOR-40 collection produced no retained decisions")
    jsonl = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    manifest = build_manifest(output, jsonl, collected, plan)
    _write_atomic(output, jsonl)
    _write_atomic(output.with_name(f"{output.stem}.manifest.json"),
                  (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    return manifest


@dataclass(frozen=True, slots=True)
class For40Example:
    match_id: str
    seed: int
    tick: int
    split: str
    observation: tuple[float, ...]
    mask: Any
    teacher_logits: tuple[float, ...]
    teacher_action: ActionV2
    context_id: str
    map_id: str
    opponent_id: str
    source: str
    behavior: str
    disturbed: bool
    trajectory_sha256: str
    decision_sha256: str


_ROW_FIELDS = {
    "schema", "match_id", "seed", "teacher_side", "tick", "split", "context_id", "map_id", "opponent_id",
    "observation", "mask", "teacher_logits", "teacher_action", "decision_sha256", "trajectory_group_id",
    "trajectory_sha256", "source", "behavior", "disturbed", "seed_context",
}


def load_for40_dataset(path: str | Path, *, expect_full_protocol: bool = True) -> tuple[dict[str, Any], list[For40Example]]:
    """Verify hashes, protocol maps, contracts, fingerprints, masks, labels and split isolation."""
    path = Path(path)
    try:
        manifest = json.loads(path.with_name(f"{path.stem}.manifest.json").read_text(encoding="utf-8"))
        raw = path.read_bytes()
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"cannot read FOR-40 dataset and manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != FOR40_SCHEMA:
        raise DatasetError("unsupported FOR-40 manifest schema")
    if manifest.get("dataset_file") != path.name or hashlib.sha256(raw).hexdigest() != manifest.get("dataset_sha256"):
        raise DatasetError("FOR-40 dataset file or SHA-256 does not match its manifest")
    if manifest.get("contracts") != {"observation": OBSERVATION_VERSION, "observation_length": OBS_V2_LENGTH,
                                     "action": ACTION_VERSION, "decoder": DECODER_VERSION}:
        raise DatasetError("FOR-40 dataset contract versions do not match this simulator")
    teacher_spec = baseline_spec(QUALITY_TEACHER)
    teacher = manifest.get("teacher")
    if (not isinstance(teacher, dict) or teacher.get("id") != teacher_spec.full_id
            or teacher.get("parameters") != teacher_spec.parameter_object()
            or teacher.get("parameter_sha256") != teacher_spec.digest()):
        raise DatasetError("FOR-40 teacher descriptor does not match balanced:v2")

    plan_maps = {m.map_id: m for m in map_plan()}
    declared = manifest.get("map_plan", {})
    if map_plan_sha256() != MAP_PLAN_SHA256 or declared.get("sha256") != MAP_PLAN_SHA256:
        raise DatasetError("FOR-40 map plan does not match the preregistered checksum")
    declared_maps = {m["map_id"]: m for m in declared.get("maps", [])}
    if set(declared_maps) != set(plan_maps):
        raise DatasetError("FOR-40 manifest maps differ from the preregistered plan")
    for map_id, info in declared_maps.items():
        spec = plan_maps[map_id]
        if (info.get("split") != spec.split or info.get("vector") != list(spec.vector)
                or info.get("config_sha256") != config_hash(config_from_vector(spec.vector))):
            raise DatasetError(f"FOR-40 map {map_id} differs from the preregistered plan")
    matches = manifest.get("matches")
    if not isinstance(matches, list):
        raise DatasetError("FOR-40 manifest has no match records")
    index: dict[str, dict[str, Any]] = {}
    for match in matches:
        match_id = match.get("match_id") if isinstance(match, dict) else None
        if not isinstance(match_id, str) or match_id in index:
            raise DatasetError("FOR-40 manifest has malformed or duplicate match records")
        spec = plan_maps.get(match.get("map_id"))
        if spec is None or match.get("split") != spec.split:
            raise DatasetError("FOR-40 match split differs from its map's preregistered split")
        if match.get("config_sha256") != declared_maps[spec.map_id]["config_sha256"]:
            raise DatasetError("FOR-40 match config hash differs from its map")
        if match.get("removed_by_disturbed_cap"):
            index[match_id] = match
            continue
        sequence = match.get("trajectory_decision_sha256s")
        if (not isinstance(sequence, list) or not sequence or len(sequence) != match.get("source_decision_count")
                or any(not isinstance(v, str) or len(v) != 64 for v in sequence)
                or _sha256(sequence) != match.get("trajectory_sha256")
                or match.get("trajectory_group_id") != f"trajectory:{match.get('trajectory_sha256')}"):
            raise DatasetError("FOR-40 match trajectory fingerprint is invalid")
        index[match_id] = match
    if len(index) != manifest.get("match_count"):
        raise DatasetError("FOR-40 manifest match_count does not match its records")
    if expect_full_protocol:
        if manifest["plan"]["sha256"] != plan_sha256(plan_for40()) or len(index) != sum(EXPECTED_MATCHES.values()):
            raise DatasetError("FOR-40 match plan differs from the preregistered plan")

    examples: list[For40Example] = []
    seen_decision: dict[str, str] = {}
    seen_trajectory: dict[str, str] = {}
    seen_tick: set[tuple[str, int]] = set()
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        try:
            item = json.loads(line)
            if not isinstance(item, dict) or set(item) != _ROW_FIELDS or item["schema"] != FOR40_SCHEMA:
                raise DatasetError(f"line {line_number}: FOR-40 row fields have invalid shape")
            match = index.get(item["match_id"])
            if match is None or not match.get("retained"):
                raise DatasetError(f"line {line_number}: row has no retained source match")
            for key in ("split", "seed", "context_id", "map_id", "opponent_id", "source", "behavior", "disturbed",
                        "seed_context"):
                if item[key] != match[key]:
                    raise DatasetError(f"line {line_number}: row {key} differs from its match")
            if (item["teacher_side"] != 0 or type(item["tick"]) is not int or item["tick"] < 0
                    or item["trajectory_sha256"] != match["trajectory_sha256"]
                    or item["trajectory_group_id"] != match["trajectory_group_id"]):
                raise DatasetError(f"line {line_number}: row provenance differs from its match")
            observation = ObservationV2(tuple(float(v) for v in item["observation"]))
            flat = tuple(item["mask"])
            if len(flat) != 30 or any(type(v) is not bool for v in flat):
                raise DatasetError(f"line {line_number}: invalid mask")
            mask = mask_from_flat(flat)
            if mask != get_action_mask(observation):
                raise DatasetError(f"line {line_number}: stored mask differs from ObservationV2")
            logits = PolicyLogitsV2(item["teacher_logits"])
            action = _parse_action(item["teacher_action"])
            if action != validate_action(observation, mask, decode_action_logits(logits, mask)).action:
                raise DatasetError(f"line {line_number}: teacher label differs from the shared decoder")
            decision_sha = _sha256(_decision_payload(item))
            if item["decision_sha256"] != decision_sha or decision_sha not in match["trajectory_decision_sha256s"]:
                raise DatasetError(f"line {line_number}: decision fingerprint is invalid")
            if decision_sha in seen_decision:
                if seen_decision[decision_sha] != item["split"]:
                    raise DatasetError("identical decision fingerprint crosses dataset splits")
                raise DatasetError(f"line {line_number}: duplicate decision was not removed")
            seen_decision[decision_sha] = item["split"]
            if seen_trajectory.setdefault(item["trajectory_sha256"], item["split"]) != item["split"]:
                raise DatasetError("identical trajectory fingerprint crosses dataset splits")
            if (item["match_id"], item["tick"]) in seen_tick:
                raise DatasetError(f"line {line_number}: duplicate decision tick")
            seen_tick.add((item["match_id"], item["tick"]))
            examples.append(For40Example(
                item["match_id"], item["seed"], item["tick"], item["split"], observation.values, mask, logits.values,
                action, item["context_id"], item["map_id"], item["opponent_id"], item["source"], item["behavior"],
                item["disturbed"], item["trajectory_sha256"], decision_sha))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            if isinstance(exc, DatasetError):
                raise
            raise DatasetError(f"line {line_number}: malformed FOR-40 row ({exc})") from exc
    if len(examples) != manifest.get("example_count") or not examples:
        raise DatasetError("FOR-40 manifest example_count does not match the dataset")
    for match in matches:
        if match.get("removed_by_disturbed_cap"):
            continue
        if match.get("retained_decision_count") != sum(1 for e in examples if e.match_id == match["match_id"]):
            raise DatasetError("FOR-40 retained decision counts do not match dataset rows")
    return manifest, examples


# ---------------------------------------------------------------------------------------------------------
# Data gate
# ---------------------------------------------------------------------------------------------------------

def evaluate_gate(manifest: dict[str, Any], examples: list[For40Example], *, full_protocol: bool = True) -> dict[str, Any]:
    rows = example_rows(examples)
    maps_info = manifest["map_plan"]["maps"]
    plan_maps = {m.map_id: m for m in map_plan()}
    train_lengths = Counter(length for m in maps_info if m["split"] == "train" for _, _, length in m["edges"])
    pair_distances = [vector_distance(plan_maps[a["map_id"]].vector, plan_maps[b["map_id"]].vector)
                      for i, a in enumerate(maps_info) for b in maps_info[i + 1:]]
    d1 = {
        "map_counts": {s: sum(1 for m in maps_info if m["split"] == s) for s in SPLITS},
        "plan_sha256_matches": manifest["map_plan"]["sha256"] == MAP_PLAN_SHA256 == map_plan_sha256(),
        "train_edge_length_usage": dict(sorted(train_lengths.items())),
        "min_pairwise_distance": min(pair_distances),
        "min_distance_to_references": min(m["min_distance_to_references"] for m in maps_info),
    }
    d1["pass"] = (d1["map_counts"] == MAP_COUNTS and d1["plan_sha256_matches"]
                  and all(train_lengths.get(length, 0) >= 4 for length in (1, 2, 3, 4))
                  and d1["min_pairwise_distance"] >= MIN_MAP_DISTANCE
                  and d1["min_distance_to_references"] >= MIN_MAP_DISTANCE)
    matches = manifest["matches"]
    cap = manifest["disturbed_cap"]
    d2 = {
        "planned_matches": manifest["plan"]["matches"], "recorded_matches": len(matches),
        "per_split": manifest["plan"]["per_split"],
        "wall_seconds": manifest["runtime"]["collection_wall_seconds"],
        "disturbed_share": cap["share_after"], "disturbed_matches_removed": len(cap["matches_removed"]),
        "max_matches": MAX_MATCHES, "max_wall_seconds": MAX_WALL_SECONDS,
    }
    d2["pass"] = ((not full_protocol or (manifest["plan"]["per_split"] == EXPECTED_MATCHES
                                         and d2["recorded_matches"] == sum(EXPECTED_MATCHES.values())))
                  and d2["recorded_matches"] <= MAX_MATCHES and d2["wall_seconds"] <= MAX_WALL_SECONDS
                  and all(v <= DISTURBED_CAP for v in cap["share_after"].values()))
    d3 = leakage_report(rows, matches)  # recomputed from the loaded rows, not trusted from the manifest
    d3["pass"] = bool(d3["clean"])
    dedup = manifest["deduplication"]
    d4 = {}
    for split in SPLITS:
        need_decisions, need_trajectories = GATE_MIN_DECISIONS[split]
        d4[split] = {"decisions": dedup["split_example_counts"][split], "trajectories": dedup["split_unique_trajectory_counts"][split],
                     "required_decisions": need_decisions, "required_trajectories": need_trajectories,
                     "pass": dedup["split_example_counts"][split] >= need_decisions
                     and dedup["split_unique_trajectory_counts"][split] >= need_trajectories}
    d4["pass"] = all(d4[s]["pass"] for s in SPLITS)
    coverage = {s: class_coverage([r for r in rows if r["split"] == s]) for s in SPLITS}
    classes: dict[str, Any] = {}
    for split in SPLITS:
        classes[split] = {}
        for tag, (need_decisions, need_maps) in GATE_THRESHOLDS[split].items():
            got = coverage[split].get(tag, {"decisions": 0, "maps": 0, "trajectories": 0, "unique_decisions": 0})
            classes[split][tag] = {**got, "required_decisions": need_decisions, "required_maps": need_maps,
                                   "pass": got["decisions"] >= need_decisions and (need_maps is None or got["maps"] >= need_maps)}
    d5 = {"classes": classes["train"], "pass": all(v["pass"] for v in classes["train"].values())}
    d6 = {"classes": classes["validation"], "pass": all(v["pass"] for v in classes["validation"].values())}
    held = [r for r in rows if r["split"] == "heldout"]
    per_map = Counter(r["map_id"] for r in held)
    per_opponent = Counter(("random:v2" if r["source"] == "H_random" else r["opponent_id"]) for r in held)
    heldout_maps = [m["map_id"] for m in maps_info if m["split"] == "heldout"]
    d7 = {
        "classes": classes["heldout"],
        "decisions_per_map": {m: per_map.get(m, 0) for m in heldout_maps},
        "decisions_per_opponent": {o: per_opponent.get(o, 0) for o in HELDOUT_OPPONENT_KEYS},
    }
    d7["pass"] = (all(v["pass"] for v in classes["heldout"].values())
                  and all(n >= GATE_MIN_MAP_DECISIONS_HELDOUT for n in d7["decisions_per_map"].values())
                  and all(n >= GATE_MIN_OPPONENT_DECISIONS_HELDOUT for n in d7["decisions_per_opponent"].values()))
    criteria = {"D1": d1, "D2": d2, "D3": d3, "D4": d4, "D5": d5, "D6": d6, "D7": d7}
    passed = all(c["pass"] for c in criteria.values())
    return {
        "schema": "formicmind.ml-for40-data-gate/v1", "protocol": FOR40_PROTOCOL,
        "dataset_sha256": manifest["dataset_sha256"],
        "decision": "DATA GO" if passed else "DATA NO-GO",
        "failed": [name for name, c in criteria.items() if not c["pass"]],
        "criteria": criteria, "funnel": manifest["funnel"],
        "class_coverage": coverage, "class_coverage_by_source": manifest["class_coverage_by_source"],
        "explanation": ("The preregistered data gate passed; training may start." if passed else
                        "The preregistered data gate failed. No model is trained or evaluated on held-out data."),
    }


def gate_for40(dataset_path: str | Path, output: str | Path | None = None) -> dict[str, Any]:
    manifest, examples = load_for40_dataset(dataset_path)
    report = evaluate_gate(manifest, examples)
    if output is not None:
        _write_atomic(Path(output), (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    return report


__all__ = [
    "FOR40_SCHEMA", "For40Example", "For40Match", "MAP_PLAN_SHA256", "REAL_MATCH_SEEDS", "SPLITS", "TRAIN_SEEDS",
    "UNTRAINED_SEED", "army_group", "class_coverage", "collect_for40_dataset", "collect_for40_rows", "config_from_vector",
    "decision_classes", "evaluate_gate", "for40_config", "gate_for40", "load_for40_dataset", "map_plan",
    "map_plan_sha256", "maps_of_split", "plan_for40", "run_for40_match",
]
