"""Versioned, leakage-resistant multi-context dataset for FOR-37.

The FOR-36 v1 reader/writer stays untouched. This module records only policy
boundary observations and teacher outputs from the authoritative run_episode.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..actions import ActionMaskV2, ActionV2, flatten_mask_v2, get_action_mask, mask_from_flat, validate_action
from ..baselines import BaselinePolicy, baseline_spec
from ..config import EdgeSpec, SimConfig, Topology
from ..decoder import PolicyLogitsV2, decode_action_logits
from ..match import EpisodeResult, run_episode
from ..invariants import validate_config
from ..observation import OBS_V2_LENGTH, ObservationV2
from ..rng import policy_stream_id
from ..rules import default_config
from ..serialization import config_hash
from ..tick import PolicyController
from ..versions import ACTION_VERSION, DECODER_VERSION, OBSERVATION_VERSION
from .dataset import DatasetError, _write_atomic

QUALITY_DATASET_SCHEMA = "formicmind.behavior-cloning/v2"
QUALITY_CONTEXTS = {
    "train": (("classic", "turtle:v2"), ("classic", "random:v2"),
              ("cross", "turtle:v2"), ("cross", "random:v2")),
    "validation": (("classic", "turtle:v2"), ("classic", "random:v2"),
                   ("cross", "turtle:v2"), ("cross", "random:v2")),
    "heldout": (("dense", "turtle:v2"), ("dense", "random:v2"), ("dense", "rush:v2"),
                ("classic", "rush:v2"), ("cross", "rush:v2")),
}
QUALITY_TEACHER = "balanced:v2"
QUALITY_SEED_RANGES = {
    "train": (370_000_001, 370_000_024),
    "validation": (371_000_001, 371_000_008),
    "heldout": (372_000_001, 372_000_008),
}
QUALITY_MATCH_SEEDS = tuple(range(373_000_001, 373_000_005))
QUALITY_MATCH_OPPONENTS = ("balanced:v2", "turtle:v2", "rush:v2")
QUALITY_TRAIN_SEED = 370_900_037
QUALITY_UNTRAINED_SEED = 370_900_038
QUALITY_DEDUP_PRIORITY = {"train": 0, "validation": 1, "heldout": 2}


@dataclass(frozen=True, slots=True)
class QualityExample:
    match_id: str
    seed: int
    teacher_side: int
    tick: int
    split: str
    observation: tuple[float, ...]
    mask: ActionMaskV2
    teacher_logits: tuple[float, ...]
    teacher_action: ActionV2
    context_id: str
    map_id: str
    opponent_id: str
    trajectory_group_id: str
    trajectory_sha256: str
    decision_sha256: str


def quality_config(map_id: str) -> SimConfig:
    """Return a mirror-valid topology variant without changing balance or rules."""
    config = default_config()
    edges = {(edge.a, edge.b): edge.length for edge in config.topology.edges}
    if map_id == "classic":
        return config
    if map_id == "cross":
        edges.pop((2, 5))
        edges[(1, 5)] = 4
        edges[(2, 4)] = 4
    elif map_id == "dense":
        edges[(1, 5)] = 4
        edges[(2, 4)] = 4
    else:
        raise DatasetError(f"unknown FOR-37 map_id: {map_id}")
    topology = Topology(
        nodes=config.topology.nodes,
        edges=tuple(EdgeSpec(a, b, length) for (a, b), length in sorted(edges.items())),
    )
    variant = SimConfig(
        rules_version=config.rules_version,
        balance_version=config.balance_version,
        topology=topology,
        balance=config.balance,
        policy_interval_k=config.policy_interval_k,
        t_max=config.t_max,
    )
    validate_config(variant)
    return variant


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _action_record(action: ActionV2) -> dict[str, Any]:
    return {
        "lay_order": action.lay_order,
        "forage_share_q": list(action.forage_share_q),
        "army_choice": action.army_choice,
        "avoid_center": action.avoid_center,
    }


def _parse_action(value: object) -> ActionV2:
    if not isinstance(value, dict) or set(value) != {
        "lay_order", "forage_share_q", "army_choice", "avoid_center",
    }:
        raise DatasetError("v2 teacher action has an invalid shape")
    try:
        return ActionV2(
            lay_order=value["lay_order"], forage_share_q=tuple(value["forage_share_q"]),
            army_choice=value["army_choice"], avoid_center=value["avoid_center"],
        )
    except (TypeError, ValueError) as exc:
        raise DatasetError("v2 teacher action has invalid values") from exc


def _decision_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in ("observation", "mask", "teacher_logits", "teacher_action")}


def _git_provenance() -> tuple[str | None, bool | None]:
    repository = Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=repository, check=True, capture_output=True, text=True,
        ).stdout.strip())
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


class _RecordingTeacher:
    def __init__(
        self, policy: BaselinePolicy, stream_id: str, rows: list[dict[str, Any]], *,
        seed: int, teacher_side: int, match_id: str, split: str, context_id: str,
        map_id: str, opponent_id: str,
    ) -> None:
        self.policy = policy
        self.stream_id = stream_id
        self.rows = rows
        self.seed = seed
        self.teacher_side = teacher_side
        self.match_id = match_id
        self.split = split
        self.context_id = context_id
        self.map_id = map_id
        self.opponent_id = opponent_id

    def act(self, tick: int, observation: ObservationV2, run: Any):
        mask = get_action_mask(observation)
        logits = PolicyLogitsV2.coerce(self.policy(observation, mask, run.stream(self.stream_id)))
        action = validate_action(observation, mask, decode_action_logits(logits, mask)).action
        row = {
            "schema": QUALITY_DATASET_SCHEMA,
            "match_id": self.match_id,
            "seed": self.seed,
            "teacher_side": self.teacher_side,
            "tick": tick,
            "split": self.split,
            "context_id": self.context_id,
            "map_id": self.map_id,
            "opponent_id": self.opponent_id,
            "observation": list(observation.values),
            "mask": list(flatten_mask_v2(mask)),
            "teacher_logits": list(logits.values),
            "teacher_action": _action_record(action),
        }
        row["decision_sha256"] = _sha256(_decision_payload(row))
        self.rows.append(row)
        return validate_action(observation, mask, action)


def _validate_seed_sets(seed_sets: dict[str, tuple[int, ...] | list[int]]) -> dict[str, tuple[int, ...]]:
    if set(seed_sets) != {"train", "validation", "heldout"}:
        raise DatasetError("quality collection requires train, validation, and heldout seed sets")
    normalized: dict[str, tuple[int, ...]] = {}
    seen: set[int] = set()
    for split in ("train", "validation", "heldout"):
        seeds = tuple(seed_sets[split])
        if (not seeds or any(type(seed) is not int or not 0 <= seed <= (1 << 64) - 1 for seed in seeds)
                or len(set(seeds)) != len(seeds)):
            raise DatasetError(f"{split} seeds must be a non-empty list of unique unsigned 64-bit integers")
        overlap = seen.intersection(seeds)
        if overlap:
            raise DatasetError(f"seed sets overlap before collection: {sorted(overlap)}")
        seen.update(seeds)
        normalized[split] = seeds
    return normalized


def _deduplicate_rows(
    rows_by_match: dict[str, list[dict[str, Any]]], matches: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    match_index = {match["match_id"]: match for match in matches}
    for match_id, rows in rows_by_match.items():
        ordered = sorted(rows, key=lambda row: row["tick"])
        if not ordered:
            raise DatasetError(f"match {match_id} produced no teacher decisions")
        decision_sequence = [row["decision_sha256"] for row in ordered]
        trajectory_sha = _sha256(decision_sequence)
        match = match_index[match_id]
        match["trajectory_sha256"] = trajectory_sha
        match["trajectory_group_id"] = f"trajectory:{trajectory_sha}"
        match["trajectory_decision_sha256s"] = decision_sequence
        match["source_decision_count"] = len(ordered)
        for row in ordered:
            row["trajectory_sha256"] = trajectory_sha
            row["trajectory_group_id"] = f"trajectory:{trajectory_sha}"

    matches_ordered = sorted(
        matches,
        key=lambda match: (QUALITY_DEDUP_PRIORITY[match["split"]], match["context_id"],
                           match["seed"], match["teacher_side"], match["match_id"]),
    )
    canonical_trajectory: dict[str, str] = {}
    duplicate_trajectory_count = 0
    for match in matches_ordered:
        fingerprint = match["trajectory_sha256"]
        previous = canonical_trajectory.get(fingerprint)
        if previous is None:
            canonical_trajectory[fingerprint] = match["match_id"]
            match["trajectory_retained"] = True
            match["duplicate_of_match_id"] = None
        else:
            match["trajectory_retained"] = False
            match["duplicate_of_match_id"] = previous
            duplicate_trajectory_count += 1

    all_rows = [row for values in rows_by_match.values() for row in values]
    trajectory_keep = {match_id for match_id, match in match_index.items() if match["trajectory_retained"]}
    candidates = [row for row in all_rows if row["match_id"] in trajectory_keep]
    candidates.sort(key=lambda row: (QUALITY_DEDUP_PRIORITY[row["split"]], row["context_id"],
                                     row["seed"], row["teacher_side"], row["tick"], row["match_id"]))
    canonical_decision: dict[str, str] = {}
    retained: list[dict[str, Any]] = []
    duplicate_decision_count = 0
    for row in candidates:
        previous = canonical_decision.get(row["decision_sha256"])
        if previous is None:
            canonical_decision[row["decision_sha256"]] = row["match_id"]
            retained.append(row)
        else:
            duplicate_decision_count += 1
    retained_match_counts: dict[str, int] = {}
    for row in retained:
        retained_match_counts[row["match_id"]] = retained_match_counts.get(row["match_id"], 0) + 1
    for match in matches:
        match["retained_decision_count"] = retained_match_counts.get(match["match_id"], 0)
        match["retained"] = match["trajectory_retained"] and match["retained_decision_count"] > 0
    retained.sort(key=lambda row: (QUALITY_DEDUP_PRIORITY[row["split"]], row["context_id"],
                                   row["seed"], row["teacher_side"], row["tick"], row["match_id"]))
    return retained, {
        "duplicate_trajectories_removed": duplicate_trajectory_count,
        "duplicate_decisions_removed": duplicate_decision_count,
    }


def collect_quality_dataset(
    output: str | Path,
    *,
    train_seeds: tuple[int, ...] | list[int],
    validation_seeds: tuple[int, ...] | list[int],
    heldout_seeds: tuple[int, ...] | list[int],
    teacher: str = QUALITY_TEACHER,
    max_ticks: int = 3000,
) -> dict[str, Any]:
    """Collect the pre-registered FOR-37 context matrix and global deduplicated splits."""
    if teacher != QUALITY_TEACHER:
        raise DatasetError(f"FOR-37 protocol fixes teacher={QUALITY_TEACHER}")
    if type(max_ticks) is not int or not 1 <= max_ticks <= default_config().t_max:
        raise DatasetError(f"max_ticks must be within 1..T_max ({default_config().t_max})")
    seed_sets = _validate_seed_sets({
        "train": train_seeds,
        "validation": validation_seeds,
        "heldout": heldout_seeds,
    })
    teacher_spec = baseline_spec(teacher)
    if teacher_spec.version != "v2" or teacher_spec.name != "balanced":
        raise DatasetError("FOR-37 requires the fixed balanced:v2 teacher")

    context_specs: dict[str, dict[str, Any]] = {}
    rows_by_match: dict[str, list[dict[str, Any]]] = {}
    matches: list[dict[str, Any]] = []
    for split in ("train", "validation", "heldout"):
        for map_id, opponent_id in QUALITY_CONTEXTS[split]:
            config = quality_config(map_id)
            opponent_spec = baseline_spec(opponent_id)
            if opponent_spec.version != "v2":
                raise DatasetError("all FOR-37 opponents must be explicitly versioned Baselines v2")
            context_id = f"{map_id}__{opponent_spec.full_id}"
            context_specs[context_id] = {
                "context_id": context_id,
                "map_id": map_id,
                "opponent": {
                    "id": opponent_spec.full_id,
                    "policy_id": opponent_spec.policy_id,
                    "parameters": opponent_spec.parameter_object(),
                    "parameter_sha256": opponent_spec.digest(),
                },
                "teacher": {
                    "id": teacher_spec.full_id,
                    "policy_id": teacher_spec.policy_id,
                    "parameters": teacher_spec.parameter_object(),
                    "parameter_sha256": teacher_spec.digest(),
                },
                "config": {
                    "sha256": config_hash(config),
                    "rules_version": config.rules_version,
                    "balance_version": config.balance_version,
                    "topology": {
                        "nodes": [{"node_id": node.node_id, "name": node.name, "home_side": node.home_side}
                                  for node in config.topology.nodes],
                        "edges": [{"a": edge.a, "b": edge.b, "length": edge.length}
                                  for edge in config.topology.edges],
                    },
                },
            }
            for seed in seed_sets[split]:
                for teacher_side in (0, 1):
                    match_id = f"{split}/{context_id}/seed-{seed}/teacher-seat-{teacher_side}"
                    local_rows: list[dict[str, Any]] = []
                    teacher_stream = policy_stream_id(teacher_spec.policy_id, 0)
                    opponent_stream = policy_stream_id(opponent_spec.policy_id, 1)
                    recording = _RecordingTeacher(
                        teacher_spec.build(), teacher_stream, local_rows,
                        seed=seed, teacher_side=teacher_side, match_id=match_id, split=split,
                        context_id=context_id, map_id=map_id, opponent_id=opponent_spec.full_id,
                    )
                    opponent = PolicyController(opponent_spec.build(), opponent_stream)
                    controllers = (recording, opponent) if teacher_side == 0 else (opponent, recording)
                    result: EpisodeResult = run_episode(config, seed, controllers[0], controllers[1], max_ticks=max_ticks)
                    rows_by_match[match_id] = local_rows
                    matches.append({
                        "match_id": match_id,
                        "split": split,
                        "seed": seed,
                        "teacher_side": teacher_side,
                        "context_id": context_id,
                        "map_id": map_id,
                        "opponent_id": opponent_spec.full_id,
                        "ticks_executed": result.ticks_executed,
                        "final_tick": result.final_tick,
                        "truncated": result.truncated,
                        "winner": result.winner,
                        "reason": result.reason,
                        "score": None if result.score is None else list(result.score),
                        "config_sha256": result.config_hash,
                    })

    rows, dedup_summary = _deduplicate_rows(rows_by_match, matches)
    if not rows:
        raise DatasetError("quality collection produced no independent teacher decisions")
    split_counts = {split: sum(row["split"] == split for row in rows)
                    for split in ("train", "validation", "heldout")}
    if any(count == 0 for count in split_counts.values()):
        raise DatasetError(f"global trajectory/decision deduplication emptied a split: {split_counts}")
    jsonl = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    dataset_sha = hashlib.sha256(jsonl).hexdigest()
    git_sha, git_dirty = _git_provenance()
    split_trajectory_counts = {
        split: len({row["trajectory_group_id"] for row in rows if row["split"] == split})
        for split in ("train", "validation", "heldout")
    }
    split_decision_counts = {
        split: len({row["decision_sha256"] for row in rows if row["split"] == split})
        for split in ("train", "validation", "heldout")
    }
    split_context_counts = {
        split: {
            context_id: sum(row["split"] == split and row["context_id"] == context_id for row in rows)
            for context_id in sorted({row["context_id"] for row in rows if row["split"] == split})
        }
        for split in ("train", "validation", "heldout")
    }
    split_by_seed = {str(seed): split for split, seeds in seed_sets.items() for seed in seeds}
    manifest: dict[str, Any] = {
        "schema": QUALITY_DATASET_SCHEMA,
        "created_by": "formicmind.ml.quality_dataset.collect_quality_dataset",
        "dataset_file": Path(output).name,
        "dataset_sha256": dataset_sha,
        "example_count": len(rows),
        "match_count": len(matches),
        "contracts": {
            "observation": OBSERVATION_VERSION,
            "observation_length": OBS_V2_LENGTH,
            "action": ACTION_VERSION,
            "decoder": DECODER_VERSION,
        },
        "teacher": {
            "id": teacher_spec.full_id,
            "policy_id": teacher_spec.policy_id,
            "parameters": teacher_spec.parameter_object(),
            "parameter_sha256": teacher_spec.digest(),
        },
        "seed_sets": {split: list(seeds) for split, seeds in seed_sets.items()},
        "split_by_seed": split_by_seed,
        "context_matrix": context_specs,
        "max_ticks": max_ticks,
        "matches": matches,
        "deduplication": {
            **dedup_summary,
            "decision_fingerprint": "observation+mask+teacher_logits+teacher_action; excludes episode metadata",
            "trajectory_fingerprint": "ordered decision fingerprints; excludes seed, side, map, opponent, and tick",
            "canonical_split_priority": ["train", "validation", "heldout"],
            "split_example_counts": split_counts,
            "split_unique_trajectory_counts": split_trajectory_counts,
            "split_unique_decision_counts": split_decision_counts,
            "split_context_counts": split_context_counts,
        },
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "source": {"git_sha": git_sha, "git_dirty": git_dirty},
    }
    output = Path(output)
    manifest_path = output.with_name(f"{output.stem}.manifest.json")
    _write_atomic(output, jsonl)
    _write_atomic(manifest_path, (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    return manifest


def load_quality_dataset(path: str | Path) -> tuple[dict[str, Any], list[QualityExample]]:
    """Validate v2 provenance, contracts, fingerprints, masks and global split isolation."""
    path = Path(path)
    manifest_path = path.with_name(f"{path.stem}.manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw = path.read_bytes()
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"cannot read quality dataset and manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != QUALITY_DATASET_SCHEMA:
        raise DatasetError("unsupported quality dataset manifest schema")
    if manifest.get("dataset_file") != path.name:
        raise DatasetError("quality manifest dataset_file does not match selected dataset")
    if hashlib.sha256(raw).hexdigest() != manifest.get("dataset_sha256"):
        raise DatasetError("quality dataset SHA-256 does not match its manifest")
    expected_contracts = {
        "observation": OBSERVATION_VERSION,
        "observation_length": OBS_V2_LENGTH,
        "action": ACTION_VERSION,
        "decoder": DECODER_VERSION,
    }
    if manifest.get("contracts") != expected_contracts:
        raise DatasetError("quality dataset contract versions do not match this simulator")
    teacher = manifest.get("teacher")
    contexts = manifest.get("context_matrix")
    seed_sets = manifest.get("seed_sets")
    split_by_seed = manifest.get("split_by_seed")
    if not isinstance(teacher, dict) or not isinstance(contexts, dict):
        raise DatasetError("quality manifest is missing teacher or context matrix")
    if not isinstance(seed_sets, dict) or set(seed_sets) != {"train", "validation", "heldout"}:
        raise DatasetError("quality manifest is missing explicit train/validation/heldout seed sets")
    normalized_sets = _validate_seed_sets(seed_sets)
    expected_split_by_seed = {str(seed): split for split, seeds in normalized_sets.items() for seed in seeds}
    if split_by_seed != expected_split_by_seed:
        raise DatasetError("quality manifest seed split map does not match the explicit seed sets")
    try:
        teacher_spec = baseline_spec(teacher["id"])
    except (KeyError, ValueError) as exc:
        raise DatasetError("quality teacher baseline is unavailable") from exc
    if (teacher_spec.full_id != QUALITY_TEACHER or teacher.get("policy_id") != teacher_spec.policy_id
            or teacher.get("parameters") != teacher_spec.parameter_object()
            or teacher.get("parameter_sha256") != teacher_spec.digest()):
        raise DatasetError("quality teacher descriptor does not match balanced:v2")
    for split, declared_contexts in QUALITY_CONTEXTS.items():
        expected_context_ids = {f"{map_id}__{opponent}" for map_id, opponent in declared_contexts}
        actual_context_ids = {
            context_id for context_id in contexts
            if any(match.get("split") == split and match.get("context_id") == context_id
                   for match in manifest.get("matches", []) if isinstance(match, dict))
        }
        if actual_context_ids != expected_context_ids:
            raise DatasetError(f"quality {split} contexts do not match the committed protocol")

    match_index: dict[str, dict[str, Any]] = {}
    matches = manifest.get("matches")
    if not isinstance(matches, list):
        raise DatasetError("quality manifest is missing match records")
    for match in matches:
        if not isinstance(match, dict) or not isinstance(match.get("match_id"), str):
            raise DatasetError("quality manifest contains malformed match metadata")
        match_id = match["match_id"]
        if match_id in match_index:
            raise DatasetError("quality manifest has duplicate match IDs")
        context = contexts.get(match.get("context_id"))
        if (context is None or match.get("map_id") != context.get("map_id")
                or match.get("opponent_id") != context.get("opponent", {}).get("id")
                or match.get("split") not in normalized_sets
                or match.get("seed") not in normalized_sets[match["split"]]
                or match.get("split") != split_by_seed.get(str(match.get("seed")))):
            raise DatasetError("quality match metadata does not match its seed split/context")
        try:
            opponent_spec = baseline_spec(match["opponent_id"])
        except (KeyError, ValueError) as exc:
            raise DatasetError("quality opponent baseline is unavailable") from exc
        if (opponent_spec.version != "v2" or context["opponent"].get("policy_id") != opponent_spec.policy_id
                or context["opponent"].get("parameters") != opponent_spec.parameter_object()
                or context["opponent"].get("parameter_sha256") != opponent_spec.digest()):
            raise DatasetError("quality opponent descriptor does not match Baselines v2")
        if context.get("teacher") != teacher:
            raise DatasetError("quality context teacher descriptor differs from manifest teacher")
        config = quality_config(match["map_id"])
        if (context.get("config", {}).get("sha256") != config_hash(config)
                or match.get("config_sha256") != config_hash(config)):
            raise DatasetError("quality context config hash does not match the declared map")
        sequence = match.get("trajectory_decision_sha256s")
        if (not isinstance(sequence, list) or not sequence or len(sequence) != match.get("source_decision_count")
                or any(not isinstance(value, str) or len(value) != 64 for value in sequence)
                or _sha256(sequence) != match.get("trajectory_sha256")
                or match.get("trajectory_group_id") != f"trajectory:{match.get('trajectory_sha256')}"):
            raise DatasetError("quality match trajectory fingerprint is invalid")
        match_index[match_id] = match
    if len(match_index) != manifest.get("match_count"):
        raise DatasetError("quality manifest match_count does not match match records")

    expected_fields = {
        "schema", "match_id", "seed", "teacher_side", "tick", "split", "context_id", "map_id",
        "opponent_id", "trajectory_group_id", "trajectory_sha256", "decision_sha256", "observation",
        "mask", "teacher_logits", "teacher_action",
    }
    examples: list[QualityExample] = []
    seen_decisions: dict[str, str] = {}
    seen_groups: dict[str, str] = {}
    per_match_decisions: set[tuple[str, int]] = set()
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        try:
            item = json.loads(line)
            if not isinstance(item, dict) or set(item) != expected_fields:
                raise DatasetError(f"line {line_number}: quality example fields have invalid shape")
            if item.get("schema") != QUALITY_DATASET_SCHEMA:
                raise DatasetError(f"line {line_number}: unsupported quality example schema")
            match_id = item["match_id"]
            match = match_index.get(match_id)
            if match is None or not match.get("retained"):
                raise DatasetError(f"line {line_number}: example has no retained source match")
            if (type(item.get("seed")) is not int or type(item.get("teacher_side")) is not int
                    or item["teacher_side"] not in (0, 1) or type(item.get("tick")) is not int
                    or item["tick"] < 0 or item.get("split") != match["split"]
                    or item.get("seed") != match["seed"] or item.get("context_id") != match["context_id"]
                    or item.get("map_id") != match["map_id"] or item.get("opponent_id") != match["opponent_id"]
                    or item["trajectory_sha256"] != match.get("trajectory_sha256")
                    or item["trajectory_group_id"] != match.get("trajectory_group_id")):
                raise DatasetError(f"line {line_number}: example provenance differs from its retained match")
            observation = ObservationV2(tuple(float(value) for value in item["observation"]))
            flat_mask = tuple(item["mask"])
            if len(flat_mask) != 30 or any(type(value) is not bool for value in flat_mask):
                raise DatasetError(f"line {line_number}: invalid v2 mask")
            mask = mask_from_flat(flat_mask)
            if mask != get_action_mask(observation):
                raise DatasetError(f"line {line_number}: stored mask differs from ObservationV2")
            logits = PolicyLogitsV2(item["teacher_logits"])
            action = _parse_action(item["teacher_action"])
            decoded = validate_action(observation, mask, decode_action_logits(logits, mask)).action
            if action != decoded:
                raise DatasetError(f"line {line_number}: teacher action differs from shared decoder output")
            decision_sha = _sha256(_decision_payload(item))
            if item.get("decision_sha256") != decision_sha:
                raise DatasetError(f"line {line_number}: decision fingerprint is invalid")
            sequence = match["trajectory_decision_sha256s"]
            if decision_sha not in sequence:
                raise DatasetError(f"line {line_number}: decision is absent from its full trajectory fingerprint")
            prior_split = seen_decisions.get(decision_sha)
            if prior_split is not None:
                if prior_split != item["split"]:
                    raise DatasetError("identical decision fingerprint crosses dataset splits")
                raise DatasetError(f"line {line_number}: duplicate decision was not removed")
            seen_decisions[decision_sha] = item["split"]
            prior_group_split = seen_groups.get(item["trajectory_sha256"])
            if prior_group_split is not None and prior_group_split != item["split"]:
                raise DatasetError("identical trajectory fingerprint crosses dataset splits")
            seen_groups[item["trajectory_sha256"]] = item["split"]
            decision_key = (match_id, item["tick"])
            if decision_key in per_match_decisions:
                raise DatasetError(f"line {line_number}: duplicate decision tick")
            per_match_decisions.add(decision_key)
            examples.append(QualityExample(
                match_id=match_id,
                seed=item["seed"],
                teacher_side=item["teacher_side"],
                tick=item["tick"],
                split=item["split"],
                observation=observation.values,
                mask=mask,
                teacher_logits=logits.values,
                teacher_action=action,
                context_id=item["context_id"],
                map_id=item["map_id"],
                opponent_id=item["opponent_id"],
                trajectory_group_id=item["trajectory_group_id"],
                trajectory_sha256=item["trajectory_sha256"],
                decision_sha256=decision_sha,
            ))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            if isinstance(exc, DatasetError):
                raise
            raise DatasetError(f"line {line_number}: malformed quality example ({exc})") from exc
    if len(examples) != manifest.get("example_count") or not examples:
        raise DatasetError("quality manifest example_count does not match dataset")
    if any(match.get("retained_decision_count") != sum(example.match_id == match["match_id"] for example in examples)
           for match in matches):
        raise DatasetError("quality retained decision counts do not match dataset rows")
    split_sets = {split: [example for example in examples if example.split == split]
                  for split in ("train", "validation", "heldout")}
    if any(not values for values in split_sets.values()):
        raise DatasetError("quality dataset must retain examples in every split")
    expected_match_ids = {match["match_id"] for match in matches if match.get("retained")}
    if {example.match_id for example in examples} != expected_match_ids:
        raise DatasetError("quality manifest retained match IDs do not match dataset rows")
    dedup = manifest.get("deduplication")
    if not isinstance(dedup, dict):
        raise DatasetError("quality manifest is missing deduplication counts")
    expected_examples = {split: len(rows) for split, rows in split_sets.items()}
    expected_trajectories = {split: len({row.trajectory_sha256 for row in rows})
                             for split, rows in split_sets.items()}
    expected_decisions = {split: len({row.decision_sha256 for row in rows})
                          for split, rows in split_sets.items()}
    if (dedup.get("split_example_counts") != expected_examples
            or dedup.get("split_unique_trajectory_counts") != expected_trajectories
            or dedup.get("split_unique_decision_counts") != expected_decisions):
        raise DatasetError("quality manifest effective sample counts do not match dataset")
    return manifest, examples


__all__ = [
    "QUALITY_CONTEXTS", "QUALITY_DATASET_SCHEMA", "QUALITY_MATCH_OPPONENTS", "QUALITY_MATCH_SEEDS",
    "QUALITY_SEED_RANGES", "QUALITY_TEACHER", "QUALITY_TRAIN_SEED", "QUALITY_UNTRAINED_SEED",
    "QualityExample", "collect_quality_dataset", "load_quality_dataset", "quality_config",
]
