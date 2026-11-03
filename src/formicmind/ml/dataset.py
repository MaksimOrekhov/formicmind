"""Collect and validate behavior-cloning examples from real simulator matches.

The dataset boundary contains only ObservationV2, its derived ActionMaskV2,
teacher logits/action, and public run metadata. SimState is never serialized.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..actions import ActionMaskV2, ActionV2, flatten_mask_v2, get_action_mask, mask_from_flat, validate_action
from ..baselines import BaselinePolicy, baseline_spec
from ..config import SimConfig
from ..decoder import PolicyLogitsV2, decode_action_logits
from ..match import EpisodeResult, run_episode
from ..observation import OBS_V2_LENGTH, ObservationV2
from ..rng import policy_stream_id
from ..rules import default_config
from ..serialization import config_hash
from ..tick import PolicyController
from ..versions import ACTION_VERSION, DECODER_VERSION, OBSERVATION_VERSION

DATASET_SCHEMA = "formicmind.behavior-cloning/v1"


@dataclass(frozen=True, slots=True)
class DatasetExample:
    match_id: str
    seed: int
    teacher_side: int
    tick: int
    split: str
    observation: tuple[float, ...]
    mask: ActionMaskV2
    teacher_logits: tuple[float, ...]
    teacher_action: ActionV2


class DatasetError(ValueError):
    """Dataset integrity, provenance, or contract validation failed."""


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
        raise DatasetError("example action has an invalid shape")
    try:
        action = ActionV2(
            lay_order=value["lay_order"],
            forage_share_q=tuple(value["forage_share_q"]),
            army_choice=value["army_choice"],
            avoid_center=value["avoid_center"],
        )
    except (TypeError, ValueError) as exc:
        raise DatasetError("example action has invalid values") from exc
    return action


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


def _split_seeds(seeds: tuple[int, ...], validation_fraction: float) -> dict[int, str]:
    if len(seeds) < 2:
        raise DatasetError("at least two distinct seeds are required for match-level train/validation split")
    if not math.isfinite(validation_fraction) or not 0.0 < validation_fraction < 1.0:
        raise DatasetError("validation_fraction must be between 0 and 1")
    validation_count = max(1, min(len(seeds) - 1, math.ceil(len(seeds) * validation_fraction)))
    ranked = sorted(seeds, key=lambda seed: hashlib.sha256(f"{DATASET_SCHEMA}:split:{seed}".encode()).digest())
    validation = set(ranked[:validation_count])
    return {seed: ("validation" if seed in validation else "train") for seed in seeds}


class _RecordingTeacherController:
    """Call the exact v2 policy once, recording only its policy-boundary inputs/outputs."""

    def __init__(
        self,
        policy: BaselinePolicy,
        stream_id: str,
        rows: list[dict[str, Any]],
        seed: int,
        teacher_side: int,
        match_id: str,
        split: str,
    ) -> None:
        self.policy = policy
        self.stream_id = stream_id
        self.rows = rows
        self.seed = seed
        self.teacher_side = teacher_side
        self.match_id = match_id
        self.split = split

    def act(self, tick: int, observation: ObservationV2, run: Any):
        mask = get_action_mask(observation)
        logits = PolicyLogitsV2.coerce(self.policy(observation, mask, run.stream(self.stream_id)))
        action = decode_action_logits(logits, mask)
        validated = validate_action(observation, mask, action)
        self.rows.append({
            "schema": DATASET_SCHEMA,
            "match_id": self.match_id,
            "seed": self.seed,
            "teacher_side": self.teacher_side,
            "tick": tick,
            "split": self.split,
            "observation": list(observation.values),
            "mask": list(flatten_mask_v2(mask)),
            "teacher_logits": list(logits.values),
            "teacher_action": _action_record(validated.action),
        })
        return validated


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    try:
        temp_path.write_bytes(data)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def collect_dataset(
    output: str | Path,
    *,
    seeds: tuple[int, ...] | list[int],
    teacher: str = "balanced:v2",
    opponent: str = "turtle:v2",
    max_ticks: int = 120,
    validation_fraction: float = 0.2,
    config: SimConfig | None = None,
) -> dict[str, Any]:
    """Collect one teacher-controlled match per seed and seat using ``run_episode``."""
    output = Path(output)
    config = default_config() if config is None else config
    map_id = "default_config" if config == default_config() else "custom_config"
    normalized_seeds = tuple(seeds)
    if any(type(seed) is not int or not 0 <= seed <= (1 << 64) - 1 for seed in normalized_seeds):
        raise DatasetError("seeds must be unsigned 64-bit integers")
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise DatasetError("seeds must be unique")
    if type(max_ticks) is not int or not 1 <= max_ticks <= config.t_max:
        raise DatasetError(f"max_ticks must be within 1..T_max ({config.t_max})")

    teacher_spec, opponent_spec = baseline_spec(teacher), baseline_spec(opponent)
    if teacher_spec.version != "v2" or opponent_spec.version != "v2":
        raise DatasetError("behavior-cloning collection requires explicitly selected Baselines v2")
    if teacher_spec.name not in ("balanced", "rush"):
        raise DatasetError("the technical teacher must be balanced:v2 or rush:v2, not Random or another profile")
    splits = _split_seeds(normalized_seeds, validation_fraction)
    rows: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []

    for seed in normalized_seeds:
        for teacher_side in (0, 1):
            match_id = f"seed-{seed}-teacher-seat-{teacher_side}"
            teacher_stream = policy_stream_id(teacher_spec.policy_id, 0)
            opponent_stream = policy_stream_id(opponent_spec.policy_id, 1)
            teacher_controller = _RecordingTeacherController(
                teacher_spec.build(), teacher_stream, rows, seed, teacher_side, match_id, splits[seed],
            )
            opponent_controller = PolicyController(opponent_spec.build(), opponent_stream)
            controllers = ((teacher_controller, opponent_controller) if teacher_side == 0
                           else (opponent_controller, teacher_controller))
            result: EpisodeResult = run_episode(
                config, seed, controllers[0], controllers[1], max_ticks=max_ticks,
            )
            matches.append({
                "match_id": match_id,
                "seed": seed,
                "teacher_side": teacher_side,
                "split": splits[seed],
                "map_id": map_id,
                "teacher_policy_id": teacher_spec.full_id,
                "teacher_stream_id": teacher_stream,
                "opponent_policy_id": opponent_spec.full_id,
                "opponent_stream_id": opponent_stream,
                "ticks_executed": result.ticks_executed,
                "final_tick": result.final_tick,
                "truncated": result.truncated,
                "winner": result.winner,
                "score": None if result.score is None else list(result.score),
                "config_hash": result.config_hash,
            })

    if not rows:
        raise DatasetError("collection produced no policy decisions")
    identifiers = [(row["match_id"], row["tick"]) for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise DatasetError("duplicate policy decisions were collected")

    jsonl = "".join(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
                    for row in rows).encode("utf-8")
    dataset_sha = hashlib.sha256(jsonl).hexdigest()
    git_sha, git_dirty = _git_provenance()
    manifest: dict[str, Any] = {
        "schema": DATASET_SCHEMA,
        "created_by": "formicmind.ml.dataset.collect_dataset",
        "dataset_file": output.name,
        "dataset_sha256": dataset_sha,
        "example_count": len(rows),
        "match_count": len(matches),
        "contracts": {
            "observation": OBSERVATION_VERSION,
            "observation_length": OBS_V2_LENGTH,
            "action": ACTION_VERSION,
            "decoder": DECODER_VERSION,
        },
        "simulator": {
            "map_id": map_id,
            "config_sha256": config_hash(config),
            "rules_version": config.rules_version,
            "balance_version": config.balance_version,
        },
        "teacher": {
            "id": teacher_spec.full_id,
            "policy_id": teacher_spec.policy_id,
            "parameters": teacher_spec.parameter_object(),
            "parameter_sha256": teacher_spec.digest(),
        },
        "opponent": {
            "id": opponent_spec.full_id,
            "policy_id": opponent_spec.policy_id,
            "parameters": opponent_spec.parameter_object(),
            "parameter_sha256": opponent_spec.digest(),
        },
        "seeds": list(normalized_seeds),
        "split_by_seed": {str(seed): splits[seed] for seed in normalized_seeds},
        "validation_fraction_requested": validation_fraction,
        "max_ticks": max_ticks,
        "matches": matches,
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "source": {"git_sha": git_sha, "git_dirty": git_dirty},
    }
    manifest_path = output.with_name(f"{output.stem}.manifest.json")
    _write_atomic(output, jsonl)
    _write_atomic(manifest_path, (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    return manifest


def load_dataset(path: str | Path) -> tuple[dict[str, Any], list[DatasetExample]]:
    """Verify hashes, contract, masks and labels before examples reach training/inference."""
    path = Path(path)
    manifest_path = path.with_name(f"{path.stem}.manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw = path.read_bytes()
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"cannot read dataset and manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != DATASET_SCHEMA:
        raise DatasetError("unsupported dataset manifest schema")
    if manifest.get("dataset_file") != path.name:
        raise DatasetError("manifest dataset_file does not match the selected dataset")
    if hashlib.sha256(raw).hexdigest() != manifest.get("dataset_sha256"):
        raise DatasetError("dataset SHA-256 does not match its manifest")
    contracts = manifest.get("contracts", {})
    if contracts != {
        "observation": OBSERVATION_VERSION,
        "observation_length": OBS_V2_LENGTH,
        "action": ACTION_VERSION,
        "decoder": DECODER_VERSION,
    }:
        raise DatasetError("dataset contract versions do not match this simulator")
    for key in ("teacher", "opponent"):
        descriptor = manifest.get(key)
        if not isinstance(descriptor, dict):
            raise DatasetError(f"manifest is missing the {key} baseline descriptor")
        try:
            spec = baseline_spec(descriptor["id"])
        except (KeyError, ValueError) as exc:
            raise DatasetError(f"manifest {key} baseline is unavailable") from exc
        if (spec.version != "v2" or (key == "teacher" and spec.name not in ("balanced", "rush"))
                or descriptor.get("policy_id") != spec.policy_id
                or descriptor.get("parameters") != spec.parameter_object()
                or descriptor.get("parameter_sha256") != spec.digest()):
            raise DatasetError(f"manifest {key} baseline parameters do not match the selected v2 policy")
    teacher_descriptor = manifest["teacher"]
    opponent_descriptor = manifest["opponent"]
    teacher_spec = baseline_spec(teacher_descriptor["id"])
    opponent_spec = baseline_spec(opponent_descriptor["id"])
    if manifest.get("split_by_seed") is None or not isinstance(manifest["split_by_seed"], dict):
        raise DatasetError("manifest is missing its whole-seed split map")
    simulator = manifest.get("simulator")
    if not isinstance(simulator, dict) or not isinstance(simulator.get("config_sha256"), str):
        raise DatasetError("manifest is missing the simulator config hash")
    match_metadata: dict[str, dict[str, Any]] = {}
    matches = manifest.get("matches")
    if not isinstance(matches, list):
        raise DatasetError("manifest is missing its match records")
    for match in matches:
        if not isinstance(match, dict) or not isinstance(match.get("match_id"), str):
            raise DatasetError("manifest contains malformed match metadata")
        if match["match_id"] in match_metadata:
            raise DatasetError("manifest contains duplicate match metadata")
        match_metadata[match["match_id"]] = match
    if len(match_metadata) != manifest.get("match_count"):
        raise DatasetError("manifest match_count does not match its match metadata")

    examples: list[DatasetExample] = []
    split_by_seed = manifest.get("split_by_seed")
    seen: set[tuple[str, int]] = set()
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        try:
            item = json.loads(line)
            expected_fields = {
                "schema", "match_id", "seed", "teacher_side", "tick", "split",
                "observation", "mask", "teacher_logits", "teacher_action",
            }
            if not isinstance(item, dict) or set(item) != expected_fields:
                raise DatasetError(f"line {line_number}: example fields do not match the policy boundary")
            if item.get("schema") != DATASET_SCHEMA:
                raise DatasetError(f"line {line_number}: unsupported example schema")
            match_id = item["match_id"]
            seed = item["seed"]
            teacher_side = item["teacher_side"]
            tick = item["tick"]
            split = item["split"]
            if (not isinstance(match_id, str) or type(seed) is not int or type(teacher_side) is not int
                    or teacher_side not in (0, 1) or type(tick) is not int or tick < 0
                    or split not in ("train", "validation") or split_by_seed.get(str(seed)) != split):
                raise DatasetError(f"line {line_number}: invalid match provenance")
            match = match_metadata.get(match_id)
            if (match is None or match.get("match_id") != f"seed-{seed}-teacher-seat-{teacher_side}"
                    or match.get("seed") != seed or match.get("teacher_side") != teacher_side
                    or match.get("split") != split or match.get("map_id") != simulator.get("map_id")
                    or match.get("teacher_policy_id") != teacher_spec.full_id
                    or match.get("teacher_stream_id") != policy_stream_id(teacher_spec.policy_id, 0)
                    or match.get("opponent_policy_id") != opponent_spec.full_id
                    or match.get("opponent_stream_id") != policy_stream_id(opponent_spec.policy_id, 1)
                    or match.get("config_hash") != simulator["config_sha256"]):
                raise DatasetError(f"line {line_number}: example does not match recorded real-match metadata")
            key = (match_id, tick)
            if key in seen:
                raise DatasetError(f"line {line_number}: duplicate decision")
            seen.add(key)
            observation = ObservationV2(tuple(float(value) for value in item["observation"]))
            flat_mask = tuple(item["mask"])
            if len(flat_mask) != 30 or any(type(value) is not bool for value in flat_mask):
                raise DatasetError(f"line {line_number}: invalid mask")
            mask = mask_from_flat(flat_mask)
            if mask != get_action_mask(observation):
                raise DatasetError(f"line {line_number}: stored mask differs from ObservationV2 contract")
            logits = PolicyLogitsV2(item["teacher_logits"])
            action = _parse_action(item["teacher_action"])
            decoded = validate_action(observation, mask, decode_action_logits(logits, mask)).action
            if action != decoded:
                raise DatasetError(f"line {line_number}: teacher label is not the shared-decoder action")
            examples.append(DatasetExample(
                match_id, seed, teacher_side, tick, split, observation.values, mask, logits.values, action,
            ))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            if isinstance(exc, DatasetError):
                raise
            raise DatasetError(f"line {line_number}: malformed example ({exc})") from exc
    if len(examples) != manifest.get("example_count") or not examples:
        raise DatasetError("manifest example_count does not match the dataset")
    if {example.match_id for example in examples} != set(match_metadata):
        raise DatasetError("manifest match_count does not match the dataset")
    return manifest, examples


__all__ = ["DATASET_SCHEMA", "DatasetError", "DatasetExample", "collect_dataset", "load_dataset"]
