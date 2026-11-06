"""FOR-40 single held-out evaluation, controls, pre-registered Q1 gate and Q2 real matches."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..baselines import baseline_spec
from ..decoder import PolicyLogitsV2
from ..match import run_episode
from ..rng import policy_stream_id
from ..tick import PolicyController
from .checkpoint import load_checkpoint
from .dataset import DatasetError, _write_atomic
from .diagnostics import _ShadowController, summarize_rollout_log
from .for40 import (
    FOR40_PROTOCOL,
    FOR40_PROTOCOL_COMMIT,
    REAL_MATCH_SEEDS,
    UNTRAINED_SEED,
    for40_config,
    load_for40_dataset,
    maps_of_split,
)
from .for40_metrics import (
    compact,
    majority_summary,
    records_from_logits,
    stratify,
    summarize,
    train_reference,
)
from .for40_train import build_tensors, predict
from .model import MLPPolicy, PolicyMLP, configure_determinism, resolve_device
from .quality_dataset import quality_config

GATE_THRESHOLDS_Q1 = {
    "macro_recall_min": 0.50, "forage_tv_ratio_max": 0.80, "forage_support_margin": 0.05,
    "maps_beating_constant_min": 6, "validation_gap_max": 0.20, "stable_seeds_min": 2,
}
DEFERRED_OPPONENTS = ("rush:v2", "raider:v2")
MATCH_DET_OPPONENTS = ("turtle:v2", "balanced:v2", "rush:v2", "raider:v2")
MATCH_POLICIES = ("candidate", "other", "constant_mean_logits", "teacher")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _forage_tv_by_teacher_support(examples: list[Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[int, list[float]] = defaultdict(list)
    for example, record in zip(examples, records, strict=True):
        if record["pair"] is not None:
            groups[sum(1 for q in example.teacher_action.forage_share_q if q)].append(record["pair"]["tv"])
    return {f"teacher_support_{k}": {"examples": len(v), "tv_mean": statistics.fmean(v)} for k, v in sorted(groups.items())}


def evaluate_q1_criteria(
    model: dict[str, Any], constant: dict[str, Any], majority: dict[str, Any], by_map: dict[str, dict[str, Any]],
    constant_by_map: dict[str, dict[str, Any]], deferred: dict[str, Any], deferred_constant: dict[str, Any],
    validation_composite: float | None,
) -> dict[str, Any]:
    """G0–G9 for one model; G10 (stability) is added across seeds by the caller."""
    t = GATE_THRESHOLDS_Q1
    forage, cforage = model["forage"], constant["forage"]
    beating = sum(1 for map_id, s in by_map.items()
                  if s.get("composite") is not None and constant_by_map[map_id].get("composite") is not None
                  and s["composite"] > constant_by_map[map_id]["composite"])
    checks = {
        "G0_legal_rate_1": model["legal_rate"] == 1.0,
        "G1_no_collapse": not model["collapse"],
        "G2_lay": (model["lay"]["macro_recall"] or 0.0) >= t["macro_recall_min"]
        and model["lay"]["accuracy"] >= majority["lay"]["accuracy"],
        "G3_army": (model["army"]["group_macro_recall"] or 0.0) >= t["macro_recall_min"]
        and model["army"]["accuracy_exact_choice"] >= majority["army"]["accuracy_exact_choice"],
        "G4_avoid_conditional": (model["avoid_conditional"]["accuracy"] or 0.0)
        >= (majority["avoid_conditional"]["accuracy"] or 0.0),
        "G5_forage": bool(forage.get("tv_mean") is not None
                          and forage["tv_mean"] <= t["forage_tv_ratio_max"] * cforage["tv_mean"]
                          and forage["support_equal_rate"] >= cforage["support_equal_rate"] + t["forage_support_margin"]
                          and forage["top1_match_rate"] >= cforage["top1_match_rate"]),
        "G6_logit_mse_below_constant": all(model["mse"][g] < constant["mse"][g] for g in ("lay", "forage", "army", "avoid")),
        "G7_maps_beating_constant": beating >= t["maps_beating_constant_min"],
        "G8_deferred_opponents": bool(deferred.get("composite") is not None and deferred_constant.get("composite") is not None
                                      and deferred["composite"] >= deferred_constant["composite"] and not deferred["collapse"]),
        "G9_validation_gap": bool(validation_composite is not None and model["composite"] is not None
                                  and model["composite"] >= validation_composite - t["validation_gap_max"]),
    }
    return {"checks": checks, "maps_beating_constant": beating, "maps": len(by_map)}


def _evaluate_model(model: PolicyMLP, examples: list[Any], tensors: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = records_from_logits(examples, predict(model, tensors))
    return records, summarize(records)


class ConstantPolicy:
    """Observation-free control: always the train-mean teacher logits."""

    def __init__(self, logits: list[float]) -> None:
        self.logits = PolicyLogitsV2(logits)

    def __call__(self, observation: Any, mask: Any, policy_rng: Any) -> PolicyLogitsV2:
        return self.logits


def _match_stats(result: Any, side: int) -> dict[str, Any]:
    starvation = sum(1 for e in result.events if e.kind == "starvation" and e.side == side)
    soldiers = sum(1 for e in result.events if e.kind == "hatch" and e.side == side and e.caste == 1)
    workers = sum(1 for e in result.events if e.kind == "hatch" and e.side == side and e.caste == 0)
    return {
        "terminal": result.outcome is not None, "truncated": result.truncated,
        "winner": result.winner, "reason": result.reason, "terminal_tick": result.terminal_tick,
        "final_tick": result.final_tick, "score": None if result.score is None else result.score[side],
        "queen_hp": result.queen_hp[side], "food": result.food[side], "workers": result.workers[side],
        "soldiers": result.soldiers[side], "opponent_queen_hp": result.queen_hp[1 - side],
        "starvation_deaths": starvation, "soldiers_hatched": soldiers, "workers_hatched": workers,
        "state_sha256": result.state_hash, "events_sha256": result.events_hash,
    }


def run_real_matches(policies: dict[str, tuple[Any, str]], max_ticks: int | None = None,
                     maps: list[tuple[str, Any, str]] | None = None) -> dict[str, Any]:
    """Q2: every policy against Baselines v2 on held-out maps (+ ``classic`` as dev-reference), both seats."""
    if maps is None:
        maps = [(spec.map_id, for40_config(spec.map_id), "heldout") for spec in maps_of_split("heldout")]
        maps.append(("classic", quality_config("classic"), "dev_reference"))
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    teacher_policy = baseline_spec("balanced:v2")
    for map_id, config, kind in maps:
        ticks = config.t_max if max_ticks is None else max_ticks
        matchups = [(opp, REAL_MATCH_SEEDS[0]) for opp in MATCH_DET_OPPONENTS] + [("random:v2", s) for s in REAL_MATCH_SEEDS]
        for opponent_id, seed in matchups:
            spec = baseline_spec(opponent_id)
            for name, (policy, stream_policy_id) in policies.items():
                for side in (0, 1):
                    log: list[dict[str, Any]] = []
                    shadow = _ShadowController(policy, policy_stream_id(stream_policy_id, 0), teacher_policy.build(), log)
                    opponent = PolicyController(spec.build(), policy_stream_id(spec.policy_id, 1))
                    controllers = (shadow, opponent) if side == 0 else (opponent, shadow)
                    result = run_episode(config, seed, controllers[0], controllers[1], max_ticks=ticks)
                    records.append({
                        "policy": name, "map_id": map_id, "map_kind": kind, "opponent": spec.full_id, "seed": seed,
                        "side": side, **_match_stats(result, side), "shadow": summarize_rollout_log(log),
                        "decisions_logged": len(log),
                    })
    return {"records": records, "seconds": time.perf_counter() - started}


def summarize_matches(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kind in ("heldout", "dev_reference"):
        for policy in sorted({r["policy"] for r in records}):
            rows = [r for r in records if r["policy"] == policy and r["map_kind"] == kind]
            by_opponent = {}
            for opponent in sorted({r["opponent"] for r in rows}):
                group = [r for r in rows if r["opponent"] == opponent]
                completed = [r for r in group if r["terminal"] and not r["truncated"] and r["score"] is not None]
                shadows = [r["shadow"] for r in group if r["shadow"].get("decisions")]
                by_opponent[opponent] = {
                    "matches": len(group), "completed": len(completed),
                    "truncated": sum(r["truncated"] for r in group),
                    "wins": sum(r["score"] == 1.0 for r in completed), "draws": sum(r["score"] == 0.5 for r in completed),
                    "losses": sum(r["score"] == 0.0 for r in completed),
                    "reasons": dict(sorted(Counter(str(r["reason"]) for r in group).items())),
                    "distinct_state_trajectories": len({(r["map_id"], r["seed"], r["state_sha256"]) for r in group}),
                    "distinct_state_hashes": len({r["state_sha256"] for r in group}),
                    "mean_final_tick": statistics.fmean(r["final_tick"] for r in group),
                    "starvation_deaths_total": sum(r["starvation_deaths"] for r in group),
                    "soldiers_hatched_mean": statistics.fmean(r["soldiers_hatched"] for r in group),
                    "shadow_lay_agree": statistics.fmean(s["lay_agree_rate"] for s in shadows) if shadows else None,
                    "shadow_army_agree": statistics.fmean(s["army_agree_rate"] for s in shadows) if shadows else None,
                    "shadow_avoid_agree": statistics.fmean(s["avoid_agree_rate"] for s in shadows) if shadows else None,
                    "shadow_forage_support_equal": statistics.fmean(s["forage_support_equal_rate"] for s in shadows) if shadows else None,
                    "shadow_forage_tv_mean": statistics.fmean(s["forage_tv_mean"] for s in shadows) if shadows else None,
                }
            out[f"{kind}/{policy}"] = by_opponent
    return out


def evaluate_for40(
    dataset_path: str | Path, selection_path: str | Path, output_dir: str | Path, *, device_name: str = "cpu",
    real_matches: bool = True, match_max_ticks: int | None = None, expect_full_protocol: bool = True,
) -> dict[str, Any]:
    """Evaluate every frozen checkpoint once on held-out data; also controls, the Q1 gate and Q2 matches."""
    dataset_manifest, examples = load_for40_dataset(dataset_path, expect_full_protocol=expect_full_protocol)
    selection_path = Path(selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("dataset_sha256") != dataset_manifest["dataset_sha256"] or not selection.get("frozen_before_heldout"):
        raise DatasetError("selection.json is not bound to this dataset or was not frozen before held-out evaluation")
    output_dir = Path(output_dir)
    device = resolve_device(device_name)
    train = [e for e in examples if e.split == "train"]
    validation = [e for e in examples if e.split == "validation"]
    heldout = [e for e in examples if e.split == "heldout"]
    held_t = build_tensors(heldout, device)
    reference = train_reference(train)

    models: dict[str, dict[str, Any]] = {}
    for run in selection["runs"]:
        directory = output_dir / "checkpoints" / run["checkpoint"]
        model, checkpoint_manifest, _ = load_checkpoint(directory, device_name)
        if (checkpoint_manifest["weights"]["sha256"] != run["weights_sha256"]
                or checkpoint_manifest["data"]["sha256"] != dataset_manifest["dataset_sha256"]):
            raise DatasetError(f"checkpoint {run['checkpoint']} differs from the frozen selection record")
        records, summary = _evaluate_model(model, heldout, held_t)
        models[run["checkpoint"]] = {"run": run, "model": model, "manifest": checkpoint_manifest,
                                     "records": records, "summary": summary}

    # Controls.
    constant_logits = reference["mean_logits"]
    constant_records = records_from_logits(heldout, [constant_logits] * len(heldout))
    constant = summarize(constant_records)
    majority = majority_summary(heldout, reference)
    configure_determinism(UNTRAINED_SEED, device)
    untrained_model = PolicyMLP().to(device)
    untrained_records, untrained = _evaluate_model(untrained_model, heldout, held_t)
    oracle = summarize(records_from_logits(heldout, [list(e.teacher_logits) for e in heldout]))

    constant_by_map = stratify(heldout, constant_records, "map_id")
    deferred_idx = [i for i, e in enumerate(heldout) if e.opponent_id in DEFERRED_OPPONENTS]
    deferred_constant = summarize([constant_records[i] for i in deferred_idx])
    validation_composites = {run["checkpoint"]: run["validation_composite"] for run in selection["runs"]}

    per_model: dict[str, Any] = {}
    gate_by_checkpoint: dict[str, Any] = {}
    for name, entry in models.items():
        records, summary = entry["records"], entry["summary"]
        by_map = stratify(heldout, records, "map_id")
        deferred = summarize([records[i] for i in deferred_idx])
        gate_by_checkpoint[name] = evaluate_q1_criteria(
            summary, constant, majority, by_map, constant_by_map, deferred, deferred_constant, validation_composites[name])
        per_model[name] = {
            "condition": entry["run"]["condition"], "seed": entry["run"]["seed"],
            "weights_sha256": entry["run"]["weights_sha256"], "converged": entry["run"]["converged"],
            "stopped_by": entry["run"]["stopped_by"], "optimizer_steps": entry["run"]["optimizer_steps"],
            "overall": summary, "by_map": {k: compact(v) for k, v in by_map.items()},
            "by_opponent": {k: compact(v) for k, v in stratify(heldout, records, "opponent_id").items()},
            "deferred_opponents": compact(deferred) if deferred.get("examples") else None,
            "forage_tv_by_teacher_support": _forage_tv_by_teacher_support(heldout, records),
            "q1_criteria": gate_by_checkpoint[name],
        }

    candidate_condition = selection["candidate_condition"]
    candidate_name = selection["selected"][candidate_condition]["checkpoint"]
    base_checks = ["G0_legal_rate_1", "G1_no_collapse", "G2_lay", "G3_army", "G4_avoid_conditional", "G5_forage",
                   "G6_logit_mse_below_constant"]
    seeds_passing = sum(
        all(gate_by_checkpoint[n]["checks"][c] for c in base_checks)
        for n, e in models.items() if e["run"]["condition"] == candidate_condition)
    final_checks = dict(gate_by_checkpoint[candidate_name]["checks"])
    final_checks["G10_stability_2_of_3_seeds"] = seeds_passing >= GATE_THRESHOLDS_Q1["stable_seeds_min"]
    failed = sorted(k for k, v in final_checks.items() if not v)
    candidate_converged = selection["selected"][candidate_condition] and next(
        r for r in selection["runs"] if r["checkpoint"] == candidate_name)["converged"]
    q1 = {
        "decision": "GO to separate design of the next stage" if not failed else "NO-GO",
        "candidate_condition": candidate_condition, "candidate_checkpoint": candidate_name,
        "candidate_converged": candidate_converged, "checks": final_checks, "failed": failed,
        "seeds_passing_G0_G6": seeds_passing, "thresholds": GATE_THRESHOLDS_Q1,
        "collapse_warning_models": sorted(n for n, m in models.items() if m["summary"]["collapse"]),
        "note": "GO authorises only a separately designed next experiment; it is not a claim of strategic strength.",
    }

    report: dict[str, Any] = {
        "schema": "formicmind.ml-for40-evaluation/v1",
        "protocol": {"path": FOR40_PROTOCOL, "commit": FOR40_PROTOCOL_COMMIT},
        "dataset_sha256": dataset_manifest["dataset_sha256"],
        "selection_sha256": _file_sha256(selection_path),
        "heldout": {"examples": len(heldout), "maps": len({e.map_id for e in heldout}),
                    "trajectories": len({e.trajectory_sha256 for e in heldout}),
                    "teacher": {"lay": majority["lay"]["teacher_counts"], "army": majority["army"]["teacher_counts"]}},
        "validation_reference": {n: validation_composites[n] for n in models},
        "models": per_model,
        "controls": {
            "constant_mean_logits": {"overall": constant, "by_map": {k: compact(v) for k, v in constant_by_map.items()},
                                     "deferred_opponents": compact(deferred_constant)},
            "majority_actions": majority,
            "untrained_mlp": {"overall": untrained, "seed": UNTRAINED_SEED},
            "teacher_oracle": compact(oracle),
            "train_reference": {k: v for k, v in reference.items() if k != "mean_logits"},
            "train_mean_logits": reference["mean_logits"],
        },
        "q1_gate": q1,
        "device": str(device),
    }
    if real_matches:
        policies: dict[str, tuple[Any, str]] = {}
        other_condition = "A" if candidate_condition == "B" else "B"
        for label, condition in (("candidate", candidate_condition), ("other", other_condition)):
            name = selection["selected"][condition]["checkpoint"]
            weights = models[name]["manifest"]["weights"]["sha256"]
            policies[label] = (MLPPolicy(models[name]["model"], device), f"mlp-{weights[:16]}")
        policies["constant_mean_logits"] = (ConstantPolicy(constant_logits), "constant-mean-logits")
        policies["teacher"] = (baseline_spec("balanced:v2").build(), baseline_spec("balanced:v2").policy_id)
        played = run_real_matches(policies, max_ticks=match_max_ticks)
        report["real_matches"] = {
            "policies": {"candidate": candidate_condition, "other": other_condition},
            "summary": summarize_matches(played["records"]), "seconds": played["seconds"],
            "match_count": len(played["records"]),
            "technical_failures": 0,
            "note": "Deterministic pairs repeat across seats; distinct_state_trajectories counts independent trials.",
        }
        report["real_matches_records"] = played["records"]
    _write_atomic(output_dir / "evaluation.json",
                  (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    return report


__all__ = ["ConstantPolicy", "GATE_THRESHOLDS_Q1", "evaluate_for40", "evaluate_q1_criteria", "run_real_matches",
           "summarize_matches"]
