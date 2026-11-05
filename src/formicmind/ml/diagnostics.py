"""FOR-38: forensic diagnostics of a frozen FOR-37 checkpoint (read-only; no training, no decoder change).

The functions that analyse predictions are pure (no PyTorch): they take teacher examples and 17-logit
predictions and measure *why* the decoded ``forage_share_q`` does or does not match the teacher.  Only
``diagnose_quality`` loads a checkpoint (and therefore PyTorch).  Nothing here writes to the frozen inputs.

All thresholds below are *descriptive* bins fixed before the analysis was run (``docs/ml_for38_protocol.md``);
none of them is a pass/fail criterion.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import Counter
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from ..actions import ActionMaskV2, ActionV2, get_action_mask, validate_action
from ..decoder import PolicyLogitsV2, _as_float32, decode_action_logits, decode_forage_logits
from ..observation import ObservationV2
from ..rules import FORAGE_CATEGORIES, FORAGE_SUPPORT_MAX, Q_TOTAL
from .dataset import DatasetError, _write_atomic

DIAGNOSTICS_SCHEMA = "formicmind.ml-action-diagnostics/v1"

# Descriptive bins, declared in docs/ml_for38_protocol.md before any number was computed.
TV_NEAR_THRESHOLDS = (0.001, 0.005, 0.01, 0.02, 0.05, 0.10)
SMALL_SHARE_TV = 0.05  # same support and TV <= this -> "small share error"
NEAR_ZERO_BANDS = (0.05, 0.10, 0.25, 0.50)
NOISE_SIGMAS = (0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
NOISE_REPEATS = 3
NOISE_SEED = 38_000_001

LOGIT_GROUPS = {"lay": (0, 3), "forage": (3, 9), "army": (9, 16), "avoid_center": (16, 17)}

FROZEN_FOR37_SHA256 = {
    "quality.jsonl": "a2d03800ff93c897782fb1483d0872a5440019c6c6620bc9b837944b34230002",
    "quality.manifest.json": "2c5ba19d02aa499b4bdd7912dad82674f8ca47041bd141913468212a9dfd5d89",
    "report.json": "c214af1b7430d89a7177b6764d300d743bb1602b32f37d3ca264472d6d9007da",
    "report_ru.md": "c874f1e974de3bd07570f91ac36172953bf776dbe5d3049de1bafd13ed93780c",
    "checkpoint/manifest.json": "784a36e8a2aa593251696d1b9e01416648eb753f6d7c2f61b1d10d265baef1f6",
    "checkpoint/weights.pt": "56278fbe7d3e5d13bee6ca8bac4061535584d0b167f8a7abf567b273dd292b8e",
}
_TEXT_ARTIFACTS = {"quality.manifest.json", "report.json", "report_ru.md", "checkpoint/manifest.json"}


def file_sha256(path: str | Path, *, normalize_lf: bool = False) -> str:
    """SHA-256 of a file; ``normalize_lf`` hashes text with CRLF folded to LF (checkout-independent)."""
    raw = Path(path).read_bytes()
    if normalize_lf:
        raw = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def verify_frozen_artifacts(directory: str | Path) -> dict[str, dict[str, Any]]:
    """Hash the frozen FOR-37 files that exist under ``directory`` and compare them with the pinned values."""
    root = Path(directory)
    result: dict[str, dict[str, Any]] = {}
    for name, expected in FROZEN_FOR37_SHA256.items():
        path = root / name
        if not path.is_file():
            result[name] = {"present": False}
            continue
        actual = file_sha256(path, normalize_lf=name in _TEXT_ARTIFACTS)
        result[name] = {"present": True, "sha256": actual, "expected_sha256": expected, "match": actual == expected,
                        "hashed_with_lf_normalization": name in _TEXT_ARTIFACTS}
    return result


# ---------------------------------------------------------------------------------------------------------
# forage pair metrics
# ---------------------------------------------------------------------------------------------------------

def _support(q: Sequence[int]) -> frozenset[int]:
    return frozenset(i for i, value in enumerate(q) if value > 0)


def _argmax_first(q: Sequence[int]) -> int:
    best = 0
    for i, value in enumerate(q):
        if value > q[best]:
            best = i
    return best


def total_variation(q_pred: Sequence[int], q_teacher: Sequence[int]) -> float:
    """``sum(|q_pred - q_teacher|) / (2 * Q)``; both vectors must sum to ``Q``."""
    return sum(abs(a - b) for a, b in zip(q_pred, q_teacher, strict=True)) / (2 * Q_TOTAL)


def forage_pair_metrics(q_pred: Sequence[int], q_teacher: Sequence[int]) -> dict[str, Any]:
    """Compare two decoded ``forage_share_q`` vectors and assign exactly one error category."""
    pred, teacher = tuple(q_pred), tuple(q_teacher)
    if len(pred) != FORAGE_CATEGORIES or len(teacher) != FORAGE_CATEGORIES:
        raise ValueError("forage_share_q needs 6 entries")
    if sum(pred) != Q_TOTAL or sum(teacher) != Q_TOTAL:
        raise ValueError(f"forage_share_q must sum to {Q_TOTAL}")
    sp, st = _support(pred), _support(teacher)
    tv = total_variation(pred, teacher)
    union = sp | st
    jaccard = len(sp & st) / len(union)
    outside = sum(pred[i] for i in range(FORAGE_CATEGORIES) if i not in st) / Q_TOTAL
    if pred == teacher:
        category = "exact"
    elif sp != st:
        if st < sp:
            category = "support_pred_superset"
        elif sp < st:
            category = "support_pred_subset"
        elif sp & st:
            category = "support_partial_overlap"
        else:
            category = "support_disjoint"
    elif tv <= SMALL_SHARE_TV:
        category = "same_support_small_share_error"
    else:
        category = "same_support_large_redistribution"
    return {
        "exact": pred == teacher,
        "support_equal": sp == st,
        "support_pred": len(sp),
        "support_teacher": len(st),
        "jaccard": jaccard,
        "top1_match": _argmax_first(pred) == _argmax_first(teacher),
        "tv": tv,
        "mass_outside_teacher_support": outside,
        "category": category,
    }


FORAGE_CATEGORY_ORDER = (
    "exact", "same_support_small_share_error", "same_support_large_redistribution",
    "support_pred_superset", "support_pred_subset", "support_partial_overlap", "support_disjoint",
)


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] * (1 - (position - lower)) + ordered[upper] * (position - lower)


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def aggregate_forage(pairs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(pairs)
    if n == 0:
        return {"examples": 0}
    tvs = [p["tv"] for p in pairs]
    cats = Counter(p["category"] for p in pairs)
    return {
        "examples": n,
        "exact_rate": _rate(sum(p["exact"] for p in pairs), n),
        "support_equal_rate": _rate(sum(p["support_equal"] for p in pairs), n),
        "top1_match_rate": _rate(sum(p["top1_match"] for p in pairs), n),
        "mean_jaccard": statistics.fmean(p["jaccard"] for p in pairs),
        "tv_mean": statistics.fmean(tvs),
        "tv_median": statistics.median(tvs),
        "tv_p90": _quantile(tvs, 0.90),
        "tv_max": max(tvs),
        "mass_outside_teacher_support_mean": statistics.fmean(p["mass_outside_teacher_support"] for p in pairs),
        "tv_at_most": {str(t): _rate(sum(v <= t for v in tvs), n) for t in TV_NEAR_THRESHOLDS},
        "support_size_confusion": {
            f"teacher_{a}_pred_{b}": c for (a, b), c in sorted(
                Counter((p["support_teacher"], p["support_pred"]) for p in pairs).items())
        },
        "categories": {name: cats.get(name, 0) for name in FORAGE_CATEGORY_ORDER},
        "category_rates": {name: _rate(cats.get(name, 0), n) for name in FORAGE_CATEGORY_ORDER},
    }


# ---------------------------------------------------------------------------------------------------------
# continuous (unquantized) view of the decoder: softmax vs largest-remainder quantization
# ---------------------------------------------------------------------------------------------------------

def forage_continuous_probs(logits: Sequence[float], target_mask: Sequence[bool]) -> tuple[float, ...]:
    """The decoder's softmax over the selected support *before* quantization (same selection rule)."""
    values = [_as_float32(v) for v in logits]
    allowed = [i for i in range(FORAGE_CATEGORIES) if target_mask[i]]
    ranked = sorted(allowed, key=lambda i: (-values[i], i))
    positive = sum(1 for i in allowed if values[i] > 0)
    k = min(FORAGE_SUPPORT_MAX, max(1, positive), len(allowed))
    chosen = ranked[:k]
    peak = max(values[i] for i in chosen)
    exps = [math.exp(values[i] - peak) for i in chosen]
    norm = math.fsum(exps)
    probs = [0.0] * FORAGE_CATEGORIES
    for i, e in zip(chosen, exps):
        probs[i] = e / norm
    return tuple(probs)


def _tv_probs(a: Sequence[float], b: Sequence[float]) -> float:
    return 0.5 * math.fsum(abs(x - y) for x, y in zip(a, b, strict=True))


# ---------------------------------------------------------------------------------------------------------
# per-example analysis
# ---------------------------------------------------------------------------------------------------------

def _active_logit_flags(mask: ActionMaskV2, teacher_action: ActionV2) -> list[bool]:
    """Which of the 17 logits can influence the decoded action under this mask (conditional ``avoid_center``)."""
    flags = [*mask.lay, *mask.forage_target, *mask.army_choice]
    no_avoid, avoid_ok = mask.army_avoid_center[teacher_action.army_choice]
    flags.append(bool(no_avoid and avoid_ok))
    return flags


def _decode_prediction(example: Any, prediction: Sequence[float]) -> ActionV2:
    logits = PolicyLogitsV2(list(prediction))
    return validate_action(ObservationV2(example.observation), example.mask,
                           decode_action_logits(logits, example.mask)).action


def _forage_logit_geometry(logits: Sequence[float], target_mask: Sequence[bool]) -> dict[str, float | int]:
    values = [_as_float32(v) for v in logits[3:9]]
    allowed = [i for i in range(6) if target_mask[i]]
    ranked = sorted(allowed, key=lambda i: (-values[i], i))
    positive = sum(1 for i in allowed if values[i] > 0)
    k = min(FORAGE_SUPPORT_MAX, max(1, positive), len(allowed))
    boundary = values[ranked[k - 1]] - values[ranked[k]] if k < len(ranked) else math.inf
    return {
        "positive_count": positive,
        "support_k": k,
        "min_abs_logit": min(abs(values[i]) for i in allowed),
        "support_boundary_margin": boundary,
    }


def analyze_predictions(examples: Sequence[Any], predictions: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Aggregate every diagnostic for a set of examples; ``predictions[i]`` is the 17-logit output for example i."""
    if len(examples) != len(predictions):
        raise ValueError("examples and predictions must align")
    n = len(examples)
    if n == 0:
        return {"examples": 0}
    sq = {name: 0.0 for name in LOGIT_GROUPS}
    cnt = {name: 0 for name in LOGIT_GROUPS}
    active_sq = {name: 0.0 for name in LOGIT_GROUPS}
    active_cnt = {name: 0 for name in LOGIT_GROUPS}
    masked_sq, masked_cnt = 0.0, 0
    pairs: list[dict[str, Any]] = []
    cont_tv: list[float] = []
    quant_tv_pred: list[float] = []
    quant_tv_teacher: list[float] = []
    exact = lay_ok = army_ok = avoid_ok_n = full = 0
    avoid_cond_total = avoid_cond_ok = 0
    lay_conf: Counter = Counter()
    army_conf: Counter = Counter()
    avoid_conf: Counter = Counter()
    geometry_pred: list[dict[str, float | int]] = []
    geometry_teacher: list[dict[str, float | int]] = []
    logit_error_vs_margin = {"selected_below_zero": 0, "unselected_above_zero": 0, "wrong_rank_within_positive": 0}
    forage_err_abs: list[float] = []
    same_k_member_mismatch = diff_k = 0
    for example, prediction in zip(examples, predictions, strict=True):
        pred = PolicyLogitsV2(list(prediction)).values
        teacher = example.teacher_logits
        flags = _active_logit_flags(example.mask, example.teacher_action)
        for name, (lo, hi) in LOGIT_GROUPS.items():
            for i in range(lo, hi):
                e2 = (pred[i] - teacher[i]) ** 2
                sq[name] += e2
                cnt[name] += 1
                if flags[i]:
                    active_sq[name] += e2
                    active_cnt[name] += 1
                else:
                    masked_sq += e2
                    masked_cnt += 1
        action = _decode_prediction(example, prediction)
        t = example.teacher_action
        pair = forage_pair_metrics(action.forage_share_q, t.forage_share_q)
        pairs.append(pair)
        full += action == t
        lay_ok += action.lay_order == t.lay_order
        army_ok += action.army_choice == t.army_choice
        avoid_ok_n += action.avoid_center == t.avoid_center
        no_avoid, can_avoid = example.mask.army_avoid_center[t.army_choice]
        if no_avoid and can_avoid and action.army_choice == t.army_choice:
            avoid_cond_total += 1
            avoid_cond_ok += action.avoid_center == t.avoid_center
        lay_conf[(t.lay_order, action.lay_order)] += 1
        army_conf[(t.army_choice, action.army_choice)] += 1
        avoid_conf[(t.avoid_center, action.avoid_center)] += 1
        p_pred = forage_continuous_probs(pred[3:9], example.mask.forage_target)
        p_teacher = forage_continuous_probs(teacher[3:9], example.mask.forage_target)
        cont_tv.append(_tv_probs(p_pred, p_teacher))
        quant_tv_pred.append(_tv_probs(p_pred, [v / Q_TOTAL for v in action.forage_share_q]))
        quant_tv_teacher.append(_tv_probs(p_teacher, [v / Q_TOTAL for v in t.forage_share_q]))
        gp = _forage_logit_geometry(pred, example.mask.forage_target)
        gt = _forage_logit_geometry(teacher, example.mask.forage_target)
        geometry_pred.append(gp)
        geometry_teacher.append(gt)
        forage_err_abs.extend(abs(pred[i] - teacher[i]) for i in range(3, 9))
        if not pair["support_equal"]:
            teacher_support = _support(t.forage_share_q)
            pred_support = _support(action.forage_share_q)
            if len(teacher_support) == len(pred_support):
                same_k_member_mismatch += 1
            else:
                diff_k += 1
            if any(pred[3 + i] <= 0 for i in teacher_support):
                logit_error_vs_margin["selected_below_zero"] += 1
            if any(pred[3 + i] > 0 for i in range(6) if i not in teacher_support):
                logit_error_vs_margin["unselected_above_zero"] += 1
            if gp["positive_count"] > FORAGE_SUPPORT_MAX:
                logit_error_vs_margin["wrong_rank_within_positive"] += 1

    def _mse(total: float, count: int) -> float | None:
        return total / count if count else None

    all_pred_flat = [PolicyLogitsV2(list(p)).values for p in predictions]
    near_zero = {}
    for band in NEAR_ZERO_BANDS:
        pred_hits = sum(abs(pr[3 + i]) < band for pr in all_pred_flat for i in range(6))
        teacher_hits = sum(abs(ex.teacher_logits[3 + i]) < band for ex in examples for i in range(6))
        near_zero[str(band)] = {"pred_fraction_of_forage_logits": pred_hits / (6 * n),
                                "teacher_fraction_of_forage_logits": teacher_hits / (6 * n)}
    teacher_support_k = Counter(g["support_k"] for g in geometry_teacher)
    pred_support_k = Counter(g["support_k"] for g in geometry_pred)
    decoded = [_decode_prediction(e, p) for e, p in zip(examples, predictions, strict=True)]
    teacher_dist = {
        "lay": dict(sorted(Counter(e.teacher_action.lay_order for e in examples).items())),
        "army": dict(sorted(Counter(e.teacher_action.army_choice for e in examples).items())),
        "avoid_center": {str(int(k)): v for k, v in sorted(Counter(e.teacher_action.avoid_center for e in examples).items())},
    }
    teacher_dist["forage_support"] = dict(sorted(Counter(
        "".join(str(i) for i in sorted(_support(e.teacher_action.forage_share_q))) for e in examples).items()))
    majority = {
        key: max(counter.values()) / n for key, counter in (
            ("lay", Counter(e.teacher_action.lay_order for e in examples)),
            ("army", Counter(e.teacher_action.army_choice for e in examples)),
            ("avoid_center", Counter(e.teacher_action.avoid_center for e in examples)),
            ("forage_support_set", Counter(_support(e.teacher_action.forage_share_q) for e in examples)),
        )
    }
    variability = prediction_variability(examples, predictions)
    variability["distinct_decoded_lay_army_avoid"] = len({(a.lay_order, a.army_choice, a.avoid_center) for a in decoded})
    variability["distinct_decoded_forage_supports"] = len({_support(a.forage_share_q) for a in decoded})
    variability["distinct_decoded_forage_tuples"] = len({a.forage_share_q for a in decoded})
    variability["distinct_teacher_forage_tuples"] = len({e.teacher_action.forage_share_q for e in examples})
    variability["distinct_teacher_lay_army_avoid"] = len({
        (e.teacher_action.lay_order, e.teacher_action.army_choice, e.teacher_action.avoid_center) for e in examples})
    return {
        "examples": n,
        "teacher_distribution": teacher_dist,
        "majority_class_accuracy": majority,
        "variability": variability,
        "unique_decisions": len({getattr(e, "decision_sha256", i) for i, e in enumerate(examples)}),
        "action_accuracy": {
            "full_action": full / n,
            "lay": lay_ok / n,
            "army": army_ok / n,
            "avoid_center_unconditional": avoid_ok_n / n,
            "avoid_center_when_real_choice_and_army_correct": _rate(avoid_cond_ok, avoid_cond_total),
            "avoid_center_real_choice_examples": avoid_cond_total,
            "forage_share_exact": _rate(sum(p["exact"] for p in pairs), n),
        },
        "logit_mse": {
            "all": sum(sq.values()) / sum(cnt.values()),
            "by_group_all_entries": {name: sq[name] / cnt[name] for name in LOGIT_GROUPS},
            "by_group_mask_active_entries": {name: _mse(active_sq[name], active_cnt[name]) for name in LOGIT_GROUPS},
            "masked_or_inactive_entries": {"count": masked_cnt, "mse": _mse(masked_sq, masked_cnt)},
        },
        "forage": {
            **aggregate_forage(pairs),
            "continuous_softmax_tv_mean": statistics.fmean(cont_tv),
            "continuous_softmax_tv_median": statistics.median(cont_tv),
            "quantization_only_tv_pred_mean": statistics.fmean(quant_tv_pred),
            "quantization_only_tv_teacher_mean": statistics.fmean(quant_tv_teacher),
            "quantization_only_tv_max": max(max(quant_tv_pred), max(quant_tv_teacher)),
            "forage_logit_abs_error": {
                "mean": statistics.fmean(forage_err_abs),
                "median": statistics.median(forage_err_abs),
                "p90": _quantile(forage_err_abs, 0.90),
                "min": min(forage_err_abs),
            },
            "support_k_distribution": {"teacher": dict(sorted(teacher_support_k.items())),
                                       "pred": dict(sorted(pred_support_k.items()))},
            "support_mismatch_breakdown": {
                "same_k_different_members": same_k_member_mismatch,
                "different_k": diff_k,
                **{f"mismatches_with_{k}": v for k, v in logit_error_vs_margin.items()},
            },
            "near_zero_logit_bands": near_zero,
            "teacher_min_abs_logit": min(g["min_abs_logit"] for g in geometry_teacher),
            "teacher_min_support_boundary_margin": min(g["support_boundary_margin"] for g in geometry_teacher),
            "pred_min_abs_logit_median": statistics.median(g["min_abs_logit"] for g in geometry_pred),
            "pred_support_boundary_margin_median": statistics.median(
                g["support_boundary_margin"] for g in geometry_pred if math.isfinite(g["support_boundary_margin"])
            ) if any(math.isfinite(g["support_boundary_margin"]) for g in geometry_pred) else None,
        },
        "confusion": {
            "lay": {f"teacher_{a}_pred_{b}": c for (a, b), c in sorted(lay_conf.items())},
            "army": {f"teacher_{a}_pred_{b}": c for (a, b), c in sorted(army_conf.items())},
            "avoid_center": {f"teacher_{int(a)}_pred_{int(b)}": c for (a, b), c in sorted(avoid_conf.items())},
        },
    }



def prediction_variability(examples: Sequence[Any], predictions: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Does the output depend on the input?  Spread and per-logit R^2 of predictions against the teacher."""
    n = len(examples)
    pred = [PolicyLogitsV2(list(p)).values for p in predictions]
    teach = [e.teacher_logits for e in examples]
    out_std_pred, out_std_teacher, r2 = [], [], []
    for i in range(17):
        pv = [row[i] for row in pred]
        tv = [row[i] for row in teach]
        mean_t = statistics.fmean(tv)
        sst = math.fsum((v - mean_t) ** 2 for v in tv)
        sse = math.fsum((a - b) ** 2 for a, b in zip(pv, tv))
        out_std_pred.append(statistics.pstdev(pv) if n > 1 else 0.0)
        out_std_teacher.append(statistics.pstdev(tv) if n > 1 else 0.0)
        r2.append(None if sst == 0 else 1.0 - sse / sst)
    groups = {}
    for name, (lo, hi) in LOGIT_GROUPS.items():
        groups[name] = {
            "pred_std_mean": statistics.fmean(out_std_pred[lo:hi]),
            "teacher_std_mean": statistics.fmean(out_std_teacher[lo:hi]),
            "r2_over_logits_with_teacher_variance": statistics.fmean([v for v in r2[lo:hi] if v is not None])
            if any(v is not None for v in r2[lo:hi]) else None,
        }
    return {"by_group": groups, "r2_per_logit": r2}


def mean_logit_baseline(train_examples: Sequence[Any], examples: Sequence[Any]) -> dict[str, Any]:
    """Reference predictor that ignores the observation: the per-logit mean of the training teacher logits."""
    mean = [statistics.fmean(e.teacher_logits[i] for e in train_examples) for i in range(17)]
    summary = analyze_predictions(examples, [mean] * len(examples))
    return {
        "mean_logits": mean,
        "logit_mse": summary["logit_mse"]["all"],
        "logit_mse_by_group": summary["logit_mse"]["by_group_all_entries"],
        "action_accuracy": summary["action_accuracy"],
        "forage_support_equal_rate": summary["forage"]["support_equal_rate"],
        "forage_top1_match_rate": summary["forage"]["top1_match_rate"],
        "forage_tv_mean": summary["forage"]["tv_mean"],
    }

# ---------------------------------------------------------------------------------------------------------
# decoder strictness: how precise must the logits be for an *exact* forage match?
# ---------------------------------------------------------------------------------------------------------

def teacher_oracle_roundtrip(examples: Sequence[Any]) -> dict[str, Any]:
    """Decode the stored teacher logits again: must reproduce every stored action (metric/decoder control)."""
    exact = forage = 0
    for example in examples:
        action = _decode_prediction(example, example.teacher_logits)
        exact += action == example.teacher_action
        forage += action.forage_share_q == example.teacher_action.forage_share_q
    n = len(examples)
    return {"examples": n, "full_action_exact_rate": _rate(exact, n), "forage_exact_rate": _rate(forage, n)}


def noise_sensitivity(examples: Sequence[Any], *, sigmas: Sequence[float] = NOISE_SIGMAS,
                      repeats: int = NOISE_REPEATS, seed: int = NOISE_SEED) -> list[dict[str, Any]]:
    """Add Gaussian noise ``N(0, sigma)`` to the six teacher forage logits only and decode again.

    This is not a model: it measures how small a *per-logit* error must be before ``forage_share_q`` can match.
    """
    rows = []
    for sigma in sigmas:
        rng = random.Random(f"{seed}:{sigma!r}")
        pairs: list[dict[str, Any]] = []
        for _ in range(repeats):
            for example in examples:
                noisy = list(example.teacher_logits)
                for i in range(3, 9):
                    noisy[i] = noisy[i] + rng.gauss(0.0, sigma)
                q = decode_forage_logits(PolicyLogitsV2(noisy).values[3:9], example.mask.forage_target)
                pairs.append(forage_pair_metrics(q, example.teacher_action.forage_share_q))
        agg = aggregate_forage(pairs)
        rows.append({
            "sigma": sigma,
            "samples": len(pairs),
            "forage_exact_rate": agg["exact_rate"],
            "support_equal_rate": agg["support_equal_rate"],
            "tv_mean": agg["tv_mean"],
            "tv_at_most_0.01": agg["tv_at_most"]["0.01"],
        })
    return rows


def share_sensitivity(examples: Sequence[Any]) -> dict[str, Any]:
    """Derivative of the quantized share w.r.t. a unit logit: ``Q * p_i * (1 - p_i)`` counts per logit unit."""
    slopes = []
    for example in examples:
        probs = forage_continuous_probs(example.teacher_logits[3:9], example.mask.forage_target)
        slopes.extend(Q_TOTAL * p * (1 - p) for p in probs if 0 < p < 1)
    if not slopes:
        return {"entries": 0}
    return {
        "entries": len(slopes),
        "counts_per_logit_unit_median": statistics.median(slopes),
        "counts_per_logit_unit_max": max(slopes),
        "logit_error_for_one_count_median": 1.0 / statistics.median(slopes),
        "logit_error_for_one_count_min": 1.0 / max(slopes),
    }


# ---------------------------------------------------------------------------------------------------------
# model-facing parts (PyTorch is imported only here)
# ---------------------------------------------------------------------------------------------------------

def predict_logits(model: Any, examples: Sequence[Any], device: Any) -> list[list[float]]:
    from .model import torch

    model.eval()
    x = torch.tensor([e.observation for e in examples], dtype=torch.float32, device=device)
    with torch.inference_mode():
        out = model(x)
    if out.dtype != torch.float32 or tuple(out.shape) != (len(examples), 17):
        raise DatasetError(f"model output has dtype {out.dtype} and shape {tuple(out.shape)}, expected float32 (N, 17)")
    if not bool(torch.isfinite(out).all().item()):
        raise DatasetError("model output contains non-finite logits")
    return out.detach().to("cpu").tolist()


def _by_context(examples: Sequence[Any], predictions: Sequence[Sequence[float]]) -> dict[str, dict[str, Any]]:
    result = {}
    for context_id in sorted({e.context_id for e in examples}):
        idx = [i for i, e in enumerate(examples) if e.context_id == context_id]
        summary = analyze_predictions([examples[i] for i in idx], [predictions[i] for i in idx])
        summary["trajectory_groups"] = len({examples[i].trajectory_sha256 for i in idx})
        result[context_id] = summary
    return result


def _independent_forage_reference(logits: Sequence[float], mask: Sequence[bool]) -> tuple[int, ...]:
    """Second implementation of the forage rule using exact ``Fraction`` arithmetic after ``math.exp``."""
    vals = [float(v) for v in logits]
    allowed = [i for i in range(6) if mask[i]]
    order = sorted(allowed, key=lambda i: (-vals[i], i))
    m = sum(1 for i in allowed if vals[i] > 0)
    k = min(3, max(1, m), len(allowed))
    chosen = order[:k]
    peak = max(vals[i] for i in chosen)
    w = {i: Fraction(repr(math.exp(vals[i] - peak))) for i in chosen}
    total = sum(w.values(), Fraction(0))
    ideal = {i: Q_TOTAL * w[i] / total for i in chosen}
    floors = {i: ideal[i].numerator // ideal[i].denominator for i in chosen}
    left = Q_TOTAL - sum(floors.values())
    for i in sorted(chosen, key=lambda j: (-(ideal[j] - floors[j]), j))[:left]:
        floors[i] += 1
    out = [0] * 6
    for i, v in floors.items():
        out[i] = v
    return tuple(out)


def _pipeline_checks(examples: Sequence[Any], predictions_by_split: dict[str, list[list[float]]]) -> dict[str, Any]:
    all_pred = [p for values in predictions_by_split.values() for p in values]
    shape_ok = all(len(p) == 17 for p in all_pred)
    finite = all(math.isfinite(v) for p in all_pred for v in p)
    float32_exact = all(PolicyLogitsV2(list(p)).values == tuple(p) for p in all_pred)
    mask_ok = all(e.mask == get_action_mask(ObservationV2(e.observation)) for e in examples)
    ref_agree = sum(
        _independent_forage_reference(e.teacher_logits[3:9], e.mask.forage_target) == e.teacher_action.forage_share_q
        for e in examples
    )
    return {
        "prediction_rows": len(all_pred),
        "outputs_have_17_values": shape_ok,
        "outputs_finite": finite,
        "outputs_are_exact_float32_values": float32_exact,
        "stored_masks_equal_rederived_masks": mask_ok,
        "independent_forage_reference_agrees_with_teacher_labels": {
            "agree": ref_agree, "of": len(examples),
        },
        "teacher_oracle_roundtrip": teacher_oracle_roundtrip(examples),
    }


def _compare_with_reported(reported: dict[str, Any], split_summaries: dict[str, Any]) -> dict[str, Any]:
    """Compare recomputed headline numbers with the frozen FOR-37 ``report.json`` (CPU torch vs the original GPU)."""
    out: dict[str, Any] = {}
    trained = reported.get("imitation", {}).get("trained", {})
    for split, old in trained.items():
        new = split_summaries[split]
        out[split] = {
            "reported_logit_mse": old["teacher_logit_mse"],
            "recomputed_logit_mse": new["logit_mse"]["all"],
            "abs_diff": abs(old["teacher_logit_mse"] - new["logit_mse"]["all"]),
            "reported_forage_exact": old["forage_share_exact_accuracy"],
            "recomputed_forage_exact": new["action_accuracy"]["forage_share_exact"],
            "reported_full_action": old["exact_action_accuracy"],
            "recomputed_full_action": new["action_accuracy"]["full_action"],
            "reported_lay": old["lay_accuracy"],
            "recomputed_lay": new["action_accuracy"]["lay"],
            "reported_army": old["army_accuracy"],
            "recomputed_army": new["action_accuracy"]["army"],
            "reported_avoid": old["avoid_center_accuracy"],
            "recomputed_avoid": new["action_accuracy"]["avoid_center_unconditional"],
        }
    return out


def diagnose_quality(
    model_dir: str | Path,
    dataset_path: str | Path,
    *,
    device: str = "auto",
    report_path: str | Path | None = None,
    include_rollout: bool = False,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Read-only forensic analysis of a frozen checkpoint against its bound v2 dataset (no training)."""
    from .checkpoint import load_checkpoint
    from .quality import _checkpoint_quality_data
    from .quality_dataset import load_quality_dataset
    from .model import PolicyMLP, configure_determinism, torch
    from .quality_dataset import QUALITY_UNTRAINED_SEED

    dataset_path = Path(dataset_path)
    dataset_manifest, examples = load_quality_dataset(dataset_path)
    model, checkpoint_manifest, resolved_device = load_checkpoint(model_dir, device)
    _checkpoint_quality_data(checkpoint_manifest, dataset_manifest)  # checkpoint <-> dataset bind, fail closed

    splits = ("train", "validation", "heldout")
    by_split = {s: [e for e in examples if e.split == s] for s in splits}
    preds = {s: predict_logits(model, rows, resolved_device) for s, rows in by_split.items()}
    split_summaries = {s: analyze_predictions(by_split[s], preds[s]) for s in splits}
    contexts = {s: _by_context(by_split[s], preds[s]) for s in splits}

    configure_determinism(QUALITY_UNTRAINED_SEED, resolved_device)
    untrained = PolicyMLP().to(resolved_device)
    untrained.eval()
    untrained_pred = predict_logits(untrained, by_split["heldout"], resolved_device)
    untrained_summary = analyze_predictions(by_split["heldout"], untrained_pred)

    reported = None
    if report_path is None:
        candidate = dataset_path.with_name("report.json")
        report_path = candidate if candidate.is_file() else None
    if report_path is not None:
        reported = json.loads(Path(report_path).read_text(encoding="utf-8"))

    artifacts_dir = dataset_path.parent
    teacher_all_noise = noise_sensitivity(examples)
    checkpoint_training = checkpoint_manifest.get("training", {})
    epoch_losses = checkpoint_training.get("epoch_teacher_logit_mse", [])
    train_n = len(by_split["train"])
    batch_size = checkpoint_training.get("batch_size", 128)
    steps_per_epoch = math.ceil(train_n / batch_size) if batch_size else None

    report: dict[str, Any] = {
        "schema": DIAGNOSTICS_SCHEMA,
        "protocol": "docs/ml_for38_protocol.md",
        "classification": "read-only forensic analysis of the frozen FOR-37 checkpoint; no retraining, no decoder change",
        "bins": {
            "tv_near_thresholds": list(TV_NEAR_THRESHOLDS),
            "small_share_tv": SMALL_SHARE_TV,
            "near_zero_bands": list(NEAR_ZERO_BANDS),
            "noise_sigmas": list(NOISE_SIGMAS),
            "noise_repeats": NOISE_REPEATS,
            "noise_seed": NOISE_SEED,
        },
        "provenance": {
            "dataset_sha256": dataset_manifest["dataset_sha256"],
            "weights_sha256": checkpoint_manifest["weights"]["sha256"],
            "checkpoint_dataset_bound": True,
            "device": str(resolved_device),
            "torch": torch.__version__,
            "frozen_artifacts": verify_frozen_artifacts(artifacts_dir),
            "examples": len(examples),
        },
        "training_facts": {
            "train_examples": train_n,
            "epochs": checkpoint_training.get("epochs"),
            "batch_size": batch_size,
            "optimizer_steps_per_epoch": steps_per_epoch,
            "optimizer_steps_total": checkpoint_manifest.get("performance", {}).get("optimizer_steps"),
            "epoch_teacher_logit_mse": epoch_losses,
            "learning_rate": checkpoint_training.get("learning_rate"),
        },
        "pipeline_checks": _pipeline_checks(examples, preds),
        "reproduces_reported_metrics": _compare_with_reported(reported, split_summaries) if reported else None,
        "splits": split_summaries,
        "contexts": contexts,
        "untrained_heldout": untrained_summary,
        "mean_logit_baseline": {s_name: mean_logit_baseline(by_split["train"], by_split[s_name]) for s_name in splits},
        "decoder_strictness": {
            "noise_on_teacher_forage_logits": teacher_all_noise,
            "share_sensitivity": share_sensitivity(examples),
        },
    }
    if include_rollout:
        report["rollout"] = rollout_divergence(model, checkpoint_manifest, resolved_device, reported)
    if output is not None:
        _write_atomic(Path(output), (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    return report


# ---------------------------------------------------------------------------------------------------------
# on-policy divergence: what does the teacher say in the states the model itself reaches?
# ---------------------------------------------------------------------------------------------------------

ROLLOUT_SEED = 373_000_001
ROLLOUT_OPPONENTS = ("balanced:v2", "turtle:v2", "rush:v2")


class _ShadowController:
    """Plays the model, and asks the teacher what it would do on the same observation (no RunState access)."""

    def __init__(self, policy: Any, stream_id: str, teacher: Any, log: list[dict[str, Any]]) -> None:
        from ..rng import new_stream

        self.policy, self.stream_id, self.teacher, self.log = policy, stream_id, teacher, log
        self._scratch = new_stream(0, "policy:diagnostic-shadow:0")

    def act(self, tick: int, observation: ObservationV2, run: Any):
        mask = get_action_mask(observation)
        logits = PolicyLogitsV2.coerce(self.policy(observation, mask, run.stream(self.stream_id)))
        action = decode_action_logits(logits, mask)
        validated = validate_action(observation, mask, action)
        teacher_logits = PolicyLogitsV2.coerce(self.teacher(observation, mask, self._scratch))
        teacher_action = decode_action_logits(teacher_logits, mask)
        pair = forage_pair_metrics(action.forage_share_q, teacher_action.forage_share_q)
        self.log.append({
            "tick": tick,
            "lay_agree": action.lay_order == teacher_action.lay_order,
            "army_agree": action.army_choice == teacher_action.army_choice,
            "avoid_agree": action.avoid_center == teacher_action.avoid_center,
            "forage_exact": pair["exact"],
            "forage_support_equal": pair["support_equal"],
            "forage_tv": pair["tv"],
            "forage_category": pair["category"],
            "full_agree": action == teacher_action,
            "model_action": [action.lay_order, list(action.forage_share_q), action.army_choice, action.avoid_center],
            "teacher_action": [teacher_action.lay_order, list(teacher_action.forage_share_q),
                               teacher_action.army_choice, teacher_action.avoid_center],
        })
        return validated


def summarize_rollout_log(log: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(log)
    if n == 0:
        return {"decisions": 0}
    first_full = next((row["tick"] for row in log if not row["full_agree"]), None)
    first_nonforage = next((row["tick"] for row in log
                            if not (row["lay_agree"] and row["army_agree"] and row["avoid_agree"])), None)
    return {
        "decisions": n,
        "full_agree_rate": _rate(sum(r["full_agree"] for r in log), n),
        "lay_agree_rate": _rate(sum(r["lay_agree"] for r in log), n),
        "army_agree_rate": _rate(sum(r["army_agree"] for r in log), n),
        "avoid_agree_rate": _rate(sum(r["avoid_agree"] for r in log), n),
        "forage_exact_rate": _rate(sum(r["forage_exact"] for r in log), n),
        "forage_support_equal_rate": _rate(sum(r["forage_support_equal"] for r in log), n),
        "forage_tv_mean": statistics.fmean(r["forage_tv"] for r in log),
        "first_divergence_tick_full_action": first_full,
        "first_divergence_tick_lay_army_avoid": first_nonforage,
    }


def rollout_divergence(model: Any, checkpoint_manifest: dict[str, Any], device: Any,
                       reported: dict[str, Any] | None = None, *, seed: int = ROLLOUT_SEED,
                       opponents: Sequence[str] = ROLLOUT_OPPONENTS, sides: Sequence[int] = (0, 1),
                       max_ticks: int | None = None) -> dict[str, Any]:
    """Replay the frozen model against baselines and compare each decision with Balanced v2 on the same state."""
    from ..baselines import baseline_spec
    from ..match import run_episode
    from ..rng import policy_stream_id
    from ..tick import PolicyController
    from .model import MLPPolicy
    from .quality_dataset import quality_config

    config = quality_config("classic")
    ticks = config.t_max if max_ticks is None else max_ticks
    policy = MLPPolicy(model, device)
    stream = policy_stream_id(f"mlp-{checkpoint_manifest['weights']['sha256'][:16]}", 0)
    teacher = baseline_spec("balanced:v2").build()
    reported_index = {}
    if reported:
        for match in reported.get("matches", {}).get("all", []):
            reported_index[(match["seed"], match["model_side"], match["opponent"])] = match
    games = []
    for opponent_id in opponents:
        spec = baseline_spec(opponent_id)
        for side in sides:
            log: list[dict[str, Any]] = []
            shadow = _ShadowController(policy, stream, teacher, log)
            opponent = PolicyController(spec.build(), policy_stream_id(spec.policy_id, 1))
            controllers = (shadow, opponent) if side == 0 else (opponent, shadow)
            result = run_episode(config, seed, controllers[0], controllers[1], max_ticks=ticks)
            old = reported_index.get((seed, side, spec.full_id))
            games.append({
                "opponent": spec.full_id,
                "model_side": side,
                "terminal_reason": result.reason,
                "final_tick": result.final_tick,
                "state_sha256": result.state_hash,
                "reported_state_sha256": None if old is None else old["state_sha256"],
                "reproduces_reported_match": None if old is None else old["state_sha256"] == result.state_hash,
                "summary": summarize_rollout_log(log),
                "decisions": log,
            })
    return {"seed": seed, "map_id": "classic", "games": games}


__all__ = [
    "DIAGNOSTICS_SCHEMA", "FROZEN_FOR37_SHA256", "analyze_predictions", "diagnose_quality", "file_sha256",
    "forage_continuous_probs", "forage_pair_metrics", "mean_logit_baseline", "noise_sensitivity",
    "predict_logits", "prediction_variability",
    "rollout_divergence", "share_sensitivity", "summarize_rollout_log", "teacher_oracle_roundtrip",
    "total_variation", "verify_frozen_artifacts",
]
