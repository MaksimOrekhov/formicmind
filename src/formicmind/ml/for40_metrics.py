"""FOR-40 decision-level metrics shared by validation selection, held-out evaluation and training curves."""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence
from typing import Any

from ..actions import ActionV2, validate_action
from ..decoder import PolicyLogitsV2, decode_action_logits
from ..observation import ObservationV2
from .diagnostics import aggregate_forage, forage_pair_metrics
from .for40 import army_group

LOGIT_GROUPS = {"lay": (0, 3), "forage": (3, 9), "army": (9, 16), "avoid": (16, 17)}
LAY_NAMES = ("none", "worker", "soldier")
ARMY_GROUPS = ("garrison", "defend", "raid")


def conditional_avoid(example: Any) -> bool:
    """Both ``avoid_center`` values are allowed for the teacher's army choice."""
    return tuple(example.mask.army_avoid_center[example.teacher_action.army_choice]) == (True, True)


def train_reference(train_examples: Sequence[Any]) -> dict[str, Any]:
    """Observation-free references fitted on train only: mean teacher logits and per-field majority."""
    n = len(train_examples)
    mean_logits = [statistics.fmean(e.teacher_logits[i] for e in train_examples) for i in range(17)]
    lay = Counter(e.teacher_action.lay_order for e in train_examples)
    army = Counter(e.teacher_action.army_choice for e in train_examples)
    avoid = Counter(e.teacher_action.avoid_center for e in train_examples if conditional_avoid(e))
    forage = Counter(e.teacher_action.forage_share_q for e in train_examples)
    return {
        "examples": n,
        "mean_logits": mean_logits,
        "lay_majority": lay.most_common(1)[0][0],
        "army_majority": army.most_common(1)[0][0],
        "avoid_conditional_majority": avoid.most_common(1)[0][0] if avoid else False,
        "forage_majority": list(forage.most_common(1)[0][0]),
    }


def _decode(example: Any, prediction: Sequence[float]) -> ActionV2 | None:
    try:
        action = decode_action_logits(PolicyLogitsV2(list(prediction)), example.mask)
        return validate_action(ObservationV2(example.observation), example.mask, action).action
    except (ValueError, OverflowError, TypeError):
        return None


def records_from_logits(examples: Sequence[Any], predictions: Sequence[Sequence[float]]) -> list[dict[str, Any]]:
    records = []
    for example, prediction in zip(examples, predictions, strict=True):
        record = _record(example, _decode(example, prediction))
        record["sq"] = {name: sum((float(prediction[i]) - example.teacher_logits[i]) ** 2 for i in range(lo, hi))
                        for name, (lo, hi) in LOGIT_GROUPS.items()}
        records.append(record)
    return records


def records_from_actions(examples: Sequence[Any], actions: Sequence[ActionV2 | None]) -> list[dict[str, Any]]:
    return [_record(example, action) for example, action in zip(examples, actions, strict=True)]


def _record(example: Any, action: ActionV2 | None) -> dict[str, Any]:
    teacher = example.teacher_action
    record: dict[str, Any] = {
        "legal": action is not None, "action": action, "teacher": teacher,
        "conditional_avoid": conditional_avoid(example),
        "map_id": example.map_id, "opponent_id": example.opponent_id, "source": example.source,
        "pair": None if action is None else forage_pair_metrics(action.forage_share_q, teacher.forage_share_q),
    }
    return record


def _macro_recall(correct: Counter, total: Counter) -> tuple[float | None, dict[str, float]]:
    recalls = {str(k): correct[k] / total[k] for k in total if total[k]}
    return (statistics.fmean(recalls.values()) if recalls else None), recalls


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate decision records. Classes absent from the subset are excluded from macro averages."""
    n = len(records)
    if n == 0:
        return {"examples": 0}
    legal = [r for r in records if r["legal"]]
    lay_total, lay_ok, army_total, army_ok = Counter(), Counter(), Counter(), Counter()
    lay_conf: Counter = Counter()
    army_conf: Counter = Counter()
    avoid_total = avoid_ok = 0
    for r in records:
        teacher = r["teacher"]
        lay_total[LAY_NAMES[teacher.lay_order]] += 1
        group = army_group(teacher.army_choice)
        army_total[group] += 1
        action = r["action"]
        if action is None:
            lay_conf[(LAY_NAMES[teacher.lay_order], "illegal")] += 1
            army_conf[(group, "illegal")] += 1
            if r["conditional_avoid"]:
                avoid_total += 1
            continue
        lay_conf[(LAY_NAMES[teacher.lay_order], LAY_NAMES[action.lay_order])] += 1
        army_conf[(group, army_group(action.army_choice))] += 1
        lay_ok[LAY_NAMES[teacher.lay_order]] += action.lay_order == teacher.lay_order
        army_ok[group] += army_group(action.army_choice) == group
        if r["conditional_avoid"]:
            avoid_total += 1
            avoid_ok += action.avoid_center == teacher.avoid_center
    lay_macro, lay_recall = _macro_recall(lay_ok, lay_total)
    army_macro, army_recall = _macro_recall(army_ok, army_total)
    pairs = [r["pair"] for r in legal]
    forage = aggregate_forage(pairs) if pairs else {"examples": 0}
    exact_choice = sum(r["action"].army_choice == r["teacher"].army_choice for r in legal)
    lay_acc = sum(lay_ok.values()) / n
    distinct_lay = len({r["action"].lay_order for r in legal})
    distinct_army = len({r["action"].army_choice for r in legal})
    teacher_distinct_lay = len({r["teacher"].lay_order for r in records})
    teacher_distinct_army = len({r["teacher"].army_choice for r in records})
    support_equal = forage.get("support_equal_rate")
    tv_mean = forage.get("tv_mean")
    parts = [lay_macro, army_macro, support_equal, None if tv_mean is None else 1.0 - tv_mean]
    composite = statistics.fmean(parts) if all(p is not None for p in parts) else None
    result: dict[str, Any] = {
        "examples": n,
        "legal_rate": len(legal) / n,
        "lay": {"accuracy": lay_acc, "macro_recall": lay_macro, "recall": lay_recall,
                "confusion": {f"teacher_{a}_pred_{b}": c for (a, b), c in sorted(lay_conf.items())},
                "teacher_counts": dict(lay_total)},
        "army": {"accuracy_group": sum(army_ok.values()) / n, "accuracy_exact_choice": exact_choice / n,
                 "group_macro_recall": army_macro, "group_recall": army_recall,
                 "confusion_groups": {f"teacher_{a}_pred_{b}": c for (a, b), c in sorted(army_conf.items())},
                 "teacher_counts": dict(army_total)},
        "avoid_conditional": {"examples": avoid_total, "accuracy": avoid_ok / avoid_total if avoid_total else None},
        "forage": forage,
        "distinct_decoded": {
            "lay": distinct_lay, "army_choice": distinct_army,
            "lay_army_avoid": len({(r["action"].lay_order, r["action"].army_choice, r["action"].avoid_center) for r in legal}),
            "teacher_lay": teacher_distinct_lay, "teacher_army_choice": teacher_distinct_army,
        },
        "collapse": distinct_lay < min(2, teacher_distinct_lay) or distinct_army < min(2, teacher_distinct_army),
        "composite": composite,
    }
    if all("sq" in r for r in records):
        mse = {name: sum(r["sq"][name] for r in records) / (n * (hi - lo)) for name, (lo, hi) in LOGIT_GROUPS.items()}
        mse["all"] = sum(sum(r["sq"].values()) for r in records) / (n * 17)
        result["mse"] = mse
    return result


def majority_actions(examples: Sequence[Any], reference: dict[str, Any]) -> list[ActionV2 | None]:
    """Observation-free majority predictor; fields are compared, not validated against the mask."""
    forage = tuple(reference["forage_majority"])
    return [ActionV2(reference["lay_majority"], forage, reference["army_majority"], bool(reference["avoid_conditional_majority"]))
            for _ in examples]


def majority_summary(examples: Sequence[Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Majority baseline accuracies computed per field (legality of the combination is irrelevant here)."""
    n = len(examples)
    forage = tuple(reference["forage_majority"])
    actions = majority_actions(examples, reference)
    records = records_from_actions(examples, actions)
    base = summarize(records)
    base["lay"]["accuracy"] = sum(e.teacher_action.lay_order == reference["lay_majority"] for e in examples) / n
    cond = [e for e in examples if conditional_avoid(e)]
    base["avoid_conditional"] = {
        "examples": len(cond),
        "accuracy": (sum(e.teacher_action.avoid_center == bool(reference["avoid_conditional_majority"]) for e in cond) / len(cond))
        if cond else None,
    }
    base["army"]["accuracy_exact_choice"] = sum(e.teacher_action.army_choice == reference["army_majority"] for e in examples) / n
    base["army"]["accuracy_group"] = sum(army_group(e.teacher_action.army_choice) == army_group(reference["army_majority"])
                                         for e in examples) / n
    base["forage_majority_tuple"] = list(forage)
    return base


def stratify(examples: Sequence[Any], records: Sequence[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        value = record[key]
        groups.setdefault("random:v2" if key == "opponent_id" and record["source"] == "H_random" else value, []).append(record)
    return {name: summarize(group) for name, group in sorted(groups.items())}


def compact(summary: dict[str, Any]) -> dict[str, Any]:
    """Short view used in curves and tables."""
    forage = summary.get("forage", {})
    return {
        "examples": summary["examples"], "composite": summary.get("composite"),
        "lay_macro_recall": summary["lay"]["macro_recall"], "lay_accuracy": summary["lay"]["accuracy"],
        "army_group_macro_recall": summary["army"]["group_macro_recall"],
        "army_accuracy_group": summary["army"]["accuracy_group"],
        "avoid_conditional_accuracy": summary["avoid_conditional"]["accuracy"],
        "forage_support_equal": forage.get("support_equal_rate"), "forage_top1": forage.get("top1_match_rate"),
        "forage_tv_mean": forage.get("tv_mean"), "forage_tv_median": forage.get("tv_median"),
        "distinct_decoded": summary["distinct_decoded"], "collapse": summary["collapse"],
        "mse": summary.get("mse"),
    }


__all__ = [
    "ARMY_GROUPS", "LAY_NAMES", "LOGIT_GROUPS", "compact", "conditional_avoid", "majority_actions", "majority_summary",
    "records_from_actions", "records_from_logits", "stratify", "summarize", "train_reference",
]
