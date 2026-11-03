"""FOR-35 POST-HOC tables (not part of the preregistered verdict).

Written after the preregistered summary was seen. Reads the stored rows of ``results/starvation_confirm`` only and
writes ``results/starvation_confirm_posthoc/posthoc.json``. Every number here is exploratory.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from formicmind import starvation_confirm as sc  # noqa: E402


PREVIEWED_GROUP = ("R0-T0", "turtle", "raider", 20270501)  # the schedule group looked at before the full run (E2.1)


def sensitivity_without_previewed_group(source: Path) -> dict:
    """POST-HOC ONLY. Recompute the preregistered analyses with every row of the previewed (map, pair, seed) removed.

    Both seats and all five variants of that seed are dropped (a strictly larger exclusion than the single
    ``first_A`` group that was previewed), because clusters need complete seat pairs. The integrity gate (complete
    4800-row schedule) is deliberately not applied: this is a sensitivity table, not a replacement verdict. The formal
    verdict and the 4800 published rows are untouched."""
    rows = [json.loads(line) for line in (source / "matches.jsonl").read_text(encoding="utf-8").splitlines()]
    demo = json.loads((source / "minimal_demo.json").read_text(encoding="utf-8"))
    anchors = json.loads((source / "anchors.json").read_text(encoding="utf-8"))["anchors"]
    original = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    pair = f"{PREVIEWED_GROUP[1]}-{PREVIEWED_GROUP[2]}"
    kept_rows = [r for r in rows if (r["map_id"], r["first"], r["second"], r["seed"]) != PREVIEWED_GROUP]
    kept_anchors = [a for a in anchors
                    if a["row_id"].split("|")[1:4] != [PREVIEWED_GROUP[0], pair, str(PREVIEWED_GROUP[3])]]
    summary = sc.build_summary(kept_rows, demo, kept_anchors)
    exposure, variants = summary["exposure"], summary["variants"]
    passing = {v: bool(exposure["E"] > 0 and all(variants[v]["criteria"].values())) for v in sc.CANDIDATES}
    collateral = {v: (variants[v]["e1"]["x_changed"], variants[v]["e1"]["t_loss"] - variants[v]["e1"]["t_loss_a0"])
                  for v in sc.CANDIDATES}
    decision = sc.decide(integrity_ok=True, k1=summary["stage_m"]["k1"], exposure=exposure["E"],
                         spread_ok=exposure["spread_ok"], n_clusters=summary["design"]["a0_clusters"],
                         passing=passing, collateral=collateral)
    keep = ("classes", "viable", "criteria")
    return {
        "schema": "formicmind.starvation-confirm-posthoc-sensitivity/v1", "status": "post-hoc only; not a verdict",
        "excluded": {"map": PREVIEWED_GROUP[0], "pair": pair, "seed": PREVIEWED_GROUP[3],
                     "rows_removed": len(rows) - len(kept_rows), "anchors_removed": len(anchors) - len(kept_anchors),
                     "note": "both seats and all variants of the previewed seed"},
        "integrity_gate_applied": False,
        "rows_kept": len(kept_rows), "a0_clusters": summary["design"]["a0_clusters"],
        "exposure": {"E": exposure["E"], "clusters": [c["cluster"] for c in exposure["clusters"]],
                     "seeds_in_exposure_clusters": {c["cluster"]: c["seeds"] for c in exposure["clusters"]},
                     "spread_ok": exposure["spread_ok"]},
        "variants": {v: {k: variants[v][k] for k in keep} | {"e1": {k: variants[v]["e1"][k] for k in (
            "t_loss", "t_loss_a0", "collapses", "collapses_a0", "x_changed", "n_non_exposed")}}
            for v in sc.CANDIDATES},
        "decision_if_the_gate_were_ignored": decision,
        "same_as_formal": {
            "E": exposure["E"] == original["exposure"]["E"],
            "exposure_cells": [(c["map"], c["pair"]) for c in exposure["clusters"]]
            == [(c["map"], c["pair"]) for c in original["exposure"]["clusters"]],
            "classes": all(variants[v]["classes"] == original["variants"][v]["classes"] for v in sc.CANDIDATES),
            "criteria": all(variants[v]["criteria"] == original["variants"][v]["criteria"] for v in sc.CANDIDATES),
            "verdict": decision["verdict"] == original["decision"]["verdict"]},
    }


def main(source: Path, out: Path) -> None:
    rows = [json.loads(line) for line in (source / "matches.jsonl").read_text(encoding="utf-8").splitlines()]
    by_variant = {v: {} for v in sc.VARIANTS}
    for r in rows:
        by_variant[r["variant"]][sc.row_key(r)] = r
    a0 = by_variant["A0"]
    clusters = sc.build_clusters(a0.values())
    result: dict = {"schema": "formicmind.starvation-confirm-posthoc/v1", "status": "post-hoc, exploratory",
                    "source_rows_sha256": sc._sha(sc._lf((source / "matches.jsonl").read_bytes()))}

    # 1. Deficit-stagnant A0 clusters with and without the forgiveness attribution (the preregistered gate).
    table = []
    for c in clusters:
        ks = c.keys()
        rs = [a0[k] for k in ks]
        if not any(r["stagnation"]["persistent_deficit"] or r["stagnation"]["persistent"] for r in rs):
            continue
        sd_m = max(r["stagnation"]["sd_missing"] for r in rs)
        sd_f = max(r["stagnation"]["sd_forgiven"] for r in rs)
        table.append({
            "cluster": c.cluster_id, "pair": "-".join(c.pair), "seeds": len(c.seeds),
            "reasons": sorted({r["terminal"]["reason"] for r in rs}),
            "stagnant": max(r["stagnation"]["stagnant"] for r in rs),
            "stagnant_deficit": max(r["stagnation"]["stagnant_deficit"] for r in rs),
            "stagnant_sat": max(r["stagnation"]["stagnant_sat"] for r in rs),
            "persistent_deficit": any(r["stagnation"]["persistent_deficit"] for r in rs),
            "forgiveness_supported": any(r["stagnation"]["forgiveness_supported"] for r in rs),
            "sd_missing": sd_m, "sd_forgiven": sd_f})
    result["a0_stagnant_clusters"] = table

    # 2. Sensitivity: what the classes would be if the forgiveness attribution were dropped (all persistent-deficit rows).
    sens = {}
    loose = [c for c in clusters if any(a0[k]["stagnation"]["persistent_deficit"] for k in c.keys())]
    for v in sc.CANDIDATES:
        classes = Counter()
        for c in loose:
            worst = None
            for k in c.keys():
                if a0[k]["stagnation"]["persistent_deficit"]:
                    cls = sc.resolution_class(a0[k], by_variant[v][k])
                    if worst is None or sc.CLASS_ORDER.index(cls) > sc.CLASS_ORDER.index(worst):
                        worst = cls
            classes[worst] += 1
        sens[v] = {k: classes[k] for k in sc.CLASS_ORDER}
    result["sensitivity_without_forgiveness_gate"] = {"clusters": len(loose), "classes": sens}

    # 3. Where do the variants differ from A0 at all (rows with a different trajectory), by pair.
    diff_by_pair = {}
    for v in sc.CANDIDATES:
        cnt = Counter()
        for k, r in by_variant[v].items():
            if r["trajectory_sha256"] != a0[k]["trajectory_sha256"]:
                cnt["-".join((k[1], k[2]))] += 1
        diff_by_pair[v] = dict(sorted(cnt.items()))
    result["rows_with_different_trajectory_by_pair"] = diff_by_pair

    # 4. Terminal reasons by variant for Turtle pairs (raw rows).
    turtle_terminals = {}
    for v in sc.VARIANTS:
        cnt = Counter()
        for r in by_variant[v].values():
            if "turtle" in (r["first"], r["second"]):
                cnt[r["terminal"]["reason"]] += 1
        turtle_terminals[v] = dict(sorted(cnt.items()))
    result["turtle_rows_terminal_reasons_raw"] = turtle_terminals

    out.mkdir(parents=True, exist_ok=True)
    (out / "posthoc.json").write_bytes(sc._dump_json(result))
    (out / "sensitivity_without_previewed_group.json").write_bytes(
        sc._dump_json(sensitivity_without_previewed_group(source)))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
