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


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
