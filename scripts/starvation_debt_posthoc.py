"""FOR-25 post-hoc descriptive tables (NOT part of the preregistered verdict).

Reads only ``results/starvation_debt/matches.jsonl`` and ``summary.json`` (no engine run) and writes
``results/starvation_debt_posthoc/posthoc.json``. These tables were produced after the primary summary had been
inspected; they explain the preregistered numbers and must not be used to change a threshold or the verdict.

    python scripts/starvation_debt_posthoc.py [--out results/starvation_debt_posthoc/posthoc.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from formicmind import starvation_debt as sd  # noqa: E402

VARIANTS = sd.VARIANTS


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def policy_side(row: dict, policy: str) -> dict:
    return row["sides"]["first"] if row["policies"][0] == policy else row["sides"]["second"]


def build(rows: list[dict]) -> dict:
    arms = {v: [r for r in rows if r["variant"] == v] for v in VARIANTS}
    a0_blocks = sd._a0_blocks(arms["A0"])
    index = {v: {(r["map_id"], r["matchup"], r["seed"], r["seat"]): r for r in arms[v]} for v in VARIANTS}

    # A. which policy carries the zombie windows (distinct A0 blocks, both seat rows of the representative seed)
    by_policy: dict[str, dict] = defaultdict(lambda: {"windows": 0, "blocks": [], "chronic_windows": 0})
    zombie_blocks = []
    for block in a0_blocks:
        rep = sd._representative(block)
        for row in rep:
            for key in ("first", "second"):
                side = row["sides"][key]
                rec = by_policy[side["policy"]]
                rec["windows"] += side["flags"]["zombie"]
                rec["chronic_windows"] += side["flags"]["chronic"]
                if side["flags"]["zombie"]:
                    cell = f"{block['map_id']}|{block['matchup']}|seed{min(block['seeds'])}|{row['seat']}"
                    if cell not in rec["blocks"]:
                        rec["blocks"].append(cell)
        if block["zombie"]:
            zombie_blocks.append(block)

    # B. the 19 zombie blocks: fate of the zombie colony and of the match under every variant
    detail = []
    for block in zombie_blocks:
        rep = sd._representative(block)
        for row in rep:
            for key in ("first", "second"):
                side = row["sides"][key]
                if not side["flags"]["zombie"]:
                    continue
                k = (row["map_id"], row["matchup"], row["seed"], row["seat"])
                entry = {
                    "cell": f"{row['map_id']}|{row['matchup']}|seed{row['seed']}|{row['seat']}",
                    "zombie_policy": side["policy"], "zombie_windows": side["flags"]["zombie"], "variants": {},
                }
                for v in VARIANTS:
                    r = index[v][k]
                    s = policy_side(r, side["policy"])
                    entry["variants"][v] = {
                        "reason": r["terminal"]["reason"], "terminal_tick": r["terminal"]["terminal_tick"],
                        "adults_end": s["adults_end"], "adults_best": s["adults_best"],
                        "starvation_deaths": s["starvation_deaths"], "lay_ok": s["lay_ok"], "hatch": s["hatch"],
                        "queen_alive": s["queen_alive"], "zombie_windows": s["flags"]["zombie"],
                        "chronic_windows": s["flags"]["chronic"], "forgiven": s["forgiven"], "missing": s["missing"],
                    }
                detail.append(entry)

    # C. spill-over: blocks that have no zombie window under A0
    spill = {}
    plain_blocks = [b for b in a0_blocks if not b["zombie"]]
    for v in VARIANTS[1:]:
        differs = changed = 0
        trans = Counter()
        for block in plain_blocks:
            for row in sd._representative(block):
                k = (row["map_id"], row["matchup"], row["seed"], row["seat"])
                other = index[v][k]
                differs += int(other["events_trace_sha256"] != row["events_trace_sha256"])
                before, after = row["terminal"]["reason"], other["terminal"]["reason"]
                changed += int(before != after)
                trans[f"{before}->{after}"] += 1
        spill[v] = {"rows": 2 * len(plain_blocks), "trajectory_differs": differs, "outcome_changed": changed,
                    "transitions": dict(sorted(trans.items()))}

    # D. starvation per colony-tick and chronic windows by policy and arm (distinct representative rows)
    per_policy: dict[str, dict] = {}
    for v in VARIANTS:
        reps = [r for b in sd._group_blocks(arms[v]) for r in sd._representative(b)]
        acc: dict[str, Counter] = defaultdict(Counter)
        for r in reps:
            for key in ("first", "second"):
                s = r["sides"][key]
                a = acc[s["policy"]]
                a["colony_ticks"] += r["terminal"]["state_tick"]
                a["starvation_deaths"] += s["starvation_deaths"]
                a["chronic_windows"] += s["flags"]["chronic"]
                a["zombie_windows"] += s["flags"]["zombie"]
                a["missing"] += s["missing"]
                a["forgiven"] += s["forgiven"]
                a["repaid"] += s["repaid"]
                a["lay_ok"] += s["lay_ok"]
                a["lay_rejected"] += s["lay_rejected"]
                a["adults_end"] += s["adults_end"]
                a["queens_alive"] += int(s["queen_alive"])
                a["colonies"] += 1
        per_policy[v] = {p: dict(c) for p, c in sorted(acc.items())}

    # E. mirror/seat sanity: how many distinct blocks per arm are heterogeneous w.r.t. the seat rows
    return {
        "note": "post-hoc descriptive tables; not used by the preregistered verdict",
        "zombie_windows_by_policy_in_A0": {p: v for p, v in sorted(by_policy.items())},
        "a0_zombie_block_detail": detail, "spillover_into_blocks_without_a0_zombies": spill,
        "per_policy_by_arm_distinct_rows": per_policy,
        "a0_zombie_blocks": [
            {"cell": f"{b['map_id']}|{b['matchup']}", "seeds_in_block": len(b["seeds"]),
             "deterministic_matchup": "random" not in b["matchup"], "zombie_windows": b["zombie_windows"]}
            for b in zombie_blocks
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "results/starvation_debt/matches.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "results/starvation_debt_posthoc/posthoc.json"))
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(load_rows(Path(args.input))), ensure_ascii=False, indent=1, sort_keys=True)
                   + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
