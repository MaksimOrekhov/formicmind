"""Command line of Ecosystem v1: ``python -m formicmind.eco {map,verify,bench,sweep,run,sim-sweep,interaction-sweep,bench-run}``.

* ``map``    generate one shared world and print stable JSON (config, map, optional fairness report and tick-0 state summary)
* ``verify`` re-check a saved ``map`` document: strict schema, hashes, regeneration from the seed, report
* ``bench``  latency of map generation and of the fairness report
* ``sweep``  fairness statistics over many seeds, sizes and scenarios
* ``run``    one headless world: physical cohort travel and food transport; ``--role`` adds scouting, guarding and raiding
             colonies, ``--timeline`` writes the compact event timeline (contacts, discoveries, fights, queen deaths)
* ``sim-sweep`` multi-map, multi-scenario comparison of peaceful diagnostic fixtures
* ``interaction-sweep`` named role assignments (control, mixed roles, four raiders) on the same maps, event statistics only
* ``bench-run`` simulation throughput and traced peak memory

The JSON holds the engine/viewer view of the world (full map). It contains no policy observation and no policy metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from typing import Any

from ..canonical import CanonicalError, canonical_dumps, canonical_loads
from .codec import to_canonical
from .config import SCENARIOS, SIZES, ConfigE1, ConfigError, config_from_canonical, config_hash, default_config
from .fairness import evaluate, report_hash
from .initial import build_world, initial_world_state
from .mapgen import MapGenerationError, generate
from .serialization import compute_map_hash, map_from_canonical, state_hash
from .fixtures import FIXTURE_CHOICES, normalize_fixture
from .simulation import benchmark_simulation, interaction_sweep, run_simulation, sweep_simulations
from .versions import (
    ACTION_VERSION, CLI_SCHEMA, ECO_BALANCE_VERSION, ECO_RULES_VERSION, GENERATOR_ALGORITHM, OBSERVATION_VERSION,
)


def _versions() -> dict:
    return {"eco_rules_version": ECO_RULES_VERSION, "eco_balance_version": ECO_BALANCE_VERSION,
            "observation_version": OBSERVATION_VERSION, "action_version": ACTION_VERSION, "generator": GENERATOR_ALGORITHM}


def _config_from_args(args: argparse.Namespace) -> ConfigE1:
    return default_config(args.colonies, args.scenario, getattr(args, "nodes", None))


def build_document(config: ConfigE1, seed: int, *, report: bool, state: bool) -> dict:
    world = build_world(config, seed)
    doc: dict[str, Any] = {
        "schema": CLI_SCHEMA,
        "versions": _versions(),
        "map_seed": str(seed),
        "config": to_canonical(config),
        "config_hash": config_hash(config),
        "map": to_canonical(world.world_map),
        "map_hash": world.world_map.map_hash,
    }
    if report:
        rep = evaluate(config, world.world_map)
        doc["report"] = rep
        doc["report_hash"] = report_hash(rep)
    if state:
        doc["initial_state"] = _initial_state_summary(world.state)
    return doc


def _initial_state_summary(state) -> dict:
    return {
        "state_hash": state_hash(state),
        "tick": state.tick,
        "colonies": [
            {"colony_id": c.colony_id, "food": c.food,
             "workers": sum(x.count for x in state.cohorts if x.colony_id == c.colony_id and x.caste == "worker"),
             "soldiers": sum(x.count for x in state.cohorts if x.colony_id == c.colony_id and x.caste == "soldier"),
             "known_nodes": len(c.memory.nodes), "visited_nodes": sum(1 for n in c.memory.nodes if n.visited),
             "known_edges": len(c.memory.edges)}
            for c in state.colonies
        ],
        "patch_stock_total": sum(p.stock for p in state.patches),
    }


def dumps(doc: dict, *, pretty: bool) -> str:
    if pretty:
        return json.dumps(doc, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    return canonical_dumps(doc).decode("ascii") + "\n"


def summary_text(doc: dict) -> str:
    rep = doc.get("report")
    c = doc["config"]
    lines = [f"Ecosystem v1 map  seed={doc['map_seed']}  colonies={c['n_colonies']}  scenario={c['scenario']}  "
             f"map_hash={doc['map_hash'][:16]}"]
    if rep:
        n, f, b = rep["counts"], rep["fairness"], rep["resource_budget"]
        r = rep["ranges"]
        lines += [
            f"  graph      nodes {n['nodes']} (junctions {n['junctions']})  edges {n['edges']}  patches {n['patches']} "
            f"{n['patches_by_zone']}  chokepoints {n['chokepoints']}",
            f"  edges      length {r['edge_length']['min']}..{r['edge_length']['max']} ticks  degree {r['degree']['min']}..{r['degree']['max']}",
            f"  nests      nearest-neighbour path {r['nearest_nest_path']['min']}..{r['nearest_nest_path']['max']} ticks  "
            f"closest pair {r['nest_pair_path']['min']}",
            f"  budget     regen {b['total_regen']} food/tick total, {b['regen_per_colony_x1000'] / 1000:.1f} per colony  "
            f"capacity {b['total_capacity']}",
            f"  fairness   near access CV {f['near_access_x1000']['cv_permille'] / 10:.1f}%  shared supply CV "
            f"{f['radius_supply_x1000']['cv_permille'] / 10:.1f}%  attempts {doc['map']['generation']['attempts']}",
            "  checks     " + ("ALL PASSED" if rep["all_checks_passed"] else "FAILED: " + ", ".join(
                k["name"] for k in rep["checks"] if not k["ok"])),
        ]
    return "\n".join(lines) + "\n"


def _write(text: str, out: str | None) -> None:
    if out:
        with open(out, "w", encoding="ascii", newline="\n") as handle:
            handle.write(text)
    else:
        sys.stdout.write(text)


def cmd_map(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    doc = build_document(config, args.seed, report=args.report, state=args.state)
    doc["document_sha256"] = hashlib.sha256(canonical_dumps(doc)).hexdigest()
    _write(dumps(doc, pretty=not args.compact), args.out)
    if args.summary:
        sys.stderr.write(summary_text(doc))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    with open(args.file, "rb") as handle:
        raw = handle.read()
    doc = json.loads(raw.decode("utf-8"))
    required = {"schema", "versions", "map_seed", "config", "config_hash", "map", "map_hash", "document_sha256"}
    optional = {"report", "report_hash", "initial_state"}
    if type(doc) is not dict or not required <= set(doc):
        raise CanonicalError("not an Ecosystem v1 map document (missing keys)")
    keys = set(doc)
    if keys - required - optional:
        raise CanonicalError("map document contains unknown top-level keys")
    if ("report" in doc) != ("report_hash" in doc):
        raise CanonicalError("report and report_hash must be present together")
    claimed = doc["document_sha256"]
    if type(claimed) is not str or claimed != hashlib.sha256(
        canonical_dumps({key: value for key, value in doc.items() if key != "document_sha256"})
    ).hexdigest():
        raise CanonicalError("document_sha256 does not match the document")
    if doc.get("schema") != CLI_SCHEMA:
        raise CanonicalError("not an Ecosystem v1 map document")
    if doc["versions"] != _versions():
        raise CanonicalError("versions do not match this Ecosystem v1 CLI")
    config = config_from_canonical(doc["config"])
    if config_hash(config) != doc["config_hash"]:
        raise CanonicalError("config_hash mismatch")
    world_map = map_from_canonical(doc["map"], config)
    if world_map.map_hash != doc["map_hash"] or world_map.map_hash != compute_map_hash(world_map):
        raise CanonicalError("map_hash mismatch")
    if type(doc["map_seed"]) is not str or not doc["map_seed"].isdigit():
        raise CanonicalError("map_seed must be a decimal string")
    seed = int(doc["map_seed"])
    regenerated = generate(config, seed).world_map
    if regenerated != world_map:
        raise CanonicalError("the map is not what the generator produces for this seed and config")
    if "report" in doc:
        rep = evaluate(config, world_map)
        if canonical_dumps(rep) != canonical_dumps(doc["report"]) or report_hash(rep) != doc["report_hash"]:
            raise CanonicalError("report differs from the recomputed fairness report")
    if "initial_state" in doc:
        expected_state = _initial_state_summary(initial_world_state(config, regenerated))
        if canonical_dumps(expected_state) != canonical_dumps(doc["initial_state"]):
            raise CanonicalError("initial_state differs from the recomputed tick-0 world state")
    sys.stdout.write(f"OK  map_hash={world_map.map_hash}  seed={seed}  colonies={config.n_colonies}  scenario={config.scenario}\n")
    return 0


def _percentile(sorted_values: list[int], q: int) -> int:
    return sorted_values[min(len(sorted_values) - 1, (len(sorted_values) * q + 99) // 100 - 1)]


def cmd_bench(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    gen_us, rep_us, attempts = [], [], []
    for i in range(args.maps):
        seed = args.seed0 + i
        t0 = time.perf_counter_ns()
        result = generate(config, seed)
        t1 = time.perf_counter_ns()
        evaluate(config, result.world_map)
        t2 = time.perf_counter_ns()
        gen_us.append((t1 - t0) // 1000)
        rep_us.append((t2 - t1) // 1000)
        attempts.append(result.attempts)

    def stats(values: list[int]) -> dict:
        s = sorted(values)
        return {"mean": sum(s) // len(s), "median": int(statistics.median(s)), "p95": _percentile(s, 95), "max": s[-1], "min": s[0]}

    doc = {
        "benchmark": "map generation latency (no simulation, not worlds per second)",
        "maps": args.maps, "seed0": args.seed0, "colonies": config.n_colonies, "scenario": config.scenario,
        "nodes": config.map_params.n_nodes, "unit": "microseconds",
        "generation_incl_acceptance_checks_us": stats(gen_us),
        "fairness_report_only_us": stats(rep_us),
        "attempts": {"mean_x100": sum(attempts) * 100 // len(attempts), "max": max(attempts)},
        "platform": {"python": sys.version.split()[0], "platform": sys.platform},
    }
    _write(json.dumps(doc, sort_keys=True, indent=2) + "\n", args.out)
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    rows = []
    for colonies in args.sizes:
        for scenario in args.scenarios:
            config = default_config(colonies, scenario)
            near, rad, atts, choke, fails = [], [], [], 0, 0
            for i in range(args.maps):
                try:
                    result = generate(config, args.seed0 + i)
                except MapGenerationError:
                    fails += 1
                    continue
                rep = evaluate(config, result.world_map)
                near.append(rep["fairness"]["near_access_x1000"]["cv_permille"])
                rad.append(rep["fairness"]["radius_supply_x1000"]["cv_permille"])
                atts.append(result.attempts)
                choke += rep["counts"]["chokepoints"] > 0
            ok = len(atts)
            rows.append({
                "colonies": colonies, "scenario": scenario, "maps": args.maps, "failed": fails,
                "mean_attempts_x100": sum(atts) * 100 // max(1, ok), "max_attempts": max(atts, default=0),
                "near_access_cv_permille": {"median": int(statistics.median(near)), "max": max(near)} if near else {},
                "shared_supply_cv_permille": {"median": int(statistics.median(rad)), "max": max(rad)} if rad else {},
                "maps_with_chokepoint": choke,
            })
    _write(json.dumps({"sweep": rows, "seed0": args.seed0}, sort_keys=True, indent=2) + "\n", args.out)
    return 0 if all(r["failed"] == 0 for r in rows) else 1


def _parse_roles(items: list[str], colonies: int) -> dict[int, str]:
    roles: dict[int, str] = {}
    for item in items:
        colony, sep, name = item.partition("=")
        if not sep or not colony.isdigit():
            raise ValueError(f"--role expects COLONY=NAME, got {item!r}")
        if int(colony) >= colonies:
            raise ValueError(f"--role: colony {colony} does not exist in a world of {colonies}")
        if int(colony) in roles:
            raise ValueError(f"--role: colony {colony} given twice")
        roles[int(colony)] = normalize_fixture(name)
    return roles


def cmd_run(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    roles = _parse_roles(args.role, config.n_colonies)
    want_events = args.timeline is not None or args.events
    result = run_simulation(config, args.seed, args.fixture, ticks=args.ticks,
                            sample_interval=args.sample_every, include_map=not args.omit_map,
                            collect_timeline=not args.no_time_series, roles=roles, include_events=want_events)
    timeline = result.report.get("event_timeline", [])
    if args.timeline is not None:
        text = "".join(line + "\n" for line in timeline)
        if args.timeline == "-":
            sys.stderr.write(text)
        else:
            with open(args.timeline, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
    if not args.events:
        result.report.pop("event_timeline", None)
    _write(json.dumps(result.report, sort_keys=True, indent=None if args.compact else 2,
                      separators=(",", ":") if args.compact else None, ensure_ascii=True) + "\n", args.out)
    if args.summary:
        report = result.report
        signals, final, totals = report["first_signals"], report["final"], report["totals"]
        sys.stderr.write(
            f"Ecosystem simulation seed={args.seed} fixture={report['fixture']} "
            f"colonies={config.n_colonies} scenario={config.scenario} ticks={report['ticks_completed']}\n"
            f"  population adults={final['adults']} maintenance_WE={final['maintenance_we']} "
            f"survivors={sum(c['status'] == 'alive' for c in final['colonies'])}/{config.n_colonies}\n"
            f"  food delivered={totals['delivered']} warehouse={final['food_in_warehouses']} "
            f"patches={final['food_on_patches']} cargo={final['food_in_transit']} "
            f"ledger_delta={report['food_ledger']['difference']}\n"
            f"  first contention={signals['contention_tick']} depletion={signals['patch_depletion_tick']} "
            f"shortfall={signals['upkeep_shortfall_tick']} starvation={signals['starvation_death_tick']}\n"
            f"  performance {report['performance']['ticks_per_second']} ticks/s "
            f"({report['performance']['wall_seconds']} s)\n"
        )
        inter = report["interaction"]
        sys.stderr.write(
            f"  roles={report['roles'] or 'none'} events={inter['event_count']} hash={inter['events_hash'][:16]}\n"
            f"  first contact={inter['first_contact_tick']} pairs in contact={inter['colony_pairs_in_contact']} "
            f"nests found={inter['nests_found']} explored nodes per colony={inter['colonies_explored_nodes']}\n"
            f"  attacks={len(inter['attacks_dispatched'])} combat ticks={inter['combat_ticks']} "
            f"combat deaths={inter['combat_deaths']} queen deaths={[(q['colony'], q['tick']) for q in inter['queen_deaths']]} "
            f"extinct={[(e['colony'], e['tick']) for e in inter['extinctions']]}\n"
        )
    return 0


def cmd_sim_sweep(args: argparse.Namespace) -> int:
    def progress(completed: int, total: int, colonies: int, scenario: str, fixture: str) -> None:
        sys.stderr.write(f"completed {completed}/{total}: {colonies} colonies, {scenario}, {fixture}, "
                         f"{args.seeds} maps × {args.ticks} ticks\n")
        sys.stderr.flush()

    report = sweep_simulations(sizes=tuple(args.sizes), scenarios=tuple(args.scenarios),
                               fixtures=tuple(args.fixtures), seeds=args.seeds, seed0=args.seed0,
                               ticks=args.ticks, progress=progress)
    _write(json.dumps(report, sort_keys=True, indent=2, ensure_ascii=True) + "\n", args.out)
    return 0


def cmd_interaction_sweep(args: argparse.Namespace) -> int:
    def progress(done: int, total: int, name: str, seed: int) -> None:
        sys.stderr.write(f"completed {done}/{total}: {name} seed {seed}\n")
        sys.stderr.flush()

    report = interaction_sweep(colonies=args.colonies, scenario=args.scenario, seeds=args.seeds, seed0=args.seed0,
                               ticks=args.ticks, progress=progress)
    _write(json.dumps(report, sort_keys=True, indent=2, ensure_ascii=True) + "\n", args.out)
    return 0


def cmd_sim_bench(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    report = benchmark_simulation(config, args.seed0, args.fixture, runs=args.runs, ticks=args.ticks)
    _write(json.dumps(report, sort_keys=True, indent=2, ensure_ascii=True) + "\n", args.out)
    return 0


def _add_world_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--colonies", type=int, default=8, choices=SIZES, help="number of colonies (default 8)")
    p.add_argument("--scenario", default="moderate", choices=SCENARIOS, help="resource scenario (default moderate)")
    p.add_argument("--nodes", type=int, default=None, help="node count, 80..120 (default 8*colonies+32 = 96 for 8 colonies)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m formicmind.eco", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    m = sub.add_parser("map", help="generate one map and print stable JSON")
    m.add_argument("--seed", type=int, default=42, help="map seed (u64)")
    _add_world_args(m)
    m.add_argument("--report", action="store_true", help="include the fairness report")
    m.add_argument("--state", action="store_true", help="include a summary and hash of the tick-0 world state")
    m.add_argument("--compact", action="store_true", help="canonical compact JSON instead of indented JSON")
    m.add_argument("--summary", action="store_true", help="print a human-readable summary to stderr")
    m.add_argument("--out", default=None, help="write to a file (LF newlines) instead of stdout")
    m.set_defaults(run=cmd_map)

    v = sub.add_parser("verify", help="re-check a saved map document")
    v.add_argument("file")
    v.set_defaults(run=cmd_verify)

    b = sub.add_parser("bench", help="latency of map generation and of the fairness report")
    _add_world_args(b)
    b.add_argument("--maps", type=int, default=100)
    b.add_argument("--seed0", type=int, default=1000)
    b.add_argument("--out", default=None)
    b.set_defaults(run=cmd_bench)

    s = sub.add_parser("sweep", help="fairness statistics over many seeds, sizes and scenarios")
    s.add_argument("--maps", type=int, default=200)
    s.add_argument("--seed0", type=int, default=100000)
    s.add_argument("--sizes", type=int, nargs="+", default=list(SIZES), choices=SIZES)
    s.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=SCENARIOS)
    s.add_argument("--out", default=None)
    s.set_defaults(run=cmd_sweep)

    r = sub.add_parser("run", help="run a headless simulation on a generated real map (peaceful unless --role says otherwise)")
    r.add_argument("--seed", type=int, default=42, help="map seed (u64)")
    _add_world_args(r)
    r.add_argument("--fixture", default="peaceful_fair", choices=FIXTURE_CHOICES,
                   help="behaviour of every colony not named by --role")
    r.add_argument("--role", action="append", default=[], metavar="COLONY=NAME",
                   help="give one colony another fixture/role, e.g. --role 0=raider (repeatable)")
    r.add_argument("--timeline", nargs="?", const="-", default=None, metavar="FILE",
                   help="write the compact event timeline to FILE (stderr when no file is given)")
    r.add_argument("--events", action="store_true", help="include the event timeline in the JSON report")
    r.add_argument("--ticks", type=int, default=None, help="world ticks (default: configured horizon)")
    r.add_argument("--sample-every", type=int, default=100, help="time-series sampling interval")
    r.add_argument("--no-time-series", action="store_true", help="omit per-tick summary snapshots")
    r.add_argument("--omit-map", action="store_true", help="omit the full map from the JSON report")
    r.add_argument("--compact", action="store_true", help="write compact stable JSON")
    r.add_argument("--summary", action="store_true", help="print key metrics to stderr")
    r.add_argument("--out", default=None, help="write JSON to a file instead of stdout")
    r.set_defaults(run=cmd_run)

    ss = sub.add_parser("sim-sweep", help="compare fixtures across real maps, colony counts and resource scenarios")
    ss.add_argument("--seeds", type=int, default=30)
    ss.add_argument("--seed0", type=int, default=50000)
    ss.add_argument("--sizes", type=int, nargs="+", default=list(SIZES), choices=SIZES)
    ss.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=SCENARIOS)
    ss.add_argument("--fixtures", nargs="+", default=["peaceful_fair", "greedy_scarcity"], choices=FIXTURE_CHOICES)
    ss.add_argument("--ticks", type=int, default=6000)
    ss.add_argument("--out", default=None)
    ss.set_defaults(run=cmd_sim_sweep)

    isw = sub.add_parser("interaction-sweep", help="run named role assignments (control, mixed roles, raiders) on the same maps")
    _add_world_args(isw)
    isw.add_argument("--seeds", type=int, default=6)
    isw.add_argument("--seed0", type=int, default=50000)
    isw.add_argument("--ticks", type=int, default=6000)
    isw.add_argument("--out", default=None)
    isw.set_defaults(run=cmd_interaction_sweep)

    sb = sub.add_parser("bench-run", help="measure real-map simulation throughput and peak traced memory")
    _add_world_args(sb)
    sb.add_argument("--fixture", default="peaceful_fair", choices=FIXTURE_CHOICES)
    sb.add_argument("--ticks", type=int, default=6000)
    sb.add_argument("--runs", type=int, default=3)
    sb.add_argument("--seed0", type=int, default=90000)
    sb.add_argument("--out", default=None)
    sb.set_defaults(run=cmd_sim_bench)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.run(args)
    except (CanonicalError, ConfigError, MapGenerationError, ValueError, OSError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
