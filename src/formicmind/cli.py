"""``sim`` entry point: deterministic run/eval/benchmark and Replay v1 verification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections.abc import Sequence

from .evaluation import (
    SCENARIOS,
    TECHNICAL_SCENARIO,
    EvaluationError,
    aggregate_evaluation,
    build_manifest,
    controller_pair,
    evaluate_episodes,
    execute_episode,
    load_manifest,
    make_result_document,
    manifest_digest,
    run_benchmark,
    validate_seed,
    write_json,
)
from .replay.cli import EXIT_IO, EXIT_OK, EXIT_USAGE, build_parser as build_replay_parser
from .replay.cli import run as run_replay
from .rules import default_config

SCRIPTED = "scripted"


class _UsageParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def _positive_int(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _u64(value: str) -> int:
    try:
        parsed = int(value, 10)
        validate_seed(parsed)
        return parsed
    except (ValueError, EvaluationError) as exc:
        raise argparse.ArgumentTypeError("must be a decimal u64 (0..18446744073709551615)") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = _UsageParser(prog="sim", description="FormicMind deterministic L0 tools.")
    sub = parser.add_subparsers(dest="command", parser_class=_UsageParser)

    run = sub.add_parser("run", help="run one deterministic episode")
    run.add_argument("--manifest", metavar="PATH", help="replay a previously generated run manifest")
    run.add_argument("--manifest-out", metavar="PATH", help="write the resolved, reusable manifest")
    run.add_argument("--json-out", metavar="PATH", help="write manifest and final episode result as JSON")
    run.add_argument("--seed", type=_u64, help="root seed as a decimal u64")
    run.add_argument("--a", choices=[SCRIPTED], help="controller on side A")
    run.add_argument("--b", choices=[SCRIPTED], help="controller on side B")
    run.add_argument("--scenario", choices=SCENARIOS, help="technical or an existing scripted smoke fixture")
    run.add_argument("--max-ticks", type=_nonnegative_int, help="explicit harness cutoff (0..T_max)")

    evaluate = sub.add_parser("eval", help="run a deterministic seed schedule and save per-episode results")
    evaluate.add_argument("--manifest", metavar="PATH", help="replay a previously generated eval manifest")
    evaluate.add_argument("--manifest-out", metavar="PATH", help="write the resolved, reusable manifest")
    evaluate.add_argument("--json-out", metavar="PATH", help="write manifest, per-episode rows, and aggregates")
    evaluate.add_argument("--episodes", type=_positive_int, help="number of seeds, or seed pairs with --paired")
    evaluate.add_argument("--seed-start", type=_u64, help="first root seed; remaining seeds are seed_start + i")
    evaluate.add_argument("--a", choices=[SCRIPTED], help="controller on side A")
    evaluate.add_argument("--b", choices=[SCRIPTED], help="controller on side B")
    evaluate.add_argument("--scenario", choices=SCENARIOS, help="technical or an existing scripted smoke fixture")
    evaluate.add_argument("--paired", dest="paired_seat", action="store_const", const=True, default=None,
                          help="run both A/B seat assignments for each seed")
    evaluate.add_argument("--max-ticks", type=_nonnegative_int, help="harness cutoff (0..T_max)")

    benchmark = sub.add_parser("benchmark", help="measure repeated full step_tick workloads without replay/events")
    benchmark.add_argument("--manifest", metavar="PATH", help="replay a previously generated benchmark manifest")
    benchmark.add_argument("--manifest-out", metavar="PATH", help="write the resolved, reusable manifest")
    benchmark.add_argument("--json-out", metavar="PATH", help="write the manifest, raw samples, and machine profile")
    benchmark.add_argument("--runs", type=_positive_int, help="measurement samples (recommended: 5)")
    benchmark.add_argument("--ticks-per-run", type=_positive_int,
                           help="exact step_tick calls per sample (recommended: 100000)")
    benchmark.add_argument("--warmup-ticks", type=_nonnegative_int,
                           help="unmeasured full-path warmup tick calls (recommended: 10000)")
    benchmark.add_argument("--seed-start", type=_u64, help="first root seed; reset each sample to this schedule")
    benchmark.add_argument("--a", choices=[SCRIPTED], help="controller on side A")
    benchmark.add_argument("--b", choices=[SCRIPTED], help="controller on side B")
    benchmark.add_argument("--scenario", choices=SCENARIOS, help="technical or an existing scripted smoke fixture")
    benchmark.add_argument("--pin-core", type=_nonnegative_int,
                           help="logical CPU index; default auto-pins one core when the OS supports it")
    benchmark.add_argument("--phase-timing-ticks", type=_nonnegative_int, default=None,
                           help="separate diagnostic tick budget with per-phase clocks; default 0")

    replay = sub.add_parser("replay", help="verify a recorded Replay v1 file")
    replay.add_argument("--verify", metavar="FILE", help="re-execute and verify the replay")
    return parser


def _reject_manifest_overrides(args: argparse.Namespace, fields: tuple[str, ...]) -> None:
    supplied = [field.replace("_", "-") for field in fields if getattr(args, field) is not None]
    if supplied:
        raise EvaluationError("--manifest cannot be combined with execution options: " + ", ".join(supplied))


def _resolve_parameters(args: argparse.Namespace, command: str, config) -> tuple[dict, dict]:
    if args.manifest:
        overrides = {
            "run": ("seed", "a", "b", "scenario", "max_ticks"),
            "eval": ("episodes", "seed_start", "a", "b", "scenario", "paired_seat", "max_ticks"),
            "benchmark": ("runs", "ticks_per_run", "warmup_ticks", "seed_start", "a", "b", "scenario",
                          "pin_core", "phase_timing_ticks"),
        }[command]
        _reject_manifest_overrides(args, overrides)
        manifest = load_manifest(args.manifest, command, config)
        return manifest["parameters"], manifest

    a, b = args.a, args.b
    if a != SCRIPTED or b != SCRIPTED:
        raise EvaluationError("direct runs require explicit --a scripted and --b scripted.")
    scenario = args.scenario or TECHNICAL_SCENARIO
    if command == "run":
        if args.seed is None:
            raise EvaluationError("sim run requires --seed (or --manifest).")
        params = {"scenario": scenario, "seed": args.seed, "max_ticks": args.max_ticks}
    elif command == "eval":
        if args.episodes is None or args.seed_start is None:
            raise EvaluationError("sim eval requires --episodes and --seed-start (or --manifest).")
        paired = args.paired_seat is True
        if args.seed_start + args.episodes - 1 > (1 << 64) - 1:
            raise EvaluationError("seed_start + episodes - 1 exceeds the u64 seed range.")
        params = {
            "scenario": scenario, "episodes": args.episodes, "seed_start": args.seed_start,
            "paired_seat": paired, "max_ticks": args.max_ticks,
            "seed_schedule": "seed_start + episode_index; one seed per pair" if paired else "seed_start + episode_index",
        }
    else:
        if None in (args.runs, args.ticks_per_run, args.warmup_ticks, args.seed_start):
            raise EvaluationError("sim benchmark requires --runs, --ticks-per-run, --warmup-ticks, and --seed-start "
                                  "(or --manifest).")
        if args.seed_start + max(args.ticks_per_run, args.warmup_ticks) - 1 > (1 << 64) - 1:
            raise EvaluationError("benchmark seed schedule may exceed the u64 range.")
        params = {
            "scenario": scenario,
            "runs": args.runs,
            "ticks_per_run": args.ticks_per_run,
            "warmup_ticks": args.warmup_ticks,
            "seed_start": args.seed_start,
            "seed_schedule": "seed_start + episode_index; reset to the same schedule for warmup and every sample",
            "pin_core": args.pin_core,
            "pin_core_policy": "auto on supported platforms when pin_core is null",
            "phase_timing_ticks": args.phase_timing_ticks or 0,
        }
    return params, build_manifest(command, config, params)


def _prepare_output_paths(json_out: str | None, manifest_out: str | None) -> None:
    if json_out and manifest_out and Path(json_out).resolve() == Path(manifest_out).resolve():
        raise EvaluationError("--json-out and --manifest-out must be different paths.")
    for value in (json_out, manifest_out):
        if not value:
            continue
        path = Path(value)
        if path.exists() and path.is_dir():
            raise OSError(f"output path is a directory: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)


def _render_episode(row: dict, scenario: str, manifest: dict) -> str:
    score = row["score"]
    score_text = "n/a (truncated)" if score is None else f"{score['A']:.1f}/{score['B']:.1f}"
    assignment = row["seat_assignment"]
    return "\n".join((
        f"seed={row['seed']} scenario={scenario} A={assignment['A']} B={assignment['B']}",
        f"winner={row['winner']} reason={row['reason']} terminal_tick={row['terminal_tick']} "
        f"state_tick={row['state_tick']} tick_completed={row['tick_completed']} "
        f"ticks_executed={row['ticks_executed']} truncated={row['truncated']} score_A/B={score_text}",
        f"queen_hp_A/B={row['queen_hp']['A']}/{row['queen_hp']['B']} "
        f"food_A/B={row['food']['A']}/{row['food']['B']} "
        f"workers_A/B={row['workers']['A']}/{row['workers']['B']} "
        f"soldiers_A/B={row['soldiers']['A']}/{row['soldiers']['B']} "
        f"brood_A/B={row['brood']['A']}/{row['brood']['B']}",
        f"config_hash={row['config_hash']}",
        f"state_hash={row['state_hash']}",
        f"checkpoint_hash={row['checkpoint_hash']}",
        f"manifest_sha256={manifest_digest(manifest)}",
    ))


def _write_manifest_if_requested(path: str | None, manifest: dict) -> None:
    if path:
        write_json(path, manifest)


def _run_command(args: argparse.Namespace) -> int:
    config = default_config()
    params, manifest = _resolve_parameters(args, "run", config)
    _prepare_output_paths(args.json_out, args.manifest_out)
    _write_manifest_if_requested(args.manifest_out, manifest)
    pair = controller_pair(params["scenario"])
    summary = execute_episode(params["seed"], pair, config=config, max_ticks=params["max_ticks"])
    row = summary.to_record(episode_index=0, pair_id=None, assignment=pair.ids)
    row["controller_config"] = {
        "A": {"id": pair.ids[0], "kind": "scripted", "script_sha256": pair.script_hashes[0]},
        "B": {"id": pair.ids[1], "kind": "scripted", "script_sha256": pair.script_hashes[1]},
    }
    document = make_result_document("run", manifest, results=[row], aggregate=aggregate_evaluation([row]))
    if args.json_out:
        write_json(args.json_out, document)
    print(_render_episode(row, params["scenario"], manifest))
    if args.json_out:
        print(f"json_out={args.json_out}")
    return EXIT_OK


def _eval_command(args: argparse.Namespace) -> int:
    config = default_config()
    params, manifest = _resolve_parameters(args, "eval", config)
    _prepare_output_paths(args.json_out, args.manifest_out)
    _write_manifest_if_requested(args.manifest_out, manifest)
    records, aggregate = evaluate_episodes(
        episodes=params["episodes"], seed_start=params["seed_start"], scenario=params["scenario"],
        paired_seat=params["paired_seat"], max_ticks=params["max_ticks"], config=config,
    )
    document = make_result_document("eval", manifest, results=records, aggregate=aggregate)
    if args.json_out:
        write_json(args.json_out, document)
        print(f"eval attempted={aggregate['attempted']} finished={aggregate['finished']} "
              f"truncated={aggregate['truncated']} wins_A/B={aggregate['wins_A']}/{aggregate['wins_B']} "
              f"draws={aggregate['draws']} json_out={args.json_out}")
        print(f"manifest_sha256={document['manifest_sha256']}")
    else:
        print(json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return EXIT_OK


def _benchmark_command(args: argparse.Namespace) -> int:
    config = default_config()
    params, manifest = _resolve_parameters(args, "benchmark", config)
    _prepare_output_paths(args.json_out, args.manifest_out)
    _write_manifest_if_requested(args.manifest_out, manifest)
    document = run_benchmark(
        runs=params["runs"], ticks_per_run=params["ticks_per_run"], warmup_ticks=params["warmup_ticks"],
        seed_start=params["seed_start"], scenario=params["scenario"], pin_core=params["pin_core"],
        phase_timing_ticks=params["phase_timing_ticks"], config=config, manifest=manifest,
    )
    if args.json_out:
        write_json(args.json_out, document)
    measurement = document["measurement"]
    throughput = measurement["throughput"]
    print(f"scenario={params['scenario']} warmup_ticks={params['warmup_ticks']} "
          f"runs={params['runs']} ticks_per_run={params['ticks_per_run']}")
    for sample in measurement["samples"]:
        print(f"sample={sample['sample_index'] + 1} ticks={sample['ticks']} episodes={sample['episodes']} "
              f"terminals={sample['full_terminal_episodes']} "
              f"benchmark_cutoffs={sample['benchmark_boundary_cutoffs']} "
              f"elapsed_s={sample['elapsed_seconds']:.6f} ticks_per_second={sample['ticks_per_second']:.3f}")
    print(f"p10_ticks_per_second={throughput['ticks_per_second_p10']:.3f} "
          f"p50_ticks_per_second={throughput['ticks_per_second_p50']:.3f} "
          f"p10_method={throughput['p10_method']}")
    print(f"affinity_applied={measurement['affinity']['affinity_applied']} "
          f"selected_core={measurement['affinity']['selected_core']} "
          f"gate0_status={throughput['gate0_status']}")
    if not throughput["qualified_for_windows_single_core_comparison"]:
        print("Windows single-core Gate 0 performance: NOT MEASURED")
    print(f"manifest_sha256={document['manifest_sha256']}")
    if args.json_out:
        print(f"json_out={args.json_out}")
    return EXIT_OK


def _run_new_command(args: argparse.Namespace) -> int:
    try:
        if args.command == "run":
            return _run_command(args)
        if args.command == "eval":
            return _eval_command(args)
        if args.command == "benchmark":
            return _benchmark_command(args)
        raise EvaluationError(f"unsupported command: {args.command}")
    except EvaluationError as exc:
        print(f"FAIL usage: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except OSError as exc:
        print(f"FAIL io: {exc}", file=sys.stderr)
        return EXIT_IO
    except Exception as exc:
        # A controller/engine error remains a technical failure, never a fabricated loss.
        print(f"FAIL {args.command}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "replay":
        try:
            return run_replay(build_replay_parser("sim replay").parse_args(args[1:]))
        except SystemExit as exc:
            return int(exc.code or 0)

    parser = build_parser()
    if not args:
        parser.print_usage(sys.stderr)
        return EXIT_USAGE
    try:
        parsed = parser.parse_args(args)
    except SystemExit as exc:
        return int(exc.code or 0)
    if parsed.command == "replay":
        # Keep the historical replay parser, exit codes, and output contract.
        try:
            return run_replay(build_replay_parser("sim replay").parse_args(args[1:]))
        except SystemExit as exc:
            return int(exc.code or 0)
    return _run_new_command(parsed)


if __name__ == "__main__":
    sys.exit(main())
