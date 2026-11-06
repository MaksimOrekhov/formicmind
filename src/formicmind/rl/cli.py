"""``python -m formicmind.rl {train,eval,bench}`` (FOR-42)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

from .config import RunConfig


def _coerce(field_type: Any, raw: str) -> Any:
    text = str(field_type)
    if raw.lower() == "none":
        return None
    if "bool" in text:
        return raw.lower() in ("1", "true", "yes")
    if "int" in text and "float" not in text and "tuple" not in text:
        return int(raw)
    if "float" in text:
        return float(raw)
    if "tuple" in text:
        return tuple(int(x) for x in raw.split(","))
    return raw


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    types = {f.name: f.type for f in fields(RunConfig)}
    values: dict[str, Any] = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        if key not in types:
            raise SystemExit(f"unknown config key {key!r}; known: {sorted(types)}")
        values[key] = _coerce(types[key], raw)
    return values


def _cmd_train(args: argparse.Namespace) -> int:
    from .train import train

    run_dir = Path(args.out)
    overrides = parse_overrides(args.set or [])
    if args.resume:
        base = RunConfig.from_json(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
        cfg = RunConfig.from_json({**base.to_json(), **overrides})
    else:
        cfg = RunConfig.from_json({**RunConfig().to_json(), **overrides})
    summary = train(cfg, run_dir, resume=args.resume)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .checkpoint import latest_checkpoint, load_checkpoint
    from .evaluate import headline, run_evaluation
    from .network import actor_state_to_bytes
    from .parallel import MatchExecutor
    from .pool import SnapshotPool, atomic_write_json

    run_dir = Path(args.run)
    directory = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(run_dir)
    if directory is None:
        raise SystemExit("no checkpoint found")
    model, manifest, _ = load_checkpoint(directory)
    pool = SnapshotPool(run_dir / "snapshots")
    snapshots = [pool.load_info(i["snapshot_id"]) for i in manifest["snapshots"]]
    opponents: dict[str, dict[str, Any]] = {"init": {"path": snapshots[0].path, "sha256": snapshots[0].sha256}}
    for snap in snapshots[1:]:
        if args.vs_snapshots and snap.snapshot_id in args.vs_snapshots:
            opponents[snap.ref] = {"path": snap.path, "sha256": snap.sha256}
    with MatchExecutor(args.workers, args.chunk_size) as executor:
        report = run_evaluation(executor, {"bytes": actor_state_to_bytes(model.actor_state_dict())}, opponents,
                                args.suite, iteration=manifest["iteration"])
    out = Path(args.output) if args.output else run_dir / "eval" / f"eval_{args.suite}_{manifest['iteration']:06d}.json"
    atomic_write_json(out, report)
    print(json.dumps(headline(report), indent=2, sort_keys=True))
    print(f"wrote {out} ({report['matches']} matches in {report['wall_seconds']:.1f}s)")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from .bench import run_benchmark

    result = run_benchmark(workers=args.workers, matches=args.matches, chunk_size=args.chunk_size,
                           out=Path(args.output) if args.output else None)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="formicmind-rl", description="FOR-42 PPO / self-play")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="train (or resume) a PPO self-play run")
    train.add_argument("--out", required=True)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a RunConfig field")
    train.set_defaults(func=_cmd_train)
    evaluate = sub.add_parser("eval", help="independent evaluation of a checkpoint")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--checkpoint")
    evaluate.add_argument("--suite", default="full")
    evaluate.add_argument("--workers", type=int, default=10)
    evaluate.add_argument("--chunk-size", type=int, default=4)
    evaluate.add_argument("--vs-snapshots", type=int, nargs="*")
    evaluate.add_argument("--output")
    evaluate.set_defaults(func=_cmd_eval)
    bench = sub.add_parser("bench", help="collection throughput and profile (no learning)")
    bench.add_argument("--workers", type=int, default=10)
    bench.add_argument("--matches", type=int, default=64)
    bench.add_argument("--chunk-size", type=int, default=4)
    bench.add_argument("--output")
    bench.set_defaults(func=_cmd_bench)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    code = args.func(args)
    print(f"done in {time.perf_counter() - started:.1f}s", file=sys.stderr)
    return code


__all__ = ["main", "parse_overrides"]
