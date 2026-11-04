"""CLI for collecting, training and evaluating the first FormicMind MLP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .dataset import collect_dataset
from .quality_dataset import QUALITY_SEED_RANGES


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="formicmind-ml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="собрать примеры из настоящих матчей Baselines v2")
    collect.add_argument("--output", type=Path, required=True, help="путь к JSONL датасету")
    collect.add_argument("--seeds", type=int, nargs="+", required=True, help="не закрытые технические seeds матчей")
    collect.add_argument("--teacher", default="balanced:v2")
    collect.add_argument("--opponent", default="turtle:v2")
    collect.add_argument("--max-ticks", type=int, default=120)
    collect.add_argument("--validation-fraction", type=float, default=0.2)

    train = subparsers.add_parser("train", help="детерминированное обучение MLP на CPU или CUDA")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True, help="каталог checkpoint")
    train.add_argument("--seed", type=int, default=20261136)
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    evaluate = subparsers.add_parser("eval", help="проверить imitation и сыграть короткие настоящие матчи")
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--seed-start", type=int, required=True)
    evaluate.add_argument("--episodes", type=int, default=2, help="число seeds, каждый сыграет за обе стороны")
    evaluate.add_argument("--opponent", default="balanced:v2")
    evaluate.add_argument("--max-ticks", type=int, default=120)
    evaluate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    evaluate.add_argument("--output", type=Path, required=True, help="JSON отчёт")

    collect_quality = subparsers.add_parser(
        "collect-quality", help="собрать заранее зафиксированные multi-context данные FOR-37/v2",
    )
    collect_quality.add_argument("--output", type=Path, required=True, help="путь к объединённому JSONL датасету")

    train_quality = subparsers.add_parser(
        "train-quality", help="обучить MLP по фиксированному протоколу FOR-37/v2",
    )
    train_quality.add_argument("--dataset", type=Path, required=True)
    train_quality.add_argument("--output", type=Path, required=True, help="каталог checkpoint")
    train_quality.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    evaluate_quality = subparsers.add_parser(
        "eval-quality", help="сравнить FOR-37 checkpoint на held-out, с controls и реальными матчами",
    )
    evaluate_quality.add_argument("--model", type=Path, required=True)
    evaluate_quality.add_argument("--dataset", type=Path, required=True)
    evaluate_quality.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    evaluate_quality.add_argument("--output", type=Path, required=True, help="JSON отчёт")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "collect":
            result = collect_dataset(
                args.output,
                seeds=args.seeds,
                teacher=args.teacher,
                opponent=args.opponent,
                max_ticks=args.max_ticks,
                validation_fraction=args.validation_fraction,
            )
            print(json.dumps({
                "dataset": str(args.output),
                "manifest": str(args.output.with_name(f"{args.output.stem}.manifest.json")),
                "examples": result["example_count"],
                "matches": result["match_count"],
                "sha256": result["dataset_sha256"],
            }, indent=2, ensure_ascii=False))
            return 0
        if args.command == "train":
            from .training import train_model

            result = train_model(
                args.dataset, args.output, seed=args.seed, epochs=args.epochs,
                batch_size=args.batch_size, learning_rate=args.learning_rate,
                requested_device=args.device,
            )
            print(json.dumps({
                "checkpoint": str(args.output),
                "manifest": str(args.output / "manifest.json"),
                "weights_sha256": result["weights"]["sha256"],
                "device": result["performance"]["training_device"],
                "validation": result["training"]["validation"],
            }, indent=2, ensure_ascii=False))
            return 0
        if args.command == "eval":
            if args.episodes <= 0:
                raise ValueError("episodes must be positive")
            if args.seed_start < 0 or args.seed_start + args.episodes - 1 > (1 << 64) - 1:
                raise ValueError("evaluation seed range must fit unsigned 64-bit values")
            from .evaluation import evaluate_model

            result = evaluate_model(
                args.model, args.dataset,
                seeds=range(args.seed_start, args.seed_start + args.episodes),
                opponent=args.opponent,
                max_ticks=args.max_ticks,
                device=args.device,
                output=args.output,
            )
            print(json.dumps({
                "report": str(args.output),
                "matches": result["match_summary"]["matches"],
                "validation_imitation": result["imitation"],
                "classification": result["classification"],
            }, indent=2, ensure_ascii=False))
            return 0
        if args.command == "collect-quality":
            from .quality_dataset import collect_quality_dataset

            seed_sets = {
                split: tuple(range(bounds[0], bounds[1] + 1))
                for split, bounds in QUALITY_SEED_RANGES.items()
            }
            result = collect_quality_dataset(
                args.output,
                train_seeds=seed_sets["train"],
                validation_seeds=seed_sets["validation"],
                heldout_seeds=seed_sets["heldout"],
                max_ticks=3000,
            )
            print(json.dumps({
                "dataset": str(args.output),
                "manifest": str(args.output.with_name(f"{args.output.stem}.manifest.json")),
                "examples": result["example_count"],
                "matches": result["match_count"],
                "sha256": result["dataset_sha256"],
                "deduplication": result["deduplication"],
            }, indent=2, ensure_ascii=False))
            return 0
        if args.command == "train-quality":
            from .quality import train_quality_model

            result = train_quality_model(args.dataset, args.output, requested_device=args.device)
            print(json.dumps({
                "checkpoint": str(args.output),
                "weights_sha256": result["weights"]["sha256"],
                "device": result["performance"]["training_device"],
                "train": result["training"]["train"],
                "validation": result["training"]["validation"],
            }, indent=2, ensure_ascii=False))
            return 0
        if args.command == "eval-quality":
            from .quality import evaluate_quality_model

            result = evaluate_quality_model(
                args.model, args.dataset, device=args.device, output=args.output,
            )
            print(json.dumps({
                "report": str(args.output),
                "heldout": result["imitation"]["trained"]["heldout"],
                "untrained_heldout": result["imitation"]["untrained_heldout"],
                "matches": result["matches"]["summary_by_opponent"],
            }, indent=2, ensure_ascii=False))
            return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"formicmind-ml: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
