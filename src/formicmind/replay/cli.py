"""Minimal Windows-friendly CLI for replays: ``--verify FILE``.

Exit codes are fixed: 0 verified; 1 the replay is valid but the engine does not reproduce it; 2 not a valid Replay v1
file (corrupt, truncated, unknown version); 3 the file could not be read; 64 wrong usage. Output is plain ASCII
(non-ASCII path characters are escaped), one short block on stdout for success, one ``FAIL`` line on stderr otherwise.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .format import ReplayFormatError, ReplayVerifyError
from .verify import VerifyReport, verify

EXIT_OK, EXIT_MISMATCH, EXIT_FORMAT, EXIT_IO, EXIT_USAGE = 0, 1, 2, 3, 64


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # argparse would exit with 2, which this CLI uses for "invalid file"
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def render(path: str, report: VerifyReport) -> str:
    if report.finished:
        how = "tick completed" if report.tick_completed else "cut mid-tick at QueenTerminal"
        outcome = (f"winner={report.winner} reason={report.reason} terminal_tick={report.terminal_tick} "
                   f"terminal_phase={report.terminal_phase} ({how})")
    else:
        outcome = "outcome=None (truncated harness run)"
    return "\n".join([
        f"OK   replay v1 verified: {ascii(path)[1:-1]}",
        f"     ticks={report.ticks} decisions={report.decisions} events={report.events} records={report.records} "
        f"state_tick={report.state_tick}",
        f"     {outcome}",
        f"     events_hash={report.events_hash}",
        f"     state_hash={report.final_state_hash}",
        f"     checkpoint_hash={report.final_checkpoint_hash}",
    ])


def build_parser(prog: str = "python -m formicmind.replay") -> argparse.ArgumentParser:
    parser = _Parser(prog=prog, description="FormicMind Replay v1 tools.")
    parser.add_argument("--verify", metavar="FILE", required=True,
                        help="re-execute FILE through step_tick and compare events, frames, outcome and hashes")
    return parser


def run(args: argparse.Namespace) -> int:
    path = args.verify
    try:
        report = verify(path)
    except ReplayVerifyError as exc:
        print(f"FAIL verify: {ascii(str(exc))[1:-1]}", file=sys.stderr)
        return EXIT_MISMATCH
    except ReplayFormatError as exc:
        print(f"FAIL format: {ascii(str(exc))[1:-1]}", file=sys.stderr)
        return EXIT_FORMAT
    except OSError as exc:
        print(f"FAIL io: {ascii(str(exc))[1:-1]}", file=sys.stderr)
        return EXIT_IO
    print(render(path, report))
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))
