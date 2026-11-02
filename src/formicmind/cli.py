"""``sim`` entry point (``[project.scripts] sim``): ``sim replay --verify FILE``. Nothing else yet."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from .replay.cli import EXIT_USAGE, build_parser, run


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] != "replay":
        print("usage: sim replay --verify FILE", file=sys.stderr)
        return EXIT_USAGE
    return run(build_parser("sim replay").parse_args(args[1:]))


if __name__ == "__main__":
    sys.exit(main())
