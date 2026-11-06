"""Fail-closed path policy for deciding whether a PR needs pinned full history checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


_INDEPENDENT_PREFIXES = (
    "src/formicmind/ml/",
    "tests/test_ml_",
    "viewer/",
    "docs/ml_",
    # Ecosystem v1 (FOR-44): a separate package. It is "independent" only because tests/test_eco_boundary.py proves with an AST
    # check that formicmind.eco imports nothing from L0 except canonical/rng and that no L0 module (nor L0 test) imports it.
    # A change that also touches any other path is sensitive (unknown paths fail closed), so mixed L0 + eco PRs run the history.
    "src/formicmind/eco/",
    "tests/test_eco_",
    "tests/test_ecosystem_",
    "docs/ecosystem_",
    "scripts/ecosystem_",
)
_INDEPENDENT_EXACT = {"docs/ci_strategy.md"}


def _is_traversal(path: str) -> bool:
    """A path with a ``..`` or ``.`` segment, an absolute path or a drive could hide a sensitive file behind an allowed prefix."""
    parts = path.split("/")
    return ".." in parts or "." in parts or path.startswith("/") or ":" in path or "//" in path


def classify_paths(paths: Iterable[str]) -> dict[str, object]:
    """Classify known independent areas as fast-only; unknown paths require full history."""
    normalized = sorted({path.strip().replace("\\", "/").lower() for path in paths if path.strip()})
    if not normalized:
        return {
            "historical_required": True,
            "sensitive_paths": ["<empty change list; fail closed>"],
        }
    sensitive = [
        path for path in normalized
        if _is_traversal(path) or (path not in _INDEPENDENT_EXACT and not path.startswith(_INDEPENDENT_PREFIXES))
    ]
    return {"historical_required": bool(sensitive), "sensitive_paths": sensitive}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--github-output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = classify_paths(sys.stdin.read().splitlines())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    with args.github_output.open("a", encoding="utf-8", newline="\n") as output:
        output.write(f"historical_required={str(result['historical_required']).lower()}\n")
        output.write("sensitive_paths=" + json.dumps(result["sensitive_paths"], ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
