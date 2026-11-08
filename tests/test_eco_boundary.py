"""FOR-44: Ecosystem v1 is isolated from L0 in both directions, L0 is unchanged, and the CI path policy for eco is fail-closed.

The AST checks below are what justifies treating eco-only changes as "fast" in scripts/ci_historical_policy.py.
"""

from __future__ import annotations

import ast
import re
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "formicmind"
ECO = SRC / "eco"
ALLOWED_L0 = {"formicmind.canonical", "formicmind.rng"}
BANNED_NAMES = {"importlib", "__import__", "exec", "eval", "runpy", "pkgutil"}


RL = ECO / "rl"
# the one bridge from the torch-free core to the learning stack: cli.py may import formicmind.eco.rl (inside functions, so that PyTorch is only
# loaded when a learning command runs)
RL_BRIDGE = {"cli.py"}


def _core_files() -> list[Path]:
    """The rules, engine, observation, masks, decoder, replay ...: everything in formicmind/eco except the learning subpackage."""
    files = sorted(path for path in ECO.rglob("*.py") if RL not in path.parents)
    assert files, "formicmind/eco has no sources"
    return files


def _rl_files() -> list[Path]:
    files = sorted(RL.glob("*.py"))
    assert files, "formicmind/eco/rl has no sources"
    return files


def _eco_files() -> list[Path]:
    return _core_files()


def _module_name(path: Path) -> str:
    base = ROOT / "src" if (ROOT / "src") in path.parents else ROOT
    rel = path.relative_to(base).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def imports_of_source(text: str, module: str, is_package: bool = False) -> list[str]:
    """Absolute dotted names imported by ``text`` (relative imports resolved against ``module``, ``from pkg import name`` expanded)."""
    package = module.split(".") if is_package else module.split(".")[:-1]
    found: list[str] = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package[: len(package) - (node.level - 1)]
                base = ".".join(anchor + ([node.module] if node.module else []))
            found.append(base)
            found += [f"{base}.{alias.name}" for alias in node.names]
    return found


def _imports(path: Path) -> list[str]:
    return imports_of_source(path.read_text(encoding="utf-8"), _module_name(path), path.name == "__init__.py")


def is_eco(name: str) -> bool:
    """``formicmind.eco`` or a submodule - NOT the L0 module ``formicmind.economy``."""
    return name == "formicmind.eco" or name.startswith("formicmind.eco.")


def allowed_in_eco(name: str) -> bool:
    return (name.split(".")[0] in sys.stdlib_module_names or name == "formicmind" or is_eco(name)
            or any(name == a or name.startswith(a + ".") for a in ALLOWED_L0))


def test_eco_imports_only_stdlib_itself_and_two_l0_utilities():
    for path in _eco_files():
        for name in _imports(path):
            assert allowed_in_eco(name), f"{path.relative_to(ROOT)} imports {name!r}: only stdlib, formicmind.eco, canonical and rng are allowed"


@pytest.mark.parametrize("source, module, expected_violation", [
    ("import formicmind.economy", "formicmind.eco.x", True),
    ("from formicmind import economy", "formicmind.eco.x", True),
    ("from formicmind.economy import ledger", "formicmind.eco.x", True),
    ("from .. import rules", "formicmind.eco.x", True),
    ("from ..tick import step_tick", "formicmind.eco.x", True),
    ("from ... import rules", "formicmind.eco.x", True),
    ("import formicmind.rules as r", "formicmind.eco.x", True),
    ("from formicmind.versions import RULES_VERSION", "formicmind.eco.x", True),
    ("import numpy", "formicmind.eco.x", True),
    ("from ..canonical import canonical_dumps", "formicmind.eco.x", False),
    ("from ..rng import new_stream", "formicmind.eco.x", False),
    ("from formicmind.canonical import domain_hash", "formicmind.eco.x", False),
    ("from . import codec", "formicmind.eco.x", False),
    ("from .config import ConfigE1", "formicmind.eco.x", False),
    ("import json, heapq, dataclasses", "formicmind.eco.x", False),
])
def test_the_boundary_checker_catches_what_it_should(source, module, expected_violation):
    violations = [n for n in imports_of_source(source, module) if not allowed_in_eco(n)]
    assert bool(violations) is expected_violation, (source, violations)


@pytest.mark.parametrize("source, module, is_eco_import", [
    ("import formicmind.eco", "formicmind.rules", True),
    ("from formicmind import eco", "formicmind.tick", True),
    ("from formicmind.eco.cli import main", "formicmind.cli", True),
    ("from .eco import config", "formicmind.cli", True),
    ("from . import eco", "formicmind.cli", True),
    ("from .economy import x", "formicmind.cli", False),
    ("import formicmind.economy", "formicmind.cli", False),
])
def test_the_reverse_checker_separates_eco_from_the_l0_economy_module(source, module, is_eco_import):
    assert any(is_eco(n) for n in imports_of_source(source, module)) is is_eco_import


RL_THIRD_PARTY = {"torch"}


def test_the_learning_subpackage_imports_only_stdlib_torch_the_core_and_two_l0_utilities():
    for path in _rl_files():
        for name in _imports(path):
            allowed = allowed_in_eco(name) or name.split(".")[0] in RL_THIRD_PARTY
            assert allowed, f"{path.relative_to(ROOT)} imports {name!r}: only stdlib, torch, formicmind.eco, canonical and rng are allowed"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # ``module.eval()`` is torch's evaluation-mode switch: only the builtin ``eval`` (a Name) is banned here
        names = ({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
                 | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr != "eval"})
        assert not (names & BANNED_NAMES), f"{path.name} uses a dynamic import mechanism: {names & BANNED_NAMES}"


def test_the_core_never_imports_the_learning_subpackage_except_the_cli_bridge():
    for path in _core_files():
        if path.name in RL_BRIDGE and path.parent == ECO:
            continue
        for name in _imports(path):
            assert not (name == "formicmind.eco.rl" or name.startswith("formicmind.eco.rl.")), f"{path.name} imports {name}"
    bridge = (ECO / "cli.py").read_text(encoding="utf-8")
    for line in bridge.splitlines():
        if "from .rl" in line and "cli_args" not in line:
            assert line.startswith(" "), "cli.py may import formicmind.eco.rl only inside a function (PyTorch must stay lazy)"


def test_importing_the_core_and_the_cli_does_not_load_torch():
    import subprocess

    script = ("import sys; sys.path.insert(0, 'src'); import formicmind.eco, formicmind.eco.cli, formicmind.eco.viability; "
              "assert 'torch' not in sys.modules, 'torch was imported'")
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True)


def test_the_network_modules_see_only_observations_and_masks():
    """The encoder, the heads and the sampler import no world state, map or engine: they cannot read a hidden fact."""
    forbidden = {"formicmind.eco.model", "formicmind.eco.engine", "formicmind.eco.initial", "formicmind.eco.mapgen", "formicmind.eco.simulation",
                 "formicmind.eco.apply", "formicmind.eco.orders", "formicmind.eco.cohorts", "formicmind.eco.combat", "formicmind.eco.serialization"}
    for name in ("features.py", "network.py", "distribution.py", "learned.py"):
        for imported in _imports(RL / name):
            assert imported not in forbidden and not any(imported.startswith(f + ".") for f in forbidden), f"{name} imports {imported}"


def test_eco_does_not_use_dynamic_imports_or_third_party_packages():
    for path in _eco_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert not (names & BANNED_NAMES), f"{path.name} uses a dynamic import mechanism: {names & BANNED_NAMES}"
        for name in _imports(path):
            assert not name.split(".")[0] in {"numpy", "torch", "scipy", "networkx"}, f"{path.name} imports {name}"


def test_no_l0_source_imports_or_mentions_eco():
    for path in sorted(SRC.rglob("*.py")):
        if ECO in path.parents:
            continue
        text = path.read_text(encoding="utf-8")
        for name in _imports(path):
            assert not is_eco(name), f"{path.relative_to(ROOT)} imports {name}"
        assert not re.search(r"formicmind\.eco(?![A-Za-z0-9_])|from \.eco(?![A-Za-z0-9_])|import eco(?![A-Za-z0-9_])", text), path.relative_to(ROOT)


def test_no_l0_test_or_script_imports_eco():
    for folder in (ROOT / "tests", ROOT / "scripts"):
        for path in sorted(folder.glob("*.py")):
            if path.name.startswith(("test_eco_", "test_ecosystem_", "ecosystem_")):
                continue
            if path.name == "test_eco_boundary.py":
                continue
            for name in _imports(path):
                assert not is_eco(name), f"{path.relative_to(ROOT)} imports {name}"


def test_pyproject_does_not_register_eco_as_an_l0_entry_point():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "formicmind.eco" not in text


def test_l0_contracts_are_unchanged():
    from formicmind import versions
    from formicmind.observation import OBS_V2_LENGTH
    from formicmind.rules import default_config

    assert versions.RULES_VERSION == "l0-proto-FOR1-v0.2.2"
    assert versions.BALANCE_VERSION == "candidate-FOR1-v0.2.1-unvalidated"
    assert versions.HASH_DOMAIN_STATE == "formicmind/state/v1" and versions.HASH_DOMAIN_CHECKPOINT == "formicmind/checkpoint/v3"
    assert OBS_V2_LENGTH == 223
    cfg = default_config()
    assert (len(cfg.topology.nodes), len(cfg.topology.edges), cfg.policy_interval_k, cfg.t_max) == (7, 9, 30, 3000)
    assert len(versions.CONTRACT_PINS) == 6


def test_eco_versions_and_hash_domains_are_disjoint_from_l0():
    from formicmind import versions as l0
    from formicmind.eco import versions as eco

    l0_domains = {v for k, v in vars(l0).items() if k.startswith("HASH_DOMAIN")}
    eco_domains = {v for k, v in vars(eco).items() if k.startswith("HASH_DOMAIN")}
    assert eco_domains and not (eco_domains & l0_domains)
    assert all(d.startswith("formicmind/eco/") for d in eco_domains)
    assert eco.ECO_RULES_VERSION != l0.RULES_VERSION and eco.OBSERVATION_VERSION != "ObservationV2"


def _classify(paths):
    return runpy.run_path(str(ROOT / "scripts" / "ci_historical_policy.py"))["classify_paths"](paths)


def test_eco_only_changes_are_fast_but_everything_else_stays_fail_closed():
    fast = [
        "src/formicmind/eco/mapgen.py", "src/formicmind/eco/sub/new.py", "tests/test_eco_mapgen.py", "tests/test_ecosystem_budget.py",
        "docs/ecosystem_v1_rfc.md", "docs/ecosystem_examples/map_seed42_moderate_8.json", "scripts/ecosystem_budget.py",
    ]
    assert _classify(fast)["historical_required"] is False
    # mixed L0 + eco must run the pinned history
    for l0_path in ("src/formicmind/rules.py", "src/formicmind/tick.py", "tests/test_economy.py", "pyproject.toml",
                    "scripts/ci_historical_policy.py", ".github/workflows/ci.yml", "results/starvation_debt/manifest.json",
                    "historical/for25/x.py", "docs/economy.md"):
        decision = _classify(fast + [l0_path])
        assert decision["historical_required"] is True and l0_path in decision["sensitive_paths"], l0_path


@pytest.mark.parametrize("path", [
    "src/formicmind/eco_extra.py",  # prefix lookalikes
    "src/formicmind/ecosystem/x.py",
    "src/formicmind/eco",
    "tests/test_eco.py",
    "tests/test_economy_pilot.py",
    "docs/ecosystem.md",
    "docs/ecology_notes.md",
    "scripts/ecosystemic.py",
    "src/formicmind/eco/../rules.py",  # traversal hidden behind an allowed prefix
    "src/formicmind/eco/./x.py",
    "src/formicmind/eco//x.py",
    "/src/formicmind/eco/x.py",
    "C:/src/formicmind/eco/x.py",
    "viewer/../src/formicmind/rules.py",
])
def test_lookalike_and_traversal_paths_are_sensitive(path):
    assert _classify([path])["historical_required"] is True, path


def test_windows_separators_are_normalised_before_classification():
    assert _classify([r"src\formicmind\eco\mapgen.py"])["historical_required"] is False
    assert _classify([r"src\formicmind\rules.py", r"src\formicmind\eco\mapgen.py"])["historical_required"] is True
    assert _classify([r"src\formicmind\eco\..\rules.py"])["historical_required"] is True


def test_the_existing_ml_and_viewer_fast_paths_are_unchanged():
    assert _classify(["src/formicmind/ml/quality.py", "tests/test_ml_quality.py", "viewer/src/App.vue", "docs/ml_x.md",
                      "docs/ci_strategy.md"])["historical_required"] is False
    assert _classify([])["historical_required"] is True


def test_ci_workflow_keeps_every_existing_check():
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for name in ("Windows CI (Python 3.12, Node 22)", "Viewer build and Vitest (Node 22)", "Historical sensitivity gate",
                 "Historical FOR-25 (frozen engine, pinned commit)", "Historical FOR-35 (frozen engine, pinned commit)"):
        assert f"name: {name}" in text, name
    assert "python -m pytest -q -ra" in text
