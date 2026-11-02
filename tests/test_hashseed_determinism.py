"""The hashes must not depend on PYTHONHASHSEED (no set/dict-order or hash() leakage)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

SCRIPT = f"""
import sys; sys.path.insert(0, {str(SRC)!r})
from formicmind import (default_config, reset, state_hash, checkpoint_hash, config_hash)
from formicmind.rng import policy_stream_id
c = default_config()
ids = [policy_stream_id('alpha', 0), policy_stream_id('beta', 0)]
s, r = reset(c, 20261102, ids)
for sid in ['world', *ids]:
    r.stream(sid).next_u32()
print(config_hash(c), state_hash(s), checkpoint_hash(s, r))
"""


def run_with_hashseed(value: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": value}
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT], capture_output=True, text=True, env=env, check=False
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_pythonhashseed_is_really_applied_to_the_subprocess():
    probe = "print(hash('probe'))"

    def probe_with(value: str) -> str:
        env = {**os.environ, "PYTHONHASHSEED": value}
        return subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                              env=env, check=True).stdout

    assert probe_with("0") == probe_with("0")
    assert probe_with("0") != probe_with("1")


@pytest.mark.parametrize("other", ["1", "12345", "4294967295", "random"])
def test_hashes_are_independent_of_pythonhashseed(other):
    assert run_with_hashseed("0") == run_with_hashseed(other)


def test_in_process_hashes_match_subprocess():
    from formicmind import checkpoint_hash, config_hash, default_config, reset, state_hash
    from formicmind.rng import policy_stream_id

    c = default_config()
    ids = [policy_stream_id("alpha", 0), policy_stream_id("beta", 0)]
    s, r = reset(c, 20261102, ids)
    for sid in ["world", *ids]:
        r.stream(sid).next_u32()
    assert run_with_hashseed("7") == f"{config_hash(c)} {state_hash(s)} {checkpoint_hash(s, r)}"
