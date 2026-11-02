import hashlib

import pytest

from formicmind.canonical import CanonicalError
from formicmind.rng import (
    Pcg32,
    derive_stream_seed,
    new_stream,
    policy_stream_id,
)
from formicmind.runstate import RunState


def test_pcg32_known_answer_reference_vector():
    # Reference vector from the pcg-c demo: pcg32_srandom_r(&rng, 42u, 54u).
    rng = Pcg32.seeded(42, 54)
    assert [rng.next_u32() for _ in range(6)] == [
        0xA15C02B7, 0x7B47F409, 0xBA1D3330, 0x83D2F293, 0xBFA4784B, 0xCBED606E,
    ]


def test_stream_derivation_matches_documented_formula():
    digest = hashlib.sha256(
        b"formicmind/rng-stream/v1\x00" + (7).to_bytes(8, "big") + (5).to_bytes(2, "big") + b"world"
    ).digest()
    expected = (int.from_bytes(digest[:8], "big"), int.from_bytes(digest[8:16], "big"))
    assert derive_stream_seed(7, "world") == expected


def test_stream_derivation_known_answers():
    # Regression vectors produced by this implementation (no external reference exists
    # for the derivation); they pin the byte layout of the derivation message.
    assert derive_stream_seed(0, "world") == (1152597289713879357, 2614177906343500131)
    assert derive_stream_seed(2**64 - 1, "policy:alpha:0") == (
        1534097309429375627, 434644365588230145,
    )
    rng = new_stream(0, "world")
    assert [rng.next_u32() for _ in range(3)] == [1254714174, 2841545534, 4235514422]
    rng = new_stream(2**64 - 1, "policy:alpha:0")
    assert [rng.below(100) for _ in range(5)] == [27, 91, 89, 41, 64]


def test_below_range_and_bound_one():
    rng = new_stream(1, "world")
    assert all(0 <= rng.below(6) < 6 for _ in range(1000))
    assert all(rng.below(1) == 0 for _ in range(10))
    assert {rng.below(2) for _ in range(200)} == {0, 1}


def test_below_rejects_biased_region():
    # For bound 3, threshold = 2**32 % 3 = 1: raw value 0 must be rejected.
    class Scripted(Pcg32):
        __slots__ = ("queue",)

        def next_u32(self):
            return self.queue.pop(0)

    rng = Scripted(0, 1)
    rng.queue = [0, 4]
    assert rng.below(3) == 4 % 3
    assert rng.queue == []


def test_below_rejects_invalid_bounds():
    rng = new_stream(1, "world")
    for bad in (0, -1, 2**32 + 1, 1.5, True):
        with pytest.raises(ValueError):
            rng.below(bad)  # type: ignore[arg-type]


def test_streams_are_independent_and_seat_free():
    a1 = new_stream(99, policy_stream_id("alpha", 0))
    a2 = new_stream(99, policy_stream_id("alpha", 0))
    beta = new_stream(99, policy_stream_id("beta", 0))
    assert a1 == a2
    assert a1 != beta
    assert policy_stream_id("alpha", 0) != policy_stream_id("alpha", 1)
    seq_a = [a1.next_u32() for _ in range(50)]
    # Drawing from another stream does not perturb this one.
    run = RunState.new(99, [policy_stream_id("alpha", 0), policy_stream_id("beta", 0)])
    for _ in range(1000):
        run.stream(policy_stream_id("beta", 0)).next_u32()
    assert [run.stream(policy_stream_id("alpha", 0)).next_u32() for _ in range(50)] == seq_a


def test_root_seed_changes_every_stream():
    assert derive_stream_seed(1, "world") != derive_stream_seed(2, "world")
    assert new_stream(1, "world") != new_stream(2, "world")


def test_length_prefix_prevents_concatenation_ambiguity():
    assert derive_stream_seed(1, "a:1") != derive_stream_seed(1, "a:10")
    assert derive_stream_seed(0x0102030405060708, "x") != derive_stream_seed(0x0102030405060700, "x8")


@pytest.mark.parametrize("bad", ["", "has space", "ü", "x" * 129, "a/b"])
def test_invalid_stream_ids(bad):
    with pytest.raises(ValueError):
        new_stream(1, bad)


@pytest.mark.parametrize("seed", [-1, 2**64, 1.0, True])
def test_invalid_root_seed(seed):
    with pytest.raises(ValueError):
        new_stream(seed, "world")  # type: ignore[arg-type]


def test_policy_stream_id_validation():
    assert policy_stream_id("baseline-1.v2", 3) == "policy:baseline-1.v2:3"
    for policy_id, rep in (("a:b", 0), ("", 0), ("ok", -1), ("ok", 1.0)):
        with pytest.raises(ValueError):
            policy_stream_id(policy_id, rep)  # type: ignore[arg-type]


def test_pcg32_canonical_state_is_strings_and_round_trips():
    rng = new_stream(2**64 - 1, "world")
    for _ in range(7):
        rng.next_u32()
    data = rng.to_canonical()
    assert all(type(v) is str for v in data.values())
    restored = Pcg32.from_canonical(data)
    assert restored == rng
    assert [restored.next_u32() for _ in range(10)] == [rng.next_u32() for _ in range(10)]


@pytest.mark.parametrize(
    "data",
    [
        {"state": "1"},
        {"state": 1, "inc": "1"},
        {"state": "01", "inc": "1"},
        {"state": "1", "inc": "2"},  # even inc
        {"state": str(2**64), "inc": "1"},
        {"state": "-1", "inc": "1"},
    ],
)
def test_pcg32_from_canonical_rejects_bad_data(data):
    with pytest.raises(CanonicalError):
        Pcg32.from_canonical(data)


def test_run_state_stream_registry():
    run = RunState.new(5)
    assert list(run.streams) == ["world"]
    with pytest.raises(ValueError):
        run.add_stream("world")
    with pytest.raises(KeyError):
        run.stream("policy:nobody:0")
