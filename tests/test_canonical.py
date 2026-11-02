import pytest

from formicmind.canonical import (
    MAX_SAFE_INT,
    CanonicalError,
    canonical_dumps,
    canonical_loads,
    domain_hash,
    u64_from_str,
    u64_to_str,
)


def test_dumps_sorted_compact_ascii():
    assert canonical_dumps({"b": [1, True, "x"], "a": {"z": 0, "y": -3}}) == (
        b'{"a":{"y":-3,"z":0},"b":[1,true,"x"]}'
    )
    assert canonical_dumps({"s": "ü→"}) == b'{"s":"\\u00fc\\u2192"}'


def test_dumps_independent_of_insertion_order():
    one = {"a": 1, "b": 2, "c": {"x": 1, "y": 2}}
    two = {"c": {"y": 2, "x": 1}, "b": 2, "a": 1}
    assert canonical_dumps(one) == canonical_dumps(two)


@pytest.mark.parametrize(
    "value",
    [1.0, float("nan"), float("inf"), None, (1, 2), {1: 2}, {"a": {1.5}}, b"x",
     MAX_SAFE_INT + 1, -MAX_SAFE_INT - 1, [[[0.1]]]],
)
def test_dumps_rejects_forbidden_values(value):
    with pytest.raises(CanonicalError):
        canonical_dumps(value)


def test_dumps_rejects_bool_subclasses_and_int_enums():
    import enum

    class E(enum.IntEnum):
        X = 1

    with pytest.raises(CanonicalError):
        canonical_dumps({"a": E.X})


def test_safe_integer_bounds_are_allowed():
    assert canonical_dumps([MAX_SAFE_INT, -MAX_SAFE_INT]) == b"[9007199254740991,-9007199254740991]"


def test_loads_round_trip():
    value = {"k": [1, 2, {"z": False}], "a": "s"}
    assert canonical_loads(canonical_dumps(value)) == value


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a": 1}',            # whitespace
        b'{"b":1,"a":2}',        # unsorted keys
        b'{"a":1,"a":2}',        # duplicate key
        b'{"a":1.0}',            # float
        b'{"a":1e2}',            # float exponent
        b'{"a":NaN}',
        b'{"a":Infinity}',
        b'{"a":-0}',             # non-canonical zero
        b'{"a":null}',
        b'{"a":9007199254740992}',
        '{"a":"ü"}'.encode("utf-8"),  # raw non-ascii
        b'{"a":1}\n',
        b"not json",
    ],
)
def test_loads_rejects_non_canonical_or_forbidden(raw):
    with pytest.raises(CanonicalError):
        canonical_loads(raw)


def test_loads_requires_bytes():
    with pytest.raises(CanonicalError):
        canonical_loads('{"a":1}')  # type: ignore[arg-type]


def test_u64_strings():
    assert u64_to_str(2**64 - 1) == "18446744073709551615"
    assert u64_from_str("18446744073709551615") == 2**64 - 1
    assert u64_from_str("0") == 0
    for bad in ("", "01", "-1", "1.0", " 1", str(2**64), 5, None):
        with pytest.raises(CanonicalError):
            u64_from_str(bad)
    for bad in (-1, 2**64, 1.0, True):
        with pytest.raises(CanonicalError):
            u64_to_str(bad)  # type: ignore[arg-type]


def test_domain_hash_is_separated_and_matches_sha256():
    import hashlib

    assert domain_hash("d", b"x") == hashlib.sha256(b"d\x00x").hexdigest()
    assert domain_hash("d1", b"x") != domain_hash("d2", b"x")
    assert domain_hash("ab", b"c") != domain_hash("a", b"bc")
