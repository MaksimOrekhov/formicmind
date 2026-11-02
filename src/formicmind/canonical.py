"""Canonical JSON bytes and domain-separated SHA-256.

Allowed value types: ``int`` (within +-(2**53 - 1), the JS safe-integer range),
``str``, ``bool``, ``list`` and ``dict`` with ``str`` keys. No float, NaN, None or
tuple. u64 values (RNG state, seeds) are stored as decimal strings instead.

Encoding: object keys sorted, separators ``,`` and ``:``, ASCII only
(``ensure_ascii``), no whitespace. ``canonical_loads`` accepts only bytes that
re-encode to themselves, so a non-canonical input is an error, not a silent
normalisation.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

MAX_SAFE_INT = 2**53 - 1
U64_MAX = 2**64 - 1

_U64_RE = re.compile(r"(0|[1-9][0-9]*)")


class CanonicalError(ValueError):
    """Value or bytes are not canonical."""


def check_canonical_value(value: Any, path: str = "$") -> None:
    # Exact type checks: bool/IntEnum/tuple subclasses must be converted explicitly.
    t = type(value)
    if t is bool or t is str:
        return
    if t is int:
        if not -MAX_SAFE_INT <= value <= MAX_SAFE_INT:
            raise CanonicalError(f"{path}: integer {value} outside the safe range")
        return
    if t is list:
        for i, item in enumerate(value):
            check_canonical_value(item, f"{path}[{i}]")
        return
    if t is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise CanonicalError(f"{path}: non-string key {key!r}")
            check_canonical_value(item, f"{path}.{key}")
        return
    raise CanonicalError(f"{path}: unsupported type {t.__name__}")


def canonical_dumps(value: Any) -> bytes:
    check_canonical_value(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _reject_float(text: str) -> Any:
    raise CanonicalError(f"float literal {text!r} is not allowed")


def _reject_constant(text: str) -> Any:
    raise CanonicalError(f"constant {text!r} is not allowed")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalError(f"duplicate key {key!r}")
        result[key] = value
    return result


def canonical_loads(data: bytes) -> Any:
    if not isinstance(data, (bytes, bytearray)):
        raise CanonicalError("canonical data must be bytes")
    raw = bytes(data)
    try:
        text = raw.decode("ascii")
        value = json.loads(
            text,
            object_pairs_hook=_no_duplicate_keys,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CanonicalError(f"invalid canonical JSON: {exc}") from exc
    if canonical_dumps(value) != raw:
        raise CanonicalError("input is valid JSON but not in canonical form")
    return value


def domain_hash(domain: str, data: bytes) -> str:
    """SHA-256 over ``domain || 0x00 || data``, lowercase hex."""
    return hashlib.sha256(domain.encode("ascii") + b"\x00" + data).hexdigest()


def u64_to_str(value: int) -> str:
    if type(value) is not int or not 0 <= value <= U64_MAX:
        raise CanonicalError(f"not a u64: {value!r}")
    return str(value)


def u64_from_str(text: Any) -> int:
    if type(text) is not str or not _U64_RE.fullmatch(text):
        raise CanonicalError(f"not a canonical u64 string: {text!r}")
    value = int(text)
    if value > U64_MAX:
        raise CanonicalError(f"u64 overflow: {text}")
    return value
