"""Strict, schema-driven canonical encoding of frozen/plain dataclasses (no third-party packages).

A schema is a dataclass whose field annotations are limited to ``int``, ``str``, ``bool``, ``Literal[...]``,
``U64`` (``Annotated[int, "u64"]``, stored as a decimal string), nested schema dataclasses, ``list[T]`` and
``tuple[T, ...]``. ``to_canonical`` produces ``int/str/bool/list/dict`` only (so ``canonical_dumps`` accepts it); ``from_canonical``
rejects missing/extra keys, wrong types (``bool`` is not an ``int``), out-of-range integers, unknown literals and non-canonical
u64 strings. Lists of tuple-typed fields are rebuilt as tuples, so frozen schemas stay immutable.
"""

from __future__ import annotations

import dataclasses
import typing
from typing import Annotated, Any, Literal, get_args, get_origin

from ..canonical import MAX_SAFE_INT, CanonicalError, u64_from_str, u64_to_str

U64 = Annotated[int, "u64"]


class SchemaError(CanonicalError):
    """The data does not match the schema (path included in the message)."""


_HINTS: dict[type, dict[str, Any]] = {}


def _hints(cls: type) -> dict[str, Any]:
    cached = _HINTS.get(cls)
    if cached is None:
        cached = typing.get_type_hints(cls, include_extras=True)
        _HINTS[cls] = cached
    return cached


def _is_u64(hint: Any) -> bool:
    return get_origin(hint) is Annotated and get_args(hint)[1:] == ("u64",)


_PLANS: dict[type, tuple[tuple[str, str, Any], ...]] = {}


def _plan(cls: type) -> tuple[tuple[str, str, Any], ...]:
    """Per-class encoding plan ``(field, kind, inner_hint)`` with kind in {'plain', 'u64', 'seq', 'obj'}; built once (hot path)."""
    plan = _PLANS.get(cls)
    if plan is None:
        hints = _hints(cls)
        items = []
        for f in dataclasses.fields(cls):
            hint = hints[f.name]
            origin = get_origin(hint)
            if _is_u64(hint):
                items.append((f.name, "u64", None))
            elif origin in (list, tuple):
                items.append((f.name, "seq", get_args(hint)[0]))
            elif origin is Literal or hint in (int, str, bool):
                items.append((f.name, "plain", None))
            elif isinstance(hint, type) and dataclasses.is_dataclass(hint):
                items.append((f.name, "obj", None))
            else:
                raise SchemaError(f"unsupported schema type {hint!r} in {cls.__name__}.{f.name}")
        plan = tuple(items)
        _PLANS[cls] = plan
    return plan


def to_canonical(obj: Any) -> Any:
    """Schema object -> canonical value (``dict``/``list``/``int``/``str``/``bool``)."""
    out = {}
    for name, kind, inner in _plan(type(obj)):
        value = getattr(obj, name)
        if kind == "plain":
            out[name] = value
        elif kind == "obj":
            out[name] = to_canonical(value)
        elif kind == "u64":
            out[name] = u64_to_str(value)
        else:
            out[name] = [_encode_item(item, inner) for item in value]
    return out


def _encode_item(value: Any, hint: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_canonical(value)
    if _is_u64(hint):
        return u64_to_str(value)
    if hint in (int, str, bool) or get_origin(hint) is Literal:
        return value
    raise SchemaError(f"unsupported list element type {hint!r}")


def from_canonical(cls: type, data: Any, path: str = "$") -> Any:
    """Canonical value -> schema object (strict). Does not run semantic validation; call ``validate`` on the result."""
    return _decode(data, cls, path)


def _decode(value: Any, hint: Any, path: str) -> Any:
    if isinstance(hint, type) and dataclasses.is_dataclass(hint):
        if type(value) is not dict:
            raise SchemaError(f"{path}: expected an object for {hint.__name__}")
        fields = dataclasses.fields(hint)
        names = {f.name for f in fields}
        if set(value) != names:
            missing, extra = sorted(names - set(value)), sorted(set(value) - names)
            raise SchemaError(f"{path}: bad keys for {hint.__name__} (missing={missing}, extra={extra})")
        hints = _hints(hint)
        return hint(**{f.name: _decode(value[f.name], hints[f.name], f"{path}.{f.name}") for f in fields})
    if _is_u64(hint):
        return u64_from_str(value)
    origin = get_origin(hint)
    if origin in (list, tuple):
        if type(value) is not list:
            raise SchemaError(f"{path}: expected a list")
        inner = get_args(hint)[0]
        items = [_decode(item, inner, f"{path}[{i}]") for i, item in enumerate(value)]
        return tuple(items) if origin is tuple else items
    if origin is Literal:
        if type(value) is not str or value not in get_args(hint):
            raise SchemaError(f"{path}: {value!r} is not one of {get_args(hint)}")
        return value
    if hint is bool:
        if type(value) is not bool:
            raise SchemaError(f"{path}: expected a bool")
        return value
    if hint is int:
        if type(value) is not int or abs(value) > MAX_SAFE_INT:
            raise SchemaError(f"{path}: expected a safe integer")
        return value
    if hint is str:
        if type(value) is not str:
            raise SchemaError(f"{path}: expected a string")
        return value
    raise SchemaError(f"{path}: unsupported schema type {hint!r}")
