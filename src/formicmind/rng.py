"""PRNG specification: PCG32 (XSH-RR, 64-bit state) with SHA-256 stream derivation.

Generator (O'Neill's reference ``pcg32_random_r`` / ``pcg32_srandom_r``)::

    next_u32:  old = state
               state = (old * 6364136223846793005 + inc) mod 2**64
               xorshifted = (((old >> 18) ^ old) >> 27) mod 2**32
               rot = old >> 59
               return ror32(xorshifted, rot)
    seed(initstate, initseq):
               state = 0; inc = ((initseq << 1) | 1) mod 2**64
               next_u32(); state = (state + initstate) mod 2**64; next_u32()

``below(n)`` uses rejection sampling: ``threshold = (2**32 - n) % n``; draw until
``r >= threshold``; return ``r % n``. No modulo bias, ``1 <= n <= 2**32``.

Stream derivation (never Python ``hash()``)::

    digest = SHA-256("formicmind/rng-stream/v1" || 0x00
                     || root_seed (u64, big-endian, 8 bytes)
                     || len(stream_id) (u16, big-endian) || stream_id (ASCII))
    initstate = digest[0:8] (big-endian u64); initseq = digest[8:16]

A stream id names a *role*, not a seat: ``world`` or
``policy:<policy_id>:<replicate>``. Seat (side A/B) is not part of the id, so a
paired seat-swap gives each policy the same stream.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from .canonical import CanonicalError, U64_MAX, u64_from_str, u64_to_str
from .versions import HASH_DOMAIN_RNG_STREAM

MASK64 = U64_MAX
MASK32 = 0xFFFFFFFF
PCG_MULT = 6364136223846793005

WORLD_STREAM_ID = "world"

_STREAM_ID_RE = re.compile(r"[A-Za-z0-9_.:\-]{1,128}")
_POLICY_ID_RE = re.compile(r"[A-Za-z0-9_.\-]{1,64}")


class Pcg32:
    __slots__ = ("state", "inc")

    def __init__(self, state: int, inc: int) -> None:
        if not (0 <= state <= MASK64 and 0 <= inc <= MASK64):
            raise ValueError("PCG32 state/inc must be u64")
        if inc % 2 == 0:
            raise ValueError("PCG32 inc must be odd")
        self.state = state
        self.inc = inc

    @classmethod
    def seeded(cls, initstate: int, initseq: int) -> "Pcg32":
        rng = cls(0, ((initseq << 1) | 1) & MASK64)
        rng.next_u32()
        rng.state = (rng.state + initstate) & MASK64
        rng.next_u32()
        return rng

    def next_u32(self) -> int:
        old = self.state
        self.state = (old * PCG_MULT + self.inc) & MASK64
        xorshifted = (((old >> 18) ^ old) >> 27) & MASK32
        rot = old >> 59
        return ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & MASK32

    def below(self, bound: int) -> int:
        if type(bound) is not int or not 1 <= bound <= 1 << 32:
            raise ValueError("bound must be an int in [1, 2**32]")
        threshold = ((1 << 32) - bound) % bound
        while True:
            r = self.next_u32()
            if r >= threshold:
                return r % bound

    def clone(self) -> "Pcg32":
        return Pcg32(self.state, self.inc)

    def to_canonical(self) -> dict[str, Any]:
        return {"state": u64_to_str(self.state), "inc": u64_to_str(self.inc)}

    @classmethod
    def from_canonical(cls, data: Any) -> "Pcg32":
        if type(data) is not dict or set(data) != {"state", "inc"}:
            raise CanonicalError("PCG32 state must be an object with 'state' and 'inc'")
        try:
            return cls(u64_from_str(data["state"]), u64_from_str(data["inc"]))
        except ValueError as exc:
            raise CanonicalError(str(exc)) from exc

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Pcg32) and (self.state, self.inc) == (other.state, other.inc)

    def __hash__(self) -> int:  # mutable state: not hashable
        raise TypeError("Pcg32 is mutable and unhashable")

    def __repr__(self) -> str:
        return f"Pcg32(state={self.state}, inc={self.inc})"


def check_stream_id(stream_id: str) -> str:
    if type(stream_id) is not str or not _STREAM_ID_RE.fullmatch(stream_id):
        raise ValueError(f"invalid stream id: {stream_id!r}")
    return stream_id


def policy_stream_id(policy_id: str, replicate: int) -> str:
    if type(policy_id) is not str or not _POLICY_ID_RE.fullmatch(policy_id):
        raise ValueError(f"invalid policy id: {policy_id!r}")
    if type(replicate) is not int or replicate < 0:
        raise ValueError("replicate must be a non-negative int")
    return f"policy:{policy_id}:{replicate}"


def derive_stream_seed(root_seed: int, stream_id: str) -> tuple[int, int]:
    """Return ``(initstate, initseq)`` for a stream."""
    if type(root_seed) is not int or not 0 <= root_seed <= MASK64:
        raise ValueError("root_seed must be a u64")
    sid = check_stream_id(stream_id).encode("ascii")
    digest = hashlib.sha256(
        HASH_DOMAIN_RNG_STREAM.encode("ascii")
        + b"\x00"
        + root_seed.to_bytes(8, "big")
        + len(sid).to_bytes(2, "big")
        + sid
    ).digest()
    return int.from_bytes(digest[0:8], "big"), int.from_bytes(digest[8:16], "big")


def new_stream(root_seed: int, stream_id: str) -> Pcg32:
    return Pcg32.seeded(*derive_stream_seed(root_seed, stream_id))
