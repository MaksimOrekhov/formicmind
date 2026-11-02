"""Helpers of the FOR-15 replay tests: recording of the smoke set, a real RNG-consuming policy, file forging."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from formicmind import default_config
from formicmind.decoder import PolicyLogitsV2
from formicmind.match import EpisodeResult
from formicmind.match_smoke import MATCH_SMOKE_SEED, SCENARIOS
from formicmind.replay import record_episode
from formicmind.replay.format import ZERO_CHAIN, open_record, seal_record, split_lines
from formicmind.replay.writer import ReplaySummary
from formicmind.rng import policy_stream_id
from formicmind.tick import PolicyController, ScriptedController

SEED = MATCH_SMOKE_SEED
CUTOFF_TICKS = 45
RNG_TICKS = 400
PROBES = ("raid_win", "mutual_raid_draw", "collapse", "timeout_draw")


def random_policy(observation, mask, policy_rng):
    """A policy that consumes its own RNG stream: 17 draws per decision, so the stream state really moves."""
    return PolicyLogitsV2([policy_rng.next_u32() / 2**31 - 1 for _ in range(17)])


def quiet_stream_policy(observation, mask, policy_rng):
    """Owns a stream but never draws from it (draws == 0)."""
    return PolicyLogitsV2([0.0] * 17)


def scripted(name: str):
    sc = SCENARIOS[name]
    return ScriptedController(sc.script_a), ScriptedController(sc.script_b)


def rng_controllers(policy=random_policy):
    return (PolicyController(policy, policy_stream_id("rnd-a", 0)),
            PolicyController(policy, policy_stream_id("rnd-b", 0)))


@dataclass(frozen=True)
class Recorded:
    path: Path
    data: bytes
    result: EpisodeResult
    summary: ReplaySummary


def record(path: Path, name: str) -> Recorded:
    if name in SCENARIOS:
        result, summary = record_episode(path, default_config(), SEED, *scripted(name), source=name)
    elif name == "cutoff":
        result, summary = record_episode(path, default_config(), SEED, *scripted("raid_win"),
                                         max_ticks=CUTOFF_TICKS, source=name)
    elif name == "rng_policy":
        result, summary = record_episode(path, default_config(), SEED, *rng_controllers(),
                                         max_ticks=RNG_TICKS, source=name)
    else:
        raise KeyError(name)
    return Recorded(path, path.read_bytes(), result, summary)


def record_all(directory: Path) -> dict[str, Recorded]:
    return {name: record(directory / f"{name}.jsonl", name) for name in (*PROBES, "cutoff", "rng_policy")}


# --- forging: edit records, then re-seal every line so only the *semantics* are wrong -----------------------------

def parse(data: bytes) -> list[dict[str, Any]]:
    records, chain = [], ZERO_CHAIN
    for number, line in enumerate(split_lines(data)):
        record_, chain = open_record(line, chain, number)
        records.append(record_)
    return records


def build(records: list[dict[str, Any]], *, renumber: bool = True) -> bytes:
    out, chain = [], ZERO_CHAIN
    for i, rec in enumerate(records):
        rec = dict(rec)
        if renumber:
            rec["seq"] = i
            if rec["type"] == "result":
                rec["record_count"], rec["chain_head"] = i, chain
        line, chain = seal_record(rec, chain)
        out.append(line)
    return b"".join(out)


def forge(data: bytes, edit: Callable[[list[dict[str, Any]]], None], *, renumber: bool = True) -> bytes:
    records = parse(data)
    edit(records)
    return build(records, renumber=renumber)


def find(records: list[dict[str, Any]], type_: str, nth: int = 0) -> dict[str, Any]:
    """The ``nth`` record of a type (not an event)."""
    return [r for r in records if r["type"] == type_][nth]


def find_event(records: list[dict[str, Any]], kind: str, nth: int = 0, **fields: Any) -> dict[str, Any]:
    """The ``nth`` event record of a kind whose payload matches ``fields``."""
    hits = [r for r in records if r["type"] == "event" and r["e"]["kind"] == kind
            and all(r["e"].get(k) == v for k, v in fields.items())]
    return hits[nth]


def index_of(records: list[dict[str, Any]], rec: dict[str, Any]) -> int:
    return next(i for i, r in enumerate(records) if r is rec)


def fix_event_totals(records: list[dict[str, Any]]) -> None:
    """After editing events: renumber ``i``, refresh the result's count and events_hash (a careful forger)."""
    from formicmind.replay.format import EventsHasher

    hasher, n = EventsHasher(), records[0]["cursor"]["event_index"]
    for rec in records:
        if rec["type"] == "event":
            rec["i"] = n
            n += 1
            hasher.add(rec["e"])
    result = records[-1]
    result["event_count"], result["events_hash"] = hasher.count, hasher.hexdigest()
