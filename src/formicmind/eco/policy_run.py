"""Running a world whose colonies are decided by independent ``PolicyE1`` policies (FOR-48).

``PolicyController`` is the engine's ``Controller``. At every decision (planned each ``decision_k`` ticks, or the extra ``alert`` decision of a
colony in a fight) it works in two phases on **one world snapshot**:

1. *decide* (the world is not touched): for every deciding colony it builds ``ObservationE1`` from that colony's own state, derives the masks from the
   observation alone, asks the colony's own policy and validates / decodes the answer. A policy failure raises and leaves the world exactly as it was;
2. *apply* (atomic): the actions are applied colony after colony through ``apply.py``; the closed accounting window and the rolling averages are
   updated. If anything refuses, the cohorts, orders and accumulators are restored and ``PolicyApplyError`` is raised: a technical error, never a defeat.

Orphans (queen dead) decide nothing; they keep re-running the last foraging plan (standing orders, contracts 5). Every decision appends a compact
record (tick, colony, observation hash, action) to a hash-chained ``DecisionLog``; the records never contain a world id.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..canonical import canonical_dumps, domain_hash
from .action import ActionE1, action_masks, action_to_canonical, validate_action
from .apply import apply_action, dispatch_forage
from .config import ConfigE1
from .decoder import decode
from .engine import RouteTable
from .model import MapE1, WindowStatsE1, WorldStateE1
from .observation import build_observation, next_rolling, observation_hash
from .policy import PolicyE1

DECISIONS_DOMAIN = "formicmind/eco/decisions/v1"
ACTION_DOMAIN = "formicmind/eco/action/v1"


class PolicyApplyError(RuntimeError):
    """Applying a validated action failed; the state was rolled back to the snapshot the colonies decided on."""


@dataclass(slots=True)
class DecisionLog:
    """Append-only, hash-chained list of decision records. ``head`` binds every record and its order."""

    records: list[dict] = field(default_factory=list)
    head: str = ""
    count: int = 0

    def append(self, record: dict) -> None:
        self.head = domain_hash(DECISIONS_DOMAIN, canonical_dumps({"prev": self.head, "decision": record}))
        self.records.append({**record, "chain": self.head})
        self.count += 1

    def to_bytes(self) -> bytes:
        return b"".join(canonical_dumps(record) + b"\n" for record in self.records)


class PolicyController:
    """One independent policy per colony behind the engine's controller interface.

    ``batcher`` (optional, FOR-49) is an object with ``act_batch(items) -> list[answer]`` where ``items`` is a list of ``(colony, obs, masks)``:
    the colonies that decide at the same tick are then answered by one call (one network forward pass) instead of one ``act`` each. The answers
    are validated and applied exactly as before; the batcher sees the same private observations and nothing else.
    """

    def __init__(self, policies: Sequence[PolicyE1], log: DecisionLog | None = None, batcher=None) -> None:
        self.policies = list(policies)
        self.batcher = batcher
        self.log = DecisionLog() if log is None else log
        self.last_decision = [-1] * len(self.policies)
        self.decisions = [0] * len(self.policies)

    # -- the engine calls this in the Policy phase
    def __call__(self, state: WorldStateE1, config: ConfigE1, world_map: MapE1, routes: RouteTable) -> None:
        if len(self.policies) != len(state.colonies):
            raise ValueError("one policy per colony is required")
        kind = "alert" if routes.alert else "planned"
        deciding = sorted(routes.alert) if routes.alert else [c.colony_id for c in state.colonies if c.status == "alive"]
        deciding = [cid for cid in deciding if state.colonies[cid].status == "alive"]
        tick = state.tick

        # ---- phase 1: decide on the snapshot; nothing is written to the world
        plans = []
        pending = []
        for cid in deciding:
            colony = state.colonies[cid]
            own = [c for c in state.cohorts if c.colony_id == cid]
            delta = 0 if self.last_decision[cid] < 0 else tick - self.last_decision[cid]
            workers = sum(c.count for c in own if c.caste == "worker")
            soldiers = sum(c.count for c in own if c.caste == "soldier")
            rolling = next_rolling(colony.rolling, colony.window, workers, soldiers, delta)
            obs = build_observation(colony, own, tick, config, decision_index=self.decisions[cid], ticks_since_last_decision=delta,
                                    rolling=rolling)
            masks = action_masks(obs)
            pending.append((cid, obs, masks, rolling))
        if self.batcher is not None:
            answers = self.batcher.act_batch([(cid, obs, masks) for cid, obs, masks, _ in pending])
        else:
            answers = [self.policies[cid].act(obs, masks) for cid, obs, masks, _ in pending]
        for (cid, obs, masks, rolling), answer in zip(pending, answers):
            action = answer if isinstance(answer, ActionE1) else decode(answer, masks)
            validate_action(action, masks)
            plans.append((cid, obs, action, rolling))

        # ---- phase 2: apply atomically
        touched = sorted({*deciding, *(c.colony_id for c in state.colonies if c.status == "orphan")})
        backup = (copy.deepcopy(state.cohorts), state.next_cohort_id,
                  {cid: copy.deepcopy((state.colonies[cid].orders, state.colonies[cid].window, state.colonies[cid].rolling)) for cid in touched})
        outbox = len(routes.outbox)
        try:
            for cid, _obs, action, rolling in plans:
                apply_action(state, config, world_map, cid, action, routes.outbox)
                state.colonies[cid].rolling = rolling
                state.colonies[cid].window = WindowStatsE1()
            if kind == "planned":
                for colony in state.colonies:
                    if colony.status == "orphan":
                        dispatch_forage(state, config, world_map, colony)
        except Exception as exc:
            state.cohorts, state.next_cohort_id = backup[0], backup[1]
            for cid, (orders, window, rolling) in backup[2].items():
                state.colonies[cid].orders, state.colonies[cid].window, state.colonies[cid].rolling = orders, window, rolling
            del routes.outbox[outbox:]
            raise PolicyApplyError(f"tick {tick}: applying the decisions failed ({type(exc).__name__}: {exc}); the world was restored") from exc

        for cid, obs, action, _rolling in plans:
            canonical = action_to_canonical(action)
            self.log.append({"tick": tick, "colony": cid, "kind": kind, "index": self.decisions[cid], "obs": observation_hash(obs),
                             "action": canonical, "action_hash": domain_hash(ACTION_DOMAIN, canonical_dumps(canonical))})
            self.last_decision[cid] = tick
            self.decisions[cid] += 1

    # -- checkpointing of the controller (the world has its own checkpoint)
    def snapshot(self) -> dict:
        return {"last_decision": list(self.last_decision), "decisions": list(self.decisions),
                "log": {"head": self.log.head, "count": self.log.count}, "policies": [p.snapshot() for p in self.policies]}

    def restore(self, data: dict) -> None:
        self.last_decision, self.decisions = list(data["last_decision"]), list(data["decisions"])
        self.log.head, self.log.count = data["log"]["head"], data["log"]["count"]
        for policy, item in zip(self.policies, data["policies"]):
            policy.restore(item)


# ------------------------------------------------------------------------------------------------ headless run


@dataclass(slots=True)
class PolicyRunResult:
    config: ConfigE1
    seed: int
    spec: object
    state: WorldStateE1
    events: object        # EventLog of the run
    decisions: DecisionLog
    controller: PolicyController
    state_hash: str
    events_hash: str
    decisions_hash: str


def run_policies(config: ConfigE1, seed: int, spec, ticks: int | None = None) -> PolicyRunResult:
    """Headless run of a real generated world with one independent policy per colony. Same inputs give the same hashes."""
    from .engine import step_world_tick
    from .events import EventLog
    from .initial import build_world
    from .policy import build_policies
    from .serialization import state_hash

    duration = config.horizon if ticks is None else ticks
    if type(duration) is not int or not 0 <= duration <= config.horizon:
        raise ValueError("ticks must be an integer between zero and the configured horizon")
    world = build_world(config, seed)
    routes = RouteTable(world.world_map)
    controller = PolicyController(build_policies(spec, config))
    events = EventLog()
    for _ in range(duration):
        record = step_world_tick(world.state, world.run, config, world.world_map, controller, routes=routes,
                                 validate_input=world.state.tick == 0)
        events.extend(record.events)
        if world.state.world_outcome.finished:
            break
    return PolicyRunResult(config, seed, spec, world.state, events, controller.log, controller, state_hash(world.state), events.head,
                           controller.log.head)
