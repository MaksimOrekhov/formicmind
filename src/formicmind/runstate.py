"""Run-level state outside the game model: RNG stream registry and policy-window counters.

Streams are keyed by role (``world``, ``policy:<policy_id>:<replicate>``), never
by seat. A stream is created only by an explicit ``add_stream`` call, so the
checkpoint content never depends on hidden lazy creation. The rules of L0 v0.2.1
consume no randomness; the registry exists so that harness-level choices
(opponent archetype, random policies) get independent, restorable streams.

``windows`` holds each side's ``WindowStats`` (FOR-2 own_global[25..29]). It is the single
serialized source of truth for those counters: the tick engine feeds them to the policy
snapshot and the checkpoint stores them, so a restored run builds the same observation.

``outcome`` is the serializable end of the episode (FOR-13): ``None`` while the match runs, an
``EpisodeOutcome`` once a terminal rule fired. It is run state, not game state, so ``state_hash`` stays the
mechanics and only ``checkpoint_hash`` changes when the episode ends. ``step_tick`` refuses to run when it is set.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from .rng import WORLD_STREAM_ID, Pcg32, check_stream_id, new_stream
from .snapshot import WindowStats
from .terminal import EpisodeOutcome


@dataclass(slots=True)
class RunState:
    root_seed: int
    streams: dict[str, Pcg32] = field(default_factory=dict)
    windows: list[WindowStats] = field(default_factory=lambda: [WindowStats(), WindowStats()])
    outcome: EpisodeOutcome | None = None

    @classmethod
    def new(cls, root_seed: int, stream_ids: Iterable[str] = ()) -> "RunState":
        run = cls(root_seed=root_seed)
        run.add_stream(WORLD_STREAM_ID)
        for stream_id in stream_ids:
            run.add_stream(stream_id)
        return run

    def add_stream(self, stream_id: str) -> Pcg32:
        check_stream_id(stream_id)
        if stream_id in self.streams:
            raise ValueError(f"stream already registered: {stream_id!r}")
        stream = new_stream(self.root_seed, stream_id)
        self.streams[stream_id] = stream
        return stream

    def stream(self, stream_id: str) -> Pcg32:
        try:
            return self.streams[stream_id]
        except KeyError:
            raise KeyError(f"stream not registered: {stream_id!r}") from None

    def clone(self) -> "RunState":
        return RunState(
            self.root_seed,
            {k: v.clone() for k, v in self.streams.items()},
            list(self.windows),  # WindowStats is frozen
            self.outcome,  # EpisodeOutcome is frozen
        )
