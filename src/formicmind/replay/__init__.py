"""Replay v1 (FOR-15): versioned JSONL records of full matches, ``ViewFrame`` seek and deterministic ``verify``.

Layers (see docs/replay_v1.md):

* ``writer``  - ``record_episode`` / ``record_play`` / ``ReplayWriter``: record the real ``step_tick`` outputs;
* ``reader``  - ``ReplayReader``, ``read_header``, ``iter_events``, ``seek``, ``read_result``: strict, no simulation;
* ``verify``  - ``verify``: re-execute through the one ``step_tick`` without the original policies;
* ``frames``  - the omniscient ``ViewFrame`` and its display patches.
"""

from .format import (
    ReplayError,
    ReplayFormatError,
    ReplayVerifyError,
    ReplayVersionError,
    ReplayWriteError,
    SeatSpec,
    SeekError,
)
from .frames import BOUNDARY_START, BOUNDARY_TERMINAL, ViewFrame, view_from_state
from .reader import ReplayReader, iter_events, read_header, read_result, seek
from .verify import VerifyReport, verify
from .writer import ReplayWriter, record_episode, record_play

__all__ = [
    "BOUNDARY_START", "BOUNDARY_TERMINAL", "ReplayError", "ReplayFormatError", "ReplayReader", "ReplayVerifyError",
    "ReplayVersionError", "ReplayWriteError", "ReplayWriter", "SeatSpec", "SeekError", "VerifyReport", "ViewFrame",
    "iter_events", "read_header", "read_result", "record_episode", "record_play", "seek", "verify",
    "view_from_state",
]
