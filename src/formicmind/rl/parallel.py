"""CPU-parallel match execution (Windows ``spawn``), FOR-42.

Workers run full engine matches with CPU copies of the actor weights; the learner (GPU) never runs inside a worker.
Each task is a chunk of match specs plus the *named* weight sources those specs reference:

* ``refs[name] = {"bytes": <torch-saved actor state>}``  - the current learner weights of this iteration;
* ``refs[name] = {"path": ..., "sha256": ...}``          - an immutable snapshot file (cached per worker by path).

``MatchExecutor(workers=0)`` runs the identical code inline (tests, 1-process smoke, parity checks).
"""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import Any

import torch

from .network import ActorCritic, actor_state_from_bytes, build_actor
from .pool import file_sha256
from .rollout import run_match

_PATH_CACHE: dict[tuple[str, str], ActorCritic] = {}


def _init_worker() -> None:
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def _resolve_actors(refs: dict[str, dict[str, Any]]) -> dict[str, ActorCritic]:
    actors: dict[str, ActorCritic] = {}
    for name, source in refs.items():
        if "bytes" in source:
            actors[name] = build_actor(actor_state_from_bytes(source["bytes"]))
            continue
        key = (source["path"], source["sha256"])
        if key not in _PATH_CACHE:
            if file_sha256(__import__("pathlib").Path(source["path"])) != source["sha256"]:
                raise ValueError(f"weights file {source['path']} does not match its recorded SHA-256")
            with open(source["path"], "rb") as handle:
                _PATH_CACHE[key] = build_actor(actor_state_from_bytes(handle.read()))
        actors[name] = _PATH_CACHE[key]
    return actors


def execute_chunk(task: tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    specs, refs = task
    actors = _resolve_actors(refs)
    return [run_match(spec, actors) for spec in specs]


class MatchExecutor:
    def __init__(self, workers: int, chunk_size: int = 4) -> None:
        self.workers = workers
        self.chunk_size = max(1, chunk_size)
        self._pool = None
        if workers > 0:
            self._pool = mp.get_context("spawn").Pool(workers, initializer=_init_worker)

    def run(self, specs: list[dict[str, Any]], refs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        """Results in the order of ``specs``."""
        chunks = [specs[i:i + self.chunk_size] for i in range(0, len(specs), self.chunk_size)]
        tasks = [(chunk, refs) for chunk in chunks]
        if self._pool is None:
            batches = [execute_chunk(task) for task in tasks]
        else:
            batches = list(self._pool.imap(execute_chunk, tasks))
        return [result for batch in batches for result in batch]

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self) -> "MatchExecutor":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = ["MatchExecutor", "execute_chunk"]
