# FormicMind

Headless-симулятор L0 (Python, только stdlib). Реализовано:

* фундамент (FOR-9): модель мира, `reset`, RNG, каноническая сериализация, SHA-256 хэши —
  [docs/foundation.md](docs/foundation.md);
* граница агента (FOR-10): ObservationV2 (223 float32), ActionMaskV2 (30 bool), приватная разведка,
  общий decoder 17 logits → ActionV2, policy-интерфейс — [docs/agent_boundary.md](docs/agent_boundary.md).

* каркас тика и движение (FOR-11): оркестратор фаз, применение приказов, квоты рабочих, маршруты, прибытия,
  события, `WindowStats` в checkpoint v2 — [docs/tick_movement.md](docs/tick_movement.md).
* экономика и расплод (FOR-12): Harvest, Upkeep/Starvation, Lay (партии яиц), Completion/Hatch, Regeneration,
  food ledger — [docs/economy.md](docs/economy.md).
* бой, терминальные условия и runner (FOR-13): одновременный Combat, урон солдатам/рабочим, защита и гибель матки,
  экономический коллапс, таймаут, `EpisodeOutcome` в checkpoint v3, `run_episode` — [docs/combat_terminal.md](docs/combat_terminal.md).
  `step_tick` без флагов исполняет все фазы; `movement_only`/`partial_economy` — неполные dev-режимы. ML и турниров нет.

* надёжность (FOR-14): GitHub Actions CI (**только Windows, Python 3.12**), матрица «контракт → тесты», строгий horizon-check
  загрузчика checkpoint — [docs/reliability.md](docs/reliability.md). CI подтверждает **correctness regression**, а не performance
  readiness и не научные Gate 1/2/3.

* Replay v1 (FOR-15): версионированный JSONL полного матча (hash-chain, решения с RNG-состояниями политик, события, omniscient
  `ViewFrame` + дисплейные патчи), `seek` на любом тике без исполнения правил, `verify` через тот же `step_tick` и CLI
  `python -m formicmind.replay --verify FILE` / `sim replay --verify FILE` — [docs/replay_v1.md](docs/replay_v1.md).
  Записи smoke-матчей: `python -m formicmind.replay_smoke --out replays`.

Smoke-прогоны: движения (90 тиков) `python -m formicmind.smoke`; экономики (120 тиков, PARTIAL) `python -m formicmind.economy_smoke`;
**полные матчи** (4 сценария, функциональная проверка) `python -m formicmind.match_smoke`.

## Запуск тестов

Нужен Python ≥ 3.12 (проверено на 3.12.10).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # Linux/macOS: .venv/bin/python
.venv/Scripts/python -m pytest
```

`pytest` берёт пакет из `src/` (настройка `pythonpath` в `pyproject.toml`), поэтому после
установки dev-зависимости достаточно `python -m pytest`.

## Использование

```python
from formicmind import default_config, reset, state_hash, state_to_bytes, state_from_bytes

config = default_config()
state, run = reset(config, seed=20261102, stream_ids=["policy:baseline:0"])
data = state_to_bytes(state)                      # canonical JSON bytes
assert state_hash(state_from_bytes(data, config)) == state_hash(state)
```
