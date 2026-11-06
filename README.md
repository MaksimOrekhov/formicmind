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

* Five fixed manual L0 baseline policies (`random`, `turtle`, `rush`, `balanced`, `raider`) for reproducible
  `sim run/eval` diagnostics — [docs/strategic_baselines.md](docs/strategic_baselines.md). They are not trained or
  validated as strong agents. Версии v2 выбираются явно (`имя:v2`), а диагностика, совместимость v1 и границы
  восстановления целых значений описаны в [docs/baselines_v2.md](docs/baselines_v2.md).

* Исследование голода (FOR-25): хронический дефицит и сброс `starvation_debt`, изолированные варианты правил, протокол и результаты —
  [docs/starvation_debt_protocol.md](docs/starvation_debt_protocol.md), [docs/starvation_debt_findings.md](docs/starvation_debt_findings.md),
  `python -m formicmind.starvation_debt run|verify` (проверка и provenance — [docs/starvation_debt_verification.md](docs/starvation_debt_verification.md)). Действующие правила не менялись.
* Независимое подтверждение правила голода (FOR-35): Baselines v2, новые seeds, стагнация без `forgiven`, общие A0-кластеры — [docs/starvation_confirm_protocol.md](docs/starvation_confirm_protocol.md), [docs/starvation_confirm_findings.md](docs/starvation_confirm_findings.md), `python -m formicmind.starvation_confirm preflight|run|verify|verify-static` (современные тесты и историческая воспроизводимость — [docs/starvation_confirm_verification.md](docs/starvation_confirm_verification.md)). Действующие правила не менялись.
* Ecosystem v1 (FOR-43, **только проектирование, код движка не реализован**): отдельная среда на 6–8 колоний — видение [docs/ecosystem_v1_vision.md](docs/ecosystem_v1_vision.md), RFC [docs/ecosystem_v1_rfc.md](docs/ecosystem_v1_rfc.md), контракты [docs/ecosystem_v1_contracts.md](docs/ecosystem_v1_contracts.md), экономика [docs/ecosystem_v1_economy.md](docs/ecosystem_v1_economy.md) (числа воспроизводит `python scripts/ecosystem_budget.py`), миграция и план PR [docs/ecosystem_v1_migration.md](docs/ecosystem_v1_migration.md). L0 не менялся.
* Ecosystem v1, E1-a (FOR-44, **реализовано**): независимый пакет `formicmind.eco` — типизированные схемы, детерминированный генератор общей карты 6–8 колоний, отчёт справедливости, CLI `python -m formicmind.eco map|verify|bench|sweep`; пример карты, бенчмарк и статистика на 1800 картах — [docs/ecosystem_v1_map_generator.md](docs/ecosystem_v1_map_generator.md), [docs/ecosystem_examples/](docs/ecosystem_examples/).

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
