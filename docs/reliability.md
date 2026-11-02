# Надёжность L0: CI, регрессия и что именно она доказывает (FOR-14)

Шаг 7 плана FOR-5. Механика, баланс и интерфейс агента **не менялись**; добавлены автоматический CI, точечные
регрессионные тесты и один fail-fast в загрузчике checkpoint.

## Что подтверждает CI — и что нет

| Слой | Что это | Статус |
|---|---|---|
| **Correctness regression** | `python -m pytest` на Windows / Python 3.12: unit-тесты, ручные golden-фикстуры, инварианты, checkpoint и детерминизм | **есть, автоматически** (`.github/workflows/ci.yml`) |
| **Performance readiness** | порог `p10 ≥ 5000 ticks/s` (FOR-5) | **не проверяется**: benchmark ещё не написан; CI не измеряет скорость |
| **Научные Gate 1/2/3** | баланс, baselines, closed- vs open-loop, ML | **не проводились**; кандидатский баланс `candidate-FOR1-v0.2.1-unvalidated` не калибровался |

Зелёный CI **не** означает, что Gate 0 пройден целиком, а лишь что известные контракты L0 не сломаны. Smoke-матчи
(`match_smoke`) — функциональные проверки, не результаты баланса. Sealed seeds FOR-4 нигде не используются: все seed в
тестах — инженерные константы разработчика.

## CI (`.github/workflows/ci.yml`)

* Триггеры: `push` в `main` и `for-*`, `pull_request` в `main`. Push + PR одной ветки дают два независимых прогона
  (разные `ref`), это принято; новый прогон того же `ref` отменяет предыдущий (`concurrency`), прогоны других коммитов — нет.
* Только `windows-latest`, Python 3.12 (`requires-python >= 3.12`), без матрицы, без Linux. Шаги: checkout → setup-python
  (кэш pip по `pyproject.toml`) → `pip install -e ".[dev]"` → `python -m pytest -q -ra`.
* `permissions: contents: read`, `timeout-minutes: 15`, без секретов, артефактов и внешних сервисов; actions закреплены
  по SHA коммитов (`actions/checkout` v4.2.2, `actions/setup-python` v5.6.0). `PYTHONUTF8=1` — чтобы вывод не зависел от кодовой
  страницы runner.
* Отдельных smoke-шагов нет: smoke-матчи уже входят в pytest (`tests/test_match.py`), полный матч `timeout_draw` (3000 тиков) идёт
  один раз как часть набора. ML, sealed-оценка и турниры в CI не запускаются.
* Локально то же самое: `python -m pytest`. Тест `test_ci_workflow_is_windows_python_312_read_only_and_has_no_linux_job`
  не даёт файлу workflow незаметно разойтись с этим описанием.

## Checkpoint: горизонт `T_max` (FOR-14)

`terminal.validate_outcome(state, None)` (его вызывает `checkpoint_from_bytes`) теперь отвергает незавершённый эпизод на горизонте:

* `outcome is None` и `state.tick > T_max` — **всегда** ошибка (`step_tick` отказывает уже на `tick >= T_max`, значит, состояние невозможно);
* `outcome is None` и `state.tick == T_max` — ошибка: полный эпизод всегда заканчивается таймаутом в конце тика `T_max − 1`
  (`finished:false` на горизонте — подмена);
* мёртвая матка без outcome — ошибка (как и раньше).

**Совместимость с dev-режимами.** `movement_only`/`partial_economy` — не матчи и исход не выносят; после ровно `T_max` тиков такой
прогон законно остаётся с `outcome=None`, `state.tick == T_max`. Минимальный безопасный контракт: загрузчик по умолчанию строг
(для полных эпизодов), а такой dev-checkpoint восстанавливается явным `checkpoint_from_bytes(..., dev_mode=True)`; фиктивный
исход не придумывается, `tick > T_max` не принимается даже так. Dev-checkpoint до горизонта флага не требует. Семантика
mid-tick queen-terminal (`state.tick = t`, `tick_completed = false`) не менялась.

## Матрица «контракт → тесты»

| Контракт | Где проверяется |
|---|---|
| FOR-1 §10.1: ручной fixture 4 тика (еда 52/44/44/48, запасы 68/48→76/56→76/60→76/60) + зеркало | `test_economy.py::test_manual_golden_four_ticks_without_laying_and_mirror_symmetry` |
| FOR-1 стартовое состояние, топология, зеркало узлов | `test_reset.py` |
| FOR-1 §6 бой: 1/4/8 рейдеров (120/30/15 фаз), 3 vs 2 в C, бонус 5/4 по n=1…8, перегиб, щит рабочих, одновременность | `test_combat.py` |
| FOR-1 §9 + решения FOR-13: матка > коллапс > таймаут, партия `eggs_per_lay×cost`, ничьи, граница `T_max` | `test_terminal.py` (+ `test_reliability.py` §1) |
| errata A (маршруты 8/10, AutomaticDepart), B (блок добычи), C (остаток еды), D (голод), E (новорождённые), F (партии яиц) | `test_movement.py`, `test_economy.py`, `test_brood.py`, `test_routing.py` |
| R1: side-local tie-break равностоимостных маршрутов | `test_routing.py` (все пары), `test_movement.py`, `test_reliability.py::test_equal_cost_routes_use_the_side_local_tiebreak_in_a_full_run` |
| FOR-2 G0 (вектор наблюдения A/B; маска) | `test_observation.py::test_g0_golden_vector_side_a_and_b`, `test_actions.py` |
| FOR-2 D-1 (decoder / доли forage) | `test_decoder.py` (раздел «forage decoder: goldens D-1») |
| FOR-2 H-1, I-1 (дом без взрослых; last-seen переживает смерть наблюдателя) | `test_intel.py::test_h1_…`, `::test_i1_…` |
| FOR-2 R-1 (nearest arrival по следующему узлу) | `test_observation.py::test_r1_nearest_arrival_is_keyed_by_next_edge_destination` |
| Приватность ObservationV2 (еда, расплод, приказы, юниты на рёбрах, HP матки без сенсора, свой дом без взрослых) | `test_observation.py`, `test_economy_system.py`, `test_combat.py`, `test_reliability.py` §5 (реальные mid-game состояния) |
| Food ledger и инварианты юнитов/HP/ID/яиц; бой + голод в одном тике | `test_economy_system.py`, `test_combat.py`, `test_reliability.py` §6 (`eggs_per_lay` 1/2/4, `check_invariants=True`) |
| Детерминизм `reset → run_episode`: сид не влияет на механику, 64-bit сиды, `PYTHONHASHSEED` для всех четырёх smoke | `test_match.py`, `test_hashseed_determinism.py`, `test_reliability.py` §2 |
| Checkpoint v3 (схемы, домен хэша, `outcome`, строгий загрузчик) | `test_serialization.py`, `test_terminal.py`, `test_reliability.py` §1, §7 |
| Restore на границах: перед Policy K=30 (с `WindowStats`), середина движения, перед боем / смертельным боем, конец тика collapse/timeout; **сшитый** журнал событий | `test_reliability.py` §3 |
| Симметрия сторон: целые трассы событий при обмене мест (бой, экономика, голод, маршруты, ничья) | `test_reliability.py` §4, `test_economy_system.py`, `test_movement.py`, `test_terminal.py` |
| Версии: `rules_version`, `balance_version`, checkpoint v3, golden-хэши reset | `test_reliability.py::test_contract_and_format_versions_are_pinned`, `test_tick.py`, `test_serialization.py` |
| CI-файл (только Windows, Python 3.12, read-only) | `test_reliability.py::test_ci_workflow_…` |
| Replay v1 (FOR-15): формат, цепочка хэшей, `seek` против кадров движка, `verify` по `step_tick`, RNG-политики, checkpoint+суффикс, матрица порчи, CLI — [replay_v1.md](replay_v1.md) | `test_replay_format.py`, `test_replay.py`, `test_replay_rng_checkpoint.py`, `test_replay_corruption.py`, `test_replay_cli.py` |

## Что сознательно не делается

Нет бенчмарка и профилирования, нет калибровки баланса, baselines/ES/MLP, турниров, sealed-данных, UI (Replay v1 как файл есть, см. [replay_v1.md](replay_v1.md); viewer — нет), генератора
случайных миров. Найденная в будущем настоящая ошибка физики — отдельный issue с минимальным воспроизведением, а не молчаливая правка.
