# FOR-25: строгая проверка, provenance и разделение «современных» и «исторических» проверок

Документ описывает, что именно проверяет `python -m formicmind.starvation_debt verify`, какие сведения считаются историческими, как соотносятся версия харнесса, создавшая данные, и версия, которая их проверяет, и **как проверки FOR-25 устроены, чтобы дальнейшие изменения движка (FOR-26, FOR-27, обучение ИИ) не ломали исторические исследования**. Он появился после ревью PR #17. Протокол, erratum E1, варианты правил, опубликованные `matches.jsonl`, `summary.json`, `minimal_demo.json`, `reproduction.json`, `manifest.json` и выводы **не менялись**.

## Почему проверки разделены

FOR-25 привязан к движку базового коммита `c2a887c` (FOR-24): fingerprint движка без харнесса `5fd459d1881a2316ef10f120b6d913706c77591ab22d198d9f7f1b972ef195d8`. CI на pull request тестирует **merge-ref** с актуальным `main`, а в нём уже есть FOR-26 (Baselines v2) и будут другие изменения движка. Поэтому одна и та же проверка не может одновременно быть и «тестом текущего дерева», и «регенерацией исторических данных». Решение — два окружения с разными гарантиями и **без пропусков (`skip`/`xfail`)**:

| | Современное дерево | Историческое окружение |
| --- | --- | --- |
| Где | обычный `pytest` (`testpaths = ["tests"]`), job `test` | `historical/for25/`, job `historical-for25` |
| Какой код | текущий checkout (merge-ref) | изолированный checkout **закреплённого SHA** (не ветки) |
| Требует движок базы | нет | да, и явно проверяет это |
| Что гарантирует | корректность логики харнесса, метрик, окон, блоков, вердикта; отказ при подмене полей манифеста; неизменность протокола, erratum и защищённых файлов; manifest-only проверку опубликованных данных | побайтную воспроизводимость 4320 матчей и всех артефактов неизменным движком, равенство A0 производственной `economy.upkeep`, строгий `verify` с регенерацией, отрицательные тесты end-to-end |

## Что проверяет `verify`

Режим **`--manifest-only`** (работает на любом checkout, ничего не регенерирует и прямо сообщает `reproducibility: not_verified_in_this_mode`):

1. Ядро, определяющее результаты, равно ядру генератора (`experiment_core_sha256() == GENERATOR_CORE_SHA256`, `332bd3d7…`). В ядро входят исходники функций правил, прогона, метрик, окон, блоков, сводки, вердикта, сериализации артефактов и замороженные константы. Хеш посчитан на исходнике коммита `4c2cbeb`; правки verifier его не изменили.
2. **Все детерминированные поля манифеста** равны **замороженным значениям**, а не значениям текущего дерева: `schema`, `protocol_id`, `protocol_sha256`, `erratum_sha256`, `base_commit`, `base_engine_fingerprint`, `engine_fingerprint_excluding_harness` (замороженный fingerprint базы из протокола), `configs`, `policy_parameter_sha256`, `seeds`, `seeds_sha256`, `variants`, `pairs`, `maps`, `seats`, `window_columns`, `episode_columns`, `checkpoint_columns`, `checkpoint_ticks`, `expected_rows`, `budget`, `versions` (`FROZEN_VERSIONS`), `protected_file_sha256` (`FROZEN_PROTECTED_FILE_SHA256`). Лишние и пропущенные ключи отвергаются.
3. Согласованность исторического `package_fingerprint` и `source_commit` (см. ниже).
4. Список и SHA-256 артефактов (после нормализации CRLF→LF).

**Полный режим** (по умолчанию) после этого вызывает `_check_static_inputs()` — проверку, что окружение действительно историческое: fingerprint движка равен базе, версии равны замороженным, хеши протокола, erratum и защищённых файлов равны замороженным, seeds, политики и карты не отличаются. **Если нет, `verify` явно отказывает** сообщением о необходимости изолированного checkout закреплённого коммита. Затем он регенерирует все ступени неизменным движком и харнессом, сравнивает `matches.jsonl`, `minimal_demo.json`, `reproduction.json`, `summary.json` побайтно и сверяет число вызовов `step_tick`. Молча воспроизвести исторические данные на изменённом движке `verify` не может.

## Что считается историческим и не сравнивается

Список зашит в код (`PROVENANCE_KEYS`) и не переписывается: `source_commit`, `package_fingerprint`, `source_tree_dirty` (есть только у новых запусков), `python`, `platform`, `commands`. `artifacts` и `calls` проверяются регенерацией.

**Привязка fingerprint к коммиту.** Если `source_commit` доступен в клоне, verifier вычисляет fingerprint `src/formicmind` в этом коммите тем же алгоритмом и требует равенства `package_fingerprint` (статус `bound`). Расхождение — ошибка. Грязное дерево (`source_tree_dirty: true`) — `not_checked_dirty_worktree`; коммита нет в клоне — `commit_unavailable` (не аутентифицировано, но и не принимается как доказательство; доказательство — побайтная регенерация). Job `historical-for25` берёт полную историю, поэтому для опубликованных данных получается `bound`.

## Две версии харнесса — не одна

| Роль | Коммит | Fingerprint пакета `src/formicmind` |
| --- | --- | --- |
| Генератор опубликованных данных (записан в `manifest.json`, не переписывался) | `4c2cbeb4a947865914118441cbe48a9148600df7` | `f8263c407a0509cece3206e139d5d8a6e7c800c21ee33500226f64dd6c2c2ca1` |
| Закреплённый verifier / историческое окружение (job `historical-for25`) | `737de9cd7ed2451d45a59ca21da9167476a47660` | см. `verification_record.json` (`verifier.package_fingerprint`) |
| Движок без харнесса (одинаков) | — | `5fd459d1881a2316ef10f120b6d913706c77591ab22d198d9f7f1b972ef195d8` (= база `c2a887c`) |

Правка `starvation_debt.py` меняет fingerprint пакета, поэтому текущий verifier **не выдаётся** за генератор. `verify --record` пишет канонический JSON без wall-clock (`results/starvation_debt/verification_record.json`): подлинная provenance генератора, идентичность verifier, режим (`full_regeneration` / `manifest_only`), статус привязки, проверенные и не сравнивавшиеся поля.

## Как запускать

Современное дерево (любой актуальный checkout):

```bash
python -m pytest -q -ra
python -m formicmind.starvation_debt verify --input results/starvation_debt --manifest-only
```

Историческая проверка — **в изолированном worktree закреплённого SHA** (так делает CI):

```bash
git worktree add ../formicmind-for25 737de9cd7ed2451d45a59ca21da9167476a47660
cd ../formicmind-for25
python -m pip install -e ".[dev]"
python -m pytest historical/for25 -q -ra
python -m formicmind.starvation_debt verify --input results/starvation_debt --jobs 4
```

Запуск `historical/for25` на дереве с иным движком падает громко, а не пропускается. Закреплённый SHA нельзя «обновлять вслед за `main`»: новые данные требуют нового протокола.

## Ограничение

Генератор и verifier — один и тот же код ядра, поэтому проверка подтверждает воспроизводимость и целостность данных, а не независимую реализацию.
