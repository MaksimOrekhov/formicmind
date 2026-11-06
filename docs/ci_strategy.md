# Стратегия CI: быстрые PR-проверки и историческая регенерация

## Зафиксированная база до изменений

Замеры сняты из GitHub Actions API до изменения workflow, включая job/step timestamps.

| Run | HEAD | Windows CI | FOR-25 | FOR-35 | Весь run до final status |
|---|---|---:|---:|---:|---:|
| [41877135182](https://github.com/MaksimOrekhov/formicmind/actions/runs/41877135182) | `efb8994a60c2e598bed0a67a1ac942b91799a33c` | 6:30 | 12:39 (verify 9:31) | 12:59 (verify 11:42) | 20:41 |
| [41882222024](https://github.com/MaksimOrekhov/formicmind/actions/runs/41882222024) | `d7585a08c8002a05af149669db845889765fdb51` | 3:52 | 13:44 (verify 10:43) | 13:06 (verify 11:37) | 13:49 |

Jobs запускались параллельно. В первом run последний job закончился в 17:36:26 UTC, а workflow получил final status в 17:44:05: задержка 7:39 уже после вычислений. Во втором run последний job закончился в 18:11:45, final status появился в 18:11:46. Это внешняя изменчивая задержка GitHub; перенос historical jobs не гарантирует уменьшение полного времени PR с 20:41 до длительности одного fast job.

Основные шаги первого Windows job: установка dev-зависимостей 0:10, CPU PyTorch 0:55, pytest 3:43, `npm ci` 0:49, build 0:10 и Vitest 0:11. В повторном прогоне те же pytest заняли 1:56, PyTorch 0:39 и `npm ci` 0:17. Разница указывает на влияние кэша/runner, поэтому после изменения нужно сравнивать реальные runs, а не обещать фиксированное ускорение.

## Решение

1. Оставить workflow `CI` для каждого PR и сохранить job IDs и отображаемые имена существующих проверок: `test` / `Windows CI (Python 3.12, Node 22)`, `historical-for25` / `Historical FOR-25 (frozen engine, pinned commit)` и `historical-for35` / `Historical FOR-35 (frozen engine, pinned commit)`. Эти contexts могут быть выбраны как required checks; число jobs не является контрактом.
2. Сохранить в PR полный pytest с CPU PyTorch, Vue build, Vitest, FOR-25 `verify --manifest-only` и FOR-35 `verify-static`. Вынести viewer в отдельный Windows job, чтобы он выполнялся параллельно с Python suite, не дублируя Python/PyTorch installation.
3. Всегда запускать короткий Windows path-policy gate. Изменения в CI/workflow/actions, классификаторе, исходниках симулятора, исторических тестах, протоколах, зависимостях и опубликованных historical results требуют полных pinned FOR-25/FOR-35 jobs в этом PR. Их прежние check contexts остаются настоящими успешными/неуспешными checks. Fast-only allowlist ограничен `src/formicmind/ml/`, `tests/test_ml_*`, `viewer/`, `docs/ml_*` и этим файлом стратегии. Для этих явно независимых областей jobs завершаются с явным сообщением о решении policy gate; они не заявляют, что regeneration запускалась. Любой другой, неизвестный или пустой набор путей считается sensitive. Ошибка определения diff или policy gate должна завершить run ошибкой.
4. Добавить отдельный Windows workflow `Historical verification`: полный preflight/tests/строгая byte-for-byte регенерация только на неизменных FOR-25 и FOR-35 SHA. Он запускается еженедельно и вручную. Manual run принимает выбранный target ref/SHA и выводит его в логе; исторические генераторы при этом всё равно checkout-ят только зафиксированные pin SHA. GitHub разрешает `workflow_dispatch` только если workflow-файл присутствует в default branch ([документация](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow)); поскольку PR оставлен открытым, ручной dispatch до merge не запускался.
5. Проверка branch-protection API для `main` вернула 403 `Resource not accessible by integration`; доступные через connector repo rulesets не дают основания утверждать, какие checks фактически обязательны. Поэтому сохраняем три существующих имени в workflow `CI`, не меняем и не предполагаем настройки защиты. В PR следует отдельно проверить требуемые contexts в настройках репозитория.

### Замороженные исторические pins

- FOR-25: `737de9cd7ed2451d45a59ca21da9167476a47660`
- FOR-35: `3139c7b953c38d96d35e0baf2267781138ebcb71`

Ни SHA, ни исторические документы/manifest/results в этой задаче не редактируются. Быстрые static checks явно не называются полной воспроизводимостью: полная строгая регенерация сохраняется на чувствительном PR и в weekly/manual workflow.

## Измеренный результат после изменения

Измерения взяты из GitHub Actions API для [run 41892800504](https://github.com/MaksimOrekhov/formicmind/actions/runs/41892800504), HEAD `f7836118d628d21ef7ccd5daf765ced62b0dc626`. Все пять jobs прошли успешно.

| Проверка | Время job | Полный verify |
|---|---:|---:|
| Windows CI: Python, полный pytest и ML smoke | 4:11 | — |
| Viewer: Vue build и Vitest | 0:57 | — |
| Historical sensitivity gate | 0:18 | — |
| FOR-25: tests и pinned full regeneration | 13:36 | 10:06 |
| FOR-35: preflight, tests и pinned full regeneration | 10:16 | 8:59 |
| Весь run до final status | 14:02 | status lag после последнего job: 0:01 |

Проверки fast-path выполняются параллельно: в этом run от старта workflow до завершения последней из них прошло 4:14. Старый единый modern job занял 3:52 во втором baseline run и 6:30 в первом; это не показывает гарантированного ускорения короткого пути, поскольку runner/cache и разделение viewer job меняют нагрузку. Этот PR меняет CI и потому намеренно классифицируется как sensitive: он запускал также обе полные исторические проверки, которые и определили общее время 14:02. Отдельный PR только из allowlist здесь не создавался, поэтому 4:14 — измеренное время fast jobs в составе sensitive run, а не отдельный end-to-end замер ML-only PR.

FOR-25 подтвердил 4 320 строк, 2 073 000 main и 404 004 anchor verification calls; FOR-35 подтвердил 4 800 строк и 960 anchors. Перед исправлением изоляции оба verify тоже прошли, однако job ошибался при post-cleanup; успешным CI считается только последующий run выше, где action cleanup завершился успешно.

## Ecosystem v1 (FOR-44)

Быстрым путём считаются также `src/formicmind/eco/`, `tests/test_eco_`, `tests/test_ecosystem_`, `docs/ecosystem_` и `scripts/ecosystem_`. Основание: `tests/test_eco_boundary.py` проверяет AST-анализом, что пакет `formicmind.eco`
импортирует из L0 только `canonical` и `rng`, а ни один модуль, тест или скрипт L0 не импортирует `eco`; это часть обычного полного `pytest` на Windows. Любой другой или смешанный путь (включая L0, `pyproject.toml`, `.github`, результаты)
остаётся чувствительным и запускает закреплённые FOR-25/FOR-35; пути с `..`, `.`, `//`, абсолютные и с буквой диска считаются чувствительными. PR, меняющий саму политику (FOR-44), чувствительный и выполняет полную историческую проверку.
Ни одна существующая проверка не отключена.

