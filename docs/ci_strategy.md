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
3. Всегда запускать короткий Windows path-policy gate. Изменения в CI/workflow, классификаторе, исходниках симулятора, исторических тестах, протоколах, зависимостях и опубликованных historical results требуют полных pinned FOR-25/FOR-35 jobs в этом PR. Их прежние check contexts остаются настоящими успешными/неуспешными checks. Для явно независимых областей (ML-only, viewer и неисторическая документация) jobs завершаются с явным сообщением о решении policy gate; они не заявляют, что regeneration запускалась. Ошибка определения diff должна завершить gate ошибкой. Новые/неизвестные пути классифицируются как sensitive.
4. Добавить отдельный Windows workflow `Historical verification`: полный preflight/tests/строгая byte-for-byte регенерация только на неизменных FOR-25 и FOR-35 SHA. Он запускается еженедельно и вручную. Manual run принимает выбранный target ref/SHA и выводит его в логе; исторические генераторы при этом всё равно checkout-ят только зафиксированные pin SHA.
5. Проверка branch-protection API для `main` вернула 403 `Resource not accessible by integration`; доступные через connector repo rulesets не дают основания утверждать, какие checks фактически обязательны. Поэтому сохраняем три существующих имени в workflow `CI`, не меняем и не предполагаем настройки защиты. В PR следует отдельно проверить требуемые contexts в настройках репозитория.

### Замороженные исторические pins

- FOR-25: `737de9cd7ed2451d45a59ca21da9167476a47660`
- FOR-35: `3139c7b953c38d96d35e0baf2267781138ebcb71`

Ни SHA, ни исторические документы/manifest/results в этой задаче не редактируются. Быстрые static checks явно не называются полной воспроизводимостью: полная строгая регенерация сохраняется на чувствительном PR и в weekly/manual workflow.

## Проверка результата

После изменения записать те же показатели из Actions API: время каждого PR job, исторические job и verify steps, время от run start до final status и дополнительный status lag после последнего job. Публиковать ссылку на обычный PR run и на новый manual historical run; сравнивать с таблицей выше, отмечая влияние кэшей и runner очередей.
