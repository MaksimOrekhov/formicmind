# Replay v1: версионированный JSONL, ViewFrame/seek, `verify` (FOR-15)

Шаг 8 плана FOR-5. Механика, баланс, ObservationV2/ActionV2/decoder и формула `checkpoint_hash` **не менялись**.
Replay — локальный, детерминированный файл полного матча: его можно (1) прочитать и посмотреть на любом тике **без
исполнения правил** и (2) независимо проверить (`verify`) через **тот же** `step_tick`. Реального viewer'а, сервера и
`sim run/benchmark` здесь нет; это контракт файла для следующего локального визуализатора.

Код: пакет `src/formicmind/replay/` (`format`, `frames`, `writer`, `reader`, `verify`, `cli`), `replay_smoke.py`, `cli.py`.

## 1. Файл

UTF-8 JSON Lines (фактически ASCII: канонический JSON экранирует всё остальное), перевод строки только `\n`, **одна
каноническая JSON-запись на строку**, без BOM, пустых строк и хвостового мусора. «Каноническая» = `canonical_dumps`
проекта: отсортированные ключи, без пробелов, только `int`/`str`/`bool`/`list`/`dict`, никаких float, NaN, `None`,
путей, таймзон и wall-clock; u64 (seed, состояния PRNG) — десятичные строки. Читатель принимает только байты, которые
перекодируются сами в себя.

У каждой записи есть `type`, `seq` (номер строки с 0) и `chain`. Схема версии — `replay_schema = "v1"`; она отделена от
`rules_version`, `balance_version`, версий Observation/Action/decoder, `checkpoint_v3` и т. д. (они повторены в
`header.contract`, см. §6).

### Цепочка хэшей (целостность файла)

```
chain[i] = SHA-256("formicmind/replay-record/v1" 0x00 chain[i-1] 0x00 line[i], где поле chain заменено 64 нулями)
chain[-1] = 64 нуля
```

Изменение, удаление, вставка или перестановка любой строки ломает её `chain` и все следующие; `result` повторяет голову
цепочки (`chain_head`), `record_count` и `seq`. Побитовый флип в любом месте файла даёт `ReplayFormatError` (проверено
выборкой 400 позиций). Цепочка — защита от порчи; от **намеренной** подделки с пересчитанными хэшами защищает `verify`:
он заново исполняет матч.

### Виды записей и порядок

```
header
snapshot                  start-кадр первого тика (t=0 — ДО Policy(0))
-- для каждого исполненного тика t --
decision                  только при t % K == 0 (K из config)
event*                    Event.to_canonical() в порядке движка, со сквозным индексом i
delta                     дисплейный патч: кадр до тика -> кадр после тика
snapshot                  start-кадр t+1, если (t+1) % K == 0 и матч продолжается;
                          либо terminal-кадр, если тик завершил матч
-- конец --
snapshot                  start-кадр «cutoff» — только у усечённого прогона, остановленного не на границе K
result
```

Читатель проверяет эту грамматику целиком: после terminal-тика ничего, кроме terminal-снимка и `result`; после `result`
ничего.

**header** — `format="formicmind.replay"`, `replay_schema`, `engine` (статичный pin без git-хэша), `contract` (все версии и
домены хэшей), полный `config` (canonical) и его `config_hash`, `root_seed` (decimal string), `streams` (зарегистрированные
RNG-потоки кроме `world`), `seats` (по одному дескриптору на сторону), `start`, `cursor`, `source`, `meta`.
`seats[i] = {side, policy_id, kind, stream_id, ref}` — **данные**, а не исполняемые объекты: `kind` ∈ `scripted|policy|other`,
`stream_id` — поток, которым владеет контроллер (`""`, если он не потребляет RNG), `ref` — произвольная ссылка на конфиг
(для scripted — sha256 скрипта). `source`/`meta` — метаданные разработчика: они в цепочке, но не влияют на воспроизведение
(тест: разные `source/meta` дают одинаковые `events_hash/state_hash/checkpoint_hash`).

**decision** — для обеих сторон: `action` (ActionV2 **после** общего `validate_action`), `obs_hash` (SHA-256 с доменом
`formicmind/observation-f32le/v1` над теми самыми 223·4 байтами float32 little-endian, которые получила политика), `mask`
(30 символов `0/1` — ActionMaskV2), `rng` (см. §4). Скрытое состояние противника не записывается: решение содержит только то,
что видела политика, плюс её ответ.

**event** — `{"i": сквозной индекс, "e": Event.to_canonical()}`. Журнал **омнисциентный** (события обеих сторон,
абсолютные `stock_after`, `food_after`, `queen_hp_after`, `hp_after`); ObservationV2 остаётся приватной. События не
переупорядочиваются: `events_hash` в `result` — тот же `events.events_hash` всей последовательности (считается
потоково, без хранения списка).

**result** — `finished`, `outcome` (`{"finished": false}` либо `winner, reason, terminal_tick, terminal_phase,
phases_executed, sides, tick_completed`), `state_tick`, `ticks_executed`, `decision_count`, `event_count`, `events_hash`,
`final_state_hash`, `final_checkpoint_hash`, `record_count`, `chain_head`. Усечённый harness-прогон (`max_ticks`) честно
записывается как `finished=false`, `outcome={"finished": false}`; исход не придумывается.

## 2. Время, cursor и границы

`state.tick` считает **завершённые** тики. Кадр всегда имеет явную границу:

| `boundary` | что это |
|---|---|
| `start` | состояние в начале тика `tick`: после `tick` завершённых тиков, **до** его Policy-фазы. `t=0` — до Policy(0); `t=30,60,…` — до Policy этого тика |
| `terminal` | финальное состояние после последней исполненной фазы; несёт `terminal` = `{winner, reason, terminal_tick, terminal_phase, tick_completed}` |

Три разных конца (никакого «tick t» без границы):

* **Гибель матки (QueenTerminal)** — тик **обрывается после 5 фаз** (0..4): `terminal_phase=4`, `phases_executed=5`,
  `tick_completed=false`, `state.tick` **остаётся** `terminal_tick`. Нет кадра конца тика, upkeep, regen и т. п. Terminal-кадр
  имеет `tick = terminal_tick`, и у того же тика есть `start`-кадр: оба доступны (`seek(t, "start")` и `seek(t, "terminal")`).
  Это верно и когда тик — policy-тик (`t % K == 0`): запись идёт `snapshot(start) → decision → event* → delta →
  snapshot(terminal) → result` (тест с `K=13`, где гибель на тике 39).
* **Коллапс / таймаут (Terminal/Log)** — тик исполнен полностью (11 фаз): `terminal_phase=10`, `tick_completed=true`,
  `state.tick = terminal_tick + 1`. Terminal-кадр стоит на `terminal_tick + 1`; `start`-кадра этого тика нет (тика
  `T_max` не существует), `seek(3000, "start")` — ошибка.
* **Усечённый прогон** — финальное состояние это обычный `start`-кадр с `finished=false`; `terminal`-кадра нет.

`cursor = {tick, event_index}` в header — где файл начинается в полном журнале. Для `start = {"kind": "reset"}` это всегда
`(0, 0)`. Для replay-**суффикса** (`start.kind = "checkpoint"`) в header встроен checkpoint (`start.checkpoint` — тот же
канонический объект, что кодирует `checkpoint_to_bytes`) и его `checkpoint_hash`; `cursor.tick = checkpoint.state.tick`,
`cursor.event_index` — сколько событий журнала предшествовало (его задаёт вызывающий: `event_index_base`). События суффикса
нумеруются дальше; `events_hash` суффикса — хэш именно записанной последовательности (префикс + суффикс по `events_hash`
дают хэш полного матча — это проверяется тестом).

## 3. ViewFrame и seek без правил

**ViewFrame** — омнисциентное отображаемое состояние (не `SimState`, не вторая механика). Строится чтением реального
`SimState` после тика. Содержит: `tick`, `boundary`, `terminal`, склады узлов `node_food` (7), и по каждой колонии:
`food`, `queen_hp`, `stationed` = `(node, caste, count, hp_sum)`, `transit` = `(from, to, arrival_tick, caste, count,
hp_sum)` — движение по рёбрам с `arrival_tick`, `brood` = `(caste, hatch_tick, count)`. Юниты сгруппированы по кастам
(и по ребру/прибытию для движущихся); `unit_id` в кадре нет — это отображение групп, а не L1-муравьи. Ключи групп стабильны
для display (узел+каста; ребро+прибытие+каста).

**Выбранный способ — дисплейные delta-патчи** (а не кадр на каждый тик): полный кадр на t=0, на каждой границе `K=30` и в
конце + на каждый исполненный тик `delta` — минимальный патч предыдущего кадра (изменённые склады узлов, изменённые скаляры,
`set/del` строк трёх таблиц групп) и `frame_hash` результирующего кадра. Seek в любой тик = ближайший snapshot слева + не более
`K−1` патчей. Размер: для 3000 тиков `snapshot+delta` ≈ 0.88 МБ из 10 МБ (≈9 %), остальное — журнал событий. Цена выбора:
патч осмыслен только поверх предшественника; это защищено хэшем каждого кадра, сверкой каждого snapshot с кадром, достигнутым
патчами, и цепочкой хэшей. Per-tick полный `SimState` не пишется.

```python
from formicmind.replay import ReplayReader

reader = ReplayReader.open("raid_win.jsonl")          # строгая проверка файла; правила не исполняются
frame = reader.seek(160)                              # start-кадр тика 160 (по умолчанию boundary="start")
end = reader.seek(175, "terminal")                    # финальный кадр; для гибели матки tick == terminal_tick
frame.colonies[0].transit                             # ((from, to, arrival_tick, caste, count, hp_sum), ...)
```

`seek` возвращает неизменяемый `ViewFrame` (frozen dataclass из кортежей), независимый от внутреннего состояния читателя.
Неизвестный тик или граница — `SeekError`. Корректность проверена **против кадров, снятых тестом с реального движка** на
каждом тике всех четырёх матчей (а не против собственной записи), плюс выбранные тики 0, 1, 29, 30, 31, середина рейда, terminal.
Отдельный тест «замины» подменяет функции движка/маршрутов/экономики/движения/боя на исключения и показывает, что
`open/seek/iter_events/read_*` продолжают работать; AST-тест запрещает `reader/frames/format` импортировать физику.

## 4. RNG-потоки политик и момент восстановления

Настоящий `PolicyController` может потреблять свой поток `policy:<id>:<replicate>` внутри `act`. Подстановка простого
контроллера «вернуть записанное действие» даёт те же `state_hash` и события, но **другой `checkpoint_hash`** (потоки не
сдвинулись) — тест `test_replaying_the_actions_without_restoring_the_streams_…` это показывает.

В `decision.seats[i].rng` лежит один элемент для места со своим потоком и ни одного — для места без потока:
`{stream_id, draws, state_after:{state, inc}}` — состояние потока **сразу после** `act` и число `next_u32`, которое его
сдвинуло. `verify` восстанавливает его **внутри `act` ReplayController'а, то есть внутри Policy-фазы**: после того как движок
построил обе ObservationV2 и до применения приказов, по порядку мест (A, затем B). Перед этим он сверяет, что живой поток —
ровно то, что оставило предыдущее решение (`inc` и `state`), и что записанное состояние отстоит от живого на `draws` шагов
(прыжок LCG за O(log n)). Фаза Policy атомарна: при откате `run.streams` восстанавливается движком.

Жёсткие проверки: `stream_id` в решении должен совпадать с потоком места в header; поток должен быть в `header.streams`;
`world`, чужой/неизвестный поток, смена `inc`, недостижимое состояние, два места с одним потоком — ошибки формата. Писатель в
свою очередь отказывает контроллеру, который трогает чужой/необъявленный поток (в том числе `world`), подменяет `inc` или
выставляет недостижимое состояние (`ReplayWriteError`). Формула `checkpoint_hash` и валидатор не менялись.

## 5. Строгая проверка формата

`ReplayReader.open` (никакой симуляции) отвергает с `ReplayFormatError`: пустой/обрезанный файл (нет `\n` в конце, нет
`result`), CR, BOM, пустые строки, неканонические байты, float; сломанную цепочку, неправильные/дублирующиеся/выпавшие `seq`;
записи после `result`; неизвестные ключи и типы в фиксированных схемах (header, decision, delta, snapshot, result, **каждое
из 15 событий** по явной таблице `EVENT_SCHEMA`, закреплённой тестом против `events.py`); неизвестный `kind`, невозможную
фазу для вида события, чужой тик, нарушение порядка `(phase, kind_rank)`; разрыв сквозного индекса событий; пропущенное
или лишнее `decision`; невозможный terminal (не последнее событие, неверные `tick_completed/phases_executed`, таймаут не в
последнем тике, гибель матки не на QueenTerminal); не те границы/`to_tick` у `delta`; патч, не дающий записанный
`frame_hash`; snapshot, не совпавший с кадром, достигнутым патчами; расхождение `result` с событиями и terminal-событием
(`events_hash`, `event_count`, `ticks_executed`, исход). Версии — `ReplayVersionError` (подкласс): чужая `replay_schema`
или любое различие в `header.contract`.

`verify` отвергает с `ReplayVerifyError`: расхождение ObservationV2-хэша, маски, отклонение записанного действия общим
валидатором, расхождение событий (с указанием тика, индекса и поля), патча/кадра, исхода, `events_hash`, `state_hash`,
`checkpoint_hash`, а также ошибку, с которой движок отказался исполнять тик. Ошибки формата, расхождения воспроизведения и
технические ошибки политики (их в replay нет: упавший тик не записывается, файл не появляется) — три разных класса.

## 6. Версионирование

| Что | Где | Правило |
|---|---|---|
| схема файла | `replay_schema = "v1"` | чужая — `ReplayVersionError` |
| контракты | `header.contract` | `rules/balance/observation/action/decoder/serialization/rng/stream_derivation` и домены хэшей config/state/checkpoint/events/rng-stream/observation/view/record — должны **совпасть** с движком, иначе `ReplayVersionError` |
| движок | `header.engine` (`formicmind-l0/0.0.1+for15`) | статичная метка для человека, не сравнивается |

Смена любой семантики файла или формулы хэша = новая версия (`v2`), а не молчаливое изменение `v1`. Файл записывается как
`<имя>.part` и переименовывается (`os.replace`) только после `result`; при ошибке частичный файл удаляется.

## 7. Использование

Запись полного матча (скрипты) и контроллеров:

```python
from formicmind import default_config
from formicmind.match_smoke import SCENARIOS
from formicmind.replay import record_episode, record_play
from formicmind.tick import ScriptedController

sc = SCENARIOS["raid_win"]
result, summary = record_episode(
    "raid_win.jsonl", default_config(), 20261102,
    ScriptedController(sc.script_a), ScriptedController(sc.script_b), source="raid_win")
# result: обычный EpisodeResult; summary: размер, число записей, events_hash/state_hash/checkpoint_hash
# record_play(path, state, run, controllers, ...) продолжает существующий (state, run), например после checkpoint
```

Для политик с RNG контроллеры — обычные `PolicyController(policy, "policy:<id>:<replicate>")`; потоки регистрирует
`record_episode`. Низкоуровнево: `ReplayWriter(sink, config, seats).start(state, run)`, `wrap(controllers)`,
`record_tick(state, run, tick_result)` (его вызывает `play(on_tick=…)`), `finish(state, run)`.

Проверка и CLI (код возврата: 0 — ок; 1 — валидный файл, но движок его не воспроизводит; 2 — не Replay v1 / повреждён;
3 — файл не прочитан; 64 — неверный вызов):

```bash
python -m formicmind.replay --verify raid_win.jsonl
sim replay --verify raid_win.jsonl        # console entrypoint из pyproject ([project.scripts] sim)
```

```
OK   replay v1 verified: raid_win.jsonl
     ticks=176 decisions=6 events=2210 records=2401 state_tick=175
     winner=A reason=queen_death terminal_tick=175 terminal_phase=4 (cut mid-tick at QueenTerminal)
     events_hash=2f6547ef1653a3c95b8cf3bad376d5e2aaba6cff00ea1c6cab7b91a8d61708a6
     state_hash=3fafbc8406661d3b66c3a895fffaf6e6ea29bcc90c56628600495be6aad32526
     checkpoint_hash=746208b8e9d9d3d1fc19260a863149a4899d79b650f0d7e666c03b58af0829da
```

(`ticks=176` — число вызовов `step_tick`, как в `EpisodeResult.ticks_executed`: оборванный тик 175 тоже считается.)

Программно: `verify(path)` → `VerifyReport` или исключение; `read_header/read_result/iter_events/seek(path, …)` — функции-обёртки
(каждая открывает и проверяет файл целиком; для многих обращений держите `ReplayReader`).

Smoke-записи (четыре публичных dev-матча FOR-13 + усечённый `cutoff` на 45 тиков + матч двух RNG-политик `rng_policy` на
400 тиков; затем читает и проверяет каждый файл):

```bash
python -m formicmind.replay_smoke --out replays
python -m formicmind.replay_smoke --scenario raid_win --out raid_win.jsonl
```

Seed — произвольная инженерная константа (`20261102`), **не** sealed seed FOR-4; это функциональная проверка, а не результат
баланса/производительности/Gate 0.

## 8. Размеры и время (Windows, Python 3.12, `source=replay_smoke:<имя>`)

| сценарий | тиков | событий | записей | байт | запись | чтение+проверка формата | verify (повторное исполнение) |
|---|---|---|---|---|---|---|---|
| mutual_raid_draw (ничья, t39) | 40 | 676 | 723 | 197 423 | 0.02 с | 0.02 с | 0.01 с |
| collapse (t46) | 47 | 475 | 529 | 143 702 | 0.02 с | 0.02 с | 0.01 с |
| raid_win (победа A, t175) | 176 | 2 210 | 2 401 | 648 163 | 0.08 с | 0.06 с | 0.04 с |
| timeout_draw (t2999) | 3000 | 33 335 | 36 538 | 9 996 382 | 1.1 с | 0.9 с | 0.6 с |
| cutoff (45 тиков) | 45 | — | 627 | 169 239 | 0.03 с | 0.04 с (вместе с чтением) | — |
| rng_policy (коллапс t265) | 266 | — | 4 197 | 1 133 248 | 0.13 с | — | 0.2 с (вместе с чтением) |

Времена — одиночные замеры на машине разработчика, **не** benchmark и не Gate 0 performance. Около 90 % байт — журнал событий
(на каждой записи ещё ~110 байт цепочки); кадры и патчи — около 9 %.

## 9. Что не входит

Нет Vue/React/front, WS/FastAPI/сервера, сторонних JSON-библиотек, parquet/БД, обучаемых стратегий, evaluation/benchmark,
sealed datasets, отдельного replay-симулятора, L1-муравьёв, полных `sim run`/`sim benchmark`. Анимация между тиками — задача
будущего viewer'а; replay хранит лишь состояния на границах тиков.
