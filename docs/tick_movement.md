# L0 tick skeleton и движение (FOR-11)

Реализован первый движущийся мир: **policy → приказы → движение → следующий тик**. Это описание режима
`movement_only=True`; с FOR-12 тот же `step_tick` умеет `partial_economy=True` (экономика и расплод, см.
[economy.md](economy.md)). С FOR-13 `step_tick` без флагов исполняет все фазы и возвращает `physics_complete=True` (бой,
QueenTerminal и terminal-проверки — [combat_terminal.md](combat_terminal.md)); оба описанных здесь режима остались
**неполными dev-режимами** (`physics_complete=False`, не матч).

## Фазы и граница тика

Порядок (FOR-1 v0.2.1, errata A): `Policy → Completion → Arrival → Combat → QueenTerminal → Harvest →
Upkeep → Lay → Regeneration → AutomaticDepart → Terminal/Log` (`events.Phase`, 0..10).

| Фаза | Статус |
|---|---|
| Policy | реализована (только при `t % K == 0`) |
| Arrival (+ сенсор `POST_ARRIVAL_PRE_COMBAT`) | реализована |
| AutomaticDepart | реализована |
| Terminal/Log | только учёт `window_ticks`; terminal-проверки нет |
| Completion, Harvest, Upkeep, Lay, Regeneration | в `movement_only` — заглушки (`implemented=False`); реализованы в `partial_economy` (FOR-12) |
| Combat, QueenTerminal | в `movement_only`/`partial_economy` не реализованы (`implemented=False`); реализованы в полном тике (FOR-13) |

**Граница:** `step_tick` получает состояние *в начале тика t* (`state.tick == t`, до `PRE_POLICY`), исполняет тик `t`
и оставляет `state.tick == t + 1` (`state.tick` считает **завершённые** тики). Если фаза сообщит terminal до Terminal/Log
(в полном тике — QueenTerminal), остальные фазы тика не исполняются, `state.tick` **не** увеличивается,
`TickResult.terminal` содержит причину; завершение в Terminal/Log (коллапс, таймаут) — полный тик, `state.tick == t + 1`.

## Policy-тик

`PRE_POLICY` обеих сторон → один frozen `PolicySnapshot` (окна из `RunState`) → обе `ObservationV2` (до любого
решения) → решения контроллеров → **только после обоих** атомарное применение приказов → диспетчеризация доступных
юнитов → старт нового окна. Фаза Policy — «всё или ничего»: любая ошибка (контроллер, валидатор, применение
приказов или диспетчеризация) откатывает состояние (в том числе intel; откат in place, ссылки на объекты сохраняются),
`RunState` (RNG-потоки и окна) и события, не оставляя ни приказов, ни отправлений.
`ValidatedActionV2` можно сконструировать вручную, поэтому движок **не доверяет обёртке**: оба действия
перепроверяются `validate_action` против наблюдения и маски *своей* стороны до применения; `apply_joint_orders`
дополнительно проверяет структуру действия (lay/army/shares/support/avoid, garrison+avoid).
Между policy-тиками стоящие приказы сохраняются и контроллеры не вызываются.
`PRE_POLICY` на не-policy тиках не вызывается: наблюдения строятся только на policy-тиках, а сенсор
`POST_ARRIVAL_PRE_COMBAT` каждый тик пишет ту же запись (дом + узлы со взрослыми); в движении без боя разницы нет.

Адаптеры: `PolicyController` (через общий `decide`), `ScriptedController` (тестовый, действие по тику).

## Приказы и квоты (FOR-2 §4.2, буквально)

`worker_quotas`: `D_i = N·q_i`, `base = D // Q`, остаток по наибольшим `rem_i` (ничья — порядок категорий).
Конкретные рабочие упорядочены по месту (стоящие — по узлу в системе координат стороны; затем транзит — по
следующему узлу и ETA), внутри места по `unit_id`; цели заполняются в порядке home, N1, N2, C, N4, N5. Юнит на
ребре меняет только `target_node`. Солдаты: `target = local_to_world(side, army_choice)`. `avoid_center` относится
только к солдатам. Приказ применяется атомарно (`apply_joint_orders` принимает только `ValidatedActionV2`).

## Маршруты и транзит

Стоимость — сумма длин рёбер; **при равной стоимости — routing errata R1**: лексикографически наименьшая
последовательность узлов в каноническом локальном порядке самой движущейся колонии (`home, N1, N2, C, N4, N5,
enemy_home`, `routing.side_local_rank`), а не по world ID. Поэтому `route_B(mirror(u), mirror(v)) =
mirror(route_A(u, v))` для всех `(side, src, dst, avoid)`; расстояния и `avoid_center` не менялись. Отправление в тик `t` по ребру длины `d` → прибытие в `t + d`, без доп. тика на
пересадку. Маршрут пересчитывается из текущего узла и `target_node` при каждой отправке, поэтому: стоящий на месте
юнит не путешествует; юнит, прибывший в промежуточный узел, уходит в `AutomaticDepart` **того же тика** (D1a);
`avoid_center` запрещает C как промежуточный/конечный узел, но не как текущий (D1b).

**Блокировка:** юнит в узле со стоящим вражеским солдатом не уходит (свои солдаты и вражеские рабочие не мешают);
доступность считается по состоянию до диспетчеризации для обеих сторон одновременно. В dev-режимах боя нет, поэтому две
встретившиеся армии остаются заблокированными навсегда; в полном тике Combat идёт до AutomaticDepart, и выжившие
освобождаются в тот же тик, когда погиб последний вражеский солдат узла.

## События

`order`, `retarget`, `depart`, `arrive` (`events.py`). Порядок = порядок исполнения = сортировка по
`(phase, kind, unit_id, side)`: сначала `order` по сторонам, затем `retarget`, `depart`, `arrive` по `unit_id`.
Все данные — int/str/bool/list; `events_hash` — SHA-256 с доменом `formicmind/events/v1`. Это омнисциентный лог;
политика его не видит.

## WindowStats и checkpoint (F1)

Единственный источник истины — `RunState.windows` (по стороне). Движок передаёт их в снимок; `window_ticks`
растёт в Terminal/Log каждого тика и обнуляется после построения наблюдений на policy-тике (в t=30 наблюдение
содержит `window_ticks = 30`, `own_global[17] = 1`). Будущие `harvest/upkeep_paid/egg_spend/debt_delta`
будут писать туда же. Формат **checkpoint v2** (FOR-11) добавил окна; с FOR-13 — **v3** (схемы `formicmind.run_state/3`,
`formicmind.checkpoint/3`, домен `formicmind/checkpoint/v3`, поле `outcome`); v1/v2 отвергаются по тегу схемы.
`state_hash` и `config_hash` не менялись.
Дефолтные нули в `begin_policy_phase(state, windows=None)` остались только для фикстур; живой путь их не использует.

## F2

`encode_float_shares` бросает `PolicyOutputError`, если положительных весов больше 3 (`quantize_shares` остаётся
общим квантизатором). Основной decoder не менялся.

## Открытые вопросы к правилам (не решены скрытым изменением)

1. **Tie-break маршрута и симметрия сторон — решено (R1).** Прежнее правило world ID было несимметрично для шести
   упорядоченных пар (`avoid_center=false`): N1→N2 шёл через A, а зеркальное N4→N5 — через C. Принято решение
   проекта: tie-break по локальному порядку движущейся колонии (errata R1 в FOR-1, синхронизировано с FOR-2),
   `rules_version = l0-proto-FOR1-v0.2.2`. Прежнее правило осталось только как `shortest_path_world_id` для
   сравнительного теста. Старые состояния/checkpoint версии v0.2.1 отвергаются по `rules_version`.
2. **D5.** Литеральный порядок §4.2 тратит 48 рабочих-тиков там, где достаточно 12; зафиксировано тестом.
3. **Блокировка без боя** (см. выше).
