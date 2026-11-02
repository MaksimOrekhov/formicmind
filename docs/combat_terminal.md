# L0 бой, терминальные условия и runner (FOR-13)

Завершает игровую механику L0: **одновременный Combat**, защита и гибель матки (`QueenTerminal`), экономический коллапс,
таймаут и тонкий runner `run_episode`. Источники истины — FOR-1 v0.2.1 (+ errata A–G, R1, решения по FOR-12), FOR-2 errata
v0.2.1, аудит D3 (FOR-8). Баланс остаётся кандидатским (`candidate-FOR1-v0.2.1-unvalidated`), **не калибровался**.
Ни ML, ни турниров, ни JSONL replay, ни UI нет. Smoke-матчи — **функциональная проверка**, не мета/производительность/Gate 0.

## Один полный путь исполнения

`step_tick(state, run, controllers)` без флагов исполняет все 11 фаз (`TickMode.FULL`) и возвращает `physics_complete=True`:

`Policy → Completion → Arrival(+сенсор POST_ARRIVAL_PRE_COMBAT) → Combat → QueenTerminal → Harvest → Upkeep → Lay →
Regeneration → AutomaticDepart → Terminal/Log`.

`movement_only` и `partial_economy` — те же функции фаз в урезанных таблицах (никакой второй реализации правил),
`physics_complete=False`. `IncompleteRulesError` удалён. После исхода `step_tick` бросает `EpisodeFinishedError`
(и для dev-режимов); шаг при `state.tick >= T_max` без исхода — `ValueError`.

## Combat (`combat.resolve_combat`, FOR-1 §6)

* Бой только в узле, между **стоящими** юнитами; на рёбрах не воюют и не страдают. Нет RNG, float и приоритета стороны.
* Для узла и стороны с ≥1 солдатом (по состоянию **до** боя этого тика): `base = солдаты × soldier_damage (2)`;
  солдаты в **своём доме** наносят `base × 5 // 4` (floor от суммы: 1→2, 2→5, 3→7, 4→10, 5→12). Бонус не помогает атакующим.
* Цель — ровно одна категория вражеских юнитов в этом узле: **солдаты → рабочие → матка (только в её доме и только если
  в доме нет ни одного вражеского солдата и рабочего)**. Урон льётся по `unit_id` по возрастанию, каждый юнит принимает не
  больше своего HP; **остаток не переносится** ни на следующую касту, ни на матку, ни на следующий тик (`wasted`).
* Оба урона вычислены из одного предбоевого состояния и применяются после; солдат, убитый в этой фазе, всё равно бьёт.
  Рабочие не наносят урона, но живыми в доме закрывают матку (фаза, убившая последнего, матку не бьёт).
* События (омнисциентный лог, canonical JSON, входят в `events_hash`): `combat` (сводка узел × атакующий: солдаты, base,
  `home_bonus`, damage, target, target_units/hp_total, applied, wasted), `wound` (выжил, hp до/после), `death`, `queen_damage`
  (`queen_hp_before/after` — абсолютные значения для spectator), `terminal`.
* После Combat intel **не обновляется** (нет post-combat сенсора): запись всегда не свежее правил видимости, регион,
  потерявший последнего наблюдателя, хранит предбоевую запись и стареет. Свой дом виден даже без взрослых.
* Добыча блокируется только **живым** вражеским солдатом после Combat (errata B); `AutomaticDepart` освобождает юнитов
  в **том же** тике, когда блокирующий солдат погиб.

## Терминальные условия (`terminal.py`)

Приоритет структурный и протестирован: **гибель матки > экономический коллапс > таймаут**.

| Условие | Когда | Исход |
|---|---|---|
| `queen_hp == 0` | сразу после Combat (`QueenTerminal`) | погибла одна — победа другой; обе в одном бою — **ничья** |
| коллапс | конец полного тика (Terminal/Log), после Upkeep/Lay/AutomaticDepart | коллапс одной — победа другой; обеих — **ничья** (D3) |
| таймаут | конец тика `T_max − 1` (если нет коллапса) | всегда **ничья**, никакой победы по еде/«силе» |

**Коллапс колонии:** нет взрослых рабочих (включая в пути), в расплоде нет оплаченного рабочего и
`food < eggs_per_lay × worker_egg_cost` — цена целой партии рабочих (10/20/40 при `eggs_per_lay` 1/2/4; при дефолте это
буквальное 10 из FOR-1 §9.2). Расплод-солдат «потенциальных рабочих» не даёт. Нулевые рабочие при `food ≥` цены партии — не коллапс.

**Граница тика (off-by-one).** `state.tick` считает завершённые тики.

* Гибель матки в тике `t`: исполнено 5 фаз (Policy…QueenTerminal), **Harvest/Upkeep/Lay/Regeneration/AutomaticDepart/Terminal-Log
  не исполняются**, `state.tick` остаётся `t`, `terminal_tick = t`, `phases_executed = 5`, окна `WindowStats` не получают
  фиктивного `window_ticks`.
* Коллапс/таймаут в тике `t`: исполнено 11 фаз, `state.tick = t + 1`, `terminal_tick = t`, окна включают этот тик.
  Таймаут всегда имеет `terminal_tick = T_max − 1` и `state.tick = T_max`.

## EpisodeOutcome и checkpoint

`EpisodeOutcome(winner A|B|draw, reason queen_death|economic_collapse|timeout, terminal_tick, terminal_phase, phases_executed,
sides)` хранится в `RunState.outcome` (не в `SimState`): `state_hash` — механика, `checkpoint_hash` знает, закончен ли матч.
**Checkpoint v3:** схемы `formicmind.run_state/3`, `formicmind.checkpoint/3`, домен `formicmind/checkpoint/v3`, поле
`run.outcome` = `{"finished": false}` или завершённый объект (в canonical-данных нет `None`). При загрузке исход сверяется с
состоянием (`terminal.validate_outcome`): чужой победитель, неверный tick/фазы, «матка мертва, но матч идёт» — `CanonicalError`.
v1/v2 отвергаются по тегу схемы. **Не менялись:** `rules_version`, `balance_version`, `state/config` схемы и хэши, observation/action/decoder,
`events_hash` домен; хэш событий/состояния прежних dev-сценариев (economy smoke) прежний, меняется только `checkpoint_hash`.

## Runner (`match.py`) и smoke

`run_episode(config, seed, controller_a, controller_b, max_ticks=None)` = `reset` + цикл `step_tick` до исхода; результат
`EpisodeResult` (winner/reason/terminal_tick/score 1·½·0/queen_hp/food/workers/soldiers/brood/event_counts/events/хэши).
RNG-потоки регистрируются по `PolicyController.stream_id` (`policy:<id>:<replicate>`), а не по месту; общий поток двух мест
отвергается. Ошибка контроллера или невалидное действие — техническая ошибка (Policy-фаза откатывается, исключение
пробрасывается), **не** поражение. `max_ticks` — отсечка harness (`outcome=None`, `truncated=True`), не больше `T_max`.
`play(state, run, controllers, …, stop_before_tick=)` продолжает загруженный checkpoint.

```bash
python -m formicmind.match_smoke            # все четыре сценария
python -m formicmind.match_smoke raid_win   # один
```

| Сценарий | Исход | Тик | Что показывает |
|---|---|---|---|
| `raid_win` | A, `queen_death` | 175 | бой в N4 и B, потеря 4 солдат B, 12 фаз по матке, wasted-урон |
| `mutual_raid_draw` | ничья, `queen_death` (обе) | 39 | 30 одновременных фаз по двум маткам (t=10..39) |
| `collapse` | B, `economic_collapse` | 46 (`state.tick`=47) | голод A до нуля рабочих |
| `timeout_draw` | ничья, `timeout` | 2999 (`state.tick`=3000) | матч не уходит за `T_max` |

## Трактовки FOR-1, которые пришлось выбрать (для ревью)

1. **Порядок collapse/timeout относительно §8.** Тело v0.2 §9.2 говорит «после upkeep и кладки», errata A ставит
   AutomaticDepart после Regeneration и Terminal/Log последней — проверка в Terminal/Log, т. е. после всей экономики и
   AutomaticDepart. На исход это не влияет (AutomaticDepart не меняет еду и состав).
2. **Целая партия вместо литерального 10** — по FOR-13; при дефолте совпадает.
3. **Солдаты бьют вражеских рабочих и вне дома** (нейтральный узел, чужой узел), даже без вражеских солдат: §6 «бой при наличии
   солдат хотя бы одной стороны» и приоритет целей. Бонус 5/4 — для любых солдат в их доме (в т. ч. против одних рабочих).
4. **Граница `state.tick` при гибели матки** не продвигается (тик не завершён); при коллапсе/таймауте — продвигается.
5. **Новорождённые** вылупляются до Combat (Completion раньше) и уязвимы в том же тике; старвация видит HP после боя.
6. Терминальное событие `terminal` включено в `events` и `events_hash`.
