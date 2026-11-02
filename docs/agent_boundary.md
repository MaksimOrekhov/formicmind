# L0 agent boundary (FOR-10)

Контракты: FOR-2 v0.2 + errata v0.2.1 (приоритет errata), FOR-1 v0.2.1 (пины в `versions.py`).
Реализованы только observation / mask / action / decoder / private intel / policy-граница.
Движения, добычи, боя, кладки, runner, replay и обучения здесь нет.

## Поток данных на Policy tick

```
update_local_intel(state, side, PRE_POLICY)      # для обеих сторон, пишет только приватную память
snapshot = freeze_policy_snapshot(state)          # один immutable PolicySnapshot для обеих сторон
obs      = build_observation(snapshot, side)      # чистое чтение, ObservationV2 (223 float32)
mask     = get_action_mask(obs)                   # 30 bool, функция только от obs
logits   = policy(obs, mask, policy_rng)          # PolicyLogitsV2 (17 конечных float32)
action   = decode_action_logits(logits, mask)     # единственный decoder
valid    = validate_action(obs, mask, action)     # fail-fast, без частичного применения
```

`begin_policy_phase(state, windows)` = обе `PRE_POLICY`-записи + `freeze_policy_snapshot`.
`decide(policy, obs, rng)` = шаги `mask → policy → decode → validate`: единый путь для всех стратегий.

## Модули

| Модуль | Содержимое |
|---|---|
| `intel.py` | `IntelPhase`, `update_local_intel` — единственный писатель `IntelMemory` |
| `snapshot.py` | `PolicySnapshot`, `SideView`, `WindowStats`, `freeze_policy_snapshot`, `begin_policy_phase` |
| `observation.py` | таблица раскладки `OBS_V2_BLOCKS/OFFSETS`, `ObservationV2`, `build_observation`, `flatten_v2` |
| `actions.py` | `ActionV2`, `ActionMaskV2`, `get_action_mask`, `validate_action`, `InvalidAction`, `MaskError` |
| `decoder.py` | `PolicyLogitsV2`, `decode_action_logits`, `decode_forage_logits`, `quantize_shares`, `encode_float_shares` |
| `policy.py` | `Policy` (Protocol), `decide` |

## Что гарантируется

* **Информационная граница.** `SideView` содержит только собственную колонию и собственную память разведки;
  состояния врага в снимке нет вообще. Единственные источники знаний о враге — сенсоры
  `update_local_intel`: собственный дом (всегда) и узлы со стоящими там собственными взрослыми. Юниты на рёбрах
  невидимы. Вражеская матка видна, только если свой взрослый стоит в гнезде врага.
* **Чистота.** `build_observation` ничего не меняет и не читает за пределами снимка; повторный вызов даёт те же байты.
  Снимок состоит из frozen dataclass и tuple и не меняется при последующих мутациях состояния.
* **Side-canonical кадр.** Все поэлементные массивы переставлены в систему координат стороны
  (`rules.local_to_world`), поэтому зеркальные ситуации дают идентичные векторы/маски/действия.
* **Decoder.** Любой конечный float32[17] при корректной маске даёт валидный `ActionV2`; NaN/±inf, неверная длина
  и т. п. — `PolicyOutputError` (технический отказ политики) до декодирования.

## Golden-случаи (все воспроизведены тестами)

* **G0**: 223 значения стороны A на старте FOR-1 (побитово, обе стороны), 30 bool маски G0.
* **D-1**: все logits 0; все +1; ровно два положительных; экстремальный разрыв; квантование «в ноль»;
  ±3.4e38, равные logits и tie на границе support; NaN/±inf отклоняются.
* **H-1**: дом без взрослых при вторжении: `enemy_count_last`, `enemy_current=1`, `age=0`, `queen_hp=112/240`.
* **I-1**: гибель последнего наблюдателя оставляет pre-combat запись, затем она стареет; повторный
  `build_observation` бит-в-бит и без изменения IntelMemory/state_hash.
* **R-1**: `nearest_arrival` ключуется по назначению следующего ребра (`to_node`), ETA=0 не теряется.
* Float-error кейс encoder: `[0.5, 0.5000009, 0, 0, 0, 0] → [32767, 32768, 0, 0, 0, 0]`.

## Инженерные выборы (не решения FOR-1/FOR-2)

1. `WindowStats` (own_global[25..29]) подаются вызывающим; в `SimState` их нет, по умолчанию нули. Поскольку они
   влияют на наблюдение, место накопления (и включение в checkpoint) надо решить в PR tick-движка.
2. `PRE_POLICY` и `POST_ARRIVAL_PRE_COMBAT` применяют одно правило сенсора (различие — только место вызова). Номер тика
   записи монотонен: запись «из будущего» — ошибка. Сенсор собственного дома записывает и запас еды узла (0).
3. Одна запись на узел: `food_*` и `enemy_*` разделяют `seen_tick`; `current ⇔ seen_tick == tick снимка`,
   `age = tick − seen_tick`, никогда не виденное: `known=0`, `age_norm=1`.
4. Нормализаторы — константы версии наблюдения; `check_observation_config` отвергает конфиг (queen HP ≠ 240,
   capacity > 120, regen > 12, ребро > 4), для которого они не рассчитаны. Счётчики и ETA не обрезаются.
5. Маска строится только из `ObservationV2`: достижимость — по блоку `public_edges`, «матка жива» — по `own_global`
   (при `queen_hp = 0` разрешён только `lay none`). Пустой/неразведанный ресурс и отсутствие рабочих цель не маскируют.
6. `validate_action` принимает `obs` по сигнатуре FOR-2, но источником истины считает переданную маску (с обычной
   маской это `get_action_mask(obs)`).
7. Decoder: softmax считается в float64 (`math.fsum`), квантование — `Decimal(str(p))` → точные рациональные числа;
   равенство остатков решается в пользу меньшего индекса. `PolicyLogitsV2` округляет значения до float32
   (различия ниже разрешения float32 становятся равенствами); конечное значение вне диапазона float32 — ошибка.
8. `army_choice` выбирается argmax до проверки avoid-опций; если для выбранного выбора маска не оставляет ни одного
   варианта `avoid_center`, это `MaskError` (штатная `get_action_mask` такого не создаёт).

## Не входит (следующие PR)

`apply_joint_orders` и применение приказов к юнитам, распределение рабочих по квотам (FOR-2 §4.2), назначение новорождённых,
накопление `WindowStats`, фазы тика, классификация `InvalidAction` в протоколе FOR-4, baseline-контроллеры и MLP.
Замечания аудита D1–D5 относятся к шагам движения/экономики/терминалов/replay.
