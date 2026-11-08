# Ecosystem v1 — контракты данных: состояние, наблюдение, действие, награда, replay (FOR-43, E0)

Статус: **проектное предложение**. Кода нового движка нет; блоки ниже — *контрактные эскизы* (имена и типы, на которые должны опираться будущие PR),
не реализация. Пакет документов: [видение](ecosystem_v1_vision.md) (основа) · [RFC](ecosystem_v1_rfc.md) · контракты (этот файл) ·
[экономика](ecosystem_v1_economy.md) · [миграция](ecosystem_v1_migration.md).

Метки: **[L0]** — факт существующего L0 · **[E1]** — предложение · **[ВЫБОР]** — пересматриваемый инженерный выбор · **[ОТКРЫТО]** — нужен ответ/эксперимент.
Ссылки на исходники L0 (`src/formicmind/...`) — факты main; `src/formicmind/rl/` существует только в ветке FOR-42 (PR #24, на ревью).

---

## 0. Версии и изоляция

Новая среда — отдельное пространство имён `formicmind.eco` (будущий каталог `src/formicmind/eco/`, тесты `tests/test_eco_*`). Существующие `versions.py`,
`rules.py`, `observation.py`, `actions.py`, `decoder.py` и все схемы L0 **не меняются**.

| Ключ | Значение | Примечание |
|---|---|---|
| `eco_rules_version` | `eco1-proto-v0` | механика; растёт при изменении правил |
| `eco_balance_version` | `eco1-candidate-unvalidated` | числа до калибровки E1-c (как `candidate-FOR1-v0.2.1-unvalidated` в L0) |
| `observation_version` | `ObservationE1/v2` | **не** `ObservationV2` |
| `action_version` | `ActionE1/v1` | **не** `ActionV2` |
| `eco_state_schema` | `formicmind.eco.state/2` | |
| `eco_checkpoint_schema` | `formicmind.eco.checkpoint/1` | |
| `eco_replay_schema` | `eco-replay/v1` | отдельно от `replay v1` L0 |
| домены хэшей | `formicmind/eco/{config,map,state,checkpoint,events}/v1` | префикс + `0x00`, SHA-256 hex — как в L0 `docs/foundation.md` §4 |
| `serialization_version` | `formicmind-canonical-json-v1` | **переиспользуется** [L0]: int, str, bool, list, dict; без float/NaN/null; u64 — десятичные строки |
| `rng_version` | `pcg32-xsh-rr-64-32-v1` | **переиспользуется** [L0] (`rng.py`) |

**Граница импортов.** `formicmind.eco` может импортировать только `canonical` и `rng` из L0 (чистые утилиты). Ни один модуль L0 не импортирует `eco`.
Проверяется тестом `test_eco_import_boundary` (задача E1.1).

---

## 1. Идентификаторы и порядок

* `colony_id: int 0..N-1` — мировой идентификатор (не показывается политике).
* `node_id: int 0..M-1`, M = число узлов; **детерминированная нумерация генератором** (по порядку создания, не по `hash()`).
* `cohort_id: int` — монотонный счётчик мира; не переиспользуется.
* Ограничения: `N ∈ {6,7,8}` (по умолчанию 8), `M ∈ [80,120]`, число площадок `P ∈ [24,40]`.
* Все итерации — по возрастанию id; множества в состоянии хранятся как отсортированные списки. Python `hash()` и порядок `set`/`dict` не участвуют.
* **Политика не видит** `colony_id`, `node_id`, `N`, `M`. Она видит только локальные слоты (§7). Мировые id нужны движку, replay и viewer.

---

## 2. ConfigE1 (неизменяемый)

```
ConfigE1:
  eco_rules_version, eco_balance_version
  n_colonies: 6|7|8            # конфиг, а не константа кода
  scenario: "abundance"|"moderate"|"severe"   # множитель σ экологии, см. экономику §3
  horizon: 6000                # world T_max
  decision_k: 30               # период стратегических решений
  alert_interval: 10           # внеочередные решения во время боя; 0 = выключено
  economy: EconomyParams       # таблица §3 (целые)
  map_params: MapParams        # диапазоны генератора §4
  options: {stop_when_alone: false, loot_on_queen_death: false, workers_flee_default: false}
```

`EconomyParams` (целые; числа — [E1] кандидаты, см. [экономику](ecosystem_v1_economy.md) §1):

| Поле | Значение | Поле | Значение |
|---|---:|---|---:|
| `worker_upkeep_tenths` | 4 | `soldier_upkeep_tenths` | 8 |
| `worker_egg_cost` | 10 | `soldier_egg_cost` | 20 |
| `worker_hatch_delay` | 20 | `soldier_hatch_delay` | 30 |
| `lay_interval_ticks` | 20 | `eggs_per_lay` | 1 |
| `start_workers` | 10 | `start_soldiers` | 2 |
| `start_food` | 80 | `storage_cap` | 600 |
| `worker_hp` / `soldier_hp` / `queen_hp` | 20 / 30 / 240 | `soldier_damage` | 2 |
| `home_defense_num/den` | 5 / 4 | `starvation_unit_cost` | 5 |
| `carry_load` (L) | 16 | `harvest_rate` (h, ед./тик) | 2 |
| `harvest_patience` (тиков ожидания на пустой площадке) | 8 | `scout_dwell` / `attack_dwell` | 2 / 3 |
| `initial_known_radius` (тиков от гнезда) | 5 | `alert_interval` | 10 |
| `lane_cap` (юнитов на направление ребра, обычное / узкое) | 8 / 4 | `age_bucket` (корзина журнала возрастов, тиков) | 100 |
| `lifespan_worker` / `lifespan_soldier` (тиков после вылупления; `0` = бессмертны, режим L0) | 2000 / 2000 | `start_age_spread` (доля `L`, в тысячных) | 500 |

`config_hash = SHA-256("formicmind/eco/config/v1" ‖ 0x00 ‖ canonical(ConfigE1))`.

---

## 3. MapE1 (статическая часть мира, генерируется один раз)

```
MapE1:
  map_seed: u64 (десятичная строка)          # корень потока "map"
  nodes: [NodeE1]     # по node_id
  edges: [EdgeE1]     # по (a, b), a < b
  nests: [node_id]    # nests[colony_id] = узел гнезда
  map_hash

NodeE1:  node_id, kind: "junction"|"patch"|"nest", nest_of: colony_id | -1,
         patch: PatchSpec | null, x, y (для viewer: целые координаты раскладки; правилами не используются)
PatchSpec: capacity C, regen R, site_cap Q, start_stock            # целые
EdgeE1:  a, b, length (2..6 тиков), lane_cap (4 | 8), chokepoint: bool
```

Карта связна; генерация детерминирована по `(map_seed, ConfigE1.map_params)`; алгоритм и метрики справедливости — RFC §5. Генератор — единственный потребитель RNG-потока `map`;
**правила мира после генерации не потребляют RNG** (поток `world` зарегистрирован для будущих расширений и входит в `checkpoint_hash`, но не используется).

---

## 4. WorldStateE1 (динамическая часть)

```
WorldStateE1:
  tick: int                                   # число завершённых тиков
  patches: [PatchState]                       # по node_id (только площадки): stock F
  piles:   [PileState]                        # {node_id, stock}; создаются динамически, regen = 0
  colonies: [ColonyStateE1]                   # по colony_id
  cohorts:  [CohortE1]                        # по cohort_id
  next_cohort_id: int
  edge_flow: [EdgeFlow]                       # {a,b,dir, in_flight_units}: производное, хранится явно для проверки
  outcomes: [ColonyOutcome | none]            # по colony_id; пишется один раз
  world_outcome: WorldOutcome | none
```

`state_hash = SHA-256("formicmind/eco/state/v1" ‖ 0x00 ‖ canonical(config_hash, map_hash, state))` — состояние правил **без** RNG и без состояний политик (как в L0).
Приватная память колоний (`ColonyMemoryE1`) **входит** в состояние и хэш, потому что она определяет наблюдения.

### 4.1 ColonyStateE1

```
ColonyStateE1:
  colony_id, status: "alive"|"orphan"|"extinct"
  food: int                                   # склад (≤ storage_cap)
  queen_hp: int                               # 0 = матка мертва
  next_lay_tick: int, lay_acc_q: int, soldier_acc_q: int      # аккумуляторы кладки (целые, в 1/3 и 1/4)
  upkeep_remainder10: int, starvation_debt: int
  brood: [BroodRecord]                        # {cohort_id, caste, laid_tick, hatch_tick}
  ages: AgeLedger                             # по кастам: {bucket_start → число взрослых}; журнал возрастов (см. ниже)
  orders: ColonyOrders                        # последние принятые ActionE1 после декодирования (сохраняются)
  memory: ColonyMemoryE1                      # приватная карта и сведения
  window: WindowStatsE1                       # накопители за последнее решение (для наблюдения)
  rolling: {income_ema_q, burn_ema_q, pop_ema_q}   # целые фикс-точечные EMA для reward/наблюдений
```

`AgeLedger` (**ревизия 2: старение, V-4**): `{caste → [{bucket_start, count}]}`, корзины шириной `age_bucket = 100` тиков по времени вылупления, отсортированы по `bucket_start`.
Это единственный источник возраста; когорты возрастов **не** хранят. Правила:

1. Вылупление добавляет `count` в корзину `⌊tick / age_bucket⌋ · age_bucket`. Стартовые взрослые получают возраст, равномерно разнесённый на `[0, start_age_spread · L)` (целочисленно, без RNG).
2. Смерть от старости: в фазе `Hatch/Age` корзина с `bucket_start + L ≤ tick` удаляется, и столько же юнитов удаляется из когорт колонии той же касты в порядке «дом → узлы → ребро → `cohort_id`»; груз умерших падает кучей в их узел;
   событие `old_age`. **Семантика корзин:** корзина умирает **целиком** в начале тика `bucket_start + L`. Поэтому `L` — **номинальный максимум**: фактическая продолжительность жизни юнита
   `∈ [L − age_bucket + 1, L]` (первый вылупившийся в корзине живёт ровно `L`, последний — на 99 тиков меньше), в среднем `≈ L − (age_bucket − 1)/2 ≈ 1950` при `L = 2000`. Стартовые взрослые округляются к **более старой** корзине
   (`bucket_start(−age)`, floor-деление), то есть тоже живут не дольше номинала. Точность — целые тики, без RNG; поштучный возраст не хранится. Тест T-23.
3. Смерть от голода или боя списывается с журнала **пропорционально** корзинам (наибольший остаток, при равенстве — старшая корзина).
4. Инвариант: `Σ ledger[caste] = Σ cohort.count[caste]` после каждой фазы. `L = 0` (бессмертны) — журнал не нужен, режим L0.
5. Матка не старится (ограничение v1); стоячие приказы и решения о кладке от возраста не зависят.

### 4.2 CohortE1 — единица юнитов

Отряд — не объект «муравей», а **когорта** (однородная партия одной касты одной колонии).

```
CohortE1:
  cohort_id, colony_id, caste: WORKER|SOLDIER
  count: int ≥ 1, hp_total: int             # alive = ceil(hp_total / unit_hp); 0 → когорта удалена
  loc: AtNode{node_id} | OnEdge{from, to, depart_tick, arrive_tick}
  mission: Mission, load: int (только WORKER; суммарный груз когорты)
Mission:
  kind: HOME|FORAGE|SCOUT|GUARD|ATTACK|WITHDRAW
  target_node: int, target_colony: int(-1)    # для ATTACK — колония-цель
  phase: GO|WORK|RETURN                       # FORAGE: идёт/собирает/несёт; SCOUT: идёт/осматривает/возвращается
  dwell_left: int
```

Правила (кратко; полные — RFC §7): когорты делятся и сливаются движком при применении приказов; урон списывается с `hp_total` по `cohort_id` по возрастанию;
`count = ceil(hp_total / unit_hp)` пересчитывается после боя (L0 характер: раненые не «чинятся»). Боевые стороны и целевая приоритетность — как в L0 `combat_terminal.md`.

### 4.3 ColonyMemoryE1 — единственный источник знаний политики

```
ColonyMemoryE1:
  slot_of_node: {node_id → slot}               # слоты выдаются в порядке открытия; slot 0 = собственное гнездо
  nodes: [KnownNode]                           # по slot
  edges: [KnownEdge]
  foreign: [ForeignColonyRecord]               # до 7 слотов, по порядку первого контакта
  sightings: [Sighting]                        # кольцевой буфер ≤ 32
  events: [MemoryEvent]                        # кольцевой буфер ≤ 16
  piles_seen: [KnownPile]

KnownNode: slot, visited: bool, discovered_tick, seen_tick (-1 = никогда не «видимый»), kind_known: bool,
           patch_capacity (если площадка и посещён), last_stock, last_stock_tick, last_delta_q,
           nest_owner_slot (-1 = неизвестно), queen_seen: 0 неизв.|1 жива|2 мертва, queen_seen_tick,
           danger_q (затухающий счётчик боёв здесь), harvest_ema_q
KnownEdge: slot_a, slot_b, length, lane_cap, discovered_tick, traversed: bool, traffic_ema_q
ForeignColonyRecord: slot, first_contact_tick, last_seen_tick, last_seen_workers_bucket, last_seen_soldiers_bucket,
           nest_node_slot (-1 неизвестно), they_attacked_q, i_attacked_q, kills_mine_q, kills_theirs_q
Sighting: node_slot, colony_slot, tick, workers_bucket, soldiers_bucket, aggression_observed: bool
```

**Правила записи памяти** (единственный писатель — сенсорная фаза движка; политика ничего не пишет):

0. **Старт (reset).** Узлы на кратчайшем пути ≤ `initial_known_radius = 5` тиков от гнезда помечаются *visited* на тике 0 по правилу 1 (без чужих sightings); остальной мир отсутствует в памяти.
1. Когорта колонии стоит в узле `v` на конец фазы Arrival: узел `v` становится *visited*, *видимым*, записываются: тип узла, для площадки — ёмкость и текущий запас, **все инцидентные рёбра** (длина, `lane_cap`) и их дальние концы как *discovered, not visited*.
2. Все чужие когорты, стоящие в `v` в этот момент, дают `Sighting` и (при первой встрече) новый `ForeignColonyRecord` — идентификатор колонии (слот) становится известен **при контакте**.
3. Если `v` — чужое гнездо: `nest_owner_slot`, `queen_seen` (по `queen_hp > 0` в момент наблюдения) записываются; **гнездо открыто только так**.
4. Матка наблюдает своё гнездо всегда (сенсор неподвижной матки): чужие когорты в гнезде фиксируются как `Sighting`, даже если собственных юнитов дома нет.
5. Видимость **локальна** (радиус 0 узлов + инцидентные рёбра), информация только стареет; `seen_tick` обновляется только пока кто-то стоит в узле.
6. Узлы и колонии, которых колония не открывала, **отсутствуют** в памяти и в любых производных структурах (масках, индексах слотов, счётчиках).

---

## 5. Приказы и применение (`ColonyOrders`)

**Сироты** (`status = orphan`, матка мертва; V-4): эпизод агента завершён, решений нет. Юниты живут по *стоячим приказам*: план добычи сохраняется (квоты пересчитываются на оставшихся рабочих), отряды получают `WITHDRAW → HOME` (охрана гнезда),
кладки нет, сбор прекращается при полном складе. Исчезают естественно: старение, голод, бой.

Действие политики декодируется в `ColonyOrders` и хранится в состоянии колонии как приказ до следующего решения. Одноразовые приказы (scout) потребляются сразу.
Перевод приказов в миссии когорт — детерминированный диспетчер (RFC §7.3). Применение валидирует действие против маски; невалидное действие — **техническая ошибка** (откат Policy-фазы, исключение),
не поражение колонии (как в L0 `match.py`).

---

## 6. Фазы тика и события

Один тик — фиксированная последовательность (расширяет 11 фаз L0 `docs/combat_terminal.md`):

| № | Фаза | Что делает |
|---:|---|---|
| 1 | `Policy` | если `tick % K == 0` или сработал `alert`: строит наблюдения, получает действия живых колоний, применяет приказы |
| 2 | `Hatch/Age` | расплод с `hatch_tick == tick` → когорты дома; затем смерть от старости: корзины журнала с `bucket_start + L ≤ tick` (§4.1) |
| 3 | `Arrival` | прибывшие когорты ставятся в узлы; сдача груза в гнезде; **сенсорная фаза памяти** (§4.3) |
| 4 | `Combat` | одновременный бой; затем убитые когорты, спил груза в кучу |
| 5 | `QueenTerminal` | `queen_hp == 0` → колония `orphan`, её эпизод завершён (§9.5) |
| 6 | `Harvest` | рабочие в состоянии WORK на площадках берут еду (§3 экономики) |
| 7 | `Upkeep` | содержание, долг голода, потери |
| 8 | `Lay` | кладка по аккумуляторам |
| 9 | `Regeneration` | `F = min(C, F + R)` |
| 10 | `Depart` | автоматический выход когорт (цикл FORAGE, WITHDRAW, SCOUT-return) |
| 11 | `Terminal/Log` | проверка `extinct`, терминал мира, события |

Типизированные события (омнисциентный лог, `events_hash`; политика их **не** видит; локальные проекции попадают в память по §4.3):
`hatch, old_age, depart, arrive, harvest, deposit, upkeep, starvation, lay, regenerate, contact, combat, wound, death, queen_damage, queen_dead, extinct, pile_created, storage_full, decision`.
Порядок в списке = порядок исполнения; ключ сортировки `(phase, kind_rank, ...tail)` — как `event_sort_key` в L0.

---

## 7. ObservationE1

> **Реализовано в FOR-48** (`eco/observation.py`, `eco/action.py`): `docs/ecosystem_v1_policy_interface.md`. Отличия от таблиц ниже, принятые при реализации: у `sightings` и `events` целочисленные слоты вынесены в отдельные массивы (`sighting_ints`, `event_ints`), в `cohorts` 14-й признак — фаза миссии, добавлены целочисленные `node_ints` и `facts` (нужны маскам без округления float), логарифмическая нормировка кусочно-линейная (без `libm`).

Одна запись на колонию на решение. **Переменный размер**: хранятся только валидные строки (дополнение нулями делает загрузчик батча). Все числа `float32`,
нормализаторы — константы версии. Никаких мировых `node_id`/`colony_id`, никакого `N`, никакой информации о неоткрытом.

| Блок | Форма (макс.) | Признаки |
|---|---|---|
| `global` | 40 | см. таблицу 7.1 |
| `nodes` | ≤128 × 24 + `node_mask` | таблица 7.2 |
| `edges` | ≤320 × 6 + пары слотов `int32[320,2]` | таблица 7.3 |
| `foreign` | 7 × 12 + `foreign_mask` | таблица 7.4 |
| `cohorts` | ≤48 × 14 + `cohort_mask` + слот узла `int32` | таблица 7.5 |
| `sightings` | ≤32 × 10 + `sighting_mask` | таблица 7.6 |
| `events` | ≤16 × 8 + `event_mask` | таблица 7.7 |
| `masks` | см. §8 | функция только от блоков выше |

Максимальный объём ≈ 6.2 тыс. float32 (≈ 25 КБ); средний на практике ≈ 2 тыс. (≈ 8 КБ) — открыто ~40 узлов и ~100 рёбер.

### 7.1 `global[40]` (индексы фиксированы версией; отсутствие данных → 0)
`0 tick/H · 1 decision_idx/(H/K) · 2 ticks_since_last_decision/K · 3 queen_alive · 4 queen_hp/240 · 5 food/storage_cap · 6 storage_full · 7 upkeep_rate/40 ·
8 income_ema/40 · 9 runway_norm (=min(1,runway/40)) · 10 starvation_debt/5 · 11 log1p(workers)/log1p(128) · 12 log1p(soldiers)/log1p(64) · 13 workers_home · 14 soldiers_home ·
15 brood_workers · 16 brood_soldiers · 17 egg_level_current/3 · 18 soldier_share_current/4 · 19 next_hatch_eta/30 · 20 idle_workers_frac · 21 scouts_out · 22 foragers_frac ·
23–25 squad_active[3] · 26 known_nodes/128 · 27 known_patches/40 · 28 known_colonies/7 · 29 frontier_count/32 · 30–35 окно решения: harvested, upkeep_paid, egg_spend,
losses_workers, losses_soldiers, kills_observed (log1p-нормализация) · 36 hostile_contact_recent · 37 in_combat_now · 38 expiring_soon (доля взрослых, чей срок жизни истекает в ближайшие 300 тиков) · 39 mean_age_frac (средний возраст взрослых / L)`.

Запрещено в `global`: число живых колоний мира, суммарные счётчики мира, любые величины чужих колоний, не наблюдавшиеся лично.

### 7.2 `nodes[·,24]` (слоты по порядку открытия; слот 0 — своё гнездо)
`0–3 kind one-hot (junction, patch, own_nest, foreign_nest) · 4 visited · 5 visible_now · 6 age_norm=min(age,1500)/1500 · 7 discovered_unvisited · 8 path_ticks_from_home/60 ·
9 stock_last/capacity · 10 capacity/200 · 11 last_delta/12 (знаковое) · 12 own_workers_here · 13 own_soldiers_here · 14 foragers_assigned_here ·
15 foreign_workers_bucket/7 · 16 foreign_soldiers_bucket/7 · 17 distinct_foreign_colonies/3 · 18 youngest_sighting_age_norm · 19 danger · 20 my_harvest_ema/12 ·
21 pile_stock_norm · 22 degree_known/6 · 23 nest_queen_status (0, 0.5 жива, 1 мертва)`.

### 7.3 `edges[·,6]`
`length/6 · lane_cap/12 · traversed_by_me · traffic_ema · age_since_known/1500 · danger`.

### 7.4 `foreign[7,12]` (слот выдаётся при первом контакте)
`present · first_contact_age/H · last_seen_age_norm · workers_bucket/7 · soldiers_bucket/7 · nest_known · queen_status · they_attacked · i_attacked · kills_mine · kills_theirs · peaceful_coharvest_recent`.

### 7.5 `cohorts[·,14]` — только собственные
`caste one-hot(2) · log1p(count) · hp_frac · moving · eta/60 · mission one-hot(6: HOME, FORAGE, SCOUT, GUARD, ATTACK, WITHDRAW) · carried_frac`; плюс `int32 node_slot` (или слот ближайшего конца ребра).

### 7.6 `sightings[·,10]`
`age_norm · fresh(≤K) · workers_bucket/7 · soldiers_bucket/7 · is_nest · aggression_observed · и int32: node_slot, colony_slot`.
Приблизительная сила — **корзина** `bucket(n) = 0 if n = 0 else min(7, 1 + floor(log2 n))` (видение §6: «примерная сила»), не точное число.

### 7.7 `events[·,8]` — локальные события памяти
Код (int): `patch_depleted_seen, first_contact, attacked, we_killed, we_lost, queen_hit, own_queen_dead_info, scout_arrived, starvation_death, hatch, lay_failed, storage_full`; + `node_slot`, `colony_slot`, `magnitude`, `age`.

### 7.8 Что **не** попадает в наблюдение (список для теста утечки)
Мировые id; `N`, `M`; чужие склады, запасы чужих гнёзд, HP чужих маток (только `queen_status` из личного наблюдения); неоткрытые узлы/рёбра; позиции чужих когорт вне присутствия; события, которые колония не наблюдала;
глобальные счётчики мира; состояние RNG; любые поля `state` других колоний.

---

## 8. Маски и ActionE1

> **Реализовано в FOR-48**: маски — `eco/action.py::action_masks(obs)`, декодер — `eco/decoder.py::decode`, применение — `eco/apply.py`. Размеры голов декодера — `LAYOUT` (818 латентов). Постоянные приказы отрядов хранятся в `colony.orders.squads`; повторный идентичный приказ выходящему отряду не отправляется.

### 8.1 Маски — чистые функции от `ObservationE1`
(как `get_action_mask(obs)` в L0 `actions.py` — маска не читает мир).

| Маска | Размер | Разрешено |
|---|---|---|
| `egg_level` | 4 | всегда (кладка при мёртвой матке невозможна: агент уже завершён) |
| `soldier_share` | 5 | всегда |
| `forage_frac` | 9 | 0 всегда; >0 при `workers>0` |
| `scout_n` | 5 | `0`; `k ≤ min(4, workers)` |
| `workers_flee` | 2 | всегда |
| `forage_target[128]` | узлы | `kind=patch`, `visited`, путь по известным рёбрам, не собственное гнездо |
| `scout_target[128]` | узлы | `discovered_unvisited`, либо `visited` с `age ≥ 90`; путь по известным рёбрам |
| `squad_mission[3,4]` | {NONE, GUARD, ATTACK, WITHDRAW} | `NONE`, `WITHDRAW` всегда; `GUARD/ATTACK` при `soldiers>0` |
| `squad_size[3,5]` | {0,¼,½,¾,1} | при выбранной миссии кроме NONE: ≥ ¼ и ≤ свободный остаток |
| `guard_target[128]` | узлы | любой посещённый узел с путём |
| `attack_target[32+7]` | пары (узел, колония) | свежая *sighting* (age ≤ 150 тиков) **или** известное чужое гнездо (`nest_known`); пути в известном графе; **не** неизвестные гнёзда и не собственная колония |

Следствие видения «неизвестное гнездо нельзя выбрать целью» выполняется конструктивно: `attack_target` строится только из `sightings` и `foreign.nest_known`.

### 8.2 ActionE1 (после декодера; целые)

```
ActionE1:
  egg_level: 0..3                 # интенсивность кладки: acc += level/3
  soldier_share_q: 0..4           # доля солдат среди яиц (в четвертях)
  forage_frac_q: 0..8             # доля рабочих (после разведчиков), назначаемых на добычу, в 1/8
  scout_n: 0..4                   # рабочих в разведку (одноразово)
  workers_flee: bool
  forage_targets: [slot]*k (k = 0..3, различны)    forage_shares_q: [int]*k, сумма 16
  scout_targets:  [slot]*(0..2)   # обязателен ≥1 при scout_n > 0
  squads: 3 × SquadOrder{mission: NONE|GUARD|ATTACK|WITHDRAW, size_q: 0..4, target: node_slot | (node_slot,colony_slot)}
```

Семантика размеров: отряды распределяются по слотам от меньшего номера; каждый берёт `size_q/4` от *всех* солдат, но не больше остатка; нераспределённые солдаты охраняют гнездо (GUARD дома).
`ATTACK` — персистентная миссия: когорта идёт к цели и бьёт **только** указанную колонию, пока она там есть; потом держит узел (`attack_dwell = 3`), затем `GUARD`. `WITHDRAW` — вернуться в гнездо кратчайшим известным путём.
Вражеские когорты, не указанные в `target`, не атакуются (видение: «национальность сама по себе не запускает бой»).

### 8.3 Распределение и log-prob (для PPO)
Факторизованная политика: категориальные головы (egg, soldier_share, forage_frac, scout_n, flee, миссии/размеры отрядов), pointer-головы с маской и **выбором без возвращения**
(forage ×3 после `k`, scout ×2, guard/attack ×3), доли — logistic-normal над выбранными целями с квантованием 1/16 (математика `distribution.py` FOR-42 переиспользуется; **код не переносится**
без явного преобразования вход/выход). Нелегальное = нулевая вероятность; принудительные варианты имеют log-prob 0 (как в FOR-42). Декодер: `decode(latent, mask) -> ActionE1` — единственный путь, любой
конечный латент даёт валидное действие; `NaN/inf` — `PolicyOutputError`.

### 8.4 Частота решений
Базовая — каждые `K = 30` тиков. Дополнительно (`alert_interval = 10`, **[ВЫБОР]**) колония получает внеочередное решение, пока в её узлах идёт бой (флаг `in_combat_now`):
бой длится 17–41 тик (экономика §8.1) — без этого отступление было бы технически невозможно. Метка времени решения (`ticks_since_last_decision`, он же `Δ`) входит в `global`. **Награда, дисконт и GAE считаются по прошедшим тикам мира, а не по числу решений (§9.2)**: alert сам по себе ничего не добавляет к награде и не меняет шкалу дисконтирования.

---

## 9. Награда, смерть и горизонт

Обозначения (ревизия 3): `v` — **viability**, ограниченный показатель жизнеспособности колонии (часть **награды**); `Value_critic(o)` — выход обучаемой value-сети (оценка **будущей отдачи**). Это разные величины: viability
никогда не подставляется в bootstrap. Время награды и дисконта — **тики мира**, а не номера решений.

### 9.1 Viability колонии (по тикам)

```
pop(τ)      = min(1, E_pop(τ) / 60)               # E_pop = EMA (τ = 300 тиков) от  workers + 0.5·soldiers
runway(τ)   = food / (burn_ema − income_ema)  если burn_ema > income_ema, иначе +∞
gate(τ)     = min(1, E_pop(τ) / 10)                # популяционный затвор безопасности (награда v2, ревизия 4)
security(τ) = min(1, runway(τ) / 40) × gate(τ)    # burn = upkeep (яйца — инвестиция, не burn); income = доставка на склад
v(τ)        = queen_alive(τ) × (0.6·pop(τ) + 0.4·security(τ))        ∈ [0, 1]
v_q(τ)      = floor(Q · v(τ)),  Q = 65536                              # целая фикс-точка 0..Q
```

**Версии награды (ревизия 4).** `REWARD_VERSION = 2`. Версия 1 — награда первого пилота FOR-49 (`gate ≡ 1`). Она платила `security = 1` колонии без единого взрослого: содержание
нулевое, `burn ≤ income`, `runway = +∞`. Сеть нашла это за 35 итераций: не класть яйца, накопить склад в 600, дождаться, пока рабочие умрут; матка жива, коллапса нет (§9.5: склад ≥ стоимости яйца),
штрафа нет, `v = 0.4` до горизонта (`docs/ecosystem_v1_training.md`). Версия 2 умножает **только** компонент `security` на `gate = min(1, E_pop/10)`: полная безопасность
оплачивается только при ≥ 10 рабочих-эквивалентах в EMA популяции; пустая колония получает от склада 0. Что не менялось: `pop`, веса 0.6/0.4, `runway`, ёмкость 40 тиков,
правила коллапса и вымирания §9.5, экономика, бой. Целочисленно: `security_q ← security_q · min(Q, pop_ema_q // 10) // Q`. Версия 1 остаётся в коде (`viability_q(..., version=1)`)
только для пересчёта прежних запусков; модель хранит `reward_version` в манифесте (манифест без поля = v1). Эталон: `scripts/ecosystem_reward_ref.py::viability_exact` (точные дроби, все версии).

**Версия 3 (ревизия 5, `REWARD_VERSION = 3`).** Формула `v` — та же, что в v2. Добавляется **терминал эпизода агента `functional_extinction`**: колония физически жива
(`status == alive`, `queen_hp > 0`), у неё **нет ни взрослых рабочих, ни взрослых солдат, ни расплода рабочих**, — независимо от склада. Расплод солдат один не спасает (он не
добывает). Выполнение условия в конце тика `τ` завершает эпизод агента этой колонии так же, как гибель: `v_q(τ) = 0`, штраф `P = 1.0` ровно один раз (§9.2), последний интервал
кончается в `τ + 1`, `terminated`, причина `functional_extinction`; после этого награда не начисляется, решений и обновлений памяти у сети для этой колонии нет, нового эпизода она
не начинает (даже если гнездо позже физически восстановится). Физические терминалы (`queen_death`, `collapse`) сохраняют смысл и имеют приоритет в том же тике: терминал один,
штраф один. Это определение **награды и эпизода**, а не правило мира, см. §9.5a. Эталон: `functional_extinction` и `episode_end` в `scripts/ecosystem_reward_ref.py`.
Версии v1/v2/v3 шкалой не сопоставимы: отчёты и манифесты несут `reward_version`, пересчёт старых политик выполняется заново по нужной версии.

`income_ema`, `burn_ema`, `pop_ema` — целые фикс-точечные EMA (обновление раз в тик) в состоянии колонии. `v_q(τ)` вычисляется в конце тика `τ` (после фазы Terminal/Log) и **добавляется в `viability_trace`** колонии;
в тик гибели матки `v_q = 0` и не записывается. Старение отдельного члена награды не требует: если колония не заменяет умерших, падает `pop`, а значит и `v` (экономика §7.2).

### 9.2 Награда за прошедшее время мира (ревизия 3)

Решение — любое: плановое (каждые `K = 30` тиков) или внеочередное `alert` (каждые 10 тиков в бою). Интервал решения `i` колонии — тики `[t_i, t_{i+1})`, `Δ_i = t_{i+1} − t_i` (≥ 1). Награда зависит только от **прошедших тиков**:

```
γ_30 = 0.997,  γ_tick = γ_30^(1/30),  γ_i = γ_30^(Δ_i / 30)        # time-aware: Δ_i = 10 даёт γ_30^(1/3), Δ_i = 30 даёт γ_30
r_i      = Σ_{τ=t_i}^{t_{i+1}−1} v_q(τ) / (Q·H)                      # недисконтированная: метрики, отчёты, сравнение запусков
r_i^γ    = Σ_{τ=t_i}^{t_{i+1}−1} γ_tick^(τ−t_i) · v_q(τ) / (Q·H)    # награда интервала, дисконтированная внутри интервала: цель PPO
при гибели колонии (последний интервал, terminated): r_i −= P;  r_i^γ −= P·γ_tick^(Δ_i),   P = 1.0     # штраф РОВНО один раз
```

Свойства (проверяются тестами T-21, T-22; эталон — `scripts/ecosystem_reward_ref.py`):

1. **Недисконтированная сумма не зависит от разбиения на решения:** `Σ_i r_i = Σ_τ v_q(τ)/(Q·H) − P·[гибель]`. Мирные 200 решений по 30 тиков, 600 по 10 тиков и смешанное расписание дают одинаковую награду при одинаковой `v(t)`.
   Максимум за жизнь `≈ 1`, как раньше (`H = 6000`).
2. **Дисконтированная отдача не зависит от разбиения:** `Σ_i (Π_{j<i} γ_j) · r_i^γ = Σ_τ γ_tick^τ · v_q(τ)/(Q·H) − P·γ_tick^(конец)`. Горизонт скидки — одни и те же `1/(1−γ_30) · 30 ≈ 10 000` тиков при любой частоте решений.
3. **Alert сам по себе ничего не добавляет:** он только дробит интервал. Удерживать `in_combat_now`, чтобы получать больше слагаемых, невыгодно: суммарная награда за те же тики та же. (Старая формула `r = v / N_dec` за каждое решение утраивала бы награду за постоянный бой —
   она **отменена**; регрессионный тест `test_a_per_decision_reward_would_pay_for_staying_in_combat_and_the_contract_does_not`.)
4. **Терминал один:** `terminated` ставится на последнем интервале (гибель матки, collapse или — с награды v3 — `functional_extinction`, §9.1), штраф `P` входит ровно в него; решений в тик гибели и после нет; последующее исчезновение сирот (`extinct`) штрафа не добавляет.
5. Нет члена «убийства». Убитые рабочие и солдаты соперника в `r_i` **не входят**; их полезность проявляется только через долю еды и безопасность (`pop`, `security`).
6. Вес гибели `P = 1.0` равен «максимально прожитой жизни»: смерть в любой момент хуже выживания со средней viability ≥ 0.05.
7. `λ_30 = 0.95` задаётся на 30 тиков, `λ_i = λ_30^(Δ_i/30)`. [ВЫБОР; PPO-математика — FOR-42].

`viability_trace` (≤ 30 целых на интервал, ≤ 10 при alert) входит в запись перехода рядом с `Δ_i`; награды вычисляются из неё в float64 вне правил мира. Допустимое упрощение для скорости — `r_i ≈ mean(v_q)·Δ_i/(Q·H)` и `r_i^γ ≈ r_i·γ_i^{1/2}`:
оно сохраняет независимость от разбиения только приближённо (≤ 0.3% на интервал), поэтому по умолчанию используется точная формула.

### 9.2a Критик, bootstrap и GAE для переменных интервалов

```
δ_i = r_i^γ + γ_i · (1 − terminated_i) · B_{i+1} − Value_critic(o_i)
A_i = δ_i + γ_i · λ_i · (1 − terminated_i) · A_{i+1},        λ_i = λ_30^(Δ_i / 30)
цель critic:  R_i = A_i + Value_critic(o_i)
```

* `B_{i+1} = Value_critic(o_{i+1})` для внутренних интервалов.
* **Терминал** (гибель матки, collapse): `B = 0`, цепочка `A` не продолжается; никакое значение viability или critic не подставляется.
* **Усечение горизонтом** (`tick == H`) или harness-усечение (`stop_when_alone`, V-3): награды нет, `B = Value_critic(o_H)` — **оценка обучаемого критика** по финальному наблюдению `o_H` (движок формирует его для живых колоний без действия).
  Подставлять `v(τ)` (viability) вместо `Value_critic` нельзя: это награда за одно мгновение, а не прогноз будущей отдачи.
* Критик получает то же `ObservationE1` и рекуррентный контекст, что политика (V-5); `Δ_i` (`ticks_since_last_decision`) входит в `global`, поэтому он видит длину интервала.
* **Потенциальная версия (контроль A/B):** `r'_i = γ_i · Φ(o_{i+1}) − Φ(o_i)`, `Φ = v` в момент решения; сохраняет оптимальную политику (Ng 1999) при условии time-aware `γ_i`.

### 9.2b Обязательные тесты контракта награды

| ID | Тест (имя в эталоне) | Что проверяет |
|---|---|---|
| T-21 | `test_undiscounted_reward_does_not_depend_on_how_often_the_policy_decides`, `test_an_alert_adds_no_reward`, `test_per_interval_reward_is_proportional_to_elapsed_ticks`, `test_a_per_decision_reward_would_pay_for_staying_in_combat_and_the_contract_does_not` | постоянная и переменная `v(t)`; 200×30, 600×10, смешанное и нерегулярное расписание дают одинаковую сумму; alert не прибавляет награды; регрессия по старой формуле |
| T-22 | `test_discount_is_time_aware_and_composes`, `test_discounted_return_matches_the_tick_level_definition_for_every_split`, `test_gae_with_lambda_one_equals_the_monte_carlo_advantage_for_variable_intervals`, `test_gae_is_zero_for_the_exact_value_function_whatever_the_split` | `γ_10³ = γ_30`; дисконтированная отдача = тик-уровневой для любого разбиения; GAE (λ=1) = Монте-Карло при смешанных Δ; точный critic даёт нулевое преимущество |
| T-22 | `test_death_penalty_is_paid_exactly_once_even_with_alerts`, `test_a_decision_at_or_after_the_end_of_the_episode_is_rejected`, `test_bootstrap_is_the_learned_critic_only_and_only_after_truncation` | штраф один раз; терминал не бутстрэпится и не зависит от переданного значения; усечение использует только critic; нет решения в/после тика гибели |

Те же тесты переносятся в тесты движка/тренера (E3a.3–E3a.4) на реальных `viability_trace`; эталон остаётся независимой проверкой арифметики.

### 9.3 Нет прямого приза за врага
Запрещено в коде награды: упоминание `colony_id` других, чтение состояний других колоний, события «убил матку», «убил N». Тест `test_reward_reads_only_own_state` (миграция, E3).

### 9.4 Anti-hacking по построению (проверяется тестами, RFC §14)
| Эксплойт | Почему ограничен |
|---|---|
| Бездействие (idle) | без добычи `pop` не растёт, а `runway = склад / upkeep` ≈ 14 тиков на старте (80 / 5.6): `security` быстро падает, затем голод |
| Накопление еды | `security` насыщается при `runway ≥ 40`; склад ≤ 600; рабочие перестают ходить при полном складе |
| Пустая («полая») колония на полном складе | v1 платила 0.4 за такую колонию бесконечно (найдено PPO в первом пилоте); v2: `security × min(1, E_pop/10)` → `v → 0`, но сеть всё равно выбрала «не размножаться» (пилот 2: безопасно от −1, платит 0.13 за стартовых рабочих); v3: функциональное вымирание завершает эпизод со штрафом −1, пустая стратегия стоит −0.88 (тесты `test_eco_reward_v3.py`, `test_eco_rl_v3.py`) |
| Бесконечные солдаты | вес `0.5` в `pop`; upkeep 0.8 уменьшает `runway`; голод убивает солдат первыми (экономика §5) |
| Бесконечные рабочие | `pop` насыщается на 60 WE; сверх потолка — голод; насыщенные площадки не окупают рабочих |
| Сидеть «в крепости» | остаётся `pop ≈ локальный потолок` (≈ 25 WE → `pop ≈ 0.4`), теряется рост на border/far |
| Война ради убийств | убийство не оплачивается; оплачивается только доля еды (экономика §8.2) |

### 9.5 Смерть, горизонт, эпизоды

| Событие | Уровень | Что происходит |
|---|---|---|
| `queen_hp == 0` | колония | статус `orphan`; эпизод агента **завершён** (`terminated`, reason `queen_death`); юниты остаются в мире и живут по стоячим приказам до старения, голода или боя (§5, экономика §8.3); мир **не** заканчивается |
| взрослых 0 и в расплоде нет рабочих и `food < worker_egg_cost` | колония | `collapse` (аналог L0 экономического коллапса, `docs/combat_terminal.md`); эпизод агента завершён, статус `extinct` |
| у сироты (или колонии) не осталось взрослых и расплода | колония | статус `extinct`; склад → куча в узле гнезда |
| `tick == H` | мир | `truncated`; живые колонии остаются живыми; метрики считаются на горизонте |
| все колонии не `alive` | мир | мир завершается раньше горизонта (`all_extinct`) |
| `stop_when_alone` (опция, по умолчанию выкл.) | harness | единственная живая колония → **усечение** (не терминал), записывается в replay; см. RFC V-3 |

Сироты и экстинктные колонии **не** получают решений и не дают траекторий: их траектория обрывается терминалом.

### 9.5a Эпизод агента и статус колонии в мире (награда v3)
Строки таблицы §9.5 **не менялись**: `ColonyStatus`, `queen_hp`, склад, правило `collapse` и физика остаются прежними. Награда v3 вводит **отдельное** понятие — конец эпизода агента
(`functional_extinction`, §9.1): колония `alive` без взрослых и без расплода рабочих. Оно не пишет в состояние мира, не создаёт кучу еды, не меняет статус и не завершает мир.
Различие в отчётах: `physical_survival` — колония `alive` в мире на горизонте; `functional_survival` — эпизод агента не завершался (физическим терминалом или функциональным вымиранием).

* **Обучающий rollout:** колония с завершённым эпизодом *выведена из решений* (`PolicyController.retired`): ей не строятся наблюдение и маска, сеть не вызывается (память GRU, поток RNG,
  log-prob не меняются), записей решений нет; её стоящие приказы остаются как есть, новых рассылок добычи нет (мир про это множество не знает; сироты по-прежнему обслуживаются
  существующим правилом). Остальные колонии идут как обычно. Множество по умолчанию пусто: все прочие запуски, fixtures и replay не меняются.
* **Оценка и replay:** мир естественный — политика продолжает играть, терминал виден только в счёте (`terminal_reason`, `terminal tick`, обе выживаемости). Файлы replay и их проверка не меняются.

---

## 10. Checkpoint и replay

### 10.1 Checkpoint (`formicmind.eco.checkpoint/1`)
```
checkpoint = { config, map, state, run: { root_seed, streams: [{id, state, inc}], windows... } , outcomes }
checkpoint_hash = SHA-256("formicmind/eco/checkpoint/v1" ‖ 0x00 ‖ canonical({state, run}))
```
Содержит всё для продолжения прогона **движка**: состояние мира (с памятями колоний), приказы, RNG-потоки (map, world, policy:<id>:<colony>), исходы.
Состояние политик (скрытое состояние GRU) принадлежит **ролл-ауту тренера**, не движку; для воспроизведения мира достаточно записанных действий.
Загрузка строгая (как в L0): неверная схема, дубликаты, неотсортированные массивы, невалидные инварианты — ошибка. Исходы сверяются с состоянием (как `terminal.validate_outcome`).

### 10.2 Replay (`eco-replay/v1`)
UTF-8 JSONL, одна каноническая запись на строку, цепочка хэшей — **механизм L0 `docs/replay_v1.md` переиспользуется**; записи — новые:

```
header      config, config_hash, map (полностью), map_hash, root_seed, seats (по колониям: policy_id, stream_id, ref), contract
keyframe    каждые K тиков: cohorts (позиции), patches.stock, piles, colony vitals (food, W, S, brood, queen_hp, status), memory (список открытых слотов → node_id)
decision    по (tick, colony): action (латент и декодированный ActionE1), obs_hash, mask_hash        # для verify
event*      типизированные события (§6) в порядке исполнения
delta       дисплейный патч между keyframe (не обязателен для verify)
result      исходы колоний, исход мира, chain_head, record_count, events_hash, state_hash
```

* **verify** заново исполняет движок из `header` + записанных `decision` и сверяет `events_hash`, `state_hash`, `obs_hash` — без нейросети.
* **Viewer** читает только `header/keyframe/event`; слой «знание колонии c» строится из `memory` keyframe (а не из мира) — он показывает ровно то, что видит политика.
* Оценка размера: событий ≈ 20 тыс. × ~120 Б ≈ 2.4 МБ, keyframes 200 × ~6 КБ ≈ 1.2 МБ, решений 1 600 × ~300 Б ≈ 0.5 МБ → **≈ 4–5 МБ на мир (6000 тиков)** несжатым; gzip ожидаемо ×6–10. [ОТКРЫТО] точная цифра — после E1-c.

### 10.3 Детерминизм
Одинаковые `(ConfigE1, map_seed, root_seed, действия)` ⇒ одинаковые `events_hash`/`state_hash`/`checkpoint_hash`. Разрешённые источники случайности: поток `map` (только генерация) и потоки политик
(не влияют на правила, только на выбор действий; записываются). Одновременные эффекты решаются симметричным правилом (пропорциональный делёж, остаток остаётся в узле; урон делится по силе нападающих);
никакой приоритет по `colony_id`.

---

## 11. Инварианты движка (проверяются после тика при `check_invariants=True`)

1. Сохранение еды (экономика §6) — ровно, без допуска.
2. `0 ≤ food ≤ storage_cap` у живых колоний; `0 ≤ stock ≤ capacity` на площадках (кучи — без потолка).
3. `count = ceil(hp_total/unit_hp) ≥ 1` у каждой когорты; когорта на ребре имеет `depart < arrive`; когорта не находится в двух местах.
4. Колония `alive` ⇒ `queen_hp > 0`; `orphan` ⇒ `queen_hp == 0`; `extinct` ⇒ нет взрослых и расплода.
5. Память: каждый `slot_of_node` указывает на узел, где колония действительно побывала или который видела инцидентным ребром; `visited ⇒ discovered`.
6. Сироты и экстинктные колонии не получают решений.
7. `cohort_id` уникальны и монотонны; `next_cohort_id > max(cohort_id)`.
8. Хэш состояния не зависит от порядка вставки словарей.
9. Журнал возрастов согласован с когортами (§4.1 п.4); нет взрослого старше `L + age_bucket` (при `L > 0`).

---

## 12. Спецификация проверки утечек (leak audit)

* **Свойство 1 (контрфактуальная неизменность).** Для случайных состояний `s` и `s'`, отличающихся только тем, что колония `c` не наблюдала (положение чужих когорт вне присутствия, состояние чужих складов,
  неоткрытые узлы, HP чужой матки без наблюдения), `ObservationE1(s, c)` и маски **побайтно равны**.
* **Свойство 2 (перестановка меток).** Переименование `colony_id` и `node_id` (с сохранением структуры) не меняет `ObservationE1`.
* **Свойство 3 (источник масок).** Модуль масок не импортирует `WorldStateE1`; получает только `ObservationE1`.
* **Свойство 4 (action targets).** Для всех состояний `attack_target` ⊆ `sightings ∪ known_nests`; неизвестное гнездо не выбирается ни при каком латенте.
* **Свойство 5 (критик).** Критик и функция награды получают те же `ObservationE1` (+ собственные накопители), что и политика; центральный критик не реализуется (RFC V-5).
