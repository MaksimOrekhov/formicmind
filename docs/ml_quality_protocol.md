# FOR-37 — протокол первого эксперимента качества imitation

**Статус:** зафиксирован до сбора целевых данных и обучения. После этого коммита нельзя менять контексты, split, метрики или пороги по результатам held-out.

## Вопрос и ограничения

Проверяется, воспроизводит ли MLP `223 → 64 → 64 → 17` действия одного учителя `balanced:v2` на новых состояниях и игровых контекстах. Это техническая проверка imitation, не оценка стратегического превосходства и не решение о запуске RL.

FOR-36 `formicmind.behavior-cloning/v1` записывает один `SimConfig`, одну пару teacher/opponent, две перестановки сторон и split только по целым seeds. Он проверяет уникальность `(match_id, tick)`, но не имеет test split, fingerprint траектории, глобальной дедупликации между датасетами или контекстами, групп по карте/opponent, метрик по контекстам и latency percentiles. Его схема и checkpoint остаются читаемыми без изменений.

Учитель во всех наборах — `balanced:v2`; его descriptor и parameter SHA-256 сохраняются. В тренировочных контекстах opponents — `turtle:v2` и `random:v2`; случайный baseline добавляет разнообразие состояний, но не становится teacher. `rush:v2` полностью отложен от train/validation как unseen opponent. Нейросеть видит только `ObservationV2`; mask получается из observation, а общий decoder и validator остаются без изменений.

## Игровые конфигурации

Все карты строятся из штатного `default_config()` и валидируются обычным `validate_config`. Меняются только зеркально-симметричные рёбра `Topology`; стартовые ресурсы, экономика, правила, `rules_version`, decoder и Baselines остаются неизменны.

| map_id | Топология |
| --- | --- |
| `classic` | Неизменённая `default_topology()` (9 рёбер). |
| `cross` | `classic` без `(N2, N5)`, плюс зеркальная пара `(N1, N5)` и `(N2, N4)`, длина 4 (10 рёбер). Сохраняет связность и маршруты с `avoid_center`. |
| `dense` | `classic` плюс зеркальная пара `(N1, N5)` и `(N2, N4)`, длина 4. |

Все пары узлов должны иметь маршрут как с разрешённым центром, так и с `avoid_center` (кроме destination C, запрещённого самим контрактом). В manifest записываются сериализованный `SimConfig`, map ID и config SHA-256. Идентификатор карты не подменяет проверку hash.

## Seeds, контексты и split

Seeds — новые открытые инженерные значения из непересекающихся диапазонов. Они не берутся из sealed FOR-4 или исторических FOR-25/FOR-35 результатов. Один seed прогоняется за обе стороны во всех назначенных ему контекстах; повторение одного seed внутри split не создаёт независимого примера.

| Split | Seeds | Карты | Opponents | Число контекстов |
| --- | --- | --- | --- | ---: |
| train | `370000001–370000024` (24) | classic, cross | turtle:v2, random:v2 | 4 |
| validation | `371000001–371000008` (8) | classic, cross | turtle:v2, random:v2 | 4 |
| held-out test | `372000001–372000008` (8) | classic, cross, dense | turtle:v2, random:v2, rush:v2; только dense × все три и classic/cross × rush | 5 |

Каждый контекст запускается с обеими сторонами teacher, `max_ticks=3000` (полная штатная граница `T_max`). Получается до 192 train, 64 validation и 80 held-out эпизодов до дедупликации. Held-out использует новые seeds, целиком неиспользованную карту `dense` и целиком неиспользованный профиль `rush:v2`; комбинации контекстов заранее заданы таблицей и не расширяются по увиденным метрикам. Никакие закрытые evaluation seeds не используются.

## Группировка и anti-leakage

Сначала сборщик фиксирует список всех эпизодов и их исходный split по диапазону seed и матрице контекстов. Для эпизода строится `trajectory_sha256` по упорядоченным решениям на границе policy: observation, derived mask, teacher logits и teacher action. Seed, сторона, map ID, opponent и tick label в fingerprint не входят: одинаковая последовательность решений обнаруживается и между разными seed/контекстами. Для отдельного решения строится аналогичный `decision_sha256` без metadata.

Идентичные полные траектории дедуплицируются глобально внутри объединённого набора. Если одна и та же траектория оказалась в разных splits, сохраняется один канонический экземпляр с приоритетом `train`, затем `validation`, затем `heldout`; dropped aliases, исходные источники и причина исключения записываются в manifest. Затем идентичные decision fingerprints между splits оставляются только в split с тем же приоритетом. Это исключает прямое совпадение входа и target между train/validation/test даже при разных seeds. Частично очищенные эпизоды остаются привязаны к своему `trajectory_group_id`; число исключённых решений и эффективный размер каждого split публикуются. Тест не переименовывает повторный seed в независимую репликацию.

Manifest v2 хранит source Git SHA/dirty state, SHA-256 файла данных, hashes конфигураций и Baselines, contracts, явные episode/trajectory/decision group IDs и fingerprints, назначение split и полный список удалённых дубликатов. Загрузчик отклоняет пересечение seed sets, полных trajectory hashes или decision hashes между splits, повреждённые fingerprints, несогласованные metadata и отсутствующие split. В рамках одного объединённого файла также проверяется отсутствие пересечений с train/validation при match-evaluation.

## Обучение и контрольные модели

Архитектура, objective и оптимизатор не меняются: `223 → 64 → 64 → 17`, MSE к 17 teacher logits, Adam, `lr=1e-3`, 8 эпох, batch size 128. Порядок батчей и инициализация фиксируются seed `370900037`. Никакого early stopping и подбора по held-out; validation только измеряется. Для метрик на тех же held-out примерах создаётся отдельно случайно инициализированная MLP с seed `370900038`; веса не обновляются. Teacher oracle — точное совпадение с сохранёнными teacher action/logits, верхняя техническая граница, не отдельная обученная модель.

До попытки обучения фиксируется окружение. RTX 5080 видна ОС, однако находящийся в репозитории `.venv` пока не содержит PyTorch. CUDA будет считаться использованной только после установки совместимой сборки и runtime-проверки `torch.cuda.is_available()`/имени GPU; иначе обучаем и оцениваем на CPU и так указываем в manifest.

## Метрики

Для каждого split и отдельно для каждой map/opponent группы публикуются:

- примеры после дедупликации, исходные примеры, уникальные полные траектории, seed, context и decision counts;
- MSE 17 teacher logits;
- exact match полного `ActionV2`, а также lay, army, forage tuple и avoid exact accuracy;
- доля легальных действий после общего decoder/validator и технические отказы (ожидается 100% легальности; сбой не маскируется как потеря);
- инференс latency mean/p50/p95 на одном решении и throughput с указанием устройства.

Обученная и необученная MLP оцениваются на идентичном held-out наборе. Oracle сообщает MSE 0 и exact accuracy 1.0. Если после дедупликации весь held-out содержит меньше 200 решений или меньше 4 уникальных trajectory groups, либо отдельный map/opponent контекст имеет меньше 100 решений, такой контекст помечается как маловыборочный; вывод о переносе на него не делается. Все counts показываются явно.

## Реальные матчи

После фиксации checkpoint выполняются парные матчи на `classic` с новыми seeds `373000001–373000004` отдельно против `balanced:v2`, `turtle:v2` и `rush:v2`; каждая пара включает обе стороны модели (24 матча). `max_ticks=3000`. Для каждого матча сохраняются opponent, seed, сторона, ticks, terminal reason, score только при terminal, queen HP, food, worker/soldier count, hashes состояния/событий и `truncated`. Completed и truncated матчи публикуются раздельно; незавершённый матч не получает score и не считается победой/поражением. Эти матчи описывают фактическое поведение, но не заменяют imitation метрики и не дают claim о превосходстве над Baselines.

## Заранее объявленные критерии решения

Технический **GO только к отдельному проектированию самостоятельного обучения**, не к автоматическому запуску RL, возможен, если: (1) held-out после дедупликации содержит не менее 200 решений и 4 полных уникальных траекторий, каждый из пяти held-out map/opponent contexts — не менее 100 решений; (2) полный held-out exact-action accuracy не ниже 0.80 и минимум на 0.20 выше необученной MLP; (3) held-out logit MSE ниже необученного контроля; (4) legal action rate — 100%, загрузка checkpoint стабильна; (5) разрыв validation→held-out по exact accuracy не больше 0.15; (6) provenance и split проверки проходят. Матчевые результаты при этом рассматриваются отдельно и не превращаются в утверждение о стратегической силе.

Иначе вывод — **NO-GO/недостаточно данных** с указанием конкретного провала (качество, переобучение, дубликаты, sample size, легальность или инфраструктура). Даже GO не разрешает смешивать teacher targets, менять physics/economy, decoder, rules version, baseline, запускать PPO/self-play или использовать sealed/historical данные.

## Порядок исполнения и воспроизводимость

1. Зафиксировать этот документ отдельным коммитом до обучения.
2. Добавить versioned v2 collect/train/eval путь с регрессией чтения v1 datasets/checkpoints и тестами leakage/provenance.
3. Собрать все splits одной командой/одним dataset manifest; записать SHA-256 и параметры до обучения.
4. Однократно обучить на заранее заданных параметрах; вычислить метрики по train/validation/held-out и контрольным MLP/oracle.
5. Выполнить указанные парные real matches и сохранить отчёт по-русски вместе с manifest и runtime.
6. Запустить полный pytest, Vue production build, Vitest, Windows CI и неизменённые pinned historical FOR-25/FOR-35 jobs. Не менять их pins или исходные артефакты.

PR открывается без merge; в его описании будут commit HEAD, итоги эксперимента, ссылка/состояние Windows CI и ограничения вывода.

Команды фиксированного профиля:

```powershell
formicmind-ml collect-quality --output results\ml_quality_for37\quality.jsonl
formicmind-ml train-quality --dataset results\ml_quality_for37\quality.jsonl --output results\ml_quality_for37\checkpoint --device auto
formicmind-ml eval-quality --model results\ml_quality_for37\checkpoint --dataset results\ml_quality_for37\quality.jsonl --device auto --output results\ml_quality_for37\report.json
```
