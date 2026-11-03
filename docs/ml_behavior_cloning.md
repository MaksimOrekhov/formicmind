# Обучаемый MLP: технический behavior cloning пилот

Этот модуль проверяет сквозной цикл **настоящий матч → данные → обучение → безопасный checkpoint → инференс → настоящий матч**. Это техническая имитация выбранного Baselines v2, а не обучение новой стратегии. Совпадение с учителем и результаты коротких матчей не подтверждают стратегическое превосходство и не закрывают Gate 1/2.

## Контракт и граница информации

- MLP получает только `ObservationV2` — 223 float32. В сети нет маски, состояния мира, `RunState`, RNG, скрытой информации о противнике или идентификатора соперника.
- Модель возвращает 17 float32 `PolicyLogitsV2`. Доступность действий задаёт существующий `get_action_mask`; преобразование в `ActionV2` и валидацию выполняют общий `decode_action_logits` и `validate_action`.
- Зафиксирована малая сеть `223 → 64 → 64 → 17` с ReLU. Behavior cloning минимизирует MSE между логитами модели и выбранного Baselines v2. Это простой воспроизводимый технический target; отчёт дополнительно считает точность декодированного действия.
- Сборщик запускает симулятор через `run_episode` с `balanced:v2` или `rush:v2` в роли учителя, записывает только наблюдение, полученную из него маску, логиты и действие. Случайная политика не может быть единственным учителем. Примеры двух сторон одного seed входят в один split; train и validation делятся по целым seeds/матчам. Никакие скрытые поля состояния или закрытые test seeds FOR-4 не читаются.

## Установка

Основной симулятор по-прежнему устанавливается без PyTorch. Для CPU:

```powershell
py -3.12 -m venv .venv-ml
.\.venv-ml\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Для CUDA на Windows с RTX 5080 установите PyTorch-команду, сгенерированную [официальным селектором](https://pytorch.org/get-started/locally/) для Windows / Pip / Python / CUDA, затем установите проект и проверьте runtime. Не устанавливайте поверх CUDA-сборки CPU wheel.

```powershell
# Скопируйте актуальную команду PyTorch для CUDA из официального селектора.
python -m pip install -e .
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

`--device auto` выбирает CUDA только при `torch.cuda.is_available()`, иначе CPU. Явный `--device cuda` завершится понятной ошибкой, если CUDA недоступна. Для экспериментов зафиксируйте версии драйвера, Python и PyTorch из manifest. CPU детерминирован; не обещается побитовое совпадение весов между CPU/GPU или разными устройствами.

## Воспроизводимый короткий запуск на Windows

Seeds ниже — открытые инженерные примеры команды, а не seeds какого-либо запечатанного набора.

```powershell
New-Item -ItemType Directory -Force ml_runs\demo | Out-Null
formicmind-ml collect --output ml_runs\demo\train.jsonl --seeds 20261136 20261137 20261138 20261139 --teacher balanced:v2 --opponent turtle:v2 --max-ticks 120
formicmind-ml train --dataset ml_runs\demo\train.jsonl --output ml_runs\demo\checkpoint --seed 20261140 --epochs 8 --batch-size 128 --device auto
formicmind-ml eval --model ml_runs\demo\checkpoint --dataset ml_runs\demo\train.jsonl --opponent balanced:v2 --seed-start 20261150 --episodes 2 --max-ticks 120 --device auto --output ml_runs\demo\report.json
```

То же через `python -m formicmind.ml`. Dataset и checkpoint не включаются в Git: команды создают небольшую JSONL выборку и manifest, затем `weights.pt` и `manifest.json`.

Evaluation принимает только датасет, использованный при обучении checkpoint: должны совпасть SHA-256 JSONL, полный descriptor учителя, hash конфигурации симулятора и карта разбиения seeds. Evaluation seeds проверяются отдельно против исходных seeds checkpoint и seeds переданного датасета; любое пересечение останавливает запуск с ошибкой. Поэтому нельзя подменить датасет B и тем самым забыть о train/validation seeds checkpoint, обученного на датасете A.

## Артефакты и метрики

Dataset JSONL имеет соседний manifest с SHA-256, версиями контрактов, config hash, конфигурацией и SHA-256 Baselines v2, seeds, split по seed, матчами и Git SHA/dirty state. Загрузчик повторно проверяет hash, длины, float32, маску относительно ObservationV2 и равенство teacher action результату общего decoder.

Checkpoint сохраняет только `state_dict`; загрузка сперва проверяет JSON manifest, фиксированную архитектуру и SHA-256, затем использует `torch.load(..., weights_only=True)` и строгую проверку ключей/тензоров. Произвольные pickle checkpoint не принимаются. Manifest фиксирует contract versions, архитектуру/loss, dataset hash, teacher/config hashes, seeds/split, run seed, Python/PyTorch/CUDA/device, исходный Git SHA/dirty state, ошибки имитации, время шага обучения и latency/throughput инференса.

Evaluation считает долю легальных действий, точное совпадение полного действия и компонентов с учителем на validation split того же provenance-проверенного датасета, а также ограниченное число матчей через настоящий runner с перестановкой сторон. Для коротких матчей явно указываются `truncated` и ticks; их нельзя трактовать как win-rate. Запуск не использует PPO, reward, self-play, скрытые поля или закрытые test seeds.
