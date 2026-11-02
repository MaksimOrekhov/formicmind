# FormicMind

Headless-симулятор L0 (Python, только stdlib). Сейчас реализован **только фундамент** (FOR-9):
модель мира, `reset`, RNG, каноническая сериализация, SHA-256 хэши. Тика, движения, экономики,
боя, политик, реплея и CLI пока нет.

Подробности и список инженерных выборов: [docs/foundation.md](docs/foundation.md).

## Запуск тестов

Нужен Python ≥ 3.12 (проверено на 3.12.10).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # Linux/macOS: .venv/bin/python
.venv/Scripts/python -m pytest
```

`pytest` берёт пакет из `src/` (настройка `pythonpath` в `pyproject.toml`), поэтому после
установки dev-зависимости достаточно `python -m pytest`.

## Использование

```python
from formicmind import default_config, reset, state_hash, state_to_bytes, state_from_bytes

config = default_config()
state, run = reset(config, seed=20261102, stream_ids=["policy:baseline:0"])
data = state_to_bytes(state)                      # canonical JSON bytes
assert state_hash(state_from_bytes(data, config)) == state_hash(state)
```
