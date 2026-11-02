# FormicMind

Headless-симулятор L0 (Python, только stdlib). Реализовано:

* фундамент (FOR-9): модель мира, `reset`, RNG, каноническая сериализация, SHA-256 хэши —
  [docs/foundation.md](docs/foundation.md);
* граница агента (FOR-10): ObservationV2 (223 float32), ActionMaskV2 (30 bool), приватная разведка,
  общий decoder 17 logits → ActionV2, policy-интерфейс — [docs/agent_boundary.md](docs/agent_boundary.md).

* каркас тика и движение (FOR-11): оркестратор фаз, применение приказов, квоты рабочих, маршруты, прибытия,
  события, `WindowStats` в checkpoint v2 — [docs/tick_movement.md](docs/tick_movement.md).
* экономика и расплод (FOR-12): Harvest, Upkeep/Starvation, Lay (партии яиц), Completion/Hatch, Regeneration,
  food ledger, режим `PARTIAL (NO COMBAT/TERMINAL)` — [docs/economy.md](docs/economy.md). Боя, terminal-проверок,
  runner, реплея и CLI пока нет.

Smoke-прогон движения (90 тиков): `python -m formicmind.smoke`; экономики (120 тиков): `python -m formicmind.economy_smoke`.

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
