# FormicMind Viewer

Самостоятельный read-only desktop-интерфейс наблюдения на Vue 3, TypeScript и Vite. Поддерживается запуск в настольном браузере Windows.

```powershell
cd viewer
npm ci
npm run dev
```

При первом запуске открыт демонстрационный матч (`ДЕМО · НЕ СИМУЛЯЦИЯ`). Кнопка **Открыть replay** позволяет выбрать локальный файл Replay v1 `.jsonl`. Файл читается в браузере и не отправляется в сеть. До завершения проверки текущий просмотр не меняется; повреждённый файл оставляет предыдущий источник на экране.

`ReplayDataSource` в `src/domain/replay.ts` — общий presentation-контракт demo и `Replay v1`. `src/data/replayJsonl.ts` проверяет версии и структуру, chain SHA-256 по canonical-байтам, `config_hash`, `events_hash`, `frame_hash`, обязательные snapshots, patches и result перед созданием источника. Snapshot/delta используются только для восстановления отображаемого состояния; игровые правила и физика не реализованы в JavaScript.

Лента событий показывает до 50 строк за раз и позволяет загружать более ранние события. Таймлайн использует собственный frame cursor: он не совпадает с игровым тиком, а terminal mid-tick имеет отдельный кадр. Положение группы в транзите схематично отмечено на середине ребра; ETA читается из фактического кадра.

Структурная и хэш-проверка браузера не является `python -m formicmind.replay verify`: viewer не заявляет о повторном исполнении симуляции. Проверка Python ReplayReader/verify остаётся нормативной для подтверждения игрового исполнения.

`test-fixtures/` содержит Replay JSONL, созданные Python writer, и выборочные golden кадры Python ReplayReader. Сжатый timeout fixture позволяет прогнать в frontend-тесте импорт всех 3000 тиков без хранения 10 МиБ JSONL в исходном виде.

Проверки в `viewer/`: `npm run build`, `npm run test`. Windows-only frontend job находится в `.github/workflows/ci.yml`.

## Ecosystem v1 (E1)

По умолчанию открывается viewer Ecosystem v1 (`src/eco/`): локальный replay реального прогона E1 (8 колоний, 96 узлов), карта с pan/zoom, воспроизведение по тикам мира, лента событий и режим «Вид колонии». Пример `public/samples/eco_seed50004_raider3.jsonl` загружается автоматически; свой файл создаёт `python -m formicmind.eco replay-export` (см. `docs/ecosystem_v1_viewer.md`). Viewer L0 доступен по `#/l0` и не изменён. Проверки те же: `npm run build`, `npm run test`.
