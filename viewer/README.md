# FormicMind Viewer

Самостоятельный read-only интерфейс наблюдения на Vue 3, TypeScript и Vite.

```powershell
cd viewer
npm install
npm run dev
```

`src/data/mockReplay.ts` — единственный источник явных демонстрационных кадров и событий (`ДЕМО · НЕ СИМУЛЯЦИЯ`). `ReplayDataSource` в `src/domain/replay.ts` — presentation-контракт; подключение JSONL и адаптера FOR-15 сюда пока не входит. Интерфейс отображает записанные снимки и не исполняет правила мира.

Для открытия сохранённого кадра можно передать `?cursor=8` (допустимые значения — 0–8).

Проверки: `npm run build`, `npm run test`.
