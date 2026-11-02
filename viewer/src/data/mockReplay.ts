import type {
  ColonyView,
  EventQuery,
  NodeGroups,
  NodeView,
  ReplayDataSource,
  ReplayMetadata,
  Side,
  TimelineFrame,
  TransitGroupView,
  ViewerEvent,
} from '../domain/replay'

const metadata: ReplayMetadata = {
  id: 'demo-river-01',
  title: 'Северный проход',
  subtitle: 'Контакт у центрального ресурса',
  recordedAt: '02 ноя 2026 · 14:32',
  seedLabel: 'DEMO SEED 042',
  sourceLabel: 'ДЕМО · НЕ СИМУЛЯЦИЯ',
  minCursor: 0,
  maxCursor: 8,
  terminal: {
    cursor: 8,
    terminalTick: 7,
    stateTick: 7,
    tickCompleted: false,
    stage: 'combat',
    outcome: 'win',
    winner: 'A',
  },
}

const groups = (
  aw: number,
  as: number,
  bw: number,
  bs: number,
): NodeGroups => ({ A: { workers: aw, soldiers: as }, B: { workers: bw, soldiers: bs } })

const makeNodes = (
  food: readonly [number, number, number, number, number, number, number],
  occupants: readonly NodeGroups[],
  contactAt: number | null = null,
): NodeView[] => [
  { id: 0, name: 'A', food: food[0], capacity: 0, groups: occupants[0], isHome: 'A' },
  { id: 1, name: 'N1', food: food[1], capacity: 80, groups: occupants[1], isHome: null },
  { id: 2, name: 'N2', food: food[2], capacity: 60, groups: occupants[2], isHome: null },
  { id: 3, name: 'C', food: food[3], capacity: 120, groups: occupants[3], isHome: null, ...(contactAt === 3 ? { alert: 'contact' as const } : {}) },
  { id: 4, name: 'N4', food: food[4], capacity: 80, groups: occupants[4], isHome: null },
  { id: 5, name: 'N5', food: food[5], capacity: 60, groups: occupants[5], isHome: null },
  { id: 6, name: 'B', food: food[6], capacity: 0, groups: occupants[6], isHome: 'B' },
]

const colony = (
  side: Side,
  queenHp: number,
  food: number,
  workers: number,
  soldiers: number,
  brood: number,
  queenAlive = true,
): ColonyView => ({ side, queenHp, queenMaxHp: 240, food, workers, soldiers, brood, queenAlive })

const transit = (
  id: string,
  side: Side,
  caste: TransitGroupView['caste'],
  count: number,
  from: number,
  to: number,
  progress: number,
  etaTicks: number,
): TransitGroupView => ({ id, side, caste, count, from, to, progress, etaTicks })

/** Each snapshot below is authored demo data. This mapper only packages those values. */
const frames: TimelineFrame[] = [
  { cursor: 0, state: { tick: 0, tickCompleted: true }, colonies: { A: colony('A', 240, 60, 12, 4, 3), B: colony('B', 240, 60, 12, 4, 3) }, nodes: makeNodes([0, 60, 40, 80, 60, 40, 0], [groups(0, 4, 0, 0), groups(0, 0, 0, 0), groups(0, 0, 0, 0), groups(0, 0, 0, 0), groups(0, 0, 0, 0), groups(0, 0, 0, 0), groups(0, 0, 0, 4)]), transits: [] },
  { cursor: 1, state: { tick: 1, tickCompleted: true }, colonies: { A: colony('A', 240, 52, 12, 4, 3), B: colony('B', 240, 59, 12, 4, 3) }, nodes: makeNodes([0, 52, 40, 80, 60, 40, 0], [groups(0, 4, 0, 0), groups(4, 0, 0, 0), groups(0, 0, 0, 0), groups(0, 0, 0, 0), groups(0, 0, 0, 0), groups(0, 0, 0, 0), groups(0, 0, 0, 4)]), transits: [transit('a-workers-1', 'A', 'workers', 4, 0, 1, 0.52, 1)] },
  { cursor: 2, state: { tick: 2, tickCompleted: true }, colonies: { A: colony('A', 240, 50, 12, 4, 4), B: colony('B', 240, 57, 12, 4, 3) }, nodes: makeNodes([0, 52, 40, 80, 60, 40, 0], [groups(0, 4, 0, 0), groups(4, 0, 0, 0), groups(0, 0, 0, 0), groups(0, 0, 0, 0), groups(0, 0, 0, 0), groups(0, 0, 0, 0), groups(0, 0, 0, 4)]), transits: [transit('a-workers-1', 'A', 'workers', 4, 0, 1, 0.9, 1)] },
  { cursor: 3, state: { tick: 3, tickCompleted: true }, colonies: { A: colony('A', 240, 50, 12, 4, 4), B: colony('B', 240, 56, 12, 4, 3) }, nodes: makeNodes([0, 52, 40, 80, 60, 40, 0], [groups(0, 4, 0, 0), groups(4, 0, 0, 0), groups(0, 0, 0, 0), groups(0, 0, 0, 0), groups(0, 0, 0, 0), groups(0, 0, 0, 0), groups(0, 0, 0, 4)]), transits: [transit('a-workers-1', 'A', 'workers', 4, 0, 1, 0.96, 1)] },
  { cursor: 4, state: { tick: 4, tickCompleted: true }, colonies: { A: colony('A', 240, 50, 12, 4, 4), B: colony('B', 240, 55, 12, 4, 3) }, nodes: makeNodes([0, 52, 40, 80, 60, 40, 0], [groups(0, 4, 0, 0), groups(4, 0, 0, 0), groups(0, 0, 0, 0), groups(0, 0, 0, 0), groups(0, 0, 0, 0), groups(0, 0, 0, 0), groups(0, 0, 0, 4)]), transits: [transit('a-soldiers-2', 'A', 'soldiers', 3, 1, 3, 0.45, 2), transit('b-soldiers-1', 'B', 'soldiers', 3, 5, 3, 0.38, 2)] },
  { cursor: 5, state: { tick: 5, tickCompleted: true }, colonies: { A: colony('A', 240, 50, 12, 4, 4), B: colony('B', 240, 47, 12, 4, 3) }, nodes: makeNodes([0, 52, 40, 80, 52, 40, 0], [groups(0, 4, 0, 0), groups(4, 0, 0, 0), groups(0, 0, 0, 0), groups(0, 0, 0, 0), groups(4, 0, 0, 0), groups(0, 0, 0, 4), groups(0, 0, 0, 4)]), transits: [transit('a-soldiers-2', 'A', 'soldiers', 3, 1, 3, 0.72, 1)] },
  { cursor: 6, state: { tick: 6, tickCompleted: true }, colonies: { A: colony('A', 240, 48, 12, 4, 4), B: colony('B', 204, 47, 12, 4, 3) }, nodes: makeNodes([0, 52, 40, 80, 52, 40, 0], [groups(0, 4, 0, 0), groups(4, 0, 0, 0), groups(0, 0, 0, 0), groups(0, 0, 0, 0), groups(4, 0, 0, 0), groups(0, 0, 0, 4), groups(0, 0, 0, 4)]), transits: [transit('a-soldiers-2', 'A', 'soldiers', 3, 1, 3, 0.88, 1)] },
  { cursor: 7, state: { tick: 7, tickCompleted: true }, colonies: { A: colony('A', 240, 48, 12, 4, 4), B: colony('B', 156, 46, 12, 2, 3) }, nodes: makeNodes([0, 52, 40, 80, 52, 40, 0], [groups(0, 4, 0, 0), groups(4, 0, 0, 0), groups(0, 0, 0, 0), groups(0, 3, 0, 1), groups(4, 0, 0, 0), groups(0, 0, 0, 4), groups(0, 0, 0, 2)], 3), transits: [transit('a-workers-2', 'A', 'workers', 4, 1, 3, 0.62, 1)] },
  { cursor: 8, state: { tick: 7, tickCompleted: false }, colonies: { A: colony('A', 240, 48, 12, 4, 4), B: colony('B', 0, 46, 12, 1, 3, false) }, nodes: makeNodes([0, 52, 40, 80, 52, 40, 0], [groups(0, 4, 0, 0), groups(4, 0, 0, 0), groups(0, 0, 0, 0), groups(0, 3, 0, 1), groups(4, 0, 0, 0), groups(0, 0, 0, 4), groups(0, 0, 0, 0)], 3), transits: [transit('a-workers-2', 'A', 'workers', 4, 1, 3, 0.88, 1)] },
]

const events: ViewerEvent[] = [
  { id: 'e0', cursor: 0, category: 'economy', kind: 'upkeep', side: 'A', nodeId: 0, summary: 'Стартовые запасы колонии A', detail: 'В кадре записано 60 единиц еды в хранилище матки.' },
  { id: 'e1', cursor: 1, category: 'movement', kind: 'arrival', side: 'A', nodeId: 1, summary: 'Группа рабочих A движется к N1', detail: 'В кадре: 4 рабочих на ребре A — N1, ETA 1 тик.' },
  { id: 'e2', cursor: 2, category: 'economy', kind: 'lay', side: 'A', nodeId: 0, summary: 'Зафиксирована кладка A', detail: 'В записи кадра число расплода A изменилось до 4.' },
  { id: 'e3', cursor: 4, category: 'movement', kind: 'arrival', side: 'A', nodeId: 3, summary: 'Отряды направлены к узлу C', detail: 'В кадре отмечены транзитные группы обеих колоний.' },
  { id: 'e4', cursor: 5, category: 'economy', kind: 'harvest', side: 'B', nodeId: 4, summary: 'Сбор еды в N4', detail: 'В кадре записан запас N4: 52 / 80.' },
  { id: 'e5', cursor: 6, category: 'combat', kind: 'queen_damage', side: 'B', nodeId: 6, summary: 'Матка B получила урон', detail: 'Значение HP в кадре: 204.' },
  { id: 'e6', cursor: 7, category: 'combat', kind: 'combat', side: null, nodeId: 3, summary: 'Контакт у центрального узла C', detail: 'Кадр отмечает присутствие боевых групп A и B.' },
  { id: 'e7', cursor: 8, category: 'combat', kind: 'death', side: 'B', nodeId: 6, summary: 'Матка B погибла', detail: 'Терминальное событие записано во время стадии боя.' },
  { id: 'e8', cursor: 8, category: 'terminal', kind: 'terminal', side: 'A', nodeId: 6, summary: 'В записи указан исход: победа A', detail: 'Демонстрационная запись содержит исход матча; UI его не вычисляет.' },
]

export class MockReplayDataSource implements ReplayDataSource {
  getMetadata(): ReplayMetadata {
    return metadata
  }

  getFrame(cursor: number): TimelineFrame {
    const frame = frames.find((item) => item.cursor === cursor)
    if (!frame) throw new RangeError(`No demo frame exists at cursor ${cursor}`)
    return frame
  }

  getEvents(query: EventQuery = {}): ViewerEvent[] {
    return events.filter((event) => {
      const categoryMatches = !query.category || query.category === 'all' || event.category === query.category
      const sideMatches = !query.side || query.side === 'all' || event.side === query.side
      return categoryMatches && sideMatches
    })
  }

  getCursorForTick(tick: number): number | undefined {
    const frame = frames.find((item) => item.state.tick === tick && item.state.tickCompleted)
    if (frame) return frame.cursor
    return metadata.terminal?.terminalTick === tick ? metadata.terminal.cursor : undefined
  }
}

export const mockReplay = new MockReplayDataSource()
