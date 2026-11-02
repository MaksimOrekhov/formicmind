export type Side = 'A' | 'B'
export type Caste = 'workers' | 'soldiers'
export type EventCategory = 'combat' | 'economy' | 'movement' | 'terminal'
export type EventKind =
  | 'lay'
  | 'hatch'
  | 'harvest'
  | 'upkeep'
  | 'arrival'
  | 'combat'
  | 'death'
  | 'starvation'
  | 'queen_damage'
  | 'terminal'

export interface ColonyView {
  side: Side
  queenHp: number
  queenMaxHp: number
  food: number
  workers: number
  soldiers: number
  brood: number
  queenAlive: boolean
}

export interface NodeGroups {
  A: { workers: number; soldiers: number }
  B: { workers: number; soldiers: number }
}

export interface NodeView {
  id: number
  name: string
  food: number
  capacity: number
  groups: NodeGroups
  isHome: Side | null
  alert?: 'contact'
}

export interface TransitGroupView {
  id: string
  side: Side
  caste: Caste
  count: number
  from: number
  to: number
  /** Position recorded by this demo frame; the viewer does not route the group. */
  progress: number
  etaTicks: number
}

export interface TimelineFrame {
  /** Replay cursor. This is distinct from state.tick at a mid-tick terminal. */
  cursor: number
  state: {
    tick: number
    tickCompleted: boolean
  }
  colonies: Record<Side, ColonyView>
  nodes: NodeView[]
  transits: TransitGroupView[]
}

export interface ViewerEvent {
  id: string
  cursor: number
  category: EventCategory
  kind: EventKind
  side: Side | null
  nodeId: number | null
  summary: string
  detail: string
}

export interface ReplayTerminal {
  cursor: number
  terminalTick: number
  stateTick: number
  tickCompleted: boolean
  stage: string
  outcome: 'win' | 'draw' | 'collapse' | 'timeout'
  winner: Side | null
}

export interface ReplayMetadata {
  id: string
  title: string
  subtitle: string
  recordedAt: string
  seedLabel: string
  sourceLabel: string
  minCursor: number
  maxCursor: number
  terminal: ReplayTerminal | null
}

export interface EventQuery {
  category?: EventCategory | 'all'
  side?: Side | 'all'
}

/** Presentation-only seam for replacing the demo source after FOR-15 is ready. */
export interface ReplayDataSource {
  getMetadata(): ReplayMetadata
  getFrame(cursor: number): TimelineFrame
  getEvents(query?: EventQuery): ViewerEvent[]
  getCursorForTick(tick: number): number | undefined
}

export const NODE_LAYOUT = [
  { id: 0, name: 'A', x: 112, y: 278, isHome: 'A' },
  { id: 1, name: 'N1', x: 316, y: 165, isHome: null },
  { id: 2, name: 'N2', x: 316, y: 405, isHome: null },
  { id: 3, name: 'C', x: 556, y: 285, isHome: null },
  { id: 4, name: 'N4', x: 796, y: 165, isHome: null },
  { id: 5, name: 'N5', x: 796, y: 405, isHome: null },
  { id: 6, name: 'B', x: 1000, y: 278, isHome: 'B' },
] as const

export const WORLD_EDGES = [
  { a: 0, b: 1, length: 2 },
  { a: 4, b: 6, length: 2 },
  { a: 0, b: 2, length: 3 },
  { a: 5, b: 6, length: 3 },
  { a: 1, b: 3, length: 2 },
  { a: 3, b: 4, length: 2 },
  { a: 2, b: 3, length: 3 },
  { a: 3, b: 5, length: 3 },
  { a: 2, b: 5, length: 4 },
] as const
