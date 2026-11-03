export type Side = 'A' | 'B'
export type Caste = 'workers' | 'soldiers'
export type ReplayBoundary = 'start' | 'terminal'
export type EventCategory = 'combat' | 'economy' | 'movement' | 'terminal'
export type EventKind =
  | 'order' | 'retarget' | 'depart' | 'arrive' | 'hatch' | 'harvest' | 'upkeep' | 'starvation'
  | 'lay' | 'regenerate' | 'combat' | 'wound' | 'death' | 'queen_damage' | 'terminal'

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
  arrivalTick: number
  /** The replay contract has no departure tick or exact position. Real markers use the edge midpoint. */
  progress: number
  positionIsApproximate: boolean
  etaTicks: number
}

export interface TimelineFrame {
  /** Presentation-frame cursor. Multiple boundaries may share one engine tick. */
  cursor: number
  state: {
    tick: number
    tickCompleted: boolean
    boundary: ReplayBoundary
  }
  colonies: Record<Side, ColonyView>
  nodes: NodeView[]
  transits: TransitGroupView[]
}

export interface ViewerEvent {
  id: string
  /** Frame cursor after the event's executed tick; the event is hidden before this frame. */
  cursor: number
  /** Engine tick from Event.to_canonical(), distinct from cursor. */
  tick: number
  category: EventCategory
  kind: EventKind
  side: Side | null
  nodeId: number | null
  summary: string
  detail: string
}

export interface ReplayTerminal {
  /** Cursor of the terminal presentation frame. */
  cursor: number
  /** Domain tick carried by the terminal event. */
  terminalTick: number
  /** state.tick stored in the terminal ViewFrame (terminalTick or terminalTick + 1). */
  stateTick: number
  tickCompleted: boolean
  stage: string
  reason: 'queen_death' | 'economic_collapse' | 'timeout'
  outcome: 'win' | 'draw'
  winner: Side | null
}

export interface WorldNodeDefinition {
  id: number
  name: string
  isHome: Side | null
  capacity: number
}

export interface WorldEdgeDefinition {
  a: number
  b: number
  length: number
}

export interface ReplayMetadata {
  id: string
  title: string
  subtitle: string
  recordedAt: string | null
  seedLabel: string
  sourceLabel: string
  sourceKind: 'demo' | 'replay'
  fileName: string | null
  minCursor: number
  maxCursor: number
  terminal: ReplayTerminal | null
  finished: boolean
  ticksExecuted: number
  eventCount: number
  timeLimit: number
  policyInterval: number
  policies: Record<Side, string>
  world: { nodes: WorldNodeDefinition[]; edges: WorldEdgeDefinition[] }
}

export interface EventQuery {
  category?: EventCategory | 'all'
  side?: Side | 'all'
  throughCursor?: number
  offset?: number
  limit?: number
}

/** Presentation-only seam shared by demo and locally validated Replay v1 data. */
export interface ReplayDataSource {
  getMetadata(): ReplayMetadata
  getFrame(cursor: number): TimelineFrame
  getEvents(query?: EventQuery): ViewerEvent[]
  getEventCount(query?: EventQuery): number
  getCursorForTick(tick: number, boundary?: ReplayBoundary): number | undefined
}

export const NODE_LAYOUT = [
  { id: 0, name: 'A', x: 112, y: 278 },
  { id: 1, name: 'N1', x: 316, y: 165 },
  { id: 2, name: 'N2', x: 316, y: 405 },
  { id: 3, name: 'C', x: 556, y: 285 },
  { id: 4, name: 'N4', x: 796, y: 165 },
  { id: 5, name: 'N5', x: 796, y: 405 },
  { id: 6, name: 'B', x: 1000, y: 278 },
] as const

export const WORLD_EDGES: WorldEdgeDefinition[] = [
  { a: 0, b: 1, length: 2 }, { a: 0, b: 2, length: 3 }, { a: 1, b: 3, length: 2 },
  { a: 2, b: 3, length: 3 }, { a: 2, b: 5, length: 4 }, { a: 3, b: 4, length: 2 },
  { a: 3, b: 5, length: 3 }, { a: 4, b: 6, length: 2 }, { a: 5, b: 6, length: 3 },
]

export const DEMO_WORLD = {
  nodes: NODE_LAYOUT.map((node) => ({
    id: node.id,
    name: node.name,
    isHome: node.id === 0 ? 'A' as const : node.id === 6 ? 'B' as const : null,
    capacity: [0, 80, 60, 120, 80, 60, 0][node.id],
  })),
  edges: WORLD_EDGES,
}
