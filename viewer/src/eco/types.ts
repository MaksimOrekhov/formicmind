/** Presentation types of the Ecosystem v1 (E1) replay. Separate from the L0 two-side replay in ../domain/replay.ts. */

export type EventKind =
  | 'discover' | 'contact' | 'nest_found' | 'dispatch' | 'combat' | 'death' | 'queen_damage' | 'queen_dead' | 'collapse' | 'extinct'
export const EVENT_KINDS: readonly EventKind[] = [
  'discover', 'contact', 'nest_found', 'dispatch', 'combat', 'death', 'queen_damage', 'queen_dead', 'collapse', 'extinct',
]
export type ColonyStatus = 'alive' | 'orphan' | 'extinct'
export type MissionKind = 'home' | 'forage' | 'scout' | 'guard' | 'attack' | 'withdraw'
export type MissionPhase = 'idle' | 'go' | 'work' | 'return'
export type FrameReason = 'start' | 'interval' | 'key_event' | 'final'

export interface EcoNode {
  id: number
  kind: 'junction' | 'patch' | 'nest'
  nestOf: number
  x: number
  y: number
  /** Patch capacity (0 for other nodes). */
  capacity: number
  zone: string
}

export interface EcoEdge { a: number; b: number; length: number; laneCap: number; chokepoint: boolean }

export interface EcoEvent {
  index: number
  tick: number
  kind: EventKind
  colony: number
  other: number
  node: number
  a: number
  b: number
}

export interface CohortState {
  id: number
  colony: number
  caste: 'worker' | 'soldier'
  count: number
  hp: number
  load: number
  kind: MissionKind
  phase: MissionPhase
  atNode: boolean
  /** The node (atNode) or the node the cohort left (on an edge). */
  from: number
  /** The node the cohort walks to; -1 when it stands in a node. */
  to: number
  depart: number
  arrive: number
  targetNode: number
  targetColony: number
}

export interface ColonyFrame {
  id: number
  status: ColonyStatus
  food: number
  queenHp: number
  workers: number
  soldiers: number
  broodWorkers: number
  broodSoldiers: number
  starvationDebt: number
}

export interface VisitedNode { node: number; seenTick: number; lastStock: number; lastStockTick: number; queenSeen: 0 | 1 | 2; danger: number }
export interface ForeignRecord {
  colony: number
  firstContact: number
  lastSeen: number
  workersBucket: number
  soldiersBucket: number
  nestNode: number
  theyAttacked: number
  iAttacked: number
  killsMine: number
  killsTheirs: number
}
export interface SightingRecord { node: number; colony: number; tick: number; workersBucket: number; soldiersBucket: number; aggression: boolean }

/** The private memory of one colony at one frame: the only thing the colony knows. */
export interface MemoryFrame {
  colony: number
  visited: VisitedNode[]
  /** Nodes the colony has only heard of through an edge of a visited node. */
  discovered: number[]
  foreign: ForeignRecord[]
  sightings: SightingRecord[]
}

export interface EcoFrame {
  index: number
  tick: number
  reason: FrameReason
  stateHash: string
  /** True stock per patch node, in ``EcoReplay.patchNodes`` order. */
  patches: number[]
  piles: { node: number; stock: number }[]
  colonies: ColonyFrame[]
  cohorts: CohortState[]
  memory: MemoryFrame[]
}

/** The independent policies that decided a FOR-48 run (one per colony) and the binding to the sidecar decision records. */
export interface EcoPolicyInfo {
  seed: string
  policies: { colony: number; kind: string; variant: string }[]
  decisionCount: number
  decisionsHash: string
  /** FOR-50: the action-mask profile the run was played under (`none` = the full action space; `nest-v1` = a diagnostic stage without soldier orders, scouts and squads). */
  curriculum: string
}

export interface EcoReplay {
  fileName: string
  seed: string
  fixture: string
  roles: Record<string, string>
  colonyCount: number
  ticksRequested: number
  ticksCompleted: number
  frameEvery: number
  configHash: string
  mapHash: string
  rulesVersion: string
  nodes: EcoNode[]
  edges: EcoEdge[]
  nests: number[]
  patchNodes: number[]
  worldSide: number
  queenMaxHp: number
  /** Decision cadence of the run (informational). */
  decisionK: number
  frames: EcoFrame[]
  events: EcoEvent[]
  eventsHash: string
  finalStateHash: string
  chainHead: string
  worldOutcome: { finished: boolean; reason: string; terminalTick: number }
  colonyOutcomes: { colony: number; status: ColonyStatus; reason: string; terminalTick: number }[]
  /** What the browser re-checked itself (everything else is covered by the Python verifier). */
  verified: { chain: boolean; configHash: boolean; mapHash: boolean; eventsHash: boolean }
  /** ``null`` for a run of the diagnostic fixtures. */
  policy: EcoPolicyInfo | null
}
