import type {
  CohortState, ColonyFrame, EcoEdge, EcoEvent, EcoFrame, EcoReplay, EventKind, ForeignRecord, MemoryFrame, SightingRecord,
} from './types'

/** Index of the last frame whose tick is <= ``tick`` (frames are sorted by tick; frame 0 is tick 0). */
export function frameIndexAtTick(replay: EcoReplay, tick: number): number {
  const frames = replay.frames
  let low = 0
  let high = frames.length - 1
  while (low < high) {
    const mid = (low + high + 1) >> 1
    if (frames[mid].tick <= tick) low = mid
    else high = mid - 1
  }
  return low
}

export interface Placement {
  x: number
  y: number
  /**
   * ``exact``: the position follows from the recorded ``depart``/``arrive`` ticks of the trip (or the cohort stands in a node of this very frame).
   * Otherwise the cohort is shown where the last frame left it and is marked approximate: what happened between two frames is not recorded.
   */
  exact: boolean
  mode: 'node' | 'edge' | 'arrived' | 'held'
  /** 0..1 along the edge from ``from`` to ``to`` while ``mode === 'edge'``. */
  progress: number
}

/** Where a cohort of ``frame`` is drawn at ``tick`` (>= frame.tick). Only real schedules are used; nothing is invented between frames. */
export function placeCohort(replay: EcoReplay, cohort: CohortState, frame: EcoFrame, tick: number): Placement {
  const from = replay.nodes[cohort.from]
  if (cohort.atNode) {
    return { x: from.x, y: from.y, exact: tick === frame.tick, mode: tick === frame.tick ? 'node' : 'held', progress: 0 }
  }
  const to = replay.nodes[cohort.to]
  const span = cohort.arrive - cohort.depart
  if (tick >= cohort.arrive) {
    // it reached the far node at ``arrive``; where it went next is not in the file until the next frame
    return { x: to.x, y: to.y, exact: false, mode: 'arrived', progress: 1 }
  }
  const progress = Math.max(0, Math.min(1, (tick - cohort.depart) / span))
  return { x: from.x + (to.x - from.x) * progress, y: from.y + (to.y - from.y) * progress, exact: true, mode: 'edge', progress }
}

/** Number of events with ``event.tick <= tick`` (events are sorted by tick). */
export function eventCountThrough(events: EcoEvent[], tick: number): number {
  let low = 0
  let high = events.length
  while (low < high) {
    const mid = (low + high) >> 1
    if (events[mid].tick <= tick) low = mid + 1
    else high = mid
  }
  return low
}

export interface KeyMoment { tick: number; kind: EventKind; label: string; colony: number; node: number; eventIndex: number }

const MOMENT_LABEL: Partial<Record<EventKind, string>> = {
  contact: 'первая встреча колоний', nest_found: 'найдено чужое гнездо', dispatch: 'первый приказ атаки', combat: 'первый бой',
  queen_dead: 'гибель матки', collapse: 'экономический коллапс', extinct: 'колония вымерла',
}

/** The moments worth jumping to: the first of each interaction kind, and every queen death, collapse and extinction. */
export function keyMoments(replay: EcoReplay): KeyMoment[] {
  return keyMomentsOf(replay.events)
}

/** Same selection over any (already filtered, tick-ordered) event list; the "first of each kind" is relative to that list. */
export function keyMomentsOf(events: readonly EcoEvent[]): KeyMoment[] {
  const moments: KeyMoment[] = []
  const firstSeen = new Set<EventKind>()
  for (const event of events) {
    const label = MOMENT_LABEL[event.kind]
    if (!label) continue
    if (event.kind === 'dispatch' && event.a !== 3) continue // attack orders only
    const repeatable = event.kind === 'queen_dead' || event.kind === 'collapse' || event.kind === 'extinct'
    if (!repeatable && firstSeen.has(event.kind)) continue
    firstSeen.add(event.kind)
    moments.push({ tick: event.tick, kind: event.kind, label, colony: event.colony, node: event.node, eventIndex: event.index })
  }
  return moments
}

export interface Losses { workers: number; soldiers: number }

/** Cumulative combat deaths per colony through ``tick`` (from ``death`` events). */
export function combatLosses(replay: EcoReplay, tick: number): Losses[] {
  const losses = Array.from({ length: replay.colonyCount }, () => ({ workers: 0, soldiers: 0 }))
  const through = eventCountThrough(replay.events, tick)
  for (let i = 0; i < through; i += 1) {
    const event = replay.events[i]
    if (event.kind !== 'death') continue
    if (event.a === 1) losses[event.colony].soldiers += event.b
    else losses[event.colony].workers += event.b
  }
  return losses
}

// ------------------------------------------------------------------------------------------------ colony view (fog of war)

export interface FogNode {
  id: number
  x: number
  y: number
  /** ``visited``: the colony has stood there; ``discovered``: it only knows the node exists (an edge of a visited node leads to it). */
  status: 'visited' | 'discovered'
  kind: 'junction' | 'patch' | 'nest' | 'unknown'
  capacity: number
  /** Last stock the colony saw on a visited patch, or ``null`` if it never saw one. */
  stock: number | null
  stockAge: number | null
  seenTick: number
  /** Ticks since the node was last observed; ``0``/``1`` means it is in view right now. */
  age: number
  inView: boolean
  /** Colony that owns the nest, only if this colony learned it by meeting it there; ``-1`` unknown. */
  nestOwner: number
  queenSeen: 0 | 1 | 2
  danger: number
}

export interface FogView {
  colony: number
  tick: number
  nodes: FogNode[]
  edges: EcoEdge[]
  own: ColonyFrame
  ownCohorts: CohortState[]
  foreign: ForeignRecord[]
  sightings: (SightingRecord & { age: number })[]
  /** How many nodes of the map this colony has never heard of (the count itself is a viewer fact, not shown to the colony). */
  unknownNodeCount: number
}

/**
 * What colony ``colonyId`` knows at ``frame``: built only from its own memory, its own cohorts and its own vitals. The true state of other colonies,
 * unvisited nodes and true patch stocks are never read, so the view cannot reveal them (tests change those parts and expect an identical result).
 */
export function colonyView(replay: EcoReplay, frame: EcoFrame, colonyId: number): FogView {
  const memory: MemoryFrame = frame.memory.find((item) => item.colony === colonyId) ?? { colony: colonyId, visited: [], discovered: [], foreign: [], sightings: [] }
  const ownNest = replay.nests[colonyId]
  const nestOwnerByNode = new Map<number, number>()
  for (const record of memory.foreign) if (record.nestNode >= 0) nestOwnerByNode.set(record.nestNode, record.colony)
  const nodes: FogNode[] = []
  const visitedIds = new Set<number>()
  for (const item of memory.visited) {
    visitedIds.add(item.node)
    const node = replay.nodes[item.node]
    const age = frame.tick - item.seenTick
    nodes.push({
      id: node.id, x: node.x, y: node.y, status: 'visited', kind: node.kind, capacity: node.capacity,
      stock: node.kind === 'patch' && item.lastStockTick >= 0 ? item.lastStock : null,
      stockAge: node.kind === 'patch' && item.lastStockTick >= 0 ? frame.tick - item.lastStockTick : null,
      seenTick: item.seenTick, age, inView: age <= 1,
      nestOwner: node.id === ownNest ? colonyId : (nestOwnerByNode.get(node.id) ?? -1),
      queenSeen: item.queenSeen, danger: item.danger,
    })
  }
  for (const id of memory.discovered) {
    const node = replay.nodes[id]
    nodes.push({ id, x: node.x, y: node.y, status: 'discovered', kind: 'unknown', capacity: 0, stock: null, stockAge: null, seenTick: -1, age: -1,
      inView: false, nestOwner: -1, queenSeen: 0, danger: 0 })
  }
  nodes.sort((a, b) => a.id - b.id)
  const edges = replay.edges.filter((edge) => visitedIds.has(edge.a) || visitedIds.has(edge.b))
  const colonyRow = frame.colonies[colonyId]
  return {
    colony: colonyId,
    tick: frame.tick,
    nodes,
    edges,
    own: colonyRow,
    ownCohorts: frame.cohorts.filter((cohort) => cohort.colony === colonyId),
    foreign: memory.foreign,
    sightings: memory.sightings.map((item) => ({ ...item, age: frame.tick - item.tick })),
    unknownNodeCount: replay.nodes.length - nodes.length,
  }
}

/** Events a colony may see in its own view: its own, plus fights and queen kills in which it took part. */
export function eventVisibleToColony(event: EcoEvent, colony: number): boolean {
  if (event.colony === colony) return true
  return (event.kind === 'combat' || event.kind === 'queen_dead') && event.other === colony
}

/**
 * Whether the colony view may show ``event`` while the viewer displays ``tick``: the colony took part in it AND the displayed state already contains its
 * effects. The state shown at ``tick`` is the state after tick ``tick - 1`` completed, so an event of tick ``t`` becomes known from ``t + 1`` on.
 * Everything the colony view lists (feed, counters, markers, jump targets, highlight) goes through this one predicate, so no later or foreign event can leak.
 */
export function eventObservedBy(event: EcoEvent, colony: number, tick: number): boolean {
  return event.tick < tick && eventVisibleToColony(event, colony)
}

/** Key moments (first contact, queen death ...) of one colony's own story, limited to what it has observed by ``tick``. */
export function colonyKeyMoments(replay: EcoReplay, colony: number, tick: number): KeyMoment[] {
  const through = eventCountThrough(replay.events, tick)
  return keyMomentsOf(replay.events.slice(0, through).filter((event) => eventObservedBy(event, colony, tick)))
}

export const COLONY_COLORS = ['#e7b04a', '#5fb8ff', '#f0716f', '#6fd49a', '#c58bf0', '#ff9f5a', '#4fd2d2', '#d6e05a'] as const
export function colonyColor(colony: number): string {
  return COLONY_COLORS[colony % COLONY_COLORS.length]
}
