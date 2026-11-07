import { describe, expect, it } from 'vitest'
import { colonyView, combatLosses, eventCountThrough, eventVisibleToColony, frameIndexAtTick, keyMoments, placeCohort } from './model'
import { pythonRun, sampleReplay } from './testing'
import type { EcoFrame, EcoReplay } from './types'

const replay = sampleReplay()

function clone<T>(value: T): T { return JSON.parse(JSON.stringify(value)) }

describe('frames, positions and key moments (no invented physics)', () => {
  it('maps a tick to the last recorded frame', () => {
    expect(frameIndexAtTick(replay, 0)).toBe(0)
    expect(replay.frames[frameIndexAtTick(replay, 19)].tick).toBe(0)
    expect(replay.frames[frameIndexAtTick(replay, 20)].tick).toBe(20)
    expect(replay.frames[frameIndexAtTick(replay, 2621)].tick).toBe(2620)
    expect(replay.frames[frameIndexAtTick(replay, 2622)].tick).toBe(2622) // the extra frame right after the queen died
    expect(frameIndexAtTick(replay, 6000)).toBe(replay.frames.length - 1)
    expect(frameIndexAtTick(replay, 99999)).toBe(replay.frames.length - 1)
  })

  it('interpolates along an edge only with the recorded depart/arrive ticks', () => {
    const frame = replay.frames.find((f) => f.tick === 1000)!
    const cohort = frame.cohorts.find((c) => !c.atNode && c.arrive - c.depart >= 4 && c.depart < frame.tick && c.arrive > frame.tick + 1)!
    expect(cohort).toBeDefined()
    const from = replay.nodes[cohort.from]
    const to = replay.nodes[cohort.to]
    const at = placeCohort(replay, cohort, frame, frame.tick)
    const share = (frame.tick - cohort.depart) / (cohort.arrive - cohort.depart)
    expect(at.exact).toBe(true)
    expect(at.mode).toBe('edge')
    expect(at.x).toBeCloseTo(from.x + (to.x - from.x) * share, 6)
    expect(at.y).toBeCloseTo(from.y + (to.y - from.y) * share, 6)
    // one tick later the cohort is still on the same trip: still exact and strictly further along
    const later = placeCohort(replay, cohort, frame, frame.tick + 1)
    expect(later.exact).toBe(true)
    expect(later.progress).toBeGreaterThan(at.progress)
  })

  it('does not pretend to know what happens after a trip ends or while a cohort stands between frames', () => {
    const frame = replay.frames.find((f) => f.tick === 1000)!
    const walking = frame.cohorts.find((c) => !c.atNode)!
    const past = placeCohort(replay, walking, frame, walking.arrive + 5)
    expect(past).toMatchObject({ exact: false, mode: 'arrived', progress: 1, x: replay.nodes[walking.to].x, y: replay.nodes[walking.to].y })
    const standing = frame.cohorts.find((c) => c.atNode)!
    expect(placeCohort(replay, standing, frame, frame.tick)).toMatchObject({ exact: true, mode: 'node' })
    expect(placeCohort(replay, standing, frame, frame.tick + 7)).toMatchObject({ exact: false, mode: 'held' })
  })

  it('finds exactly the moments of the story and counts events through a tick', () => {
    const moments = keyMoments(replay)
    expect(moments.map((m) => [m.kind, m.tick])).toEqual(expect.arrayContaining([['nest_found', 886], ['dispatch', 2280], ['queen_dead', 2621], ['extinct', 4400]]))
    expect(moments.filter((m) => m.kind === 'queen_dead')).toHaveLength(1)
    expect(moments.filter((m) => m.kind === 'dispatch')).toHaveLength(1) // only the first attack order is a key moment
    expect(eventCountThrough(replay.events, -1)).toBe(0)
    expect(eventCountThrough(replay.events, 6000)).toBe(replay.events.length)
    const at = eventCountThrough(replay.events, 2621)
    expect(replay.events[at - 1].tick).toBeLessThanOrEqual(2621)
    expect(replay.events[at]?.tick ?? 9999).toBeGreaterThan(2621)
  })

  it('reproduces the Python combat losses from the death events', () => {
    const python = pythonRun().interaction.combat_deaths_by_colony as Record<string, { workers: number; soldiers: number }>
    const losses = combatLosses(replay, 6000)
    for (const [colony, expected] of Object.entries(python)) expect(losses[Number(colony)]).toEqual(expected)
    expect(combatLosses(replay, 0).every((l) => l.workers === 0 && l.soldiers === 0)).toBe(true)
  })
})

describe('colony view: what a colony knows, and only that', () => {
  const frame1000 = replay.frames.find((f) => f.tick === 1000)!

  it('contains only visited nodes and nodes the colony has merely heard of, with known edges only', () => {
    for (let colony = 0; colony < replay.colonyCount; colony += 1) {
      const view = colonyView(replay, frame1000, colony)
      const memory = frame1000.memory.find((m) => m.colony === colony)!
      expect(view.nodes.map((n) => n.id)).toEqual([...memory.visited.map((v) => v.node), ...memory.discovered].sort((a, b) => a - b))
      expect(view.unknownNodeCount).toBe(96 - view.nodes.length)
      expect(view.unknownNodeCount).toBeGreaterThan(0)
      const visited = new Set(memory.visited.map((v) => v.node))
      expect(view.edges.every((e) => visited.has(e.a) || visited.has(e.b))).toBe(true)
      expect(view.edges.length).toBe(replay.edges.filter((e) => visited.has(e.a) || visited.has(e.b)).length)
      for (const node of view.nodes.filter((n) => n.status === 'discovered')) {
        expect(node).toMatchObject({ kind: 'unknown', stock: null, capacity: 0, nestOwner: -1, queenSeen: 0 })
      }
      expect(view.ownCohorts.every((c) => c.colony === colony)).toBe(true)
      expect(view.foreign.every((f) => f.colony !== colony)).toBe(true)
    }
  })

  it('does not change when only hidden facts change (counterfactual)', () => {
    const colony = 3
    const base = JSON.stringify(colonyView(replay, frame1000, colony))
    const alt: EcoReplay = clone(replay)
    const altFrame: EcoFrame = alt.frames.find((f) => f.tick === 1000)!
    const memory = altFrame.memory.find((m) => m.colony === colony)!
    const known = new Set([...memory.visited.map((v) => v.node), ...memory.discovered])
    // other colonies: vitals, food, queens, armies, positions, and their private memories
    for (const row of altFrame.colonies) if (row.id !== colony) Object.assign(row, { food: row.food + 77, queenHp: 1, workers: row.workers + 40, soldiers: 99, broodWorkers: 9 })
    for (const cohort of altFrame.cohorts) if (cohort.colony !== colony) Object.assign(cohort, { count: cohort.count + 5, load: 0, from: (cohort.from + 1) % 96, atNode: true, to: -1 })
    for (const other of altFrame.memory) if (other.colony !== colony) { other.visited = []; other.foreign = []; other.sightings = []; other.discovered = [] }
    // true stocks and piles (the colony only has its last-seen copies), the true map content of nodes it has never heard of
    altFrame.patches = altFrame.patches.map((s) => s + 5)
    altFrame.piles = [{ node: 1, stock: 500 }]
    const hiddenNode = alt.nodes.find((n) => !known.has(n.id) && n.kind === 'patch')!
    Object.assign(hiddenNode, { capacity: 9999, kind: 'nest', nestOf: 5 })
    expect(JSON.stringify(colonyView(alt, altFrame, colony))).toBe(base)
  })

  it('reveals nothing for a colony that has visited nothing but its own nest', () => {
    const frame = clone(frame1000)
    const colony = 5
    const nest = replay.nests[colony]
    frame.memory.find((m) => m.colony === colony)!.visited = [{ node: nest, seenTick: frame.tick - 1, lastStock: -1, lastStockTick: -1, queenSeen: 1, danger: 0 }]
    frame.memory.find((m) => m.colony === colony)!.discovered = []
    frame.memory.find((m) => m.colony === colony)!.foreign = []
    frame.memory.find((m) => m.colony === colony)!.sightings = []
    const view = colonyView(replay, frame, colony)
    expect(view.nodes.map((n) => n.id)).toEqual([nest])
    expect(view.foreign).toEqual([])
    expect(view.sightings).toEqual([])
    expect(view.edges.every((e) => e.a === nest || e.b === nest)).toBe(true)
    expect(view.nodes[0]).toMatchObject({ kind: 'nest', nestOwner: colony, inView: true })
  })

  it('marks old information as historical and learns a foreign nest only by meeting it', () => {
    const before = colonyView(replay, replay.frames.find((f) => f.tick === 880)!, 3)
    expect(before.foreign.find((f) => f.nestNode >= 0)).toBeUndefined()
    expect(before.nodes.every((n) => n.nestOwner === -1 || n.nestOwner === 3)).toBe(true)
    const after = colonyView(replay, replay.frames.find((f) => f.tick === 900)!, 3)
    const node = after.nodes.find((n) => n.nestOwner === 0)!
    expect(node).toMatchObject({ kind: 'nest', status: 'visited' })
    expect(after.foreign.find((f) => f.colony === 0)?.nestNode).toBe(node.id)
    // late in the run the scout has long left: that node is stale, flagged as not in view and ageing
    const late = colonyView(replay, replay.frames.find((f) => f.tick === 1500)!, 3)
    const old = late.nodes.find((n) => n.nestOwner === 0)!
    expect(old.inView).toBe(false)
    expect(old.age).toBeGreaterThan(1)
    expect(old.seenTick).toBeLessThan(1500)
  })

  it('never shows another colony its rival as a live position, only as dated sightings with strength buckets', () => {
    const view = colonyView(replay, replay.frames.find((f) => f.tick === 2000)!, 3)
    expect(view.ownCohorts.every((c) => c.colony === 3)).toBe(true)
    for (const sighting of view.sightings) {
      expect(sighting.age).toBeGreaterThanOrEqual(0)
      expect(sighting.workersBucket).toBeLessThanOrEqual(7)
    }
  })

  it('lets a colony see its own events and the fights it took part in, nothing else', () => {
    const own = replay.events.find((e) => e.kind === 'discover' && e.colony === 2)!
    expect(eventVisibleToColony(own, 2)).toBe(true)
    expect(eventVisibleToColony(own, 3)).toBe(false)
    const combat = replay.events.find((e) => e.kind === 'combat' && e.colony === 3 && e.other === 0)!
    expect(eventVisibleToColony(combat, 3)).toBe(true)
    expect(eventVisibleToColony(combat, 0)).toBe(true)
    expect(eventVisibleToColony(combat, 5)).toBe(false)
    const dispatch = replay.events.find((e) => e.kind === 'dispatch' && e.a === 3)!
    expect(eventVisibleToColony(dispatch, dispatch.other)).toBe(false) // the target is not told about an order it cannot observe
  })

  it('builds a view for every frame and colony without error', () => {
    for (const frame of replay.frames) for (let colony = 0; colony < replay.colonyCount; colony += 1) colonyView(replay, frame, colony)
  })
})
