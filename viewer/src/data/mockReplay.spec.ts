import { describe, expect, it } from 'vitest'
import { mockReplay } from './mockReplay'
import { NODE_LAYOUT, WORLD_EDGES } from '../domain/replay'

describe('MockReplayDataSource', () => {
  it('exposes only authored frames and the seven-node, nine-edge L0 map', () => {
    expect(mockReplay.getMetadata().sourceLabel).toContain('ДЕМО')
    expect(mockReplay.getFrame(8).state.tickCompleted).toBe(false)
    expect(NODE_LAYOUT.map((node) => node.name)).toEqual(['A', 'N1', 'N2', 'C', 'N4', 'N5', 'B'])
    expect(WORLD_EDGES).toHaveLength(9)
    expect(() => mockReplay.getFrame(9)).toThrow(RangeError)
  })

  it('filters recorded events by category and side without inventing events', () => {
    expect(mockReplay.getEvents({ category: 'combat' }).map((event) => event.kind)).toEqual(['death', 'combat', 'queen_damage'])
    expect(mockReplay.getEvents({ side: 'B' }).every((event) => event.side === 'B')).toBe(true)
    expect(mockReplay.getEvents({ category: 'economy', side: 'A' }).map((event) => event.kind)).toEqual(['lay', 'upkeep'])
  })

  it('keeps a mid-tick terminal tick separate from its presentation cursor', () => {
    expect(mockReplay.getCursorForTick(7)).toBe(7)
    expect(mockReplay.getCursorForTick(8)).toBeUndefined()
    expect(mockReplay.getMetadata().terminal).toMatchObject({ cursor: 8, terminalTick: 7, stateTick: 7, tickCompleted: false, stage: 'combat' })
    expect(mockReplay.getFrame(7).state).toEqual({ tick: 7, tickCompleted: true, boundary: 'start' })
    expect(mockReplay.getFrame(8).state).toEqual({ tick: 7, tickCompleted: false, boundary: 'terminal' })
    expect(mockReplay.getCursorForTick(99)).toBeUndefined()
  })
})
