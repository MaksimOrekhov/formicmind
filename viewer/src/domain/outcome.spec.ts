import { describe, expect, it } from 'vitest'
import { presentOutcome } from './outcome'
import type { ReplayTerminal } from './replay'

const terminalFixture = (overrides: Partial<ReplayTerminal>): ReplayTerminal => ({
  cursor: 8,
  terminalTick: 7,
  stateTick: 7,
  tickCompleted: false,
  stage: 'combat',
  outcome: 'win',
  winner: 'A',
  ...overrides,
})

describe('presentOutcome', () => {
  it('shows a draw without assigning a winner', () => {
    expect(presentOutcome(terminalFixture({ outcome: 'draw', winner: null }))).toMatchObject({
      title: 'Ничья',
      tone: 'draw',
    })
  })

  it('never invents a winning side when a recorded outcome omits winner', () => {
    expect(presentOutcome(terminalFixture({ outcome: 'win', winner: null })).title).toBe('Победа')
    expect(presentOutcome(terminalFixture({ outcome: 'collapse', winner: null })).title).toBe('Коллапс')
    expect(presentOutcome(terminalFixture({ outcome: 'timeout', winner: null })).title).toBe('Таймаут')
  })

  it('uses the recorded winner for collapse and timeout while keeping the reason visible', () => {
    expect(presentOutcome(terminalFixture({ outcome: 'collapse', winner: 'B' }))).toMatchObject({
      title: 'Победа колонии B',
      detail: expect.stringContaining('коллапс'),
    })
    expect(presentOutcome(terminalFixture({ outcome: 'timeout', winner: 'A' }))).toMatchObject({
      title: 'Победа колонии A',
      detail: expect.stringContaining('таймаут'),
    })
  })
})
