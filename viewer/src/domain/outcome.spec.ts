import { describe, expect, it } from 'vitest'
import { presentOutcome } from './outcome'
import type { ReplayTerminal } from './replay'

const terminalFixture = (overrides: Partial<ReplayTerminal>): ReplayTerminal => ({
  cursor: 8,
  terminalTick: 7,
  stateTick: 7,
  tickCompleted: false,
  stage: 'QueenTerminal',
  reason: 'queen_death',
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

  it('uses the recorded winner for queen death and economic collapse', () => {
    expect(presentOutcome(terminalFixture({ reason: 'economic_collapse', tickCompleted: true, stateTick: 8, winner: 'B' }))).toMatchObject({
      title: 'Победа колонии B',
      detail: expect.stringContaining('экономический коллапс'),
    })
  })

  it('renders timeout as a recorded draw', () => {
    expect(presentOutcome(terminalFixture({ reason: 'timeout', outcome: 'draw', winner: null, tickCompleted: true, stateTick: 8 }))).toMatchObject({
      title: 'Ничья',
      detail: expect.stringContaining('таймаут'),
    })
  })

  it('does not invent a winner if a malformed caller omits one', () => {
    expect(presentOutcome(terminalFixture({ outcome: 'win', winner: null }))).toMatchObject({ title: 'Победа', tone: 'neutral' })
  })
})
