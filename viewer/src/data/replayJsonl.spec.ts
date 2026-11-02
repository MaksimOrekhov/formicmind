import { describe, expect, it } from 'vitest'
import { gunzipSync } from 'node:zlib'
import type { ReplayDataSource, TimelineFrame } from '../domain/replay'
import { concatBytes, hashDomain, utf8 } from './replayCrypto'
import { MAX_REPLAY_BYTES, ReplayVersionError, parseReplayBytes } from './replayJsonl'

const textFixtures = import.meta.glob<string>('../../test-fixtures/*.jsonl', { eager: true, query: '?raw', import: 'default' })
const otherFixtures = import.meta.glob<string>('../../test-fixtures/python-golden.json', { eager: true, query: '?raw', import: 'default' })
const timeoutGzipFixture = import.meta.glob<string>('../../test-fixtures/timeout_draw.jsonl.gz.b64', { eager: true, query: '?raw', import: 'default' })
const fixture = (name: string) => {
  const value = textFixtures[`../../test-fixtures/${name}`]
  if (value === undefined) throw new Error(`Missing Replay fixture: ${name}`)
  return utf8(value)
}
const goldenText = Object.values(otherFixtures)[0]
if (goldenText === undefined) throw new Error('Missing Python ReplayReader golden file')
const golden = JSON.parse(goldenText) as Record<string, GoldenFrame[]>
const DOMAIN = 'formicmind/replay-record/v1'
const ZERO = '0'.repeat(64)

interface GoldenFrame extends Omit<TimelineFrame, 'state'> {
  tick: number
  boundary: 'start' | 'terminal'
  tickCompleted: boolean
  terminal: TimelineFrame['state'] | null | Record<string, unknown>
}

function parseRecords(bytes: Uint8Array) {
  return new TextDecoder().decode(bytes).trimEnd().split('\n').map((line) => JSON.parse(line) as Record<string, any>)
}

function decompressGzipBase64(encoded: string) {
  const binary = atob(encoded.trim())
  const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0))
  return new Uint8Array(gunzipSync(bytes))
}

function sealRecords(records: Record<string, any>[], rewriteSequence = true) {
  let previous = ZERO
  const lines = records.map((record, seq) => {
    if (rewriteSequence) record.seq = seq
    if (record.type === 'result') {
      record.record_count = seq
      record.chain_head = previous
    }
    const unsigned = { ...record, chain: ZERO }
    const chain = hashDomain(DOMAIN, concatBytes(utf8(`${previous}\u0000`), utf8(canonical(unsigned))))
    record.chain = chain
    previous = chain
    return canonical(record)
  })
  return utf8(`${lines.join('\n')}\n`)
}

function canonical(value: unknown): string {
  if (typeof value === 'string') return JSON.stringify(value).replace(/[\u0080-\uFFFF]/g, (char) => `\\u${char.charCodeAt(0).toString(16).padStart(4, '0')}`)
  if (typeof value === 'number') return String(value)
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`
  if (value && typeof value === 'object') return `{${Object.keys(value).sort().map((key) => `${canonical(key)}:${canonical((value as Record<string, unknown>)[key])}`).join(',')}}`
  throw new Error('fixture canonicalizer received unsupported JSON')
}

function projectFrame(source: ReplayDataSource, expected: GoldenFrame) {
  const cursor = source.getCursorForTick(expected.tick, expected.boundary)
  if (cursor === undefined) throw new Error(`missing ${expected.boundary} frame for tick ${expected.tick}`)
  const frame = source.getFrame(cursor)
  const terminal = source.getMetadata().terminal?.cursor === cursor ? source.getMetadata().terminal : null
  return {
    cursor,
    tick: frame.state.tick,
    boundary: frame.state.boundary,
    tickCompleted: frame.state.tickCompleted,
    colonies: frame.colonies,
    nodes: frame.nodes.map(({ alert: _alert, ...node }) => node),
    transits: frame.transits,
    terminal,
  }
}

describe('Replay v1 browser adapter', () => {
  it('matches Python ReplayReader/ViewFrame golden frames for checkpoints, tick boundaries and all match outcomes', async () => {
    for (const name of ['raid_win', 'mutual_raid_draw', 'collapse', 'cutoff']) {
      const source = await parseReplayBytes(fixture(`${name}.jsonl`), `${name}.jsonl`)
      for (const expected of golden[name]) expect(projectFrame(source, expected)).toEqual({
        cursor: expected.cursor,
        tick: expected.tick,
        boundary: expected.boundary,
        tickCompleted: expected.tickCompleted,
        colonies: expected.colonies,
        nodes: expected.nodes,
        transits: expected.transits,
        terminal: expected.terminal,
      })
    }
  })

  it('keeps engine tick separate from presentation cursor around a mid-tick queen death', async () => {
    const source = await parseReplayBytes(fixture('raid_win.jsonl'), 'raid_win.jsonl')
    const metadata = source.getMetadata()
    expect(metadata.terminal).toMatchObject({ cursor: 6, terminalTick: 175, stateTick: 175, tickCompleted: false, winner: 'A' })
    expect(source.getCursorForTick(175)).toBe(5)
    expect(source.getCursorForTick(175, 'terminal')).toBe(6)
    expect(source.getCursorForTick(176)).toBeUndefined()
    expect(source.getFrame(5).state).toMatchObject({ tick: 175, boundary: 'start', tickCompleted: true })
    expect(source.getFrame(6).state).toMatchObject({ tick: 175, boundary: 'terminal', tickCompleted: false })
    expect(source.getEvents({ throughCursor: 5 }).some((event) => event.tick === 175)).toBe(false)
    expect(source.getEvents({ throughCursor: 6 }).some((event) => event.kind === 'terminal' && event.tick === 175)).toBe(true)
  })

  it('indexes a full 3000-tick replay, keeps the timeout draw and does not invent start(3000)', async () => {
    const encoded = Object.values(timeoutGzipFixture)[0]
    if (encoded === undefined) throw new Error('Missing 3000-tick Python Replay fixture')
    const bytes = decompressGzipBase64(encoded)
    expect(bytes.byteLength).toBeLessThanOrEqual(MAX_REPLAY_BYTES)
    const source = await parseReplayBytes(bytes, 'timeout_draw.jsonl')
    expect(source.getMetadata()).toMatchObject({ ticksExecuted: 3000, eventCount: 33335, maxCursor: 3000, finished: true })
    expect(source.getMetadata().terminal).toMatchObject({ terminalTick: 2999, stateTick: 3000, tickCompleted: true, outcome: 'draw', winner: null })
    expect(source.getCursorForTick(2999)).toBe(2999)
    expect(source.getCursorForTick(3000)).toBeUndefined()
    expect(source.getCursorForTick(3000, 'terminal')).toBe(3000)
    expect(source.getEvents({ limit: 50 })).toHaveLength(50)
    expect(source.getEventCount()).toBe(33335)
    expect(source.getEventCount({ category: 'terminal' })).toBe(1)
    expect(source.getEventCount({ side: 'A' })).toBeGreaterThan(0)
    expect(source.getEvents({ limit: 0 })).toHaveLength(0)
    const sampled = golden.timeout_draw
    for (const expected of sampled) expect(projectFrame(source, expected)).toMatchObject({
      cursor: expected.cursor, tick: expected.tick, boundary: expected.boundary, tickCompleted: expected.tickCompleted,
      colonies: expected.colonies, nodes: expected.nodes, transits: expected.transits, terminal: expected.terminal,
    })
  }, 30_000)

  it('fails closed for empty, truncated, wrong-version and corrupted-chain files', async () => {
    await expect(parseReplayBytes(new Uint8Array(), 'empty.jsonl')).rejects.toThrow(/пуст/i)
    const valid = fixture('raid_win.jsonl')
    await expect(parseReplayBytes(valid.slice(0, -1), 'truncated.jsonl')).rejects.toThrow(/обрезан/i)
    const badChain = parseRecords(valid)
    badChain[2].chain = `f${badChain[2].chain.slice(1)}`
    await expect(parseReplayBytes(utf8(`${badChain.map(canonical).join('\n')}\n`), 'corrupt.jsonl')).rejects.toThrow(/цепочк/i)
    const badVersion = parseRecords(valid)
    badVersion[0].replay_schema = 'v2'
    await expect(parseReplayBytes(sealRecords(badVersion), 'foreign.jsonl')).rejects.toBeInstanceOf(ReplayVersionError)
  })

  it('rejects validly resealed structural tampering and missing snapshots/results', async () => {
    const original = parseRecords(fixture('raid_win.jsonl'))
    const missingSnapshot = original.filter((record) => !(record.type === 'snapshot' && record.seq === 1))
    await expect(parseReplayBytes(sealRecords(missingSnapshot), 'missing-snapshot.jsonl')).rejects.toThrow(/начальный snapshot/i)

    const missingResult = original.slice(0, -1).map((record) => ({ ...record }))
    await expect(parseReplayBytes(sealRecords(missingResult), 'missing-result.jsonl')).rejects.toThrow(/result/i)

    const badSequence = original.map((record) => ({ ...record }))
    badSequence[2].seq = 1
    await expect(parseReplayBytes(sealRecords(badSequence, false), 'sequence.jsonl')).rejects.toThrow(/seq/i)

    const badPatch = original.map((record) => structuredClone(record))
    const delta = badPatch.find((record) => record.type === 'delta')!
    delta.frame_hash = ZERO
    await expect(parseReplayBytes(sealRecords(badPatch), 'patch.jsonl')).rejects.toThrow(/frame_hash/i)

    const unknown = original.map((record) => structuredClone(record))
    const event = unknown.find((record) => record.type === 'event')!
    event.e.extra_field = 'not in Replay v1'
    await expect(parseReplayBytes(sealRecords(unknown), 'unknown-field.jsonl')).rejects.toThrow(/неверный набор полей/i)

    const unknownType = original.map((record) => structuredClone(record))
    unknownType[2].type = 'future-record'
    await expect(parseReplayBytes(sealRecords(unknownType), 'unknown-type.jsonl')).rejects.toThrow(/decision|event|delta/i)
  })

  it('does not claim simulation verification for a fully resealed but forged final state hash', async () => {
    const records = parseRecords(fixture('raid_win.jsonl'))
    records.at(-1)!.final_state_hash = ZERO
    const source = await parseReplayBytes(sealRecords(records), 'forged.jsonl')
    expect(source.getMetadata().finished).toBe(true)
    expect(source.getMetadata().sourceLabel).toBe('REAL REPLAY')
  })
})
