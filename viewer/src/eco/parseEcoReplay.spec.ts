import { describe, expect, it } from 'vitest'
import { EcoReplayError, EcoReplayVersionError, parseEcoReplayBytes } from './parseEcoReplay'
import { joinLines, policySampleBytes, policySampleReplay, pythonRun, rechain, sampleBytes, sampleReplay, textLines } from './testing'
import { utf8 } from '../data/replayCrypto'

describe('E1 replay parser on the real seed-50004 run', () => {
  const replay = sampleReplay()

  it('reads the run that Python exported and re-verified', () => {
    expect(replay.seed).toBe('50004')
    expect(replay.colonyCount).toBe(8)
    expect(replay.nodes).toHaveLength(96)
    expect(replay.edges).toHaveLength(166)
    expect(replay.nests).toHaveLength(8)
    expect(replay.roles).toEqual({ '3': 'raider' })
    expect(replay.frames).toHaveLength(303)
    expect(replay.events).toHaveLength(629)
    expect(replay.ticksCompleted).toBe(6000)
    expect(replay.worldOutcome).toEqual({ finished: true, reason: 'horizon', terminalTick: 5999 })
    expect(replay.verified).toEqual({ chain: true, configHash: true, mapHash: true, eventsHash: true })
  })

  it('agrees with the Python run report: same events hash, same counts, same deaths', () => {
    const python = pythonRun().interaction
    expect(replay.eventsHash).toBe(python.events_hash)
    const counts: Record<string, number> = {}
    for (const event of replay.events) counts[event.kind] = (counts[event.kind] ?? 0) + 1
    expect(counts).toEqual(Object.fromEntries(Object.entries(python.events_by_kind).filter(([, n]) => (n as number) > 0)))
    const queen = replay.events.filter((e) => e.kind === 'queen_dead').map((e) => [e.colony, e.tick, e.other])
    expect(queen).toEqual(python.queen_deaths.map((q: any) => [q.colony, q.tick, q.killer]))
    const extinct = replay.events.filter((e) => e.kind === 'extinct').map((e) => [e.colony, e.tick])
    expect(extinct).toEqual(python.extinctions.map((x: any) => [x.colony, x.tick]))
    const attacks = replay.events.filter((e) => e.kind === 'dispatch' && e.a === 3).map((e) => [e.tick, e.colony, e.other, e.b])
    expect(attacks).toEqual(python.attacks_dispatched.map((a: any) => [a.tick, a.attacker, a.target, a.soldiers]))
  })

  it('shows the story of the acceptance scenario: discovery, raids, queen death, extinction, seven survivors', () => {
    const first = (kind: string, colony?: number) => replay.events.find((e) => e.kind === kind && (colony === undefined || e.colony === colony))!
    expect(first('nest_found', 3)).toMatchObject({ tick: 886, other: 0 })
    const attacks = replay.events.filter((e) => e.kind === 'dispatch' && e.a === 3 && e.other === 0)
    expect(attacks.slice(0, 2).map((e) => e.tick)).toEqual([2280, 2520])
    expect(first('queen_dead')).toMatchObject({ tick: 2621, colony: 0, other: 3 })
    expect(first('extinct')).toMatchObject({ tick: 4400, colony: 0 })
    const last = replay.frames[replay.frames.length - 1]
    expect(last.colonies.filter((c) => c.status === 'alive')).toHaveLength(7)
    expect(last.colonies[0].status).toBe('extinct')
    expect(replay.frames.find((f) => f.tick === 2622)?.colonies[0].status).toBe('orphan')
    expect(replay.frames.map((f) => f.reason).filter((r) => r === 'key_event')).toHaveLength(2)
  })

  it('keeps the recorded food and cohorts physically consistent frame by frame', () => {
    const nests = new Set(replay.nests)
    for (const frame of replay.frames) {
      expect(frame.patches.every((stock, i) => stock >= 0 && stock <= replay.nodes[replay.patchNodes[i]].capacity)).toBe(true)
      for (const cohort of frame.cohorts) {
        if (!cohort.atNode) {
          expect(cohort.depart).toBeLessThanOrEqual(frame.tick)
          expect(cohort.arrive).toBeGreaterThanOrEqual(frame.tick)
          const edge = replay.edges.find((e) => (e.a === cohort.from && e.b === cohort.to) || (e.b === cohort.from && e.a === cohort.to))
          expect(edge, 'cohorts walk real edges').toBeDefined()
          expect(cohort.arrive - cohort.depart).toBeGreaterThanOrEqual(edge!.length)
        }
        expect(cohort.load).toBeLessThanOrEqual(cohort.count * 16)
        if (cohort.caste === 'soldier') expect(cohort.load).toBe(0)
      }
      for (const colony of frame.colonies) if (colony.status === 'extinct') expect(colony.workers + colony.soldiers).toBe(0)
      expect(nests.size).toBe(8)
    }
  })
})

describe('E1 replay import is fail-closed', () => {
  const lines = textLines(sampleBytes())
  const message = (bytes: Uint8Array) => {
    try { parseEcoReplayBytes(bytes, 'x.jsonl') } catch (error) { return error as Error }
    throw new Error('the file was accepted')
  }

  it('rejects empty, truncated, BOM, CRLF and non-ASCII files with clear messages', () => {
    expect(message(new Uint8Array()).message).toMatch(/пуст/)
    expect(message(sampleBytes().slice(0, sampleBytes().length - 1)).message).toMatch(/обрезан/)
    expect(message(utf8(`﻿${lines.join('\n')}\n`)).message).toMatch(/BOM|ASCII/)
    expect(message(utf8(`${lines.join('\r\n')}\r\n`)).message).toMatch(/LF/)
    expect(message(utf8('{"a":"é"}\n{}\n{}\n')).message).toMatch(/ASCII|canonical/)
    expect(message(utf8('not json\nnot json\nnot json\n')).message).toMatch(/не JSON/)
  })

  it('rejects a line that is valid JSON but not canonical JSON', () => {
    const edited = [...lines]
    edited[5] = JSON.stringify(JSON.parse(lines[5]), null, 1).replace(/\n/g, ' ')
    expect(message(joinLines(edited))).toBeInstanceOf(EcoReplayError)
    expect(message(joinLines(edited)).message).toMatch(/canonical/)
  })

  it('rejects any edit to a recorded value: the integrity chain breaks', () => {
    const frameLine = lines.findIndex((line) => line.includes('"type":"frame"') && !line.includes('"reason":"start"'))
    const record = JSON.parse(lines[frameLine])
    record.colonies[0].food += 1
    const edited = [...lines]
    edited[frameLine] = JSON.stringify(record, Object.keys(record).sort())
    const text = message(joinLines(edited)).message
    expect(text).toMatch(/цепочк|canonical/)
    // a truncated file with its result removed is no better
    expect(message(joinLines(lines.slice(0, -1))).message).toMatch(/цепочк|result|Последняя/)
    // reordering two records
    const swapped = [...lines]
    ;[swapped[10], swapped[11]] = [swapped[11], swapped[10]]
    expect(message(joinLines(swapped)).message).toMatch(/номер записи|цепочк/)
  })

  it('rejects an unknown schema version with a version error', () => {
    const bytes = rechain(lines, (record) => (record.type === 'header' ? { ...record, schema: 'formicmind.eco.replay/2' } : record))
    const error = message(bytes)
    expect(error).toBeInstanceOf(EcoReplayVersionError)
    expect(error.message).toMatch(/Неподдерживаемая версия/)
  })

  it('rejects a validly chained file whose hashes disagree with its content', () => {
    const configEdited = rechain(lines, (record) => (record.type === 'header' ? { ...record, config: { ...record.config, horizon: record.config.horizon - 1 } } : record))
    expect(message(configEdited).message).toMatch(/config_hash/)
    const mapEdited = rechain(lines, (record) => {
      if (record.type !== 'header') return record
      const nodes = record.map.nodes.map((node: any, i: number) => (i === 3 ? { ...node, x: node.x + 1 } : node))
      return { ...record, map: { ...record.map, nodes } }
    })
    expect(message(mapEdited).message).toMatch(/map_hash/)
    let edited = 0
    const eventEdited = rechain(lines, (record) => {
      if (record.type === 'event' && record.kind === 'death' && edited === 0) { edited = 1; return { ...record, b: record.b + 1 } }
      return record
    })
    expect(message(eventEdited).message).toMatch(/events_hash/)
    const dropped = rechain(lines, (record) => (record.type === 'event' && record.tick === 2621 && record.kind === 'queen_dead' ? null : record))
    expect(message(dropped).message).toMatch(/events_hash|result/)
  })

  it('rejects structurally broken records even when the chain is valid', () => {
    const badCohort = rechain(lines, (record) => (record.type === 'frame' && record.tick === 20 ? { ...record, cohorts: [[1, 99, 0, 1, 20, 0, 0, 0, 0, 0, -1, 0, 0, -1, -1], ...record.cohorts] } : record))
    expect(message(badCohort).message).toMatch(/когорт/)
    const missingKey = rechain(lines, (record) => { if (record.type === 'frame' && record.tick === 20) { const { memory, ...rest } = record; return rest } return record })
    expect(message(missingKey).message).toMatch(/набор полей/)
    const backwards = rechain(lines, (record) => (record.type === 'frame' && record.tick === 40 ? { ...record, tick: 10 } : record))
    expect(message(backwards).message).toMatch(/кадр|заголовок/)
    let done = false
    const outOfRange = rechain(lines, (record) => { if (!done && record.type === 'event' && record.kind === 'discover') { done = true; return { ...record, colony: 12 } } return record })
    expect(message(outOfRange).message).toMatch(/вне диапазона/)
  })
})


describe('E1 replay of independent policies (FOR-48)', () => {
  const replay = policySampleReplay()

  it('reads the policy specification and binds the decision records', () => {
    expect(replay.fixture).toBe('policy')
    expect(replay.colonyCount).toBe(8)
    expect(replay.policy).not.toBeNull()
    expect(replay.policy!.seed).toBe('5')
    expect(replay.policy!.policies.map((p) => p.kind + (p.variant ? `:${p.variant}` : ''))).toEqual([
      'heuristic:explorer', 'heuristic:aggressor', 'heuristic:explorer', 'heuristic:aggressor', 'heuristic:peaceful', 'heuristic:aggressor', 'random_valid', 'heuristic:raider'])
    expect(replay.policy!.decisionCount).toBeGreaterThan(100)
    expect(replay.policy!.decisionsHash).toMatch(/^[0-9a-f]{64}$/)
    expect(replay.roles['6']).toBe('random_valid')
    expect(replay.verified).toEqual({ chain: true, configHash: true, mapHash: true, eventsHash: true })
  })

  it('shows real consequences of the actions: scouting orders, contacts, nests found, and a fight started through ActionE1', () => {
    const kinds = new Set(replay.events.map((e) => e.kind))
    for (const kind of ['discover', 'contact', 'nest_found', 'dispatch', 'combat']) expect(kinds.has(kind as any)).toBe(true)
    expect(replay.events.some((e) => e.kind === 'dispatch' && e.a === 3)).toBe(true) // an attack order exists
    expect(replay.events.some((e) => e.kind === 'dispatch' && e.a === 1)).toBe(true) // scouting orders exist
  })

  it('has no curriculum on a full-action-space run, and reads and validates the FOR-50 profile', () => {
    expect(replay.policy!.curriculum).toBe('none')
    const lines = textLines(policySampleBytes())
    const withProfile = (curriculum: unknown) => rechain(lines, (record) => (record.type === 'header' ? { ...record, policy: { ...record.policy, curriculum } } : record))
    expect(parseEcoReplayBytes(withProfile('nest-v1')).policy!.curriculum).toBe('nest-v1')
    expect(parseEcoReplayBytes(withProfile('scout-v1')).policy!.curriculum).toBe('scout-v1')
    for (const bad of ['none', 'nest', 1]) expect(() => parseEcoReplayBytes(withProfile(bad))).toThrow(EcoReplayError)
  })

  it('runs of the fixtures have no policy block', () => {
    expect(sampleReplay().policy).toBeNull()
  })

  it('rejects a policy replay whose policy header or decision binding is altered, even with a valid chain', () => {
    const lines = textLines(policySampleBytes())
    const forgedKind = rechain(lines, (record) => (record.type === 'header' ? { ...record, policy: { ...record.policy, policies: record.policy.policies.map((p: any, i: number) => (i === 0 ? { ...p, kind: 'oracle' } : p)) } } : record))
    expect(() => parseEcoReplayBytes(forgedKind)).toThrow(EcoReplayError)
    const forgedSchema = rechain(lines, (record) => (record.type === 'header' ? { ...record, policy: { ...record.policy, schema: 'formicmind.eco.policy-run/2' } } : record))
    expect(() => parseEcoReplayBytes(forgedSchema)).toThrow(EcoReplayError)
    const noHash = rechain(lines, (record) => { if (record.type !== 'result') return record; const { decisions_hash: _drop, ...rest } = record; return rest })
    expect(() => parseEcoReplayBytes(noHash)).toThrow(EcoReplayError)
    const badFixture = rechain(lines, (record) => (record.type === 'header' ? { ...record, fixture: 'peaceful_fair' } : record))
    expect(() => parseEcoReplayBytes(badFixture)).toThrow(EcoReplayError)
  })
})
