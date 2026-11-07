import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { canonicalString } from '../data/replayJsonl'
import { hashDomain, utf8 } from '../data/replayCrypto'
import { parseEcoReplayBytes } from './parseEcoReplay'
import type { EcoReplay } from './types'

/** Test helpers: the real exported run (seed 50004, colony 3 raider) and a re-chaining editor for building validly chained but altered files. */
// Vitest runs from the viewer/ directory (npm run test), so paths are taken from there.
const SAMPLE_PATH = resolve(process.cwd(), 'public/samples/eco_seed50004_raider3.jsonl')
const POLICY_SAMPLE_PATH = resolve(process.cwd(), 'public/samples/eco_policy_seed50012.jsonl')
const PYTHON_RUN_PATH = resolve(process.cwd(), '../docs/ecosystem_examples/for46_raid_seed50004.json')
const ZERO = '0'.repeat(64)

let cachedBytes: Uint8Array | null = null
let cachedReplay: EcoReplay | null = null

export function sampleBytes(): Uint8Array {
  if (!cachedBytes) cachedBytes = new Uint8Array(readFileSync(SAMPLE_PATH))
  return cachedBytes
}

export function sampleReplay(): EcoReplay {
  if (!cachedReplay) cachedReplay = parseEcoReplayBytes(sampleBytes(), 'eco_seed50004_raider3.jsonl')
  return cachedReplay
}

let cachedPolicyBytes: Uint8Array | null = null
let cachedPolicyReplay: EcoReplay | null = null

/** The FOR-48 run: 8 colonies, each decided by its own policy through ObservationE1 -> ActionE1 (``python -m formicmind.eco policy-run``). */
export function policySampleBytes(): Uint8Array {
  if (!cachedPolicyBytes) cachedPolicyBytes = new Uint8Array(readFileSync(POLICY_SAMPLE_PATH))
  return cachedPolicyBytes
}

export function policySampleReplay(): EcoReplay {
  if (!cachedPolicyReplay) cachedPolicyReplay = parseEcoReplayBytes(policySampleBytes(), 'eco_policy_seed50012.jsonl')
  return cachedPolicyReplay
}

/** The Python ``run`` report of the same scenario (events_hash, per-kind counts, queen deaths ...). */
export function pythonRun(): { interaction: Record<string, any>; hashes: Record<string, string> } {
  return JSON.parse(readFileSync(PYTHON_RUN_PATH, 'utf-8'))
}

export function textLines(bytes: Uint8Array): string[] {
  return new TextDecoder().decode(bytes).slice(0, -1).split('\n')
}

export function joinLines(lines: string[]): Uint8Array {
  return utf8(`${lines.join('\n')}\n`)
}

/** Apply ``edit`` to every record, then recompute seq/chain so only the semantic check (not the integrity chain) can notice the change. */
export function rechain(lines: string[], edit: (record: Record<string, any>, index: number) => Record<string, any> | null): Uint8Array {
  let previous = ZERO
  const out: string[] = []
  lines.forEach((line, index) => {
    const edited = edit(JSON.parse(line), index)
    if (edited === null) return
    const record = { ...edited, seq: out.length, chain: ZERO }
    const chain = hashDomain('formicmind/eco/replay-record/v1', utf8(`${previous}\n${canonicalString(record)}`))
    out.push(canonicalString({ ...record, chain }))
    previous = chain
  })
  return joinLines(out)
}
