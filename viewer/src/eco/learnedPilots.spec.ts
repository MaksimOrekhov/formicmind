import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { parseEcoReplayBytes } from './parseEcoReplay'

// The replays of the committed FOR-49 pilots (reward v2 and v3) are exported by the Python side and must open in the viewer through the same strict,
// hash-checked parser as every other E1 replay (they are not shipped as sample buttons; the viewer opens them from a file).
const PILOTS = [
  ['reward v2 (stopped at iteration 89)', '../docs/ecosystem_examples/for49_pilot_reward_v2/replay_seed90000.jsonl'],
  ['reward v3 (stopped at iteration 82)', '../docs/ecosystem_examples/for49_reward_v3/replay_seed90000.jsonl'],
] as const

describe('committed FOR-49 pilot replays open in the E1 viewer', () => {
  for (const [label, path] of PILOTS) {
    it(`parses the ${label} replay`, () => {
      const bytes = new Uint8Array(readFileSync(resolve(process.cwd(), path)))
      const replay = parseEcoReplayBytes(bytes, path.split('/').pop())
      expect(replay).toBeTruthy()
      expect(JSON.stringify(replay).length).toBeGreaterThan(10_000)
    })
  }
})
