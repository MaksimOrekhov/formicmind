import { effectScope, nextTick } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { sampleReplay } from './testing'
import { useEcoPlayback } from './useEcoPlayback'

const replay = sampleReplay()

function make() {
  const scope = effectScope()
  const playback = scope.run(() => useEcoPlayback(replay))!
  return { scope, playback }
}

describe('E1 playback', () => {
  beforeEach(() => { vi.useFakeTimers() })
  afterEach(() => { vi.useRealTimers() })

  it('starts at tick 0 on the start frame and shows world ticks, not wall-clock time', () => {
    const { scope, playback } = make()
    expect(playback.tick.value).toBe(0)
    expect(playback.frame.value.reason).toBe('start')
    expect(playback.lastTick).toBe(6000)
    scope.stop()
  })

  it('plays at the chosen number of ticks per second and pauses', async () => {
    const { scope, playback } = make()
    playback.setSpeed(120)
    playback.play()
    await nextTick()
    vi.advanceTimersByTime(1000)
    expect(playback.tick.value).toBeGreaterThanOrEqual(110)
    expect(playback.tick.value).toBeLessThanOrEqual(125)
    playback.pause()
    const stopped = playback.tick.value
    vi.advanceTimersByTime(2000)
    expect(playback.tick.value).toBe(stopped)
    playback.setSpeed(7) // not an offered speed: ignored
    expect(playback.speed.value).toBe(120)
    scope.stop()
  })

  it('seeks within bounds, steps by recorded frames and stops at the end', async () => {
    const { scope, playback } = make()
    expect(playback.seek(-50)).toBe(0)
    expect(playback.seek(99999)).toBe(6000)
    playback.seek(2621)
    expect(playback.frame.value.tick).toBe(2620)
    expect(playback.stepFrame(1)).toBe(2622)
    expect(playback.stepFrame(-1)).toBe(2620)
    expect(playback.stepFrame(-1)).toBe(2600)
    playback.seek(5990)
    playback.setSpeed(400)
    playback.play()
    await nextTick()
    vi.advanceTimersByTime(500)
    expect(playback.tick.value).toBe(6000)
    expect(playback.isPlaying.value).toBe(false)
    scope.stop()
  })

  it('jumps between key moments to the state after the tick in which they happened', () => {
    const { scope, playback } = make()
    expect(playback.jumpMoment(1)).toBeGreaterThan(0)
    const seen: number[] = [playback.tick.value]
    for (let i = 0; i < 40; i += 1) {
      const next = playback.jumpMoment(1)
      if (next === seen[seen.length - 1]) break
      seen.push(next)
    }
    expect(seen).toContain(887)   // nest found at 886
    expect(seen).toContain(2281)  // first attack order at 2280
    expect(seen).toContain(2622)  // queen died at 2621
    expect(seen).toContain(4401)  // extinct at 4400
    expect([...seen].sort((a, b) => a - b)).toEqual(seen)
    playback.seek(4401)
    expect(playback.jumpMoment(-1)).toBe(2622)
    scope.stop()
  })

  it('survives a rapid seek storm across the whole 6000-tick file', () => {
    const { scope, playback } = make()
    for (let i = 0; i < 3000; i += 1) playback.seek((i * 7919) % 6001)
    expect(playback.frame.value.tick).toBeLessThanOrEqual(playback.tick.value)
    scope.stop()
  })
})
