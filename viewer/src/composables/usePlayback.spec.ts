import { effectScope, nextTick } from 'vue'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { mockReplay } from '../data/mockReplay'
import { usePlayback } from './usePlayback'

describe('usePlayback', () => {
  afterEach(() => vi.useRealTimers())

  it('clamps seeks, steps across existing frames and keeps terminal state separate', () => {
    const scope = effectScope()
    const playback = scope.run(() => usePlayback(mockReplay))!

    expect(playback.cursor.value).toBe(0)
    expect(playback.seek(-10)).toBe(0)
    playback.step(1)
    expect(playback.frame.value.cursor).toBe(1)
    expect(playback.seek(99)).toBe(8)
    expect(playback.isTerminal.value).toBe(true)
    expect(playback.frame.value.state).toEqual({ tick: 7, tickCompleted: false })
    expect(playback.seekTick(7)).toBe(7)
    expect(playback.frame.value.state.tickCompleted).toBe(true)
    scope.stop()
  })

  it('pauses automatically at the recorded terminal cursor and disposes its timer', async () => {
    vi.useFakeTimers()
    const scope = effectScope()
    const playback = scope.run(() => usePlayback(mockReplay))!
    playback.setSpeed(20)
    playback.play()
    await nextTick()
    vi.advanceTimersByTime(720)
    expect(playback.cursor.value).toBe(8)
    expect(playback.isPlaying.value).toBe(false)
    expect(playback.isTerminal.value).toBe(true)
    expect(vi.getTimerCount()).toBe(0)
    scope.stop()
    expect(vi.getTimerCount()).toBe(0)
  })

  it('pauses on manual seek and supports all advertised speeds', () => {
    const scope = effectScope()
    const playback = scope.run(() => usePlayback(mockReplay))!
    playback.play()
    playback.seek(3)
    expect(playback.isPlaying.value).toBe(false)
    playback.setSpeed(5)
    expect(playback.speed.value).toBe(5)
    playback.setSpeed(8)
    expect(playback.speed.value).toBe(5)
    scope.stop()
  })
})
