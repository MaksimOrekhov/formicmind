import { computed, onScopeDispose, ref, watch } from 'vue'
import { frameIndexAtTick, keyMoments } from './model'
import type { EcoReplay } from './types'

/** Playback speeds in world ticks per second. The viewer shows ticks of the simulated world; it never pretends to be real time. */
export const ECO_SPEEDS = [10, 40, 120, 400] as const
const STEP_MS = 33

export function useEcoPlayback(replay: EcoReplay) {
  const lastTick = replay.ticksCompleted
  const tick = ref(0)
  const isPlaying = ref(false)
  const speed = ref<number>(40)
  const moments = keyMoments(replay)
  const frameIndex = computed(() => frameIndexAtTick(replay, tick.value))
  const frame = computed(() => replay.frames[frameIndex.value])
  let carry = 0
  let timer: ReturnType<typeof setInterval> | undefined

  function stop() {
    if (timer !== undefined) {
      clearInterval(timer)
      timer = undefined
    }
  }

  function pause() { isPlaying.value = false }

  function seek(next: number) {
    pause()
    carry = 0
    tick.value = Math.min(lastTick, Math.max(0, Math.round(next)))
    return tick.value
  }

  function advance() {
    carry += speed.value * STEP_MS / 1000
    const whole = Math.floor(carry)
    if (whole < 1) return
    carry -= whole
    tick.value = Math.min(lastTick, tick.value + whole)
    if (tick.value >= lastTick) pause()
  }

  function play() {
    if (tick.value >= lastTick) return
    isPlaying.value = true
  }

  function setSpeed(next: number) {
    if ((ECO_SPEEDS as readonly number[]).includes(next)) speed.value = next
  }

  /** Move to the previous/next recorded frame. */
  function stepFrame(direction: -1 | 1) {
    const index = frameIndex.value
    const target = direction === 1
      ? replay.frames[Math.min(replay.frames.length - 1, index + 1)].tick
      : (replay.frames[index].tick < tick.value ? replay.frames[index].tick : replay.frames[Math.max(0, index - 1)].tick)
    return seek(target)
  }

  /** Jump to the previous/next key moment (first contact, nest found, attack, combat, queen death, extinction). */
  function jumpMoment(direction: -1 | 1) {
    const at = (m: { tick: number }) => Math.min(lastTick, m.tick + 1) // the state after the tick in which the event happened
    const candidates = direction === 1 ? moments.filter((m) => at(m) > tick.value) : moments.filter((m) => at(m) < tick.value).reverse()
    return candidates.length ? seek(at(candidates[0])) : tick.value
  }

  watch([isPlaying, speed], ([playing]) => {
    stop()
    if (playing) timer = setInterval(advance, STEP_MS)
  }, { immediate: true, flush: 'sync' })
  onScopeDispose(stop)

  return {
    replay, tick, lastTick, isPlaying, speed, speeds: ECO_SPEEDS, frame, frameIndex, moments,
    play, pause, seek, setSpeed, stepFrame, jumpMoment,
    goToStart: () => seek(0),
    goToEnd: () => seek(lastTick),
  }
}
