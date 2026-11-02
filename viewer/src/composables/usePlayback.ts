import { computed, onScopeDispose, ref, watch } from 'vue'
import type { ReplayDataSource, TimelineFrame } from '../domain/replay'

const SPEEDS = [1, 5, 20] as const

export function usePlayback(source: ReplayDataSource) {
  const metadata = source.getMetadata()
  const cursor = ref(metadata.minCursor)
  const frame = ref<TimelineFrame>(source.getFrame(metadata.minCursor))
  const isPlaying = ref(false)
  const speed = ref<number>(1)
  const isTerminal = computed(() => metadata.terminal?.cursor === cursor.value)

  let timer: ReturnType<typeof setInterval> | undefined

  function stopTimer() {
    if (timer !== undefined) {
      clearInterval(timer)
      timer = undefined
    }
  }

  function pause() {
    isPlaying.value = false
  }

  function seek(nextCursor: number) {
    pause()
    const next = Math.min(metadata.maxCursor, Math.max(metadata.minCursor, Math.round(nextCursor)))
    cursor.value = next
    frame.value = source.getFrame(next)
    return next
  }

  function step(amount: -1 | 1) {
    return seek(cursor.value + amount)
  }

  function advance() {
    if (cursor.value >= metadata.maxCursor) {
      pause()
      return
    }
    cursor.value += 1
    frame.value = source.getFrame(cursor.value)
    if (cursor.value >= metadata.maxCursor || isTerminal.value) pause()
  }

  function play() {
    if (cursor.value >= metadata.maxCursor || isTerminal.value) return
    isPlaying.value = true
  }

  function setSpeed(nextSpeed: number) {
    if (SPEEDS.includes(nextSpeed as (typeof SPEEDS)[number])) speed.value = nextSpeed
  }

  watch([isPlaying, speed], ([playing, currentSpeed]) => {
    stopTimer()
    if (playing) timer = setInterval(advance, 720 / currentSpeed)
  }, { immediate: true, flush: 'sync' })

  onScopeDispose(stopTimer)

  return {
    metadata,
    cursor,
    frame,
    isPlaying,
    isTerminal,
    speed,
    speeds: SPEEDS,
    pause,
    play,
    seek,
    step,
    setSpeed,
    goToStart: () => seek(metadata.minCursor),
    goToEnd: () => seek(metadata.maxCursor),
    seekTick: (tick: number) => {
      const target = source.getCursorForTick(tick)
      return target === undefined ? cursor.value : seek(target)
    },
  }
}
