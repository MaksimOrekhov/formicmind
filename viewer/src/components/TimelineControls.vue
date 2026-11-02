<script setup lang="ts">
import type { ReplayMetadata, TimelineFrame } from '../domain/replay'

defineProps<{
  metadata: ReplayMetadata
  frame: TimelineFrame
  cursor: number
  playing: boolean
  speed: number
  speeds: readonly number[]
}>()

const emit = defineEmits<{
  toggle: []
  seek: [cursor: number]
  step: [amount: -1 | 1]
  start: []
  end: []
  speed: [value: number]
}>()

function tickLabel(tick: number) {
  return `T${String(tick).padStart(2, '0')}`
}
</script>

<template>
  <section class="timeline-panel" aria-label="Управление воспроизведением">
    <div class="timeline-topline">
      <div class="timeline-title"><span class="timeline-icon">◷</span><span><span class="eyebrow">ЛЕНТА ВОСПРОИЗВЕДЕНИЯ</span><strong>{{ metadata.title }}</strong></span></div>
      <div class="timeline-readout"><span class="readout-current">{{ tickLabel(cursor) }}</span><span class="readout-slash">/</span><span>{{ metadata.terminal ? tickLabel(metadata.terminal.terminalTick) : tickLabel(metadata.maxCursor) }}</span><span class="readout-divider"></span><span class="state-tick">СОСТОЯНИЕ {{ tickLabel(frame.state.tick) }}</span></div>
    </div>
    <div class="timeline-scrub-row">
      <span class="scrub-edge">{{ tickLabel(metadata.minCursor) }}</span>
      <input class="scrubber" type="range" :min="metadata.minCursor" :max="metadata.maxCursor" step="1" :value="cursor" :aria-label="`Тик реплея: ${cursor}`" @input="emit('seek', Number(($event.target as HTMLInputElement).value))" />
      <span class="scrub-edge">{{ tickLabel(metadata.maxCursor) }}</span>
    </div>
    <div class="timeline-controls-row">
      <div class="transport-controls">
        <button class="transport-button subtle" aria-label="В начало реплея" title="В начало" @click="emit('start')"><svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 4v12M15 5 8 10l7 5z" /></svg></button>
        <button class="transport-button subtle" aria-label="На один тик назад" title="Назад на тик" @click="emit('step', -1)"><svg viewBox="0 0 20 20" aria-hidden="true"><path d="M8 5 2 10l6 5zM17 5l-6 5 6 5z" /></svg></button>
        <button class="play-button" :aria-label="playing ? 'Пауза' : 'Воспроизвести'" :title="playing ? 'Пауза' : 'Воспроизвести'" :disabled="!playing && cursor === metadata.maxCursor" @click="emit('toggle')">
          <svg v-if="playing" viewBox="0 0 20 20" aria-hidden="true"><path d="M6 4h3v12H6zM12 4h3v12h-3z" /></svg>
          <svg v-else viewBox="0 0 20 20" aria-hidden="true"><path d="m6 4 10 6-10 6z" /></svg>
        </button>
        <button class="transport-button subtle" aria-label="На один тик вперёд" title="Вперёд на тик" @click="emit('step', 1)"><svg viewBox="0 0 20 20" aria-hidden="true"><path d="m3 5 6 5-6 5zM12 5l6 5-6 5z" /></svg></button>
        <button class="transport-button subtle" aria-label="В конец реплея" title="В конец" @click="emit('end')"><svg viewBox="0 0 20 20" aria-hidden="true"><path d="M15 4v12M5 5l7 5-7 5z" /></svg></button>
      </div>
      <div class="speed-control" aria-label="Скорость воспроизведения">
        <span>СКОРОСТЬ</span>
        <button v-for="value in speeds" :key="value" :class="{ active: speed === value }" :aria-pressed="speed === value" @click="emit('speed', value)">×{{ value }}</button>
      </div>
      <div class="terminal-status" :class="{ terminal: metadata.terminal && cursor === metadata.terminal.cursor }">
        <i></i><span v-if="metadata.terminal && cursor === metadata.terminal.cursor">ТЕРМИНАЛЬНЫЙ КАДР · {{ metadata.terminal.stage }} · ТИК НЕ ЗАВЕРШЁН</span><span v-else>{{ playing ? 'ВОСПРОИЗВЕДЕНИЕ' : 'ПАУЗА' }}</span>
      </div>
    </div>
  </section>
</template>
