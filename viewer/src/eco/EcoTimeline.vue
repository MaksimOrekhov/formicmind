<script setup lang="ts">
import { computed } from 'vue'
import { colonyColor, type KeyMoment } from './model'
import type { EcoFrame } from './types'

const props = defineProps<{ tick: number; lastTick: number; playing: boolean; speed: number; speeds: readonly number[]; moments: KeyMoment[]; frame: EcoFrame; frameCount: number }>()
const emit = defineEmits<{ toggle: []; seek: [tick: number]; frame: [direction: -1 | 1]; moment: [direction: -1 | 1]; start: []; end: []; speed: [value: number] }>()
const markers = computed(() => props.moments.map((moment, index) => ({ ...moment, left: props.lastTick ? moment.tick / props.lastTick * 100 : 0, key: index })))
const approx = computed(() => props.tick !== props.frame.tick)
</script>

<template>
  <section class="eco-timeline" aria-label="Управление воспроизведением">
    <div class="eco-readout">
      <span class="lbl">ТИК МИРА</span><strong data-testid="tick">{{ tick }}</strong><span class="lbl">/ {{ lastTick }}</span>
      <span class="sep"></span>
      <span class="lbl">КАДР</span><span data-testid="frame-tick">записан на тике {{ frame.tick }}</span><span v-if="approx" class="approx">· между кадрами позиции вне рёбер не записаны</span>
    </div>
    <div class="eco-scrub">
      <input type="range" min="0" :max="lastTick" step="1" :value="tick" aria-label="Тик мира" @input="emit('seek', Number(($event.target as HTMLInputElement).value))">
      <div class="eco-moments"><button v-for="m in markers" :key="m.key" type="button" tabindex="-1" :class="`kind-${m.kind}`" :style="{ left: `${m.left}%`, background: colonyColor(m.colony) }" :title="`${m.label} · колония ${m.colony} · тик ${m.tick}`" :aria-label="`${m.label}, тик ${m.tick}`" @click="emit('seek', Math.min(lastTick, m.tick + 1))"></button></div>
    </div>
    <div class="eco-controls">
      <button type="button" aria-label="В начало" title="В начало" @click="emit('start')">⏮</button>
      <button type="button" aria-label="Предыдущее ключевое событие" title="Предыдущее ключевое событие" @click="emit('moment', -1)">⇤</button>
      <button type="button" aria-label="Кадр назад" title="Кадр назад" @click="emit('frame', -1)">◂</button>
      <button type="button" class="play" :aria-label="playing ? 'Пауза' : 'Воспроизвести'" :disabled="!playing && tick >= lastTick" @click="emit('toggle')">{{ playing ? '⏸' : '▶' }}</button>
      <button type="button" aria-label="Кадр вперёд" title="Кадр вперёд" @click="emit('frame', 1)">▸</button>
      <button type="button" aria-label="Следующее ключевое событие" title="Следующее ключевое событие" @click="emit('moment', 1)">⇥</button>
      <button type="button" aria-label="В конец" title="В конец" @click="emit('end')">⏭</button>
      <span class="speed"><small>ТИКОВ/С</small><button v-for="value in speeds" :key="value" type="button" :class="{ active: speed === value }" :aria-pressed="speed === value" @click="emit('speed', value)">{{ value }}</button></span>
      <span class="frames">{{ frameCount }} кадров</span>
    </div>
  </section>
</template>
