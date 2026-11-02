<script setup lang="ts">
import type { ColonyView, Side } from '../domain/replay'

defineProps<{
  colony: ColonyView
  previous?: ColonyView
  selected: boolean
}>()

const emit = defineEmits<{ select: [side: Side] }>()

function delta(current: number, previous?: number) {
  if (previous === undefined || previous === current) return '—'
  return `${current - previous > 0 ? '+' : ''}${current - previous}`
}
</script>

<template>
  <button class="colony-card" :class="[`colony-${colony.side.toLowerCase()}`, { selected }]" :aria-pressed="selected" @click="emit('select', colony.side)">
    <span class="colony-head">
      <span class="side-badge">{{ colony.side }}</span>
      <span class="colony-heading">
        <span class="eyebrow">КОЛОНИЯ {{ colony.side }}</span>
        <span class="colony-home">Матка · {{ colony.queenAlive ? 'в строю' : 'погибла' }}</span>
      </span>
      <span class="selected-mark" aria-hidden="true">↗</span>
    </span>
    <span class="queen-stat">
      <span><span class="stat-label">HP матки</span><span class="delta" :class="{ down: previous && previous.queenHp > colony.queenHp, up: previous && previous.queenHp < colony.queenHp }">{{ delta(colony.queenHp, previous?.queenHp) }}</span></span>
      <strong>{{ colony.queenHp }}<small> / {{ colony.queenMaxHp }}</small></strong>
      <span class="health-track"><i :style="{ width: `${Math.max(0, Math.min(100, colony.queenHp / colony.queenMaxHp * 100))}%` }"></i></span>
    </span>
    <span class="colony-metrics">
      <span class="metric"><span>ЕДА</span><strong>{{ colony.food }}<small> ед.</small></strong><em :class="{ down: previous && previous.food > colony.food, up: previous && previous.food < colony.food }">{{ delta(colony.food, previous?.food) }}</em></span>
      <span class="metric"><span>РАБОЧИЕ</span><strong>{{ colony.workers }}</strong><em :class="{ down: previous && previous.workers > colony.workers, up: previous && previous.workers < colony.workers }">{{ delta(colony.workers, previous?.workers) }}</em></span>
      <span class="metric"><span>СОЛДАТЫ</span><strong>{{ colony.soldiers }}</strong><em :class="{ down: previous && previous.soldiers > colony.soldiers, up: previous && previous.soldiers < colony.soldiers }">{{ delta(colony.soldiers, previous?.soldiers) }}</em></span>
      <span class="metric"><span>РАСПЛОД</span><strong>{{ colony.brood }}</strong><em :class="{ down: previous && previous.brood > colony.brood, up: previous && previous.brood < colony.brood }">{{ delta(colony.brood, previous?.brood) }}</em></span>
    </span>
  </button>
</template>
