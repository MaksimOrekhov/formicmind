<script setup lang="ts">
import { computed } from 'vue'
import { bucketRange } from './describe'
import { colonyColor, type FogView, type Losses } from './model'
import type { EcoFrame, EcoReplay } from './types'

const props = defineProps<{ replay: EcoReplay; frame: EcoFrame; fog: FogView | null; selected: number; losses: Losses[] }>()
const emit = defineEmits<{ select: [colony: number] }>()

const statusLabel = { alive: 'жива', orphan: 'сирота', extinct: 'вымерла' } as const
const rows = computed(() => props.frame.colonies.map((colony) => {
  const memory = props.frame.memory.find((item) => item.colony === colony.id)
  return { colony, known: memory ? memory.foreign.length : 0 }
}))

function foreignOf(id: number) {
  return props.fog?.foreign.find((record) => record.colony === id)
}
function hidden(id: number) {
  return props.fog !== null && props.fog.colony !== id
}
</script>

<template>
  <div class="eco-colonies">
    <button v-for="row in rows" :key="row.colony.id" type="button" class="eco-colony" :class="[`status-${row.colony.status}`, { selected: selected === row.colony.id }]" :style="{ '--c': colonyColor(row.colony.id) }" :aria-pressed="selected === row.colony.id" @click="emit('select', row.colony.id)">
      <span class="ec-head">
        <span class="ec-swatch">{{ row.colony.id }}</span>
        <span class="ec-title">Колония {{ row.colony.id }}<small v-if="props.replay.roles[String(row.colony.id)]"> · {{ props.replay.roles[String(row.colony.id)] }}</small></span>
        <span v-if="!hidden(row.colony.id)" class="ec-status">{{ statusLabel[row.colony.status] }}</span>
        <span v-else class="ec-status unknown">{{ foreignOf(row.colony.id) ? 'встречена' : 'не известна' }}</span>
      </span>
      <template v-if="!hidden(row.colony.id)">
        <span class="ec-queen"><span>матка</span><b>{{ row.colony.queenHp }}<small>/{{ props.replay.queenMaxHp }}</small></b><i :style="{ width: `${Math.max(0, Math.min(100, row.colony.queenHp / props.replay.queenMaxHp * 100))}%` }"></i></span>
        <span class="ec-grid">
          <span><small>рабочие</small><b>{{ row.colony.workers }}</b></span>
          <span><small>солдаты</small><b>{{ row.colony.soldiers }}</b></span>
          <span><small>расплод</small><b>{{ row.colony.broodWorkers + row.colony.broodSoldiers }}</b></span>
          <span><small>еда</small><b>{{ row.colony.food }}</b></span>
          <span><small>знает колоний</small><b>{{ row.known }}</b></span>
          <span><small>потери в бою</small><b>{{ losses[row.colony.id].workers }}р {{ losses[row.colony.id].soldiers }}с</b></span>
        </span>
      </template>
      <template v-else>
        <span v-if="foreignOf(row.colony.id)" class="ec-known">
          <span>первая встреча: тик {{ foreignOf(row.colony.id)!.firstContact }}</span>
          <span>последний раз видели: тик {{ foreignOf(row.colony.id)!.lastSeen }} ({{ frame.tick - foreignOf(row.colony.id)!.lastSeen }} т. назад)</span>
          <span>силы (корзины): раб. {{ bucketRange(foreignOf(row.colony.id)!.workersBucket) }} · сол. {{ bucketRange(foreignOf(row.colony.id)!.soldiersBucket) }}</span>
          <span>гнездо: {{ foreignOf(row.colony.id)!.nestNode >= 0 ? `узел ${foreignOf(row.colony.id)!.nestNode}` : 'не найдено' }}</span>
          <span>потери: наши {{ foreignOf(row.colony.id)!.killsMine }} · их {{ foreignOf(row.colony.id)!.killsTheirs }}</span>
        </span>
        <span v-else class="ec-known muted">Эту колонию выбранная колония ещё не встречала: её состав, склад и матка скрыты.</span>
      </template>
    </button>
  </div>
</template>
