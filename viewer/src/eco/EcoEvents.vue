<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { describeEvent, KIND_LABEL } from './describe'
import { colonyColor, eventCountThrough, eventVisibleToColony } from './model'
import type { EcoEvent, EcoReplay, EventKind } from './types'
import { EVENT_KINDS } from './types'

const props = defineProps<{ replay: EcoReplay; tick: number; viewColony: number | null; activeIndex: number | null }>()
const emit = defineEmits<{ select: [event: EcoEvent] }>()
const PAGE = 80
const enabled = ref<Set<EventKind>>(new Set(EVENT_KINDS.filter((kind) => kind !== 'combat' && kind !== 'queen_damage')))
const showFuture = ref(false)
const colonyFilter = ref<number | 'all'>('all')
const pages = ref(1)
watch([enabled, showFuture, colonyFilter, () => props.viewColony], () => { pages.value = 1 }, { deep: true })

const base = computed(() => props.replay.events.filter((event) => props.viewColony === null || eventVisibleToColony(event, props.viewColony)))
const counts = computed(() => {
  const out: Record<string, number> = {}
  for (const event of base.value) out[event.kind] = (out[event.kind] ?? 0) + 1
  return out
})
const filtered = computed(() => {
  const limit = showFuture.value ? props.replay.events.length : eventCountThrough(props.replay.events, props.tick)
  const out: EcoEvent[] = []
  for (let i = 0; i < limit; i += 1) {
    const event = props.replay.events[i]
    if (!enabled.value.has(event.kind)) continue
    if (props.viewColony !== null && !eventVisibleToColony(event, props.viewColony)) continue
    if (colonyFilter.value !== 'all' && event.colony !== colonyFilter.value && event.other !== colonyFilter.value) continue
    out.push(event)
  }
  return out.reverse()
})
const visible = computed(() => filtered.value.slice(0, pages.value * PAGE))

function toggle(kind: EventKind) {
  const next = new Set(enabled.value)
  if (next.has(kind)) next.delete(kind)
  else next.add(kind)
  enabled.value = next
}
</script>

<template>
  <section class="eco-panel eco-events" aria-labelledby="eco-events-h">
    <div class="eco-panel-head"><div><span class="eyebrow">ХРОНИКА</span><h2 id="eco-events-h">События <span class="count">{{ filtered.length.toLocaleString('ru-RU') }}</span></h2></div></div>
    <div class="eco-chips" role="group" aria-label="Фильтр по типу события">
      <button v-for="kind in EVENT_KINDS" :key="kind" type="button" class="chip" :class="[`kind-${kind}`, { on: enabled.has(kind) }]" :aria-pressed="enabled.has(kind)" @click="toggle(kind)">{{ KIND_LABEL[kind] }} <small>{{ counts[kind] ?? 0 }}</small></button>
    </div>
    <div class="eco-filter-row">
      <label>колония <select v-model="colonyFilter"><option value="all">любая</option><option v-for="id in replay.colonyCount" :key="id" :value="id - 1">{{ id - 1 }}</option></select></label>
      <label class="check"><input v-model="showFuture" type="checkbox"> показать будущие</label>
    </div>
    <div class="eco-event-list" aria-live="polite">
      <button v-for="event in visible" :key="event.index" type="button" class="eco-event" :class="[`kind-${event.kind}`, { active: activeIndex === event.index, future: event.tick > tick }]" @click="emit('select', event)">
        <span class="ev-tick">тик {{ event.tick }}</span>
        <span class="ev-dot" :style="{ background: colonyColor(event.colony) }"></span>
        <span class="ev-text"><b>{{ KIND_LABEL[event.kind] }}</b> {{ describeEvent(event) }}</span>
      </button>
      <div v-if="visible.length === 0" class="eco-empty">Нет событий для текущего тика и фильтров.</div>
    </div>
    <div v-if="visible.length < filtered.length" class="eco-more"><span>{{ visible.length }} из {{ filtered.length }}</span><button type="button" @click="pages += 1">Показать ещё</button></div>
  </section>
</template>
