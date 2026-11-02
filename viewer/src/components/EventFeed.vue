<script setup lang="ts">
import { computed, ref } from 'vue'
import type { EventCategory, Side, ViewerEvent } from '../domain/replay'

const props = defineProps<{ events: ViewerEvent[]; currentCursor: number }>()
const emit = defineEmits<{ selectEvent: [event: ViewerEvent] }>()

const categories: { value: EventCategory | 'all'; label: string }[] = [
  { value: 'all', label: 'Все' },
  { value: 'combat', label: 'Бой' },
  { value: 'economy', label: 'Экономика' },
  { value: 'movement', label: 'Движение' },
]
const activeCategory = ref<EventCategory | 'all'>('all')
const activeSide = ref<Side | 'all'>('all')
const visibleEvents = computed(() => props.events.filter((event) => {
  const categoryMatches = activeCategory.value === 'all' || event.category === activeCategory.value
  const sideMatches = activeSide.value === 'all' || event.side === activeSide.value
  return categoryMatches && sideMatches && event.cursor <= props.currentCursor
}).slice().reverse())

function eventIcon(event: ViewerEvent) {
  if (event.category === 'combat') return '✳'
  if (event.category === 'movement') return '↗'
  if (event.category === 'terminal') return '◎'
  if (event.kind === 'lay' || event.kind === 'hatch') return '＋'
  return '·'
}
</script>

<template>
  <section class="panel event-panel" aria-labelledby="events-heading">
    <div class="panel-heading">
      <div><span class="eyebrow">ХРОНИКА РЕПЛЕЯ</span><h2 id="events-heading">События <span class="event-count">{{ visibleEvents.length }}</span></h2></div>
      <span class="feed-live"><i></i> ПО КАДРАМ</span>
    </div>
    <div class="filter-row" role="tablist" aria-label="Фильтр событий по типу">
      <button v-for="category in categories" :key="category.value" class="filter-chip" :class="{ active: activeCategory === category.value }" role="tab" :aria-selected="activeCategory === category.value" @click="activeCategory = category.value">{{ category.label }}</button>
    </div>
    <div class="side-filter" aria-label="Фильтр стороны">
      <span>СТОРОНА</span>
      <button :class="{ active: activeSide === 'all' }" :aria-pressed="activeSide === 'all'" @click="activeSide = 'all'">Обе</button>
      <button class="side-filter-a" :class="{ active: activeSide === 'A' }" :aria-pressed="activeSide === 'A'" @click="activeSide = 'A'">A</button>
      <button class="side-filter-b" :class="{ active: activeSide === 'B' }" :aria-pressed="activeSide === 'B'" @click="activeSide = 'B'">B</button>
    </div>
    <div class="event-list" aria-live="polite">
      <button v-for="event in visibleEvents" :key="event.id" class="event-item" :class="[`event-${event.category}`, { current: event.cursor === currentCursor }]" @click="emit('selectEvent', event)">
        <span class="event-rail"><span class="event-symbol">{{ eventIcon(event) }}</span><span class="event-connector"></span></span>
        <span class="event-content">
          <span class="event-meta"><span class="event-tick">КАДР {{ String(event.cursor).padStart(2, '0') }}</span><span v-if="event.side" class="event-side" :class="`event-side-${event.side.toLowerCase()}`">{{ event.side }}</span><span class="event-kind">{{ event.kind.replace('_', ' ') }}</span></span>
          <strong>{{ event.summary }}</strong>
          <span class="event-detail">{{ event.detail }}</span>
        </span>
        <span class="event-open" aria-hidden="true">↗</span>
      </button>
      <div v-if="visibleEvents.length === 0" class="empty-events"><span class="empty-icon">⌁</span><strong>Нет событий</strong><span>Для текущих фильтров событий не найдено.</span></div>
    </div>
    <div class="feed-footer"><span class="feed-footer-dot"></span>Показаны только факты из demo-записи</div>
  </section>
</template>
