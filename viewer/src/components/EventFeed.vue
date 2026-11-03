<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { EventCategory, ReplayDataSource, Side, ViewerEvent } from '../domain/replay'

const props = defineProps<{ source: ReplayDataSource; currentCursor: number }>()
const emit = defineEmits<{ selectEvent: [event: ViewerEvent] }>()
const PAGE_SIZE = 50

const categories: { value: EventCategory | 'all'; label: string }[] = [
  { value: 'all', label: 'Все' },
  { value: 'combat', label: 'Бой' },
  { value: 'economy', label: 'Экономика' },
  { value: 'movement', label: 'Движение' },
  { value: 'terminal', label: 'Исход' },
]
const activeCategory = ref<EventCategory | 'all'>('all')
const activeSide = ref<Side | 'all'>('all')
const page = ref(1)
watch([activeCategory, activeSide, () => props.currentCursor], () => { page.value = 1 })

const query = computed(() => ({ category: activeCategory.value, side: activeSide.value, throughCursor: props.currentCursor }))
const eventCount = computed(() => props.source.getEventCount(query.value))
const visibleEvents = computed(() => props.source.getEvents({ ...query.value, offset: (page.value - 1) * PAGE_SIZE, limit: PAGE_SIZE }))
const hasOlderEvents = computed(() => page.value * PAGE_SIZE < eventCount.value)

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
      <div><span class="eyebrow">ХРОНИКА РЕПЛЕЯ</span><h2 id="events-heading">События <span class="event-count">{{ eventCount.toLocaleString('ru-RU') }}</span></h2></div>
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
          <span class="event-meta"><span class="event-tick">ТИК {{ String(event.tick).padStart(4, '0') }} · КАДР {{ String(event.cursor).padStart(4, '0') }}</span><span v-if="event.side" class="event-side" :class="`event-side-${event.side.toLowerCase()}`">{{ event.side }}</span><span class="event-kind">{{ event.kind.replace('_', ' ') }}</span></span>
          <strong>{{ event.summary }}</strong>
          <span class="event-detail">{{ event.detail }}</span>
        </span>
        <span class="event-open" aria-hidden="true">↗</span>
      </button>
      <div v-if="visibleEvents.length === 0" class="empty-events"><span class="empty-icon">⌁</span><strong>Нет событий</strong><span>Для текущего кадра и фильтров событий не найдено.</span></div>
    </div>
    <div v-if="hasOlderEvents" class="feed-pagination"><span>{{ ((page - 1) * PAGE_SIZE + visibleEvents.length).toLocaleString('ru-RU') }} из {{ eventCount.toLocaleString('ru-RU') }}</span><button @click="page += 1">Загрузить более ранние</button></div>
    <div class="feed-footer"><span class="feed-footer-dot"></span>{{ source.getMetadata().sourceKind === 'demo' ? 'Демо · не симуляция' : 'Проверены структура, кадры и цепочка SHA-256 · без sim verify' }}</div>
  </section>
</template>
