<script setup lang="ts">
import { computed } from 'vue'
import type { NodeView, Side, TimelineFrame } from '../domain/replay'
import { NODE_LAYOUT, WORLD_EDGES } from '../domain/replay'

const props = defineProps<{ node: NodeView | null; frame: TimelineFrame; selectedSide: Side }>()
const emit = defineEmits<{ selectSide: [side: Side] }>()

const connectedEdges = computed(() => props.node ? WORLD_EDGES.filter((edge) => edge.a === props.node?.id || edge.b === props.node?.id).map((edge) => {
  const peerId = edge.a === props.node!.id ? edge.b : edge.a
  return { name: NODE_LAYOUT[peerId].name, length: edge.length }
}) : [])
const groupsHere = computed(() => props.node?.groups[props.selectedSide] ?? { workers: 0, soldiers: 0 })
const transitsHere = computed(() => props.frame.transits.filter((group) => group.from === props.node?.id || group.to === props.node?.id))
</script>

<template>
  <section class="panel inspector-panel" aria-labelledby="inspector-heading">
    <div class="panel-heading compact-heading"><div><span class="eyebrow">ИНСПЕКТОР УЗЛА</span><h2 id="inspector-heading">{{ node ? node.name : 'Выберите узел' }}</h2></div><span class="inspector-index">{{ node ? `#${String(node.id).padStart(2, '0')}` : '—' }}</span></div>
    <template v-if="node">
      <div class="inspector-type-row"><span class="node-type-icon" :class="node.isHome ? 'type-home' : 'type-food'">{{ node.isHome ? '⌂' : '◇' }}</span><span><strong>{{ node.isHome ? `Дом колонии ${node.isHome}` : 'Кормовой узел' }}</strong><small>{{ node.alert ? 'Зафиксирован контакт' : node.isHome ? 'Материнская колония' : 'Ресурсный пункт' }}</small></span><i v-if="node.alert" class="contact-indicator"></i></div>
      <div class="inspector-resource">
        <div><span class="stat-label">ЗАПАС ЕДЫ</span><strong>{{ node.food }}<small> / {{ node.capacity }} ед.</small></strong></div>
        <div class="resource-meter"><span :style="{ width: `${node.capacity ? node.food / node.capacity * 100 : 0}%` }"></span></div>
      </div>
      <div class="inspector-subhead"><span>ПРИСУТСТВИЕ</span><span>КОЛОНИЯ {{ selectedSide }}</span></div>
      <div class="side-switch" aria-label="Выбрать колонию для инспекции">
        <button class="side-switch-a" :class="{ active: selectedSide === 'A' }" :aria-pressed="selectedSide === 'A'" @click="emit('selectSide', 'A')">Колония A</button>
        <button class="side-switch-b" :class="{ active: selectedSide === 'B' }" :aria-pressed="selectedSide === 'B'" @click="emit('selectSide', 'B')">Колония B</button>
      </div>
      <div class="presence-grid">
        <div><span class="presence-icon worker-icon">●</span><span><strong>{{ groupsHere.workers }}</strong><small>рабочие</small></span></div>
        <div><span class="presence-icon soldier-icon">◆</span><span><strong>{{ groupsHere.soldiers }}</strong><small>солдаты</small></span></div>
      </div>
      <div class="inspector-subhead connections-title"><span>СВЯЗИ</span><span>{{ connectedEdges.length }} РЕБРА</span></div>
      <div class="connection-list"><span v-for="edge in connectedEdges" :key="edge.name" class="connection-pill"><i></i>{{ edge.name }}<small>{{ edge.length }} т.</small></span></div>
      <div class="transit-callout">
        <div class="transit-callout-head"><span>В ТРАНЗИТЕ</span><span>{{ transitsHere.length }}</span></div>
        <div v-if="transitsHere.length" class="transit-list">
          <span v-for="group in transitsHere" :key="group.id" class="transit-row" :class="`transit-row-${group.side.toLowerCase()}`"><i></i><strong>{{ group.side }} · {{ group.count }} {{ group.caste === 'workers' ? 'раб.' : 'солд.' }}</strong><small>ETA {{ group.etaTicks }} т.</small></span>
        </div>
        <span v-else class="no-transit">В этом кадре нет транзитных групп</span>
      </div>
    </template>
    <div v-else class="inspector-empty"><span>⌖</span><strong>Карта готова к осмотру</strong><p>Нажмите на узел, чтобы увидеть его ресурс, присутствие колоний и связанные рёбра.</p></div>
    <div class="inspector-footer"><span>СОСТОЯНИЕ КАДРА</span><strong>{{ frame.state.tickCompleted ? 'ТИК ЗАВЕРШЁН' : 'ПРЕРВАНО ВО ВРЕМЯ ТИКА' }}</strong></div>
  </section>
</template>
