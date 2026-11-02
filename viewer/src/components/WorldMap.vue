<script setup lang="ts">
import { computed } from 'vue'
import { NODE_LAYOUT, WORLD_EDGES, type NodeView, type TimelineFrame } from '../domain/replay'

const props = defineProps<{ frame: TimelineFrame; selectedNodeId: number | null }>()
const emit = defineEmits<{ selectNode: [nodeId: number] }>()

const positions = new Map<number, (typeof NODE_LAYOUT)[number]>(NODE_LAYOUT.map((node) => [node.id, node]))
const nodeById = computed(() => new Map(props.frame.nodes.map((node) => [node.id, node])))
const transitMarkers = computed(() => props.frame.transits.map((group) => {
  const from = positions.get(group.from)!
  const to = positions.get(group.to)!
  return { ...group, x: from.x + (to.x - from.x) * group.progress, y: from.y + (to.y - from.y) * group.progress }
}))

function formatCount(node: NodeView) {
  const a = node.groups.A
  const b = node.groups.B
  return `A ${String(a.workers).padStart(2, '0')}/${String(a.soldiers).padStart(2, '0')} · B ${String(b.workers).padStart(2, '0')}/${String(b.soldiers).padStart(2, '0')}`
}

function keySelect(event: KeyboardEvent, nodeId: number) {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault()
    emit('selectNode', nodeId)
  }
}
</script>

<template>
  <div class="map-shell">
    <div class="map-toolbar">
      <div class="map-legend">
        <span class="legend-item"><i class="legend-dot side-a-bg"></i>КОЛОНИЯ A</span>
        <span class="legend-item"><i class="legend-dot side-b-bg"></i>КОЛОНИЯ B</span>
        <span class="legend-item"><i class="legend-line"></i>ДЛИНА РЕБРА</span>
      </div>
      <span class="map-dimensions">7 УЗЛОВ <b>·</b> 9 СВЯЗЕЙ</span>
    </div>
    <svg class="world-map" viewBox="0 0 1112 550" role="img" aria-labelledby="map-title map-desc">
      <title id="map-title">Карта мира из семи узлов</title>
      <desc id="map-desc">Сеть из семи узлов и девяти рёбер. Нажмите на узел, чтобы открыть его данные.</desc>
      <defs>
        <pattern id="map-grid" width="32" height="32" patternUnits="userSpaceOnUse">
          <path d="M 32 0 L 0 0 0 32" fill="none" stroke="#263337" stroke-width=".65" opacity=".52" />
        </pattern>
        <radialGradient id="map-wash">
          <stop offset="0" stop-color="#23372f" stop-opacity=".58" />
          <stop offset="1" stop-color="#182326" stop-opacity="0" />
        </radialGradient>
        <filter id="node-glow" x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation="7" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
      </defs>
      <rect width="1112" height="550" fill="url(#map-grid)" />
      <ellipse cx="556" cy="285" rx="470" ry="250" fill="url(#map-wash)" />
      <g class="map-edge-layer">
        <g v-for="edge in WORLD_EDGES" :key="`${edge.a}-${edge.b}`" class="map-edge">
          <line :x1="positions.get(edge.a)!.x" :y1="positions.get(edge.a)!.y" :x2="positions.get(edge.b)!.x" :y2="positions.get(edge.b)!.y" />
          <g class="edge-length" :transform="`translate(${(positions.get(edge.a)!.x + positions.get(edge.b)!.x) / 2} ${(positions.get(edge.a)!.y + positions.get(edge.b)!.y) / 2})`">
            <rect x="-16" y="-11" width="32" height="22" rx="7" />
            <text text-anchor="middle" dominant-baseline="central">{{ edge.length }}</text>
          </g>
        </g>
      </g>
      <g v-for="node in NODE_LAYOUT" :key="node.id" class="map-node" :class="[`node-${node.isHome?.toLowerCase() ?? 'resource'}`, { selected: selectedNodeId === node.id, contact: nodeById.get(node.id)?.alert === 'contact' }]" :transform="`translate(${node.x} ${node.y})`" role="button" tabindex="0" :aria-label="`Узел ${node.name}, еда ${nodeById.get(node.id)?.food ?? 0} из ${nodeById.get(node.id)?.capacity ?? 0}. Открыть инспектор`" :aria-pressed="selectedNodeId === node.id" @click="emit('selectNode', node.id)" @keydown="keySelect($event, node.id)">
        <rect class="node-card" x="-72" y="-53" width="144" height="106" rx="15" />
        <circle class="node-halo" cx="-54" cy="-35" r="5" />
        <text class="node-overline" x="-42" y="-31">{{ node.isHome ? `ДОМ · КОЛОНИЯ ${node.name}` : 'КОРМОВЫЙ УЗЕЛ' }}</text>
        <text class="node-name" x="-57" y="-1">{{ node.name }}</text>
        <g v-if="nodeById.get(node.id)?.alert === 'contact'" class="contact-badge" transform="translate(22 -16)">
          <rect x="0" y="-13" width="43" height="21" rx="6" />
          <text x="21.5" y="1" text-anchor="middle">БОЙ</text>
        </g>
        <text class="node-food" x="-57" y="21">{{ nodeById.get(node.id)?.food ?? 0 }}<tspan class="node-capacity"> / {{ nodeById.get(node.id)?.capacity ?? 0 }} ЕД.</tspan></text>
        <rect class="food-track" x="-57" y="29" width="114" height="3" rx="2" />
        <rect class="food-fill" x="-57" y="29" :width="nodeById.get(node.id)?.capacity ? 114 * (nodeById.get(node.id)!.food / nodeById.get(node.id)!.capacity) : 0" height="3" rx="2" />
        <text class="node-groups" x="-57" y="45">{{ formatCount(nodeById.get(node.id)!) }}</text>
      </g>
      <g v-for="group in transitMarkers" :key="group.id" class="transit-marker" :class="`transit-${group.side.toLowerCase()}`" :transform="`translate(${group.x} ${group.y})`" :aria-label="`${group.side}: ${group.count} ${group.caste === 'workers' ? 'рабочих' : 'солдат'}, ETA ${group.etaTicks}`">
        <circle class="transit-orbit" r="15" />
        <circle class="transit-dot" r="5" />
        <g transform="translate(17 -27)"><rect width="48" height="20" rx="7" /><text x="24" y="13" text-anchor="middle">{{ group.count }} · {{ group.etaTicks }}т</text></g>
      </g>
    </svg>
    <div class="map-footnote"><span class="footnote-key">i</span><span>Координаты и транзит — значения демонстрационных кадров. Схема рёбер соответствует L0.</span><span class="map-north">N <i>↑</i></span></div>
  </div>
</template>
