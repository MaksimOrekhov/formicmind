<script setup lang="ts">
import { computed, reactive, ref } from 'vue'
import { colonyColor, placeCohort, type FogView } from './model'
import type { EcoEvent, EcoFrame, EcoReplay } from './types'

const props = defineProps<{
  replay: EcoReplay
  frame: EcoFrame
  tick: number
  fog: FogView | null
  selectedNode: number | null
  recentEvents: EcoEvent[]
  focusEvent: EcoEvent | null
}>()
const emit = defineEmits<{ selectNode: [node: number | null] }>()

const side = computed(() => props.replay.worldSide)
const pad = 46
const camera = reactive({ x: 0, y: 0, scale: 1 })
const svg = ref<SVGSVGElement | null>(null)
const dragging = ref(false)
let dragStart = { x: 0, y: 0, cx: 0, cy: 0 }

const viewBox = computed(() => {
  const full = side.value + pad * 2
  const w = full / camera.scale
  return `${-pad + camera.x} ${-pad + camera.y} ${w} ${w}`
})

function clampCamera() {
  const full = side.value + pad * 2
  const w = full / camera.scale
  camera.x = Math.max(0, Math.min(full - w, camera.x))
  camera.y = Math.max(0, Math.min(full - w, camera.y))
}

function toWorld(clientX: number, clientY: number) {
  const box = svg.value!.getBoundingClientRect()
  const full = side.value + pad * 2
  const w = full / camera.scale
  return { x: -pad + camera.x + ((clientX - box.left) / box.width) * w, y: -pad + camera.y + ((clientY - box.top) / box.height) * w }
}

function zoomAt(factor: number, clientX?: number, clientY?: number) {
  const full = side.value + pad * 2
  const next = Math.max(1, Math.min(8, camera.scale * factor))
  const anchor = clientX === undefined || !svg.value ? { x: -pad + camera.x + full / camera.scale / 2, y: -pad + camera.y + full / camera.scale / 2 } : toWorld(clientX, clientY!)
  const oldW = full / camera.scale
  const newW = full / next
  const rx = (anchor.x - (-pad + camera.x)) / oldW
  const ry = (anchor.y - (-pad + camera.y)) / oldW
  camera.scale = next
  camera.x = anchor.x - rx * newW + pad
  camera.y = anchor.y - ry * newW + pad
  clampCamera()
}

function onWheel(event: WheelEvent) {
  event.preventDefault()
  zoomAt(event.deltaY < 0 ? 1.25 : 0.8, event.clientX, event.clientY)
}
function onPointerDown(event: PointerEvent) {
  if ((event.target as Element).closest('.eco-node')) return
  dragging.value = true
  dragStart = { x: event.clientX, y: event.clientY, cx: camera.x, cy: camera.y }
  ;(event.currentTarget as Element).setPointerCapture?.(event.pointerId)
}
function onPointerMove(event: PointerEvent) {
  if (!dragging.value || !svg.value) return
  const box = svg.value.getBoundingClientRect()
  const full = side.value + pad * 2
  const w = full / camera.scale
  camera.x = dragStart.cx - ((event.clientX - dragStart.x) / box.width) * w
  camera.y = dragStart.cy - ((event.clientY - dragStart.y) / box.height) * w
  clampCamera()
}
function onPointerUp() { dragging.value = false }
function resetCamera() { camera.x = 0; camera.y = 0; camera.scale = 1 }
function focusOn(node: number) {
  const n = props.replay.nodes[node]
  const full = side.value + pad * 2
  camera.scale = Math.max(camera.scale, 2.5)
  const w = full / camera.scale
  camera.x = n.x + pad - w / 2
  camera.y = n.y + pad - w / 2
  clampCamera()
}
defineExpose({ focusOn, resetCamera })

const patchIndex = computed(() => new Map(props.replay.patchNodes.map((id, index) => [id, index])))
const pileByNode = computed(() => new Map(props.frame.piles.map((pile) => [pile.node, pile.stock])))

interface DrawNode {
  id: number; x: number; y: number; kind: string; status: 'visited' | 'discovered' | 'all'; fill: number; stockLabel: string
  stale: boolean; ownerColony: number; queenState: 0 | 1 | 2; pile: number; danger: boolean
}
const drawNodes = computed<DrawNode[]>(() => {
  const { replay, frame, fog } = props
  if (!fog) {
    return replay.nodes.map((node) => {
      const pi = patchIndex.value.get(node.id)
      const stock = pi === undefined ? 0 : frame.patches[pi]
      return {
        id: node.id, x: node.x, y: node.y, kind: node.kind, status: 'all' as const, fill: node.capacity ? stock / node.capacity : 0,
        stockLabel: node.kind === 'patch' ? `${stock}/${node.capacity}` : '', stale: false, ownerColony: node.nestOf,
        queenState: node.kind === 'nest' ? (frame.colonies[node.nestOf].queenHp > 0 ? 1 : 2) as 1 | 2 : 0, pile: pileByNode.value.get(node.id) ?? 0, danger: false,
      }
    })
  }
  return fog.nodes.map((node) => ({
    id: node.id, x: node.x, y: node.y, kind: node.kind, status: node.status,
    fill: node.stock !== null && node.capacity ? node.stock / node.capacity : 0,
    stockLabel: node.stock !== null ? `${node.stock}/${node.capacity}${node.stockAge && node.stockAge > 1 ? ` · ${node.stockAge}т назад` : ''}` : '',
    stale: node.status === 'visited' && !node.inView, ownerColony: node.nestOwner, queenState: node.queenSeen, pile: 0, danger: node.danger > 0,
  }))
})
const drawEdges = computed(() => (props.fog ? props.fog.edges : props.replay.edges).map((edge) => {
  const a = props.replay.nodes[edge.a]
  const b = props.replay.nodes[edge.b]
  return { key: `${edge.a}-${edge.b}`, x1: a.x, y1: a.y, x2: b.x, y2: b.y, narrow: edge.chokepoint, length: edge.length }
}))

interface Marker {
  key: number; colony: number; caste: 'worker' | 'soldier'; count: number; x: number; y: number; r: number; loadFrac: number
  kind: string; exact: boolean; mode: string
}
const markers = computed<Marker[]>(() => {
  const list = props.fog ? props.fog.ownCohorts : props.frame.cohorts
  const taken = new Map<string, number>()
  const out: Marker[] = []
  for (const cohort of [...list].sort((a, b) => a.colony - b.colony || a.id - b.id)) {
    const place = placeCohort(props.replay, cohort, props.frame, props.tick)
    const slotKey = `${Math.round(place.x)},${Math.round(place.y)}`
    const slot = taken.get(slotKey) ?? 0
    taken.set(slotKey, slot + 1)
    const angle = slot * 2.399963
    const spread = slot === 0 ? 0 : 7 + 2.2 * Math.sqrt(slot)
    out.push({
      key: cohort.id, colony: cohort.colony, caste: cohort.caste, count: cohort.count, x: place.x + Math.cos(angle) * spread, y: place.y + Math.sin(angle) * spread,
      r: 3.4 + Math.sqrt(cohort.count) * 1.15, loadFrac: cohort.caste === 'worker' ? Math.min(1, cohort.load / (cohort.count * 16)) : 0,
      kind: cohort.kind, exact: place.exact, mode: place.mode,
    })
  }
  return out
})
const ghosts = computed(() => (props.fog ? props.fog.sightings : []).map((item, index) => {
  const node = props.replay.nodes[item.node]
  return { key: index, x: node.x + 10 + (index % 3) * 9, y: node.y - 14 - Math.floor(index / 3) * 9, colony: item.colony, age: item.age, strength: item.workersBucket + item.soldiersBucket, aggression: item.aggression }
}))
const pulses = computed(() => props.recentEvents.filter((event) => event.node >= 0).map((event) => {
  const node = props.replay.nodes[event.node]
  return { key: event.index, x: node.x, y: node.y, kind: event.kind, age: props.tick - event.tick }
}))
const focusPoint = computed(() => (props.focusEvent && props.focusEvent.node >= 0 ? props.replay.nodes[props.focusEvent.node] : null))
const nodeR = (kind: string) => (kind === 'nest' ? 11 : kind === 'patch' ? 7.5 : kind === 'unknown' ? 4 : 3.2)

function keySelect(event: KeyboardEvent, id: number) {
  if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); emit('selectNode', id) }
}
</script>

<template>
  <div class="eco-map-shell">
    <div class="eco-map-tools">
      <div class="eco-legend">
        <span><i class="lg-worker"></i>рабочие</span>
        <span><i class="lg-soldier"></i>солдаты</span>
        <span><i class="lg-load"></i>груз</span>
        <span><i class="lg-patch"></i>площадка (запас)</span>
        <span v-if="fog"><i class="lg-ghost"></i>наблюдение (устарело)</span>
      </div>
      <div class="eco-zoom">
        <button type="button" aria-label="Приблизить" @click="zoomAt(1.4)">+</button>
        <button type="button" aria-label="Отдалить" @click="zoomAt(1 / 1.4)">−</button>
        <button type="button" aria-label="Сбросить масштаб" @click="resetCamera()">⌂</button>
        <span class="eco-zoom-level">×{{ camera.scale.toFixed(1) }}</span>
      </div>
    </div>
    <svg ref="svg" class="eco-map" :class="{ dragging }" :viewBox="viewBox" role="img" aria-label="Карта мира Ecosystem v1" @wheel="onWheel" @pointerdown="onPointerDown" @pointermove="onPointerMove" @pointerup="onPointerUp" @pointercancel="onPointerUp">
      <rect :x="-pad" :y="-pad" :width="side + pad * 2" :height="side + pad * 2" class="eco-map-bg" />
      <g class="eco-edges">
        <line v-for="edge in drawEdges" :key="edge.key" :x1="edge.x1" :y1="edge.y1" :x2="edge.x2" :y2="edge.y2" :class="{ narrow: edge.narrow }" />
      </g>
      <g class="eco-nodes">
        <g v-for="node in drawNodes" :key="node.id" class="eco-node" :class="[`kind-${node.kind}`, { discovered: node.status === 'discovered', stale: node.stale, selected: selectedNode === node.id }]" :transform="`translate(${node.x} ${node.y})`" role="button" tabindex="0" :aria-label="`Узел ${node.id}`" @click.stop="emit('selectNode', node.id)" @keydown="keySelect($event, node.id)">
          <template v-if="node.kind === 'nest'">
            <circle :r="nodeR('nest') + 4" class="nest-ring" :style="{ stroke: node.ownerColony >= 0 ? colonyColor(node.ownerColony) : '#7d8a86' }" />
            <circle :r="nodeR('nest')" class="nest-core" :style="{ fill: node.ownerColony >= 0 ? colonyColor(node.ownerColony) : '#3a4642' }" :class="{ dead: node.queenState === 2 }" />
            <text class="nest-label" y="3.6" text-anchor="middle">{{ node.ownerColony >= 0 ? node.ownerColony : '?' }}</text>
            <text v-if="node.queenState === 2" class="queen-mark" y="-19" text-anchor="middle">✝</text>
          </template>
          <template v-else-if="node.kind === 'patch'">
            <circle :r="nodeR('patch') + 2" class="patch-track" />
            <circle :r="Math.max(1.5, nodeR('patch') * Math.sqrt(node.fill))" class="patch-fill" />
          </template>
          <circle v-else :r="nodeR(node.kind)" class="junction-dot" />
          <circle v-if="node.pile > 0" cx="9" cy="9" r="3.6" class="pile-dot" />
          <circle v-if="node.danger" r="15" class="danger-ring" />
        </g>
      </g>
      <g v-if="focusPoint" class="eco-focus" :transform="`translate(${focusPoint.x} ${focusPoint.y})`"><circle r="22" /><circle r="30" /></g>
      <g v-for="pulse in pulses" :key="pulse.key" class="eco-pulse" :class="`pulse-${pulse.kind}`" :transform="`translate(${pulse.x} ${pulse.y})`">
        <circle :r="14 + pulse.age * 2.4" :opacity="Math.max(0.15, 1 - pulse.age / 9)" />
      </g>
      <g class="eco-ghosts">
        <g v-for="ghost in ghosts" :key="ghost.key" :transform="`translate(${ghost.x} ${ghost.y})`" :opacity="Math.max(0.25, 1 - ghost.age / 800)">
          <circle r="4.5" :style="{ stroke: colonyColor(ghost.colony) }" class="ghost" :class="{ aggressive: ghost.aggression }" />
          <text y="2.6" text-anchor="middle">{{ ghost.strength }}</text>
        </g>
      </g>
      <g class="eco-cohorts">
        <g v-for="marker in markers" :key="marker.key" class="eco-cohort" :class="[`mission-${marker.kind}`, { approx: !marker.exact }]" :transform="`translate(${marker.x} ${marker.y})`">
          <circle v-if="marker.kind === 'attack' || marker.kind === 'scout'" :r="marker.r + 3.4" class="mission-ring" :class="`ring-${marker.kind}`" />
          <circle v-if="marker.caste === 'worker'" :r="marker.r" :style="{ fill: colonyColor(marker.colony) }" class="cohort-worker" />
          <rect v-else :x="-marker.r" :y="-marker.r" :width="marker.r * 2" :height="marker.r * 2" :style="{ fill: colonyColor(marker.colony) }" class="cohort-soldier" transform="rotate(45)" />
          <circle v-if="marker.loadFrac > 0" :r="Math.max(1.2, marker.r * 0.55 * Math.sqrt(marker.loadFrac))" class="cohort-load" />
          <text v-if="marker.count > 1" class="cohort-count" :y="-marker.r - 2" text-anchor="middle">{{ marker.count }}</text>
        </g>
      </g>
    </svg>
    <div class="eco-map-note">{{ fog ? 'Вид колонии: показаны только её собственные сведения. Положение узлов — раскладка viewer, а не знание колонии.' : 'Все сведения: истинное состояние мира. Движение по рёбрам — по записанным depart/arrive; между кадрами положение вне ребра не записано.' }}</div>
  </div>
</template>
