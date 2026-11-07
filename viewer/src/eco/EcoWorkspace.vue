<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import EcoColonies from './EcoColonies.vue'
import EcoEvents from './EcoEvents.vue'
import EcoMap from './EcoMap.vue'
import EcoTimeline from './EcoTimeline.vue'
import { describeEvent } from './describe'
import { colonyView, combatLosses, eventCountThrough, eventVisibleToColony } from './model'
import type { EcoEvent, EcoReplay } from './types'
import { useEcoPlayback } from './useEcoPlayback'

const props = defineProps<{ replay: EcoReplay }>()
const playback = useEcoPlayback(props.replay)
const params = new URLSearchParams(typeof window === 'undefined' ? '' : window.location.search)

const raiderColony = Object.entries(props.replay.roles).find(([, role]) => role === 'raider')?.[0]
const viewMode = ref<'all' | 'colony'>(params.get('view') === 'colony' ? 'colony' : 'all')
const selectedColony = ref<number>(Number.isInteger(Number(params.get('colony'))) && params.get('colony') !== null && Number(params.get('colony')) < props.replay.colonyCount
  ? Number(params.get('colony')) : Number(raiderColony ?? 0))
const selectedNode = ref<number | null>(null)
const focusEvent = ref<EcoEvent | null>(null)
const map = ref<InstanceType<typeof EcoMap> | null>(null)
const requestedTick = Number(params.get('tick'))
if (params.get('tick') !== null && Number.isInteger(requestedTick)) playback.seek(requestedTick)

const frame = playback.frame
const tick = playback.tick
const fog = computed(() => (viewMode.value === 'colony' ? colonyView(props.replay, frame.value, selectedColony.value) : null))
const losses = computed(() => combatLosses(props.replay, tick.value))
const viewColony = computed(() => (viewMode.value === 'colony' ? selectedColony.value : null))
const recent = computed(() => {
  const through = eventCountThrough(props.replay.events, tick.value)
  const out: EcoEvent[] = []
  for (let i = through - 1; i >= 0 && props.replay.events[i].tick > tick.value - 9; i -= 1) {
    const event = props.replay.events[i]
    if (event.kind === 'discover' || event.kind === 'contact' || event.kind === 'nest_found' || event.kind === 'dispatch' || event.kind === 'combat' || event.kind === 'queen_dead' || event.kind === 'extinct' || event.kind === 'collapse') {
      if (viewColony.value === null || eventVisibleToColony(event, viewColony.value)) out.push(event)
    }
  }
  return out
})
const aliveCount = computed(() => frame.value.colonies.filter((c) => c.status === 'alive').length)
const orphanCount = computed(() => frame.value.colonies.filter((c) => c.status === 'orphan').length)
const extinctCount = computed(() => frame.value.colonies.filter((c) => c.status === 'extinct').length)

/** The state *after* tick ``event.tick`` completed contains the event's effects, so that is where the viewer jumps. */
function chooseEvent(event: EcoEvent) {
  playback.seek(Math.min(playback.lastTick, event.tick + 1))
  focusEvent.value = event
  if (event.node >= 0) {
    selectedNode.value = event.node
    map.value?.focusOn(event.node)
  }
}
function chooseMoment(direction: -1 | 1) {
  const target = playback.jumpMoment(direction)
  const moment = props.replay.events.find((event) => event.tick + 1 === target && playback.moments.some((m) => m.eventIndex === event.index))
  if (moment) { focusEvent.value = moment; if (moment.node >= 0) map.value?.focusOn(moment.node) }
}

const inspector = computed(() => {
  const id = selectedNode.value
  if (id === null) return null
  const node = props.replay.nodes[id]
  const known = fog.value?.nodes.find((n) => n.id === id)
  const here = (fog.value ? fog.value.ownCohorts : frame.value.cohorts).filter((c) => c.atNode && c.from === id)
  const byColony = new Map<number, { workers: number; soldiers: number }>()
  for (const cohort of here) {
    const row = byColony.get(cohort.colony) ?? { workers: 0, soldiers: 0 }
    if (cohort.caste === 'worker') row.workers += cohort.count
    else row.soldiers += cohort.count
    byColony.set(cohort.colony, row)
  }
  const patchIndex = props.replay.patchNodes.indexOf(id)
  const sightings = fog.value ? fog.value.sightings.filter((s) => s.node === id) : []
  return { node, known, stock: patchIndex >= 0 ? frame.value.patches[patchIndex] : null, pile: frame.value.piles.find((p) => p.node === id)?.stock ?? 0,
    groups: [...byColony.entries()].sort((a, b) => a[0] - b[0]), sightings }
})

function onKeydown(event: KeyboardEvent) {
  const target = event.target
  if (target instanceof HTMLElement && (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON', 'A'].includes(target.tagName))) return
  if (event.code === 'Space') { event.preventDefault(); playback.isPlaying.value ? playback.pause() : playback.play() }
  else if (event.key === 'ArrowLeft') { event.preventDefault(); playback.stepFrame(-1) }
  else if (event.key === 'ArrowRight') { event.preventDefault(); playback.stepFrame(1) }
  else if (event.key === 'Home') playback.goToStart()
  else if (event.key === 'End') playback.goToEnd()
}
onMounted(() => window.addEventListener('keydown', onKeydown))
onUnmounted(() => window.removeEventListener('keydown', onKeydown))
</script>

<template>
  <div class="eco-workspace">
    <div class="eco-modebar">
      <div class="eco-mode" role="group" aria-label="Режим знания">
        <button type="button" :class="{ active: viewMode === 'all' }" :aria-pressed="viewMode === 'all'" @click="viewMode = 'all'">Все сведения</button>
        <button type="button" :class="{ active: viewMode === 'colony' }" :aria-pressed="viewMode === 'colony'" @click="viewMode = 'colony'">Вид колонии</button>
        <label v-if="viewMode === 'colony'" class="eco-select">колония
          <select v-model.number="selectedColony" aria-label="Колония для режима знания"><option v-for="id in replay.colonyCount" :key="id" :value="id - 1">{{ id - 1 }}</option></select>
        </label>
      </div>
      <p class="eco-mode-note" data-testid="mode-note">
        <template v-if="viewMode === 'all'">Режим исследователя: показано истинное состояние мира, которого колонии не знают.</template>
        <template v-else>Экранный режим «колония {{ selectedColony }}»: только её память, её юниты и устаревающие наблюдения; скрыты неоткрытые узлы, чужие склады, матки и войска. Это инструмент зрителя, а не данные для ИИ.</template>
      </p>
      <div class="eco-summary"><span>живых <b>{{ aliveCount }}</b></span><span>сирот <b>{{ orphanCount }}</b></span><span>вымерло <b>{{ extinctCount }}</b></span></div>
    </div>
    <main class="eco-dashboard">
      <aside class="eco-left" aria-label="Колонии">
        <div class="eco-intro"><span class="eyebrow">ECOSYSTEM V1 · СИД {{ replay.seed }}</span><h1>{{ replay.colonyCount }} колоний · {{ replay.nodes.length }} узлов · {{ replay.edges.length }} рёбер</h1><p>{{ replay.ticksCompleted }} тиков мира · {{ replay.frames.length }} кадров · {{ replay.events.length }} событий</p></div>
        <EcoColonies :replay="replay" :frame="frame" :fog="fog" :selected="selectedColony" :losses="losses" @select="selectedColony = $event" />
      </aside>
      <section class="eco-center" aria-label="Карта и воспроизведение">
        <div class="eco-panel eco-map-card">
          <EcoMap ref="map" :replay="replay" :frame="frame" :tick="tick" :fog="fog" :selected-node="selectedNode" :recent-events="recent" :focus-event="focusEvent" @select-node="selectedNode = $event" />
        </div>
        <EcoTimeline :tick="tick" :last-tick="playback.lastTick" :playing="playback.isPlaying.value" :speed="playback.speed.value" :speeds="playback.speeds" :moments="playback.moments" :frame="frame" :frame-count="replay.frames.length"
          @toggle="playback.isPlaying.value ? playback.pause() : playback.play()" @seek="playback.seek" @frame="playback.stepFrame" @moment="chooseMoment" @start="playback.goToStart" @end="playback.goToEnd" @speed="playback.setSpeed" />
      </section>
      <aside class="eco-right" aria-label="События и инспектор">
        <EcoEvents :replay="replay" :tick="tick" :view-colony="viewColony" :active-index="focusEvent?.index ?? null" @select="chooseEvent" />
        <section class="eco-panel eco-inspector" aria-label="Инспектор узла">
          <div class="eco-panel-head"><div><span class="eyebrow">ИНСПЕКТОР</span><h2>{{ inspector ? `Узел ${inspector.node.id}` : 'Узел не выбран' }}</h2></div></div>
          <template v-if="inspector">
            <template v-if="!fog">
              <dl>
                <dt>тип</dt><dd>{{ inspector.node.kind === 'nest' ? `гнездо колонии ${inspector.node.nestOf}` : inspector.node.kind === 'patch' ? `площадка (${inspector.node.zone})` : 'перекрёсток' }}</dd>
                <template v-if="inspector.stock !== null"><dt>запас</dt><dd>{{ inspector.stock }} / {{ inspector.node.capacity }}</dd></template>
                <template v-if="inspector.pile > 0"><dt>куча еды</dt><dd>{{ inspector.pile }}</dd></template>
              </dl>
              <ul class="eco-here"><li v-for="[colony, row] in inspector.groups" :key="colony">колония {{ colony }}: {{ row.workers }} раб. · {{ row.soldiers }} сол.</li><li v-if="inspector.groups.length === 0" class="muted">юнитов в узле нет (кадр {{ frame.tick }})</li></ul>
            </template>
            <template v-else>
              <p v-if="!inspector.known" class="muted">Колония {{ selectedColony }} не знает об этом узле.</p>
              <dl v-else>
                <dt>известность</dt><dd>{{ inspector.known.status === 'visited' ? 'посещён' : 'только слышала о нём' }}</dd>
                <template v-if="inspector.known.status === 'visited'">
                  <dt>тип</dt><dd>{{ inspector.known.kind }}{{ inspector.known.nestOwner >= 0 ? ` · гнездо колонии ${inspector.known.nestOwner}` : '' }}</dd>
                  <dt>последний раз видела</dt><dd>тик {{ inspector.known.seenTick }} ({{ inspector.known.inView ? 'в поле зрения' : `${inspector.known.age} т. назад — историческое сведение` }})</dd>
                  <template v-if="inspector.known.stock !== null"><dt>запас на тот момент</dt><dd>{{ inspector.known.stock }} / {{ inspector.known.capacity }}{{ inspector.known.stockAge ? ` (${inspector.known.stockAge} т. назад)` : '' }}</dd></template>
                  <template v-if="inspector.known.queenSeen"><dt>матка в гнезде</dt><dd>{{ inspector.known.queenSeen === 1 ? 'была жива' : 'мертва' }} на момент наблюдения</dd></template>
                </template>
              </dl>
              <ul v-if="inspector.sightings.length" class="eco-here"><li v-for="(s, i) in inspector.sightings" :key="i">видела колонию {{ s.colony }} {{ s.age }} т. назад (корзины {{ s.workersBucket }}/{{ s.soldiersBucket }}){{ s.aggression ? ' · атакующую' : '' }}</li></ul>
              <ul class="eco-here"><li v-for="[colony, row] in inspector.groups" :key="colony">свои: {{ row.workers }} раб. · {{ row.soldiers }} сол.</li></ul>
            </template>
          </template>
          <p v-if="focusEvent" class="eco-focus-note">Событие: {{ describeEvent(focusEvent) }} (тик {{ focusEvent.tick }})</p>
        </section>
      </aside>
    </main>
  </div>
</template>
