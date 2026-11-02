<script setup lang="ts">
import { computed, onMounted, onUnmounted } from 'vue'
import ColonyPanel from './components/ColonyPanel.vue'
import EventFeed from './components/EventFeed.vue'
import NodeInspector from './components/NodeInspector.vue'
import ReplayHeader from './components/ReplayHeader.vue'
import TimelineControls from './components/TimelineControls.vue'
import WorldMap from './components/WorldMap.vue'
import { usePlayback } from './composables/usePlayback'
import { useSelection } from './composables/useSelection'
import type { ReplayDataSource } from './domain/replay'
import type { ParseProgress } from './data/replayJsonl'
import { presentOutcome } from './domain/outcome'

const props = defineProps<{ source: ReplayDataSource; isLoading: boolean; progress: ParseProgress | null; errorMessage: string; isDemo: boolean }>()
const emit = defineEmits<{ openReplay: []; returnDemo: [] }>()
const playback = usePlayback(props.source)
const cursorParam = new URLSearchParams(window.location.search).get('cursor')
const requestedCursor = cursorParam === null ? Number.NaN : Number(cursorParam)
if (Number.isInteger(requestedCursor) && requestedCursor >= playback.metadata.minCursor && requestedCursor <= playback.metadata.maxCursor) {
  playback.seek(requestedCursor)
}
const selection = useSelection(3)
const frame = playback.frame
const previousFrame = computed(() => playback.cursor.value > playback.metadata.minCursor ? props.source.getFrame(playback.cursor.value - 1) : undefined)
const selectedNode = computed(() => frame.value.nodes.find((node) => node.id === selection.selectedNodeId.value) ?? null)
const terminalOutcome = computed(() => playback.metadata.terminal ? presentOutcome(playback.metadata.terminal) : null)

function chooseEvent(event: { cursor: number; nodeId: number | null; side: 'A' | 'B' | null }) {
  playback.seek(event.cursor)
  if (event.nodeId !== null) selection.selectedNodeId.value = event.nodeId
  if (event.side) selection.selectedSide.value = event.side
}

function onKeydown(event: KeyboardEvent) {
  const target = event.target
  if (target instanceof HTMLElement && (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON', 'A'].includes(target.tagName))) return
  if (event.code === 'Space') {
    event.preventDefault()
    playback.isPlaying.value ? playback.pause() : playback.play()
  } else if (event.key === 'ArrowLeft') {
    event.preventDefault()
    playback.step(-1)
  } else if (event.key === 'ArrowRight') {
    event.preventDefault()
    playback.step(1)
  } else if (event.key === 'Home') playback.goToStart()
  else if (event.key === 'End') playback.goToEnd()
}

onMounted(() => window.addEventListener('keydown', onKeydown))
onUnmounted(() => window.removeEventListener('keydown', onKeydown))
</script>

<template>
  <div id="top" class="app-shell">
    <ReplayHeader :metadata="playback.metadata" :current-cursor="playback.cursor.value" :is-loading="props.isLoading" :is-demo="props.isDemo" @open-replay="emit('openReplay')" @return-demo="emit('returnDemo')" />
    <div v-if="props.isLoading || props.errorMessage" class="import-banner" :class="{ 'import-error': props.errorMessage }" role="status" aria-live="polite">
      <template v-if="props.isLoading"><span class="import-spinner"></span><span>Проверяю локальный replay · {{ Math.round((props.progress?.bytesProcessed ?? 0) / Math.max(1, props.progress?.totalBytes ?? 1) * 100) }}%</span><small>{{ (props.progress?.recordsProcessed ?? 0).toLocaleString('ru-RU') }} / {{ (props.progress?.totalRecords ?? 0).toLocaleString('ru-RU') }} строк</small></template>
      <template v-else><strong>Файл не открыт</strong><span>{{ props.errorMessage }}</span></template>
    </div>
    <main class="dashboard">
      <aside class="left-column" aria-label="Состояние колоний">
        <div class="section-intro"><span class="eyebrow">ОБЗОР МАТЧА</span><h1>{{ playback.metadata.subtitle }}</h1><p>{{ playback.metadata.world.nodes.length }} узлов · {{ playback.metadata.world.edges.length }} связей · {{ playback.metadata.ticksExecuted }} тиков в записи</p></div>
        <div class="colony-stack">
          <ColonyPanel v-for="side in ['A', 'B'] as const" :key="side" :colony="frame.colonies[side]" :previous="previousFrame?.colonies[side]" :selected="selection.selectedSide.value === side" @select="selection.selectedSide.value = $event" />
        </div>
        <div class="match-card">
          <div class="match-card-top"><span class="eyebrow">ИСХОД В ЗАПИСИ</span><span class="match-icon">◎</span></div>
          <template v-if="playback.metadata.terminal && playback.cursor.value === playback.metadata.terminal.cursor">
            <strong class="outcome-title" :class="`outcome-${terminalOutcome?.tone}`">{{ terminalOutcome?.title }}</strong>
            <span class="outcome-note">{{ terminalOutcome?.detail }}</span>
          </template>
          <template v-else-if="!playback.metadata.finished"><strong class="outcome-title outcome-neutral">Запись остановлена</strong><span class="outcome-note">Матч не завершён · победитель не записан</span></template>
          <template v-else><strong class="outcome-title outcome-pending">Просмотр до терминального кадра</strong><span class="outcome-note">Результат загружен, кадр ещё не достигнут</span></template>
          <div class="match-meta"><span>{{ playback.metadata.seedLabel }}</span><span>{{ playback.metadata.policies.A }} / {{ playback.metadata.policies.B }}</span></div>
        </div>
        <div class="keyboard-hint"><span>⌨</span><span>SPACE</span><small>старт / пауза</small><span>← →</span><small>шаг по кадрам</small></div>
      </aside>

      <section class="center-column" aria-label="Карта и временная шкала">
        <div class="world-card panel">
          <div class="world-heading">
            <div class="world-title"><span class="eyebrow">КАРТА ТЕРРИТОРИИ <i class="live-tick"></i></span><h2>Наблюдение за колониями</h2></div>
            <div class="world-frame"><span>КАДР</span><strong>{{ String(playback.cursor.value).padStart(2, '0') }}</strong><span>/ {{ String(playback.metadata.maxCursor).padStart(2, '0') }}</span></div>
          </div>
        <WorldMap :frame="frame" :world="playback.metadata.world" :source-kind="playback.metadata.sourceKind" :selected-node-id="selection.selectedNodeId.value" @select-node="selection.selectedNodeId.value = $event" />
        </div>
        <TimelineControls :metadata="playback.metadata" :frame="frame" :cursor="playback.cursor.value" :playing="playback.isPlaying.value" :speed="playback.speed.value" :speeds="playback.speeds" @toggle="playback.isPlaying.value ? playback.pause() : playback.play()" @seek="playback.seek" @step="playback.step" @start="playback.goToStart" @end="playback.goToEnd" @speed="playback.setSpeed" />
      </section>

      <aside class="right-column" aria-label="События и инспектор">
        <EventFeed :source="props.source" :current-cursor="playback.cursor.value" @select-event="chooseEvent" />
        <NodeInspector :node="selectedNode" :frame="frame" :world="playback.metadata.world" :selected-side="selection.selectedSide.value" @select-side="selection.selectedSide.value = $event" />
      </aside>
    </main>
    <footer class="app-footer"><span>FORMICMIND <b>·</b> REPLAY V1 SPECTATOR</span><span>READ-ONLY VIEWER <i></i> {{ playback.metadata.sourceLabel }}</span><span>{{ playback.metadata.fileName ?? 'Локальная демо-запись' }} · {{ playback.metadata.eventCount.toLocaleString('ru-RU') }} событий</span></footer>
  </div>
</template>
