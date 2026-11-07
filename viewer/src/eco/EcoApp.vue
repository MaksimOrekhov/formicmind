<script setup lang="ts">
import { onMounted, ref, shallowRef } from 'vue'
import EcoWorkspace from './EcoWorkspace.vue'
import { parseEcoReplayBytes, type EcoParseProgress } from './parseEcoReplay'
import type { EcoReplay } from './types'
import './eco.css'

const SAMPLE_URL = 'samples/eco_seed50004_raider3.jsonl'

const replay = shallowRef<EcoReplay | null>(null)
const generation = ref(0)
const input = ref<HTMLInputElement | null>(null)
const isLoading = ref(false)
const progress = ref<EcoParseProgress | null>(null)
const errorMessage = ref('')

async function accept(bytes: Uint8Array, name: string) {
  isLoading.value = true
  errorMessage.value = ''
  await new Promise((resolve) => setTimeout(resolve, 0)) // let the banner paint before the synchronous parse
  try {
    replay.value = parseEcoReplayBytes(bytes, name, (p) => { progress.value = p })
    generation.value += 1
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : 'Не удалось прочитать E1 replay.'
  } finally {
    isLoading.value = false
  }
}

async function readFile(event: Event) {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  target.value = ''
  if (!file) return
  await accept(new Uint8Array(await file.arrayBuffer()), file.name)
}

async function loadSample() {
  isLoading.value = true
  errorMessage.value = ''
  try {
    const response = await fetch(`${import.meta.env.BASE_URL}${SAMPLE_URL}`)
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    await accept(new Uint8Array(await response.arrayBuffer()), SAMPLE_URL.split('/').pop()!)
  } catch (error) {
    isLoading.value = false
    errorMessage.value = `Пример не загружен (${error instanceof Error ? error.message : 'ошибка'}). Откройте файл вручную: python -m formicmind.eco replay-export …`
  }
}

onMounted(() => { if (typeof fetch === 'function') void loadSample() })
</script>

<template>
  <div class="eco-app">
    <input ref="input" class="file-input" type="file" accept=".jsonl,application/jsonl" :disabled="isLoading" @change="readFile">
    <header class="eco-topbar">
      <span class="eco-brand">formic<b>mind</b></span>
      <span class="eco-title">Ecosystem v1 · зритель replay</span>
      <span v-if="replay" class="eco-chips-top">
        <span class="top-chip">{{ replay.fileName }}</span>
        <span class="top-chip ok" :title="`events_hash ${replay.eventsHash}`">цепочка ✓ · config ✓ · карта ✓ · события ✓</span>
        <span class="top-chip" :title="`chain_head ${replay.chainHead}`">hash состояния: из Python (python -m formicmind.eco replay-verify)</span>
      </span>
      <span class="eco-top-actions">
        <button type="button" :disabled="isLoading" @click="loadSample">Пример: seed 50004</button>
        <button type="button" :disabled="isLoading" @click="input?.click()">{{ isLoading ? 'Проверка…' : 'Открыть файл' }}</button>
      </span>
    </header>
    <div v-if="isLoading || errorMessage" class="eco-banner" :class="{ error: errorMessage }" role="status" aria-live="polite">
      <template v-if="isLoading"><span class="spin"></span>Проверяю replay: цепочку, хэши конфигурации, карты и событий…</template>
      <template v-else><strong>Файл не открыт.</strong> {{ errorMessage }}</template>
    </div>
    <EcoWorkspace v-if="replay" :key="generation" :replay="replay" />
    <div v-else-if="!isLoading && !errorMessage" class="eco-empty-state">Откройте E1 replay (.jsonl), созданный командой <code>python -m formicmind.eco replay-export</code>, или загрузите пример.</div>
  </div>
</template>
