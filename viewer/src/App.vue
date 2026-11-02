<script setup lang="ts">
import { computed, ref, shallowRef } from 'vue'
import ReplayWorkspace from './ReplayWorkspace.vue'
import { parseReplayFile, type ParseProgress } from './data/replayJsonl'
import { mockReplay } from './data/mockReplay'
import type { ReplayDataSource } from './domain/replay'

const source = shallowRef<ReplayDataSource>(mockReplay)
const generation = ref(0)
const input = ref<HTMLInputElement | null>(null)
const isLoading = ref(false)
const progress = ref<ParseProgress | null>(null)
const errorMessage = ref('')
const isDemo = computed(() => source.value.getMetadata().sourceKind === 'demo')

function openReplay() {
  if (isLoading.value) return
  input.value?.click()
}

async function readFile(event: Event) {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  target.value = ''
  if (!file) return
  isLoading.value = true
  errorMessage.value = ''
  progress.value = { recordsProcessed: 0, totalRecords: 0, bytesProcessed: 0, totalBytes: file.size }
  try {
    const nextSource = await parseReplayFile(file, (next) => { progress.value = next })
    source.value = nextSource
    generation.value += 1
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : 'Не удалось прочитать Replay. Проверьте JSONL-файл и попробуйте ещё раз.'
  } finally {
    isLoading.value = false
  }
}

function returnToDemo() {
  if (isLoading.value) return
  source.value = mockReplay
  generation.value += 1
  errorMessage.value = ''
}
</script>

<template>
  <input ref="input" class="file-input" type="file" accept=".jsonl,application/jsonl" :disabled="isLoading" @change="readFile">
  <ReplayWorkspace :key="generation" :source="source" :is-loading="isLoading" :progress="progress" :error-message="errorMessage" :is-demo="isDemo" @open-replay="openReplay" @return-demo="returnToDemo" />
</template>
