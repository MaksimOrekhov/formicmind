<script setup lang="ts">
import type { ReplayMetadata } from '../domain/replay'

defineProps<{ metadata: ReplayMetadata; currentCursor: number; isLoading: boolean; isDemo: boolean }>()
const emit = defineEmits<{ openReplay: []; returnDemo: [] }>()
</script>

<template>
  <header class="topbar">
    <a class="brand" href="#top" aria-label="FormicMind — к началу">
      <span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i><i></i></span>
      <span class="brand-name">formic<span>mind</span></span>
      <span class="brand-divider"></span>
      <span class="brand-section">ПОЛЕВОЙ НАБЛЮДАТЕЛЬ</span>
    </a>
    <div class="topbar-meta">
      <span class="recording-dot"></span>
      <span class="topbar-id">{{ metadata.fileName ?? metadata.id }}</span>
      <span class="topbar-divider"></span>
      <span class="topbar-date">{{ metadata.seedLabel }}</span>
    </div>
    <div class="topbar-right">
      <span class="demo-chip" :class="{ 'replay-chip': !isDemo }"><span class="demo-chip-dot"></span>{{ metadata.sourceLabel }}</span>
      <span class="cursor-chip" :aria-label="`Кадр ${String(currentCursor).padStart(2, '0')}`">КАДР {{ String(currentCursor).padStart(2, '0') }}</span>
      <button class="open-replay-button" :disabled="isLoading" @click="emit('openReplay')">{{ isLoading ? 'ПРОВЕРКА…' : 'ОТКРЫТЬ REPLAY' }}</button>
      <button v-if="!isDemo" class="return-demo-button" :disabled="isLoading" @click="emit('returnDemo')">ДЕМО</button>
    </div>
  </header>
</template>
