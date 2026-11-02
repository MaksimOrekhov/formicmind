<script setup lang="ts">
import type { ReplayMetadata } from '../domain/replay'

defineProps<{ metadata: ReplayMetadata; currentCursor: number }>()
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
      <span class="topbar-id">{{ metadata.id }}</span>
      <span class="topbar-divider"></span>
      <span class="topbar-date">{{ metadata.recordedAt }}</span>
    </div>
    <div class="topbar-right">
      <span class="demo-chip"><span class="demo-chip-dot"></span>{{ metadata.sourceLabel }}</span>
      <span class="cursor-chip" :aria-label="`Кадр ${String(currentCursor).padStart(2, '0')}`">КАДР {{ String(currentCursor).padStart(2, '0') }}</span>
    </div>
  </header>
</template>
