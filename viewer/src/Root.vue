<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import App from './App.vue'
import EcoApp from './eco/EcoApp.vue'

const hash = ref(typeof window === 'undefined' ? '' : window.location.hash)
const showL0 = computed(() => hash.value === '#/l0')
function sync() { hash.value = window.location.hash }
onMounted(() => window.addEventListener('hashchange', sync))
onUnmounted(() => window.removeEventListener('hashchange', sync))
</script>

<template>
  <nav class="root-switch" aria-label="Режим viewer">
    <a href="#/eco" :class="{ active: !showL0 }" :aria-current="!showL0 ? 'page' : undefined">Ecosystem v1 · replay E1</a>
    <a href="#/l0" :class="{ active: showL0 }" :aria-current="showL0 ? 'page' : undefined">L0 · Replay v1 (две колонии)</a>
  </nav>
  <App v-if="showL0" />
  <EcoApp v-else />
</template>
