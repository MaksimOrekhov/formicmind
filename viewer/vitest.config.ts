import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  test: {
    environment: 'jsdom',
    clearMocks: true,
  },
} as import('vite').UserConfig & { test: { environment: 'jsdom'; clearMocks: true } })
