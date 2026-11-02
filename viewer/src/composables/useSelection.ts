import { ref } from 'vue'
import type { Side } from '../domain/replay'

export function useSelection(initialNodeId = 3) {
  const selectedNodeId = ref<number | null>(initialNodeId)
  const selectedSide = ref<Side>('A')

  return { selectedNodeId, selectedSide }
}
