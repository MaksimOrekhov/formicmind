import type { EcoEvent, EventKind } from './types'

export const KIND_LABEL: Record<EventKind, string> = {
  discover: 'Исследование', contact: 'Встреча', nest_found: 'Найдено гнездо', dispatch: 'Приказ', combat: 'Бой', death: 'Гибель юнитов',
  queen_damage: 'Урон матке', queen_dead: 'Гибель матки', collapse: 'Коллапс', extinct: 'Вымирание',
}
const MISSION = ['', 'разведку', 'охрану', 'атаку', 'отступление']

/** One readable line per event; the numbers are exactly the event fields. */
export function describeEvent(event: EcoEvent): string {
  const c = `колония ${event.colony}`
  const o = `колония ${event.other}`
  switch (event.kind) {
    case 'discover': return `${c} впервые побывала в узле ${event.node}`
    case 'contact': return `${c} встретила: ${o} (узел ${event.node})`
    case 'nest_found': return `${c} нашла гнездо: ${o}, узел ${event.node}, матка ${event.b === 1 ? 'жива' : 'мертва'}`
    case 'dispatch': return `${c} отправила ${event.b} юн. на ${MISSION[event.a] ?? '?'} → узел ${event.node}${event.other >= 0 ? `, цель: ${o}` : ''}`
    case 'combat': return `${c} бьёт: ${o}, узел ${event.node}: урон ${event.a}, нанесено ${event.b}`
    case 'death': return `${c} потеряла ${event.b} ${event.a === 1 ? 'солдат' : 'рабочих'} в узле ${event.node}`
    case 'queen_damage': return `Матка: ${c}, узел ${event.node}: −${event.a}, осталось ${event.b}`
    case 'queen_dead': return `Матка: ${c} погибла в узле ${event.node}${event.other >= 0 ? `; убила: ${o}` : ''}`
    case 'collapse': return `${c}: экономический коллапс`
    case 'extinct': return `${c} вымерла; ${event.a} еды стали кучей в узле ${event.node}`
  }
}

/** ``2^(b-1) <= n < 2^b``; bucket 7 means 64 or more. */
export function bucketRange(bucket: number): string {
  if (bucket <= 0) return '0'
  if (bucket >= 7) return '64+'
  return bucket === 1 ? '1' : `${2 ** (bucket - 1)}–${2 ** bucket - 1}`
}
