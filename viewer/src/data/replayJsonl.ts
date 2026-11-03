import type {
  Caste, ColonyView, EventCategory, EventKind, EventQuery, NodeGroups, NodeView, ReplayBoundary,
  ReplayDataSource, ReplayMetadata, ReplayTerminal, Side, TimelineFrame, TransitGroupView, ViewerEvent,
  WorldEdgeDefinition, WorldNodeDefinition,
} from '../domain/replay'
import { concatBytes, hashDomain, utf8 } from './replayCrypto'

export const MAX_REPLAY_BYTES = 10 * 1024 * 1024
const MAX_RECORDS = 50_000
const MAX_LINE_BYTES = 1024 * 1024
const ZERO_CHAIN = '0'.repeat(64)
const FORMAT = 'formicmind.replay'
const SCHEMA = 'v1'
const FORMAT_DOMAIN = 'formicmind/replay-record/v1'
const VIEW_DOMAIN = 'formicmind/replay-view/v1'
const CONFIG_DOMAIN = 'formicmind/config/v1'
const CHECKPOINT_DOMAIN = 'formicmind/checkpoint/v3'
const EVENT_DOMAIN = 'formicmind/events/v1'

const CONTRACT: Record<string, string> = {
  rules_version: 'l0-proto-FOR1-v0.2.2',
  balance_version: 'candidate-FOR1-v0.2.1-unvalidated',
  observation_version: 'FOR-2-v0.2+errata-v0.2.1',
  action_version: 'ActionV2',
  decoder_version: 'FOR-2-decoder-v0.2.1',
  serialization_version: 'formicmind-canonical-json-v1',
  rng_version: 'pcg32-xsh-rr-64-32-v1',
  stream_derivation_version: 'sha256-lenprefixed-v1',
  hash_domain_config: CONFIG_DOMAIN,
  hash_domain_state: 'formicmind/state/v1',
  hash_domain_checkpoint: CHECKPOINT_DOMAIN,
  hash_domain_events: EVENT_DOMAIN,
  hash_domain_rng_stream: 'formicmind/rng-stream/v1',
  hash_domain_observation: 'formicmind/observation-f32le/v1',
  hash_domain_view: VIEW_DOMAIN,
  hash_domain_record: FORMAT_DOMAIN,
}

const BALANCE_FIELDS = [
  'start_food', 'start_workers', 'start_soldiers', 'worker_hp', 'soldier_hp', 'queen_hp', 'worker_egg_cost',
  'soldier_egg_cost', 'eggs_per_lay', 'lay_interval_ticks', 'worker_hatch_delay', 'soldier_hatch_delay',
  'worker_upkeep_tenths', 'soldier_upkeep_tenths', 'starvation_unit_cost', 'soldier_damage', 'home_defense_num',
  'home_defense_den',
]
const EVENT_SCHEMAS: Record<EventKind, { rank: number; phases: number[]; fields: Record<string, 'i' | 'b' | 's' | 'I' | 'J'> }> = {
  order: { rank: 0, phases: [0], fields: { tick: 'i', phase: 'i', side: 'i', lay_order: 'i', forage_share_q: 'I', army_choice: 'i', avoid_center: 'b', army_target_node: 'i', worker_quota: 'I' } },
  retarget: { rank: 1, phases: [0], fields: { tick: 'i', phase: 'i', side: 'i', unit_id: 'i', caste: 'i', old_target: 'i', new_target: 'i' } },
  depart: { rank: 2, phases: [0, 9], fields: { tick: 'i', phase: 'i', side: 'i', unit_id: 'i', caste: 'i', from_node: 'i', to_node: 'i', arrival_tick: 'i', target_node: 'i' } },
  arrive: { rank: 3, phases: [2], fields: { tick: 'i', phase: 'i', side: 'i', unit_id: 'i', caste: 'i', node: 'i', from_node: 'i', target_node: 'i' } },
  hatch: { rank: 4, phases: [1], fields: { tick: 'i', phase: 'i', side: 'i', unit_id: 'i', caste: 'i', node: 'i', hp: 'i', laid_tick: 'i', target_node: 'i' } },
  harvest: { rank: 5, phases: [5], fields: { tick: 'i', phase: 'i', side: 'i', node: 'i', workers: 'i', demand: 'i', blocked: 'b', claim: 'i', harvested: 'i', stock_before: 'i', stock_after: 'i' } },
  upkeep: { rank: 6, phases: [6], fields: { tick: 'i', phase: 'i', side: 'i', adult_workers: 'i', adult_soldiers: 'i', remainder_before: 'i', remainder_after: 'i', due: 'i', paid: 'i', missing: 'i', food_before: 'i', food_after: 'i', debt_before: 'i', debt_after: 'i', deaths: 'i', debt_reset: 'b' } },
  starvation: { rank: 7, phases: [6], fields: { tick: 'i', phase: 'i', side: 'i', seq: 'i', unit_id: 'i', caste: 'i', hp: 'i', node: 'i', to_node: 'i', debt_before: 'i', debt_after: 'i' } },
  lay: { rank: 8, phases: [7], fields: { tick: 'i', phase: 'i', side: 'i', laid: 'b', caste: 'i', eggs: 'i', unit_cost: 'i', total_cost: 'i', food_before: 'i', food_after: 'i', unit_ids: 'I', hatch_tick: 'i', next_lay_tick: 'i' } },
  regenerate: { rank: 9, phases: [8], fields: { tick: 'i', phase: 'i', node: 'i', before: 'i', regen: 'i', added: 'i', after: 'i', capacity: 'i' } },
  combat: { rank: 10, phases: [3], fields: { tick: 'i', phase: 'i', node: 'i', attacker_side: 'i', defender_side: 'i', attacker_soldiers: 'i', base_damage: 'i', home_bonus: 'b', damage: 'i', target: 's', target_units: 'i', target_hp_total: 'i', applied: 'i', wasted: 'i' } },
  wound: { rank: 11, phases: [3], fields: { tick: 'i', phase: 'i', side: 'i', unit_id: 'i', caste: 'i', node: 'i', attacker_side: 'i', hp_before: 'i', damage: 'i', hp_after: 'i' } },
  death: { rank: 12, phases: [3], fields: { tick: 'i', phase: 'i', side: 'i', unit_id: 'i', caste: 'i', node: 'i', attacker_side: 'i', hp_before: 'i', damage: 'i', hp_after: 'i' } },
  queen_damage: { rank: 13, phases: [3], fields: { tick: 'i', phase: 'i', side: 'i', node: 'i', attacker_side: 'i', damage: 'i', queen_hp_before: 'i', queen_hp_after: 'i' } },
  terminal: { rank: 14, phases: [4, 10], fields: { tick: 'i', phase: 'i', reason: 's', winner: 's', sides: 'I', phases_executed: 'i', queen_hp: 'J', tick_completed: 'b' } },
}
const SORT_TAIL: Partial<Record<EventKind, [string, string]>> = {
  order: ['', 'side'], retarget: ['unit_id', 'side'], depart: ['unit_id', 'side'], arrive: ['unit_id', 'side'],
  hatch: ['unit_id', 'side'], harvest: ['node', 'side'], upkeep: ['', 'side'], starvation: ['side', 'seq'],
  lay: ['', 'side'], regenerate: ['node', ''], combat: ['node', 'attacker_side'], wound: ['unit_id', 'side'],
  death: ['unit_id', 'side'], queen_damage: ['', 'side'], terminal: ['', ''],
}

export class ReplayFormatError extends Error {
  constructor(message: string) { super(message); this.name = 'ReplayFormatError' }
}
export class ReplayVersionError extends ReplayFormatError {
  constructor(message: string) { super(message); this.name = 'ReplayVersionError' }
}

export interface ParseProgress {
  recordsProcessed: number
  totalRecords: number
  bytesProcessed: number
  totalBytes: number
}

export interface ReplayFileLike {
  name: string
  size: number
  arrayBuffer(): Promise<ArrayBuffer>
}

export function canonicalString(value: unknown): string {
  if (typeof value === 'string') {
    return JSON.stringify(value).replace(/[\u0080-\uFFFF]/g, (char) => `\\u${char.charCodeAt(0).toString(16).padStart(4, '0')}`)
  }
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (typeof value === 'number') {
    if (!Number.isSafeInteger(value)) throw new ReplayFormatError('JSON Replay разрешает только безопасные целые числа.')
    return Object.is(value, -0) ? '0' : String(value)
  }
  if (Array.isArray(value)) return `[${value.map(canonicalString).join(',')}]`
  if (value && typeof value === 'object') {
    const keys = Object.keys(value).sort(compareUnicode)
    return `{${keys.map((key) => `${canonicalString(key)}:${canonicalString((value as Record<string, unknown>)[key])}`).join(',')}}`
  }
  throw new ReplayFormatError('В JSON Replay обнаружен неподдерживаемый тип значения.')
}

function compareUnicode(left: string, right: string) {
  const a = Array.from(left, (character) => character.codePointAt(0)!)
  const b = Array.from(right, (character) => character.codePointAt(0)!)
  const length = Math.min(a.length, b.length)
  for (let i = 0; i < length; i += 1) if (a[i] !== b[i]) return a[i] - b[i]
  return a.length - b.length
}

type JsonObject = Record<string, unknown>
function object(value: unknown, where: string): JsonObject {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new ReplayFormatError(`${where}: ожидался JSON-объект.`)
  return value as JsonObject
}
function exactKeys(value: unknown, keys: string[], where: string): JsonObject {
  const found = object(value, where)
  const expected = new Set(keys)
  const actual = Object.keys(found)
  if (actual.length !== keys.length || actual.some((key) => !expected.has(key))) {
    const missing = keys.filter((key) => !Object.hasOwn(found, key))
    throw new ReplayFormatError(`${where}: неверный набор полей${missing.length ? `; отсутствуют ${missing.join(', ')}` : ''}.`)
  }
  return found
}
function int(value: unknown, where: string, min?: number, max?: number): number {
  if (!Number.isSafeInteger(value) || (min !== undefined && (value as number) < min) || (max !== undefined && (value as number) > max)) {
    throw new ReplayFormatError(`${where}: ожидалось целое число${min === undefined ? '' : ` не меньше ${min}`}.`)
  }
  return value as number
}
function str(value: unknown, where: string): string {
  if (typeof value !== 'string') throw new ReplayFormatError(`${where}: ожидалась строка.`)
  return value
}
function bool(value: unknown, where: string): boolean {
  if (typeof value !== 'boolean') throw new ReplayFormatError(`${where}: ожидалось логическое значение.`)
  return value
}
function array(value: unknown, where: string): unknown[] {
  if (!Array.isArray(value)) throw new ReplayFormatError(`${where}: ожидался массив.`)
  return value
}
function same(left: unknown, right: unknown) {
  if (left === undefined || right === undefined) return left === right
  return canonicalString(left) === canonicalString(right)
}
function hash64(value: unknown, where: string): string {
  const found = str(value, where)
  if (!/^[0-9a-f]{64}$/.test(found)) throw new ReplayFormatError(`${where}: ожидался SHA-256 в hex.`)
  return found
}

interface ReplayConfig {
  tMax: number
  policyInterval: number
  queenHp: number
  nodes: WorldNodeDefinition[]
  edges: WorldEdgeDefinition[]
  capacities: number[]
  raw: JsonObject
}

function validateConfig(value: unknown): ReplayConfig {
  const config = exactKeys(value, ['schema', 'serialization_version', 'rules_version', 'balance_version', 'policy_interval_k', 't_max', 'topology', 'balance'], 'header.config')
  if (config.schema !== 'formicmind.sim_config/1' || config.serialization_version !== CONTRACT.serialization_version || config.rules_version !== CONTRACT.rules_version || config.balance_version !== CONTRACT.balance_version) {
    throw new ReplayVersionError('Конфигурация Replay создана несовместимой версией FormicMind.')
  }
  const topology = exactKeys(config.topology, ['nodes', 'edges'], 'header.config.topology')
  const nodesRaw = array(topology.nodes, 'header.config.topology.nodes')
  if (nodesRaw.length !== 7) throw new ReplayFormatError('Replay содержит конфигурацию не из 7 узлов L0.')
  const nodes: WorldNodeDefinition[] = nodesRaw.map((item, index) => {
    const node = exactKeys(item, ['node_id', 'name', 'home_side'], `node ${index}`)
    if (int(node.node_id, `node ${index}.node_id`) !== index) throw new ReplayFormatError('Узлы должны идти в порядке node_id 0..6.')
    const homeSide = int(node.home_side, `node ${index}.home_side`, -1, 1)
    return { id: index, name: str(node.name, `node ${index}.name`), isHome: homeSide === -1 ? null : sideFromIndex(homeSide), capacity: 0 }
  })
  if (nodes.filter((node) => node.isHome === 'A').length !== 1 || nodes.filter((node) => node.isHome === 'B').length !== 1) {
    throw new ReplayFormatError('Конфигурация Replay должна задавать ровно одно гнездо для каждой колонии.')
  }
  const edgesRaw = array(topology.edges, 'header.config.topology.edges')
  const edges: WorldEdgeDefinition[] = []
  let priorEdge = -1
  for (const [index, item] of edgesRaw.entries()) {
    const edge = exactKeys(item, ['a', 'b', 'length'], `edge ${index}`)
    const a = int(edge.a, `edge ${index}.a`, 0, 6)
    const b = int(edge.b, `edge ${index}.b`, 1, 6)
    const length = int(edge.length, `edge ${index}.length`, 1)
    const key = a * 7 + b
    if (a >= b || key <= priorEdge) throw new ReplayFormatError('Рёбра должны быть уникальными и отсортированными по (a, b).')
    priorEdge = key
    edges.push({ a, b, length })
  }
  const balanceKeys = [...BALANCE_FIELDS, 'resources']
  const balance = exactKeys(config.balance, balanceKeys, 'header.config.balance')
  const resources = array(balance.resources, 'header.config.balance.resources')
  if (resources.length !== 7) throw new ReplayFormatError('Replay должен содержать ресурсную запись для каждого узла.')
  const capacities = resources.map((item, index) => {
    const resource = exactKeys(item, ['node_id', 'start_food', 'capacity', 'regen', 'max_harvest'], `resource ${index}`)
    if (int(resource.node_id, `resource ${index}.node_id`) !== index) throw new ReplayFormatError('Ресурсы должны идти в порядке node_id 0..6.')
    const capacity = int(resource.capacity, `resource ${index}.capacity`, 0)
    int(resource.start_food, `resource ${index}.start_food`, 0, capacity)
    int(resource.regen, `resource ${index}.regen`, 0)
    int(resource.max_harvest, `resource ${index}.max_harvest`, 0)
    nodes[index].capacity = capacity
    return capacity
  })
  const tMax = int(config.t_max, 'header.config.t_max', 1)
  const policyInterval = int(config.policy_interval_k, 'header.config.policy_interval_k', 1)
  const queenHp = int(balance.queen_hp, 'header.config.balance.queen_hp', 1)
  for (const field of BALANCE_FIELDS) int(balance[field], `header.config.balance.${field}`, 0)
  return { tMax, policyInterval, queenHp, nodes, edges, capacities, raw: config }
}

function sideFromIndex(value: number): Side { return value === 0 ? 'A' : 'B' }
function casteFromIndex(value: number): Caste {
  if (value === 0) return 'workers'
  if (value === 1) return 'soldiers'
  throw new ReplayFormatError(`Неизвестная каста ${value}.`)
}

interface CanonicalTerminal {
  finished: boolean
  winner?: string
  reason?: string
  terminal_tick?: number
  terminal_phase?: number
  tick_completed?: boolean
}
interface CanonicalFrame {
  tick: number
  boundary: ReplayBoundary
  terminal: CanonicalTerminal
  node_food: number[]
  colonies: Array<{ side: number; food: number; queen_hp: number; stationed: number[][]; transit: number[][]; brood: number[][] }>
}
interface CanonicalBody { node_food: number[]; colonies: CanonicalFrame['colonies'] }

function compareRowKey(a: number[], b: number[]) {
  for (let i = 0; i < Math.min(a.length, b.length); i += 1) if (a[i] !== b[i]) return a[i] - b[i]
  return a.length - b.length
}
function validateRows(value: unknown, table: 'stationed' | 'transit' | 'brood', nodeCount: number, where: string) {
  const widths = { stationed: [2, 2], transit: [4, 2], brood: [2, 1] } as const
  const [keyWidth, valueWidth] = widths[table]
  let previous: number[] | null = null
  return array(value, `${where}.${table}`).map((item, index) => {
    const row = array(item, `${where}.${table}[${index}]`).map((field, at) => int(field, `${where}.${table}[${index}][${at}]`))
    if (row.length !== keyWidth + valueWidth) throw new ReplayFormatError(`${where}.${table}: строка неверной длины.`)
    const key = row.slice(0, keyWidth)
    if (previous && compareRowKey(key, previous) <= 0) throw new ReplayFormatError(`${where}.${table}: ключи должны быть уникальными и отсортированными.`)
    previous = key
    if (row.slice(keyWidth).some((field) => field < 0)) throw new ReplayFormatError(`${where}.${table}: отрицательное значение.`)
    if (table === 'stationed' && (row[0] < 0 || row[0] >= nodeCount)) throw new ReplayFormatError(`${where}.stationed: неизвестный узел.`)
    if (table === 'transit' && (row[0] < 0 || row[0] >= nodeCount || row[1] < 0 || row[1] >= nodeCount)) throw new ReplayFormatError(`${where}.transit: неизвестный узел.`)
    if (table === 'stationed' && row[1] !== 0 && row[1] !== 1 || table === 'transit' && row[3] !== 0 && row[3] !== 1 || table === 'brood' && row[0] !== 0 && row[0] !== 1) {
      throw new ReplayFormatError(`${where}.${table}: неизвестная каста.`)
    }
    return row
  })
}

function validateTerminal(value: unknown, where: string): CanonicalTerminal {
  const terminal = object(value, where)
  if (same(terminal, { finished: false })) return { finished: false }
  exactKeys(terminal, ['finished', 'winner', 'reason', 'terminal_tick', 'terminal_phase', 'tick_completed'], where)
  if (terminal.finished !== true) throw new ReplayFormatError(`${where}: неверный finished.`)
  if (!['A', 'B', 'draw'].includes(str(terminal.winner, `${where}.winner`))) throw new ReplayFormatError(`${where}: неизвестный winner.`)
  if (!['queen_death', 'economic_collapse', 'timeout'].includes(str(terminal.reason, `${where}.reason`))) throw new ReplayFormatError(`${where}: неизвестный reason.`)
  const result: CanonicalTerminal = {
    finished: true,
    winner: terminal.winner as string,
    reason: terminal.reason as string,
    terminal_tick: int(terminal.terminal_tick, `${where}.terminal_tick`, 0),
    terminal_phase: int(terminal.terminal_phase, `${where}.terminal_phase`, 0, 10),
    tick_completed: bool(terminal.tick_completed, `${where}.tick_completed`),
  }
  return result
}

function validateFrame(value: unknown, config: ReplayConfig, where: string): CanonicalFrame {
  const frame = exactKeys(value, ['tick', 'boundary', 'terminal', 'node_food', 'colonies'], where)
  const tick = int(frame.tick, `${where}.tick`, 0, config.tMax)
  if (frame.boundary !== 'start' && frame.boundary !== 'terminal') throw new ReplayFormatError(`${where}.boundary: неизвестная граница кадра.`)
  const terminal = validateTerminal(frame.terminal, `${where}.terminal`)
  if ((frame.boundary === 'terminal') !== terminal.finished) throw new ReplayFormatError(`${where}: terminal boundary должен совпадать с terminal объектом.`)
  const nodeFood = array(frame.node_food, `${where}.node_food`).map((food, index) => int(food, `${where}.node_food[${index}]`, 0, config.capacities[index]))
  if (nodeFood.length !== config.nodes.length) throw new ReplayFormatError(`${where}.node_food: ожидалось 7 запасов узлов.`)
  const colonies = array(frame.colonies, `${where}.colonies`)
  if (colonies.length !== 2) throw new ReplayFormatError(`${where}.colonies: ожидались две колонии.`)
  const mapped = colonies.map((value, side) => {
    const colony = exactKeys(value, ['side', 'food', 'queen_hp', 'stationed', 'transit', 'brood'], `${where}.colonies[${side}]`)
    if (int(colony.side, `${where}.colonies[${side}].side`) !== side) throw new ReplayFormatError(`${where}.colonies: стороны должны идти A, B.`)
    return {
      side,
      food: int(colony.food, `${where}.colonies[${side}].food`, 0),
      queen_hp: int(colony.queen_hp, `${where}.colonies[${side}].queen_hp`, 0),
      stationed: validateRows(colony.stationed, 'stationed', config.nodes.length, `${where}.colonies[${side}]`),
      transit: validateRows(colony.transit, 'transit', config.nodes.length, `${where}.colonies[${side}]`),
      brood: validateRows(colony.brood, 'brood', config.nodes.length, `${where}.colonies[${side}]`),
    }
  })
  return { tick, boundary: frame.boundary, terminal, node_food: nodeFood, colonies: mapped }
}

function cloneBody(body: CanonicalBody): CanonicalBody {
  return { node_food: body.node_food.slice(), colonies: body.colonies.map((colony) => ({
    ...colony, stationed: colony.stationed.map((row) => row.slice()), transit: colony.transit.map((row) => row.slice()), brood: colony.brood.map((row) => row.slice()),
  })) }
}
function bodyOf(frame: CanonicalFrame): CanonicalBody { return { node_food: frame.node_food, colonies: frame.colonies } }
function frameHash(frame: CanonicalFrame) { return hashDomain(VIEW_DOMAIN, utf8(canonicalString(frame))) }

function applyPatch(bodyValue: CanonicalBody, value: unknown, config: ReplayConfig, where: string): CanonicalBody {
  const patch = object(value, where)
  if (!Object.hasOwn(patch, 'c') || Object.keys(patch).some((key) => key !== 'c' && key !== 'nf')) throw new ReplayFormatError(`${where}: неизвестное поле патча.`)
  const body = cloneBody(bodyValue)
  const nf = patch.nf === undefined ? [] : array(patch.nf, `${where}.nf`)
  let previousIndex = -1
  for (const [index, item] of nf.entries()) {
    const row = array(item, `${where}.nf[${index}]`)
    if (row.length !== 2) throw new ReplayFormatError(`${where}.nf: неверная строка.`)
    const nodeIndex = int(row[0], `${where}.nf[${index}][0]`, 0, config.nodes.length - 1)
    const food = int(row[1], `${where}.nf[${index}][1]`, 0, config.capacities[nodeIndex])
    if (nodeIndex <= previousIndex) throw new ReplayFormatError(`${where}.nf: индексы должны быть уникальными и отсортированными.`)
    body.node_food[nodeIndex] = food
    previousIndex = nodeIndex
  }
  const entries = array(patch.c, `${where}.c`)
  if (entries.length !== 2) throw new ReplayFormatError(`${where}.c: ожидались изменения для A и B.`)
  const widths = { stationed: 2, transit: 4, brood: 2 } as const
  for (let side = 0; side < 2; side += 1) {
    const entry = object(entries[side], `${where}.c[${side}]`)
    if (Object.keys(entry).some((key) => !['food', 'queen_hp', 'stationed', 'transit', 'brood'].includes(key))) throw new ReplayFormatError(`${where}.c[${side}]: неизвестное поле.`)
    const colony = body.colonies[side]
    if (Object.hasOwn(entry, 'food')) colony.food = int(entry.food, `${where}.c[${side}].food`, 0)
    if (Object.hasOwn(entry, 'queen_hp')) colony.queen_hp = int(entry.queen_hp, `${where}.c[${side}].queen_hp`, 0)
    for (const table of ['stationed', 'transit', 'brood'] as const) {
      if (!Object.hasOwn(entry, table)) continue
      const changes = exactKeys(entry[table], ['set', 'del'], `${where}.c[${side}].${table}`)
      const keyWidth = widths[table]
      const rows = new Map(colony[table].map((row) => [row.slice(0, keyWidth).join(','), row]))
      const deleted = new Set<string>()
      let previous: number[] | null = null
      for (const [index, item] of array(changes.del, `${where}.c[${side}].${table}.del`).entries()) {
        const key = array(item, `${where}.c[${side}].${table}.del[${index}]`).map((part, partIndex) => int(part, `${where}.del[${index}][${partIndex}]`))
        if (key.length !== keyWidth || previous && compareRowKey(key, previous) <= 0) throw new ReplayFormatError(`${where}.c[${side}].${table}.del: ключи неверны или не отсортированы.`)
        const joined = key.join(',')
        if (!rows.has(joined)) throw new ReplayFormatError(`${where}.c[${side}].${table}.del: удаляется отсутствующая строка.`)
        rows.delete(joined)
        deleted.add(joined)
        previous = key
      }
      const setRows = validateRows(changes.set, table, config.nodes.length, `${where}.c[${side}]`)
      for (const row of setRows) {
        const joined = row.slice(0, keyWidth).join(',')
        if (deleted.has(joined)) throw new ReplayFormatError(`${where}.c[${side}].${table}: ключ одновременно удаляется и задаётся.`)
        rows.set(joined, row)
      }
      colony[table] = [...rows.values()].sort((a, b) => compareRowKey(a.slice(0, keyWidth), b.slice(0, keyWidth)))
    }
  }
  return body
}

function validateEvent(value: unknown, expectedTick: number, config: ReplayConfig, where: string) {
  const event = object(value, where)
  const kind = str(event.kind, `${where}.kind`) as EventKind
  const schema = EVENT_SCHEMAS[kind]
  if (!schema) throw new ReplayFormatError(`${where}: неизвестный вид события ${kind}.`)
  exactKeys(event, ['kind', ...Object.keys(schema.fields)], where)
  for (const [field, type] of Object.entries(schema.fields)) {
    const fieldValue = event[field]
    if (type === 'i') int(fieldValue, `${where}.${field}`)
    else if (type === 'b') bool(fieldValue, `${where}.${field}`)
    else if (type === 's') str(fieldValue, `${where}.${field}`)
    else {
      const values = array(fieldValue, `${where}.${field}`)
      if (type === 'J' && values.length !== 2) throw new ReplayFormatError(`${where}.${field}: ожидалось два значения.`)
      values.forEach((part, index) => int(part, `${where}.${field}[${index}]`))
    }
  }
  if (event.tick !== expectedTick) throw new ReplayFormatError(`${where}: событие относится к тику ${event.tick}, ожидался ${expectedTick}.`)
  if (!schema.phases.includes(event.phase as number)) throw new ReplayFormatError(`${where}: невозможная фаза для ${kind}.`)
  if ('side' in event && event.side !== 0 && event.side !== 1) throw new ReplayFormatError(`${where}: side должен быть 0 или 1.`)
  if (kind === 'terminal') {
    if (event.phases_executed !== (event.phase as number) + 1 || event.tick_completed !== (event.phase === 10)) throw new ReplayFormatError(`${where}: неверные terminal phase/tick_completed.`)
    const sides = event.sides as number[]
    if (sides.some((side) => side !== 0 && side !== 1) || same(sides, [...new Set(sides)].sort((a, b) => a - b)) === false) throw new ReplayFormatError(`${where}: terminal.sides должны быть уникальным списком 0/1.`)
    if (!['A', 'B', 'draw'].includes(event.winner as string) || !['queen_death', 'economic_collapse', 'timeout'].includes(event.reason as string)) throw new ReplayFormatError(`${where}: неизвестные terminal winner/reason.`)
    if (event.phase === 4 && event.reason !== 'queen_death' || event.phase === 10 && event.reason === 'queen_death') throw new ReplayFormatError(`${where}: reason не соответствует terminal phase.`)
    if (event.reason === 'timeout' && (event.phase !== 10 || expectedTick !== config.tMax - 1 || event.winner !== 'draw')) throw new ReplayFormatError(`${where}: timeout должен завершать последний тик ничьей.`)
  }
  for (const field of ['node', 'from_node', 'to_node'] as const) {
    if (field in event) {
      const minimum = kind === 'starvation' && field === 'to_node' ? -1 : 0
      int(event[field], `${where}.${field}`, minimum, config.nodes.length - 1)
    }
  }
  const tail = SORT_TAIL[kind]!
  const key = [event.phase as number, schema.rank, ...(tail.map((field) => field ? event[field] as number : -1))]
  return { raw: event, kind, sortKey: key }
}

function compareSortKey(a: number[], b: number[]) { return compareRowKey(a, b) }

function nodeName(config: ReplayConfig, nodeId: number) { return config.nodes[nodeId]?.name ?? `#${nodeId}` }
function casteName(caste: unknown) { return caste === 0 ? 'рабочий' : caste === 1 ? 'солдат' : `каста ${String(caste)}` }
function sideName(side: unknown) { return side === 0 ? 'A' : side === 1 ? 'B' : 'колония' }
function reasonName(reason: unknown) {
  return reason === 'queen_death' ? 'гибель матки' : reason === 'economic_collapse' ? 'экономический коллапс' : reason === 'timeout' ? 'таймаут' : String(reason)
}

function eventPresentation(raw: JsonObject, index: number, tick: number, config: ReplayConfig): Omit<ViewerEvent, 'cursor'> {
  const kind = raw.kind as EventKind
  const side = typeof raw.side === 'number' ? sideFromIndex(raw.side) : null
  const actor = side ? `Колония ${side}` : 'Матч'
  const nodeId = typeof raw.node === 'number' ? raw.node
    : typeof raw.from_node === 'number' ? raw.from_node
      : typeof raw.to_node === 'number' ? raw.to_node
        : typeof raw.army_target_node === 'number' ? raw.army_target_node
          : typeof raw.new_target === 'number' ? raw.new_target : null
  const category: EventCategory = kind === 'terminal' ? 'terminal'
    : ['combat', 'wound', 'death', 'queen_damage'].includes(kind) ? 'combat'
      : ['retarget', 'depart', 'arrive', 'order'].includes(kind) ? 'movement' : 'economy'
  let summary = `${actor}: ${kind}`
  let detail = `Тик ${tick}; поля события соответствуют записи.`
  switch (kind) {
    case 'order': summary = `Приказ колонии ${sideName(raw.side)}`; detail = `Цель армии: ${nodeName(config, raw.army_target_node as number)}; квоты рабочих записаны в событии.`; break
    case 'retarget': summary = `Изменена цель отряда ${sideName(raw.side)}`; detail = `Юнит ${raw.unit_id}: ${nodeName(config, raw.old_target as number)} → ${nodeName(config, raw.new_target as number)}.`; break
    case 'depart': summary = `Отряд ${sideName(raw.side)} вышел на ребро`; detail = `Юнит ${raw.unit_id}: ${nodeName(config, raw.from_node as number)} → ${nodeName(config, raw.to_node as number)}; прибытие в тик ${raw.arrival_tick}.`; break
    case 'arrive': summary = `Отряд ${sideName(raw.side)} прибыл в ${nodeName(config, raw.node as number)}`; detail = `Юнит ${raw.unit_id}; отправился из ${nodeName(config, raw.from_node as number)}.`; break
    case 'hatch': summary = `Вылупился ${casteName(raw.caste)}`; detail = `${sideName(raw.side)}, узел ${nodeName(config, raw.node as number)}, HP ${raw.hp}; кладка в тике ${raw.laid_tick}.`; break
    case 'harvest': summary = `Сбор ресурсов в ${nodeName(config, raw.node as number)}`; detail = `${sideName(raw.side)}: собрано ${raw.harvested} из ${raw.demand}; запас ${raw.stock_before} → ${raw.stock_after}.`; break
    case 'upkeep': summary = `Содержание колонии ${sideName(raw.side)}`; detail = `${raw.adult_workers} рабочих, ${raw.adult_soldiers} солдат; оплачено ${raw.paid} из ${raw.due}; еда ${raw.food_before} → ${raw.food_after}; потери ${raw.deaths}.`; break
    case 'starvation': summary = `Потеря от голода · ${casteName(raw.caste)}`; detail = `${sideName(raw.side)}, узел ${nodeName(config, raw.node as number)}, юнит ${raw.unit_id}.`; break
    case 'lay': summary = raw.laid ? `Кладка колонии ${sideName(raw.side)}` : `Кладка пропущена · ${sideName(raw.side)}`; detail = `${raw.eggs} яиц; стоимость ${raw.total_cost}; пища ${raw.food_before} → ${raw.food_after}; вылупление в тике ${raw.hatch_tick}.`; break
    case 'regenerate': summary = `Восстановление ресурса · ${nodeName(config, raw.node as number)}`; detail = `Запас ${raw.before} → ${raw.after}; добавлено ${raw.added} из ${raw.regen}.`; break
    case 'combat': summary = `Бой в узле ${nodeName(config, raw.node as number)}`; detail = `Атаковали ${sideName(raw.attacker_side)}; цель ${raw.target}; урон ${raw.applied} из ${raw.damage}, потери урона ${raw.wasted}.`; break
    case 'wound': summary = `Ранен ${casteName(raw.caste)} колонии ${sideName(raw.side)}`; detail = `Юнит ${raw.unit_id} в ${nodeName(config, raw.node as number)}: HP ${raw.hp_before} → ${raw.hp_after}; урон ${raw.damage}.`; break
    case 'death': summary = `Погиб ${casteName(raw.caste)} колонии ${sideName(raw.side)}`; detail = `Юнит ${raw.unit_id} в ${nodeName(config, raw.node as number)}; урон ${raw.damage}.`; break
    case 'queen_damage': summary = `Урон матке ${sideName(raw.side)}`; detail = `Узел ${nodeName(config, raw.node as number)}: HP ${raw.queen_hp_before} → ${raw.queen_hp_after}; урон ${raw.damage}.`; break
    case 'terminal': summary = raw.winner === 'draw' ? 'Ничья в матче' : `Победа колонии ${raw.winner}`; detail = `${reasonName(raw.reason)}; terminal tick ${tick}; тик ${raw.tick_completed ? 'завершён' : 'прерван'}.`; break
  }
  return { id: `event-${index}`, tick, category, kind, side, nodeId, summary, detail }
}

function framePresentation(frame: CanonicalFrame, cursor: number, config: ReplayConfig, frameEvents: readonly ViewerEvent[]): TimelineFrame {
  const groupsForNode = new Map<number, NodeGroups>(config.nodes.map((node) => [node.id, {
    A: { workers: 0, soldiers: 0 }, B: { workers: 0, soldiers: 0 },
  }]))
  const colonies = {} as Record<Side, ColonyView>
  const transits: TransitGroupView[] = []
  for (const colony of frame.colonies) {
    const side = sideFromIndex(colony.side)
    let workers = 0
    let soldiers = 0
    let brood = 0
    for (const [node, caste, count] of colony.stationed) {
      const groups = groupsForNode.get(node)![side]
      if (caste === 0) { groups.workers += count; workers += count } else { groups.soldiers += count; soldiers += count }
    }
    for (const [from, to, arrivalTick, caste, count] of colony.transit) {
      if (caste === 0) workers += count
      else soldiers += count
      transits.push({
        id: `${side}-${from}-${to}-${arrivalTick}-${caste}`,
        side,
        caste: casteFromIndex(caste),
        count,
        from,
        to,
        arrivalTick,
        progress: 0.5,
        positionIsApproximate: true,
        etaTicks: Math.max(0, arrivalTick - frame.tick),
      })
    }
    for (const [, , count] of colony.brood) brood += count
    colonies[side] = {
      side, queenHp: colony.queen_hp, queenMaxHp: config.queenHp, food: colony.food,
      workers, soldiers, brood, queenAlive: colony.queen_hp > 0,
    }
  }
  const nodes: NodeView[] = config.nodes.map((node) => ({
    id: node.id, name: node.name, food: frame.node_food[node.id], capacity: node.capacity,
    groups: groupsForNode.get(node.id)!, isHome: node.isHome,
    ...(frameEvents.some((event) => event.kind === 'combat' && event.nodeId === node.id) ? { alert: 'contact' as const } : {}),
  }))
  return {
    cursor,
    state: { tick: frame.tick, tickCompleted: frame.terminal.finished ? Boolean(frame.terminal.tick_completed) : true, boundary: frame.boundary },
    colonies,
    nodes,
    transits,
  }
}

function terminalPresentation(terminal: CanonicalTerminal, cursor: number, stateTick: number): ReplayTerminal {
  if (!terminal.finished) throw new ReplayFormatError('Неожиданно создан terminal DTO для незавершённой записи.')
  const winner = terminal.winner === 'A' || terminal.winner === 'B' ? terminal.winner : null
  const outcome = terminal.winner === 'draw' || terminal.reason === 'timeout' ? 'draw' : 'win'
  return {
    cursor,
    terminalTick: terminal.terminal_tick!,
    stateTick,
    tickCompleted: Boolean(terminal.tick_completed),
    stage: terminal.terminal_phase === 4 ? 'QueenTerminal' : 'TerminalLog',
    reason: terminal.reason as ReplayTerminal['reason'],
    outcome,
    winner,
  }
}

function validateDecision(record: JsonObject, tick: number, streams: string[], seatStreams: string[]) {
  exactKeys(record, ['type', 'seq', 'tick', 'seats'], `decision ${tick}`)
  if (int(record.tick, `decision ${tick}.tick`) !== tick) throw new ReplayFormatError(`decision: ожидался тик ${tick}.`)
  const seats = array(record.seats, `decision ${tick}.seats`)
  if (seats.length !== 2) throw new ReplayFormatError(`decision ${tick}: ожидались две стороны.`)
  for (let side = 0; side < 2; side += 1) {
    const seat = exactKeys(seats[side], ['side', 'obs_hash', 'mask', 'action', 'rng'], `decision ${tick}, seat ${side}`)
    if (int(seat.side, `decision ${tick}.side`) !== side) throw new ReplayFormatError('Порядок сторон в decision должен быть A, B.')
    hash64(seat.obs_hash, `decision ${tick}.obs_hash`)
    const mask = str(seat.mask, `decision ${tick}.mask`)
    if (mask.length !== 30 || /[^01]/.test(mask)) throw new ReplayFormatError(`decision ${tick}.mask: ожидалась строка из 30 бит.`)
    const action = exactKeys(seat.action, ['lay_order', 'forage_share_q', 'army_choice', 'avoid_center'], `decision ${tick}.action`)
    int(action.lay_order, 'action.lay_order')
    array(action.forage_share_q, 'action.forage_share_q').forEach((part, index) => int(part, `action.forage_share_q[${index}]`))
    int(action.army_choice, 'action.army_choice')
    bool(action.avoid_center, 'action.avoid_center')
    const rng = array(seat.rng, `decision ${tick}.rng`)
    const expectedStream = seatStreams[side]
    if (rng.length !== (expectedStream ? 1 : 0)) throw new ReplayFormatError(`decision ${tick}.rng: число RNG записей не совпадает с seat.`)
    if (expectedStream) {
      const entry = exactKeys(rng[0], ['stream_id', 'draws', 'state_after'], `decision ${tick}.rng[0]`)
      if (entry.stream_id !== expectedStream || !streams.includes(expectedStream)) throw new ReplayFormatError(`decision ${tick}.rng: неизвестный stream_id.`)
      int(entry.draws, `decision ${tick}.rng.draws`, 0, 2 ** 40)
      const state = exactKeys(entry.state_after, ['state', 'inc'], `decision ${tick}.rng.state_after`)
      for (const field of ['state', 'inc']) {
        const value = str(state[field], `decision ${tick}.rng.${field}`)
        if (!/^(0|[1-9][0-9]*)$/.test(value) || BigInt(value) > 0xffffffffffffffffn) throw new ReplayFormatError(`decision ${tick}.rng.${field}: ожидалась десятичная u64 строка.`)
      }
    }
  }
}

function validateHeader(record: JsonObject) {
  exactKeys(record, ['type', 'seq', 'format', 'replay_schema', 'engine', 'contract', 'config', 'config_hash', 'root_seed', 'streams', 'seats', 'start', 'cursor', 'source', 'meta'], 'header')
  if (record.format !== FORMAT) throw new ReplayFormatError('Это не файл FormicMind Replay.')
  if (record.replay_schema !== SCHEMA) throw new ReplayVersionError(`Неподдерживаемая версия Replay ${String(record.replay_schema)}; нужен v1.`)
  if (!same(record.contract, CONTRACT)) throw new ReplayVersionError('Версии контракта файла не совпадают с поддерживаемой сборкой FormicMind.')
  const config = validateConfig(record.config)
  const configHash = hash64(record.config_hash, 'header.config_hash')
  if (hashDomain(CONFIG_DOMAIN, utf8(canonicalString(config.raw))) !== configHash) throw new ReplayFormatError('header.config_hash не совпадает с конфигурацией файла.')
  const rootSeed = str(record.root_seed, 'header.root_seed')
  if (!/^(0|[1-9][0-9]*)$/.test(rootSeed) || BigInt(rootSeed) > 0xffffffffffffffffn) throw new ReplayFormatError('header.root_seed должен быть десятичной u64 строкой.')
  const streams = array(record.streams, 'header.streams').map((item, index) => str(item, `header.streams[${index}]`))
  if (streams.includes('world') || streams.some((stream, index) => index > 0 && stream <= streams[index - 1])) throw new ReplayFormatError('header.streams должны быть уникальны и отсортированы; world сюда не входит.')
  const seatsRaw = array(record.seats, 'header.seats')
  if (seatsRaw.length !== 2) throw new ReplayFormatError('header.seats должен содержать A и B.')
  const seatStreams = seatsRaw.map((item, side) => {
    const seat = exactKeys(item, ['side', 'policy_id', 'kind', 'stream_id', 'ref'], `header.seats[${side}]`)
    if (int(seat.side, `header.seats[${side}].side`) !== side) throw new ReplayFormatError('header.seats должны идти в порядке A, B.')
    str(seat.policy_id, `header.seats[${side}].policy_id`)
    if (!['scripted', 'policy', 'other'].includes(str(seat.kind, `header.seats[${side}].kind`))) throw new ReplayFormatError('Неизвестный тип места матча.')
    const stream = str(seat.stream_id, `header.seats[${side}].stream_id`)
    str(seat.ref, `header.seats[${side}].ref`)
    if (stream && !streams.includes(stream)) throw new ReplayFormatError('Seat ссылается на неописанный RNG stream.')
    return stream
  })
  if (seatStreams[0] && seatStreams[0] === seatStreams[1]) throw new ReplayFormatError('Колонии не могут совместно использовать RNG stream.')
  const cursor = exactKeys(record.cursor, ['tick', 'event_index'], 'header.cursor')
  const startTick = int(cursor.tick, 'header.cursor.tick', 0, config.tMax - 1)
  const eventIndex = int(cursor.event_index, 'header.cursor.event_index', 0)
  const start = object(record.start, 'header.start')
  if (start.kind === 'reset') {
    exactKeys(start, ['kind'], 'header.start')
    if (startTick !== 0 || eventIndex !== 0) throw new ReplayFormatError('Replay reset должен начинаться с (tick 0, event 0).')
  } else if (start.kind === 'checkpoint') {
    exactKeys(start, ['kind', 'checkpoint', 'checkpoint_hash'], 'header.start')
    const checkpoint = exactKeys(start.checkpoint, ['schema', 'serialization_version', 'state', 'run'], 'header.start.checkpoint')
    if (checkpoint.schema !== 'formicmind.checkpoint/3' || checkpoint.serialization_version !== CONTRACT.serialization_version) throw new ReplayVersionError('Checkpoint имеет неподдерживаемую версию.')
    if (hashDomain(CHECKPOINT_DOMAIN, utf8(canonicalString(checkpoint))) !== hash64(start.checkpoint_hash, 'header.start.checkpoint_hash')) throw new ReplayFormatError('Хэш checkpoint не совпадает с его содержимым.')
    const state = object(checkpoint.state, 'checkpoint.state')
    if (int(state.tick, 'checkpoint.state.tick', 0, config.tMax - 1) !== startTick || state.config_sha256 !== configHash || state.rules_version !== CONTRACT.rules_version) throw new ReplayFormatError('Checkpoint не совпадает с cursor/config Replay.')
    const run = object(checkpoint.run, 'checkpoint.run')
    if (run.root_seed !== rootSeed || !same(run.outcome, { finished: false })) throw new ReplayFormatError('Checkpoint относится к другому seed или уже завершённому матчу.')
    const runStreams = object(run.streams, 'checkpoint.run.streams')
    if (!Object.hasOwn(runStreams, 'world') || !same(Object.keys(runStreams).filter((key) => key !== 'world').sort(), streams)) throw new ReplayFormatError('Checkpoint RNG streams не совпадают с header.')
  } else throw new ReplayFormatError('header.start.kind должен быть reset или checkpoint.')
  const meta = object(record.meta, 'header.meta')
  for (const [key, value] of Object.entries(meta)) if (typeof value !== 'string') throw new ReplayFormatError(`header.meta.${key}: ожидалась строка.`)
  const policies = seatsRaw as JsonObject[]
  return {
    config, startTick, eventIndex, rootSeed, streams, seatStreams,
    fileSource: str(record.source, 'header.source'),
    policies: { A: String(policies[0].policy_id), B: String(policies[1].policy_id) } as Record<Side, string>,
    meta,
    checkpoint: start.kind === 'checkpoint' ? start.checkpoint as JsonObject : null,
  }
}

class ReplayDataIndex implements ReplayDataSource {
  private readonly eventCounts = new Map<string, Uint32Array>()

  constructor(private readonly metadata: ReplayMetadata, private readonly frames: TimelineFrame[], private readonly events: ViewerEvent[]) {
    const categories: Array<EventCategory | 'all'> = ['all', 'combat', 'economy', 'movement', 'terminal']
    const sides: Array<Side | 'all'> = ['all', 'A', 'B']
    for (const category of categories) for (const side of sides) this.eventCounts.set(`${category}|${side}`, new Uint32Array(events.length + 1))
    for (let index = 0; index < events.length; index += 1) {
      const event = events[index]
      for (const category of ['all', event.category]) {
        for (const side of ['all', ...(event.side ? [event.side] : [])]) {
          const counts = this.eventCounts.get(`${category}|${side}`)!
          counts[index + 1] = counts[index] + 1
        }
      }
      for (const category of categories) for (const side of sides) {
        if (category === 'all' || category === event.category) continue
        const counts = this.eventCounts.get(`${category}|${side}`)!
        counts[index + 1] = counts[index]
      }
      for (const side of sides) {
        if (side === 'all' || side === event.side) continue
        for (const category of ['all', event.category]) {
          const counts = this.eventCounts.get(`${category}|${side}`)!
          counts[index + 1] = counts[index]
        }
      }
    }
  }

  getMetadata() { return this.metadata }

  getFrame(cursor: number) {
    const frame = this.frames[cursor]
    if (!frame) throw new RangeError(`Кадр ${cursor} отсутствует в записи.`)
    return frame
  }

  getEvents(query: EventQuery = {}) {
    const limit = query.limit === undefined ? Number.MAX_SAFE_INTEGER : Math.max(0, query.limit)
    if (limit === 0) return []
    const offset = Math.max(0, query.offset ?? 0)
    const throughCursor = query.throughCursor ?? this.metadata.maxCursor
    const result: ViewerEvent[] = []
    let matched = 0
    for (let index = Math.min(this.events.length - 1, upperBoundEventCursor(this.events, throughCursor)); index >= 0; index -= 1) {
      const event = this.events[index]
      if (query.category && query.category !== 'all' && event.category !== query.category) continue
      if (query.side && query.side !== 'all' && event.side !== query.side) continue
      if (matched++ < offset) continue
      result.push(event)
      if (result.length >= limit) break
    }
    return result
  }

  getEventCount(query: EventQuery = {}) {
    const throughCursor = query.throughCursor ?? this.metadata.maxCursor
    const end = upperBoundEventCursor(this.events, throughCursor)
    if (end < 0) return 0
    const category = query.category ?? 'all'
    const side = query.side ?? 'all'
    return this.eventCounts.get(`${category}|${side}`)![end + 1]
  }

  getCursorForTick(tick: number, boundary: ReplayBoundary = 'start') {
    return this.frames.find((frame) => frame.state.tick === tick && frame.state.boundary === boundary)?.cursor
  }
}

function upperBoundEventCursor(events: ViewerEvent[], cursor: number) {
  let low = 0
  let high = events.length
  while (low < high) {
    const middle = (low + high) >>> 1
    if (events[middle].cursor <= cursor) low = middle + 1
    else high = middle
  }
  return low - 1
}

function checkpointFrame(checkpoint: JsonObject, config: ReplayConfig, startTick: number): CanonicalFrame {
  const state = exactKeys(checkpoint.state, ['schema', 'serialization_version', 'rules_version', 'config_sha256', 'tick', 'next_unit_id', 'node_food', 'colonies'], 'checkpoint.state')
  if (state.schema !== 'formicmind.sim_state/1' || state.serialization_version !== CONTRACT.serialization_version || state.rules_version !== CONTRACT.rules_version) throw new ReplayVersionError('Checkpoint state создан несовместимой версией FormicMind.')
  if (int(state.tick, 'checkpoint.state.tick', 0, config.tMax - 1) !== startTick || !/^[0-9a-f]{64}$/.test(str(state.config_sha256, 'checkpoint.state.config_sha256'))) throw new ReplayFormatError('Checkpoint state не совпадает с начальным тиком или не содержит config hash.')
  int(state.next_unit_id, 'checkpoint.state.next_unit_id', 1)
  const nodeFood = array(state.node_food, 'checkpoint.state.node_food').map((value, index) => int(value, `checkpoint.state.node_food[${index}]`, 0, config.capacities[index]))
  if (nodeFood.length !== config.nodes.length) throw new ReplayFormatError('Checkpoint state содержит неверное число node_food значений.')
  const rawColonies = array(state.colonies, 'checkpoint.state.colonies')
  if (rawColonies.length !== 2) throw new ReplayFormatError('Checkpoint state должен содержать две колонии.')
  const colonies = rawColonies.map((value, side) => {
    const source = exactKeys(value, ['side', 'queen_hp', 'food', 'units', 'brood', 'lay_order', 'forage_q', 'army_choice', 'avoid_center', 'next_lay_tick', 'upkeep_remainder10', 'starvation_debt', 'intel'], `checkpoint.state.colonies[${side}]`)
    if (int(source.side, `checkpoint.state.colonies[${side}].side`) !== side) throw new ReplayFormatError('Checkpoint колонии должны идти в порядке A, B.')
    const stationed = new Map<string, number[]>()
    const transit = new Map<string, number[]>()
    const units = array(source.units, `checkpoint.state.colonies[${side}].units`)
    let previousUnit = 0
    for (const [index, value] of units.entries()) {
      const unit = exactKeys(value, ['unit_id', 'caste', 'hp', 'node', 'from_node', 'to_node', 'arrival_tick', 'target_node'], `checkpoint unit ${side}:${index}`)
      const id = int(unit.unit_id, `checkpoint unit ${side}:${index}.unit_id`, 1)
      if (id <= previousUnit) throw new ReplayFormatError('Checkpoint unit ids должны быть строго возрастающими.')
      previousUnit = id
      const caste = int(unit.caste, 'checkpoint unit caste', 0, 1)
      const countKey = unit.to_node === -1 ? stationed : transit
      let key: number[]
      if (unit.to_node === -1) {
        key = [int(unit.node, 'checkpoint unit.node', 0, 6), caste]
      } else {
        key = [int(unit.from_node, 'checkpoint unit.from_node', 0, 6), int(unit.to_node, 'checkpoint unit.to_node', 0, 6), int(unit.arrival_tick, 'checkpoint unit.arrival_tick', 0), caste]
        int(unit.node, 'checkpoint transit unit.node', -1, 6)
      }
      int(unit.hp, 'checkpoint unit.hp', 1)
      const joined = key.join(',')
      const row = countKey.get(joined) ?? [...key, 0, 0]
      row[key.length] += 1
      row[key.length + 1] += unit.hp as number
      countKey.set(joined, row)
    }
    const broodRows = new Map<string, number[]>()
    for (const [index, value] of array(source.brood, `checkpoint.state.colonies[${side}].brood`).entries()) {
      const egg = exactKeys(value, ['unit_id', 'caste', 'laid_tick', 'hatch_tick'], `checkpoint brood ${side}:${index}`)
      const caste = int(egg.caste, 'checkpoint brood.caste', 0, 1)
      const hatchTick = int(egg.hatch_tick, 'checkpoint brood.hatch_tick', 0)
      const key = [caste, hatchTick]
      const joined = key.join(',')
      const row = broodRows.get(joined) ?? [...key, 0]
      row[2] += 1
      broodRows.set(joined, row)
    }
    int(source.food, 'checkpoint colony.food', 0)
    int(source.queen_hp, 'checkpoint colony.queen_hp', 0, config.queenHp)
    return {
      side,
      food: source.food as number,
      queen_hp: source.queen_hp as number,
      stationed: [...stationed.values()].sort(compareRowKey),
      transit: [...transit.values()].sort(compareRowKey),
      brood: [...broodRows.values()].sort(compareRowKey),
    }
  })
  return { tick: startTick, boundary: 'start', terminal: { finished: false }, node_food: nodeFood, colonies }
}

function validateResult(record: JsonObject, seq: number, previousChain: string, eventCount: number, eventsHash: string,
  tickCount: number, decisionCount: number, currentTick: number, finished: boolean, terminalEvent: JsonObject | null,
  lastStartTick: number, config: ReplayConfig) {
  exactKeys(record, ['type', 'seq', 'finished', 'outcome', 'state_tick', 'ticks_executed', 'decision_count', 'event_count', 'events_hash', 'final_state_hash', 'final_checkpoint_hash', 'record_count', 'chain_head'], 'result')
  if (record.finished !== finished) throw new ReplayFormatError('result.finished не совпадает с терминальным событием.')
  if (int(record.record_count, 'result.record_count') !== seq || int(record.state_tick, 'result.state_tick', 0, config.tMax) !== currentTick) throw new ReplayFormatError('result.record_count/state_tick не совпадают с кадрами записи.')
  if (int(record.ticks_executed, 'result.ticks_executed') !== tickCount || int(record.decision_count, 'result.decision_count') !== decisionCount || int(record.event_count, 'result.event_count') !== eventCount) throw new ReplayFormatError('Счётчики result не совпадают с записями replay.')
  if (hash64(record.events_hash, 'result.events_hash') !== eventsHash || hash64(record.chain_head, 'result.chain_head') !== previousChain) throw new ReplayFormatError('Итоговый hash событий или chain_head не совпадает.')
  hash64(record.final_state_hash, 'result.final_state_hash')
  hash64(record.final_checkpoint_hash, 'result.final_checkpoint_hash')
  if (!finished) {
    if (!same(record.outcome, { finished: false }) || lastStartTick !== currentTick || currentTick >= config.tMax) throw new ReplayFormatError('Незавершённая запись должна оканчиваться допустимым start-кадром без исхода.')
    return null
  }
  if (!terminalEvent) throw new ReplayFormatError('result сообщает об исходе без terminal event.')
  const outcome = exactKeys(record.outcome, ['finished', 'winner', 'reason', 'terminal_tick', 'terminal_phase', 'phases_executed', 'sides', 'tick_completed'], 'result.outcome')
  const event = terminalEvent
  if (outcome.finished !== true || outcome.winner !== event.winner || outcome.reason !== event.reason || outcome.terminal_tick !== event.tick || outcome.terminal_phase !== event.phase || outcome.phases_executed !== event.phases_executed || outcome.tick_completed !== event.tick_completed || !same(outcome.sides, event.sides)) throw new ReplayFormatError('result.outcome не совпадает с terminal event.')
  const tickCompleted = bool(event.tick_completed, 'terminal.tick_completed')
  if (currentTick !== int(event.tick, 'terminal.tick') + (tickCompleted ? 1 : 0)) throw new ReplayFormatError('state_tick не соответствует terminal_tick и tick_completed.')
  if (event.reason === 'timeout' && event.winner !== 'draw') throw new ReplayFormatError('Timeout по контракту всегда заканчивается ничьей.')
  return event
}

/** Reads and fully validates one local Replay v1 JSONL before constructing its presentation-only data source. */
export async function parseReplayBytes(bytes: Uint8Array, fileName = 'replay.jsonl', onProgress?: (progress: ParseProgress) => void): Promise<ReplayDataSource> {
  if (bytes.byteLength === 0) throw new ReplayFormatError('Файл пуст. Выберите завершённый Replay v1 JSONL.')
  if (bytes.byteLength > MAX_REPLAY_BYTES) throw new ReplayFormatError('Файл превышает лимит 10 МиБ.')
  if (bytes[0] === 0xef && bytes[1] === 0xbb && bytes[2] === 0xbf) throw new ReplayFormatError('Файл содержит BOM; Replay v1 должен быть ASCII JSONL без BOM.')
  if (bytes[bytes.byteLength - 1] !== 0x0a) throw new ReplayFormatError('Файл обрезан: последняя JSONL-строка не завершена переводом строки.')
  let totalRecords = 0
  for (const byte of bytes) {
    if (byte === 0x0a) totalRecords += 1
    if (byte === 0x0d) throw new ReplayFormatError('Replay v1 использует только LF-переводы строк, без CR/CRLF.')
    if (byte > 0x7f) throw new ReplayFormatError('Replay v1 должен содержать ASCII-совместимый canonical JSON.')
  }
  if (totalRecords < 3) throw new ReplayFormatError('Недостаточно записей: ожидаются header, snapshot и result.')
  if (totalRecords > MAX_RECORDS) throw new ReplayFormatError('Число записей превышает лимит 50 000.')
  const decoder = new TextDecoder('utf-8', { fatal: true })
  let offset = 0
  let seq = 0
  let previousChain = ZERO_CHAIN
  let header: ReturnType<typeof validateHeader> | null = null
  let body: CanonicalBody | null = null
  let currentTick = 0
  let lastStartTick = 0
  let currentCanonicalFrame: CanonicalFrame | null = null
  let finished = false
  let terminalEvent: JsonObject | null = null
  let expectedSnapshot: { boundary: ReplayBoundary; tick: number; frame: CanonicalFrame } | null = null
  let closing = false
  let eventIndex = 0
  let tickCount = 0
  let decisionCount = 0
  let eventNumber = 0
  let totalEventsCanonical: string[] = []
  const frames: TimelineFrame[] = []
  const events: ViewerEvent[] = []
  let lastProgressAt = 0

  const readRecord = async (): Promise<{ record: JsonObject; previousChain: string; number: number }> => {
    if (offset >= bytes.byteLength) throw new ReplayFormatError('Файл завершился до обязательной записи result.')
    const start = offset
    const newline = bytes.indexOf(0x0a, start)
    if (newline < 0) throw new ReplayFormatError('Файл обрезан: незавершённая строка.')
    const length = newline - start
    if (length === 0) throw new ReplayFormatError(`Пустая строка ${seq + 1} не допускается.`)
    if (length > MAX_LINE_BYTES) throw new ReplayFormatError(`Строка ${seq + 1} превышает лимит 1 МиБ.`)
    const lineBytes = bytes.subarray(start, newline)
    offset = newline + 1
    const line = decoder.decode(lineBytes)
    let parsed: unknown
    try { parsed = JSON.parse(line) } catch { throw new ReplayFormatError(`Строка ${seq + 1}: некорректный JSON.`) }
    const record = object(parsed, `строка ${seq + 1}`)
    if (canonicalString(record) !== line) throw new ReplayFormatError(`Строка ${seq + 1}: JSON не является canonical JSON Replay v1.`)
    const chain = hash64(record.chain, `строка ${seq + 1}.chain`)
    const zeroLine = utf8(canonicalString({ ...record, chain: ZERO_CHAIN }))
    const before = previousChain
    const expectedChain = hashDomain(FORMAT_DOMAIN, concatBytes(utf8(`${before}\u0000`), zeroLine))
    if (chain !== expectedChain) throw new ReplayFormatError(`Строка ${seq + 1}: нарушена SHA-256 цепочка целостности.`)
    if (int(record.seq, `строка ${seq + 1}.seq`) !== seq) throw new ReplayFormatError(`Строка ${seq + 1}: seq нарушает порядок записей.`)
    delete record.chain
    previousChain = chain
    const number = seq++
    if (number % 128 === 0 || offset - lastProgressAt > 256 * 1024) {
      onProgress?.({ recordsProcessed: seq, totalRecords, bytesProcessed: offset, totalBytes: bytes.byteLength })
      lastProgressAt = offset
      await new Promise<void>((resolve) => setTimeout(resolve, 0))
    }
    return { record, previousChain: before, number }
  }

  const first = await readRecord()
  if (first.number !== 0) throw new ReplayFormatError('Первая запись должна быть header.')
  header = validateHeader(first.record)
  eventIndex = header.eventIndex
  const initial = await readRecord()
  if (initial.record.type !== 'snapshot') throw new ReplayFormatError('После header ожидался начальный snapshot.')
  exactKeys(initial.record, ['type', 'seq', 'frame', 'frame_hash'], 'начальный snapshot')
  const initialFrame = validateFrame(initial.record.frame, header.config, 'начальный snapshot.frame')
  if (initialFrame.tick !== header.startTick || initialFrame.boundary !== 'start' || initialFrame.terminal.finished) throw new ReplayFormatError('Начальный snapshot должен быть start-кадром header.cursor.tick.')
  if (frameHash(initialFrame) !== hash64(initial.record.frame_hash, 'начальный snapshot.frame_hash')) throw new ReplayFormatError('Хэш начального snapshot не совпадает с кадром.')
  if (header.checkpoint) {
    const expected = checkpointFrame(header.checkpoint, header.config, header.startTick)
    if (!same(initialFrame, expected)) throw new ReplayFormatError('Начальный snapshot не совпадает с embedded checkpoint.')
  }
  body = bodyOf(initialFrame)
  currentCanonicalFrame = initialFrame
  currentTick = lastStartTick = header.startTick
  frames.push(framePresentation(initialFrame, 0, header.config, []))
  let pending = await readRecord()

  while (pending.record.type !== 'result') {
    const record = pending.record
    if (pending.number === totalRecords - 1) throw new ReplayFormatError('result должен быть последней записью файла.')
    if (expectedSnapshot) {
      if (record.type !== 'snapshot') throw new ReplayFormatError(`После delta ожидался ${expectedSnapshot.boundary} snapshot для state.tick=${expectedSnapshot.tick}.`)
      exactKeys(record, ['type', 'seq', 'frame', 'frame_hash'], 'snapshot')
      const snapshot = validateFrame(record.frame, header.config, 'snapshot.frame')
      if (snapshot.tick !== expectedSnapshot.tick || snapshot.boundary !== expectedSnapshot.boundary || frameHash(snapshot) !== hash64(record.frame_hash, 'snapshot.frame_hash') || !same(snapshot, expectedSnapshot.frame)) throw new ReplayFormatError('Snapshot после delta не совпадает с восстановленным кадром или frame_hash.')
      if (expectedSnapshot.boundary === 'start') lastStartTick = currentTick
      expectedSnapshot = null
      pending = await readRecord()
      continue
    }
    if (record.type === 'snapshot') {
      exactKeys(record, ['type', 'seq', 'frame', 'frame_hash'], 'cutoff snapshot')
      const snapshot = validateFrame(record.frame, header.config, 'cutoff snapshot.frame')
      if (finished || snapshot.boundary !== 'start' || snapshot.tick !== currentTick || snapshot.terminal.finished || currentTick % header.config.policyInterval === 0 || lastStartTick === currentTick || !same(snapshot, currentCanonicalFrame)) throw new ReplayFormatError('Неожиданный самостоятельный snapshot; он допустим только как кадр отсечения replay.')
      if (frameHash(snapshot) !== hash64(record.frame_hash, 'cutoff snapshot.frame_hash')) throw new ReplayFormatError('Хэш cutoff snapshot не совпадает.')
      lastStartTick = currentTick
      closing = true
      pending = await readRecord()
      if (pending.record.type !== 'result') throw new ReplayFormatError('После cutoff snapshot должен идти result.')
      break
    }
    if (closing || finished) throw new ReplayFormatError('В записи обнаружены данные после cutoff/terminal кадра.')
    if (currentTick >= header.config.tMax) throw new ReplayFormatError(`Запись пытается исполнить тик за пределами t_max=${header.config.tMax}.`)
    const tick = currentTick
    let decision = record
    if (tick % header.config.policyInterval === 0) {
      if (decision.type !== 'decision') throw new ReplayFormatError(`Тик ${tick}: перед событиями отсутствует policy decision.`)
      validateDecision(decision, tick, header.streams, header.seatStreams)
      decisionCount += 1
      pending = await readRecord()
    } else if (decision.type === 'decision') throw new ReplayFormatError(`Тик ${tick} не является policy boundary, но содержит decision.`)

    const tickEvents: JsonObject[] = []
    let previousSortKey: number[] | null = null
    while (pending.record.type === 'event') {
      const eventRecord = pending.record
      exactKeys(eventRecord, ['type', 'seq', 'i', 'e'], `event record ${pending.number}`)
      if (int(eventRecord.i, `event ${pending.number}.i`, 0) !== eventIndex) throw new ReplayFormatError(`event index ${eventRecord.i} не совпадает с ожидаемым ${eventIndex}.`)
      const validated = validateEvent(eventRecord.e, tick, header.config, `event ${pending.number}`)
      if (previousSortKey && compareSortKey(validated.sortKey, previousSortKey) <= 0) throw new ReplayFormatError(`event ${pending.number}: события нарушают engine order или повторяются.`)
      previousSortKey = validated.sortKey
      if (terminalEvent) throw new ReplayFormatError('После terminal event не должно быть других событий.')
      if (validated.kind === 'terminal') terminalEvent = validated.raw
      tickEvents.push(validated.raw)
      totalEventsCanonical.push(canonicalString(validated.raw))
      eventIndex += 1
      eventNumber += 1
      pending = await readRecord()
    }
    if (pending.record.type !== 'delta') throw new ReplayFormatError(`Тик ${tick}: после событий ожидался delta.`)
    const delta = pending.record
    exactKeys(delta, ['type', 'seq', 'tick', 'to_tick', 'to_boundary', 'patch', 'frame_hash'], `delta ${tick}`)
    if (int(delta.tick, `delta ${tick}.tick`) !== tick) throw new ReplayFormatError(`delta относится не к текущему тику ${tick}.`)
    const ended = terminalEvent !== null
    const expectedBoundary: ReplayBoundary = ended ? 'terminal' : 'start'
    if (delta.to_boundary !== expectedBoundary) throw new ReplayFormatError(`delta ${tick}: boundary не соответствует наличию terminal event.`)
    let toTick: number
    if (ended) {
      const event = terminalEvent!
      const tickCompleted = bool(event.tick_completed, `terminal ${tick}.tick_completed`)
      if (event.tick !== tick || tickCompleted !== (event.phase === 10) || event.phase === 4 && event.reason !== 'queen_death' || event.phase === 10 && event.reason === 'queen_death') throw new ReplayFormatError(`terminal ${tick}: неверная связь между phase, reason и tick_completed.`)
      if (event.reason === 'timeout' && tick !== header.config.tMax - 1) throw new ReplayFormatError('Timeout может быть только на последнем разрешённом тике.')
      toTick = tick + (tickCompleted ? 1 : 0)
    } else toTick = tick + 1
    if (int(delta.to_tick, `delta ${tick}.to_tick`, 0, header.config.tMax) !== toTick) throw new ReplayFormatError(`delta ${tick}.to_tick противоречит числу завершённых тиков.`)
    body = applyPatch(body!, delta.patch, header.config, `delta ${tick}.patch`)
    const canonicalFrame: CanonicalFrame = {
      tick: toTick,
      boundary: expectedBoundary,
      terminal: ended ? validateTerminal({
        finished: true, winner: terminalEvent!.winner, reason: terminalEvent!.reason, terminal_tick: terminalEvent!.tick,
        terminal_phase: terminalEvent!.phase, tick_completed: terminalEvent!.tick_completed,
      }, `terminal frame ${tick}.terminal`) : { finished: false },
      node_food: body.node_food,
      colonies: body.colonies,
    }
    if (frameHash(canonicalFrame) !== hash64(delta.frame_hash, `delta ${tick}.frame_hash`)) throw new ReplayFormatError(`delta ${tick}: patch не восстанавливает записанный frame_hash.`)
    const cursor = frames.length
    const presentedEvents = tickEvents.map((event, index) => ({ ...eventPresentation(event, events.length + index, tick, header!.config), cursor }))
    events.push(...presentedEvents)
    frames.push(framePresentation(canonicalFrame, cursor, header.config, presentedEvents))
    currentCanonicalFrame = canonicalFrame
    currentTick = toTick
    tickCount += 1
    if (ended) {
      finished = true
      expectedSnapshot = { boundary: 'terminal', tick: toTick, frame: canonicalFrame }
    } else if (currentTick % header.config.policyInterval === 0 && currentTick < header.config.tMax) {
      expectedSnapshot = { boundary: 'start', tick: currentTick, frame: canonicalFrame }
    }
    pending = await readRecord()
  }

  if (!pending || pending.record.type !== 'result' || pending.number !== totalRecords - 1 || offset !== bytes.byteLength) throw new ReplayFormatError('result должен быть последней записью файла без дополнительных байтов.')
  if (expectedSnapshot) throw new ReplayFormatError('Файл заканчивается до обязательного snapshot terminal/policy boundary.')
  const eventBytes = utf8(`[${totalEventsCanonical.join(',')}]`)
  const eventsHash = hashDomain(EVENT_DOMAIN, eventBytes)
  const terminal = validateResult(pending.record, pending.number, pending.previousChain, eventNumber, eventsHash, tickCount,
    decisionCount, currentTick, finished, terminalEvent, lastStartTick, header.config)
  if (terminal && (!frames[frames.length - 1].state.boundary || frames[frames.length - 1].state.boundary !== 'terminal')) throw new ReplayFormatError('Последний кадр результата должен быть terminal.')
  onProgress?.({ recordsProcessed: seq, totalRecords, bytesProcessed: bytes.byteLength, totalBytes: bytes.byteLength })
  const terminalDto = terminal ? terminalPresentation({
    finished: true, winner: str(terminal.winner, 'terminal.winner'), reason: str(terminal.reason, 'terminal.reason'),
    terminal_tick: int(terminal.tick, 'terminal.tick'), terminal_phase: int(terminal.phase, 'terminal.phase'),
    tick_completed: bool(terminal.tick_completed, 'terminal.tick_completed'),
  }, frames.length - 1, currentTick) : null
  const sourceLabel = terminalDto ? 'REAL REPLAY' : 'RECORDING CUTOFF'
  const policies = header.policies
  const title = `${policies.A} vs ${policies.B}`
  const metadata: ReplayMetadata = {
    id: `replay-${fileName}-${seq}-${pending.record.final_checkpoint_hash as string}`,
    title,
    subtitle: terminalDto ? 'Завершённый матч из Replay v1' : 'Запись остановлена · матч не завершён',
    recordedAt: null,
    seedLabel: `SEED ${header.rootSeed}`,
    sourceLabel,
    sourceKind: 'replay',
    fileName,
    minCursor: 0,
    maxCursor: frames.length - 1,
    terminal: terminalDto,
    finished: terminalDto !== null,
    ticksExecuted: tickCount,
    eventCount: eventNumber,
    timeLimit: header.config.tMax,
    policyInterval: header.config.policyInterval,
    policies,
    world: { nodes: header.config.nodes, edges: header.config.edges },
  }
  return new ReplayDataIndex(metadata, frames, events)
}

export async function parseReplayFile(file: ReplayFileLike, onProgress?: (progress: ParseProgress) => void) {
  if (!file.name.toLowerCase().endsWith('.jsonl')) throw new ReplayFormatError('Ожидается локальный файл с расширением .jsonl.')
  if (file.size <= 0 || file.size > MAX_REPLAY_BYTES) throw new ReplayFormatError('Размер Replay должен быть от 1 байта до 10 МиБ.')
  const bytes = new Uint8Array(await file.arrayBuffer())
  if (bytes.byteLength !== file.size) throw new ReplayFormatError('Файл изменился во время чтения; повторите импорт.')
  return parseReplayBytes(bytes, file.name, onProgress)
}
