import { canonicalString } from '../data/replayJsonl'
import { hashDomain, utf8 } from '../data/replayCrypto'
import type {
  CohortState, ColonyFrame, ColonyStatus, EcoEdge, EcoEvent, EcoFrame, EcoNode, EcoPolicyInfo, EcoReplay, EventKind, FrameReason, MemoryFrame,
  MissionKind, MissionPhase,
} from './types'
import { EVENT_KINDS } from './types'

export const ECO_REPLAY_SCHEMA = 'formicmind.eco.replay/1'
export const MAX_ECO_REPLAY_BYTES = 48 * 1024 * 1024
const RECORD_DOMAIN = 'formicmind/eco/replay-record/v1'
const CONFIG_DOMAIN = 'formicmind/eco/config/v1'
const MAP_DOMAIN = 'formicmind/eco/map/v1'
const EVENTS_DOMAIN = 'formicmind/eco/events/v1'
const ZERO_CHAIN = '0'.repeat(64)

const KIND_NAMES: MissionKind[] = ['home', 'forage', 'scout', 'guard', 'attack', 'withdraw']
const PHASE_NAMES: MissionPhase[] = ['idle', 'go', 'work', 'return']
const STATUSES: ColonyStatus[] = ['alive', 'orphan', 'extinct']
const FRAME_REASONS: FrameReason[] = ['start', 'interval', 'key_event', 'final']

const HEADER_KEYS = ['type', 'seq', 'chain', 'schema', 'rules_version', 'balance_version', 'seed', 'fixture', 'roles', 'ticks_requested',
  'frame_every', 'decision_k', 'alert_interval', 'colonies', 'config', 'config_hash', 'map', 'map_hash', 'cohort_fields', 'visited_fields',
  'foreign_fields', 'sighting_fields', 'kind_codes', 'phase_codes']
const FRAME_KEYS = ['type', 'seq', 'chain', 'tick', 'reason', 'state_hash', 'patches', 'piles', 'colonies', 'cohorts', 'memory']
const EVENT_KEYS = ['type', 'seq', 'chain', 'tick', 'kind', 'colony', 'other', 'node', 'a', 'b']
const RESULT_KEYS = ['type', 'seq', 'chain', 'ticks_completed', 'world_outcome', 'colony_outcomes', 'event_count', 'events_hash', 'frame_count',
  'final_state_hash', 'final_checkpoint_hash', 'record_count']
const COLONY_KEYS = ['id', 'status', 'food', 'queen_hp', 'workers', 'soldiers', 'brood_workers', 'brood_soldiers', 'starvation_debt']
const MEMORY_KEYS = ['colony', 'visited', 'discovered', 'foreign', 'sightings']
/** A run decided by independent policies (FOR-48) carries the policy specification in the header and the decision count and chain hash in the result. */
const POLICY_SCHEMA = 'formicmind.eco.policy-run/1'
const POLICY_KINDS = ['random_valid', 'heuristic', 'learned']
/** Curriculum profiles (FOR-50); a replay of the full action space carries no `curriculum` key at all. */
const CURRICULUM_PROFILES = ['nest-v1']
const COHORT_FIELDS = ['id', 'colony', 'caste', 'count', 'hp', 'load', 'kind', 'phase', 'where', 'a', 'b', 'depart', 'arrive', 'target_node', 'target_colony']
const VISITED_FIELDS = ['node', 'seen_tick', 'last_stock', 'last_stock_tick', 'queen_seen', 'danger']
const FOREIGN_FIELDS = ['colony', 'first_contact', 'last_seen', 'workers_bucket', 'soldiers_bucket', 'nest_node', 'they_attacked', 'i_attacked', 'kills_mine', 'kills_theirs']
const SIGHTING_FIELDS = ['node', 'colony', 'tick', 'workers_bucket', 'soldiers_bucket', 'aggression']

export class EcoReplayError extends Error {
  constructor(message: string) { super(message); this.name = 'EcoReplayError' }
}
export class EcoReplayVersionError extends EcoReplayError {
  constructor(message: string) { super(message); this.name = 'EcoReplayVersionError' }
}

type Rec = Record<string, unknown>

function need(condition: boolean, message: string): asserts condition {
  if (!condition) throw new EcoReplayError(message)
}
function isInt(value: unknown): value is number { return typeof value === 'number' && Number.isSafeInteger(value) }
function isRec(value: unknown): value is Rec { return typeof value === 'object' && value !== null && !Array.isArray(value) }
function exactKeys(record: Rec, keys: string[], where: string) {
  const actual = Object.keys(record).sort()
  const wanted = [...keys].sort()
  need(actual.length === wanted.length && actual.every((key, index) => key === wanted[index]), `${where}: неожиданный набор полей.`)
}
function intRow(value: unknown, length: number, where: string): number[] {
  need(Array.isArray(value) && value.length === length && value.every(isInt), `${where}: ожидается строка из ${length} целых чисел.`)
  return value as number[]
}
function sameList(value: unknown, expected: string[]) {
  return Array.isArray(value) && value.length === expected.length && value.every((item, index) => item === expected[index])
}

export interface EcoParseProgress { recordsProcessed: number; totalRecords: number }

/** Reads and validates one local E1 replay. Every failure is an ``EcoReplayError`` with a readable message; nothing is half-loaded. */
export function parseEcoReplayBytes(bytes: Uint8Array, fileName = 'eco-replay.jsonl', onProgress?: (progress: EcoParseProgress) => void): EcoReplay {
  need(bytes.byteLength > 0, 'Файл пуст. Выберите E1 replay (.jsonl).')
  need(bytes.byteLength <= MAX_ECO_REPLAY_BYTES, 'Файл превышает лимит 48 МиБ.')
  need(!(bytes[0] === 0xef && bytes[1] === 0xbb && bytes[2] === 0xbf), 'Файл содержит BOM; E1 replay — ASCII JSONL без BOM.')
  need(bytes[bytes.byteLength - 1] === 0x0a, 'Файл обрезан: последняя строка не завершена переводом строки.')
  let totalRecords = 0
  for (const byte of bytes) {
    if (byte === 0x0a) totalRecords += 1
    need(byte !== 0x0d, 'E1 replay использует только LF-переводы строк.')
    need(byte <= 0x7f, 'E1 replay должен быть ASCII-совместимым canonical JSON.')
  }
  need(totalRecords >= 3, 'Недостаточно записей: ожидаются header, frame и result.')
  const text = new TextDecoder('utf-8', { fatal: true }).decode(bytes)
  const lines = text.slice(0, -1).split('\n')

  const records: Rec[] = []
  let previous = ZERO_CHAIN
  for (let seq = 0; seq < lines.length; seq += 1) {
    const line = lines[seq]
    let record: unknown
    try { record = JSON.parse(line) } catch { throw new EcoReplayError(`Строка ${seq + 1}: не JSON.`) }
    need(isRec(record), `Строка ${seq + 1}: ожидается объект.`)
    need(canonicalString(record) === line, `Строка ${seq + 1}: JSON не является canonical JSON.`)
    need(record.seq === seq, `Строка ${seq + 1}: номер записи должен быть ${seq}.`)
    need(typeof record.chain === 'string', `Строка ${seq + 1}: нет chain.`)
    const expected = hashDomain(RECORD_DOMAIN, utf8(`${previous}\n${canonicalString({ ...record, chain: ZERO_CHAIN })}`))
    need(record.chain === expected, `Строка ${seq + 1}: нарушена SHA-256 цепочка целостности (файл изменён или повреждён).`)
    previous = expected
    records.push(record)
    if (onProgress && seq % 200 === 0) onProgress({ recordsProcessed: seq + 1, totalRecords })
  }

  const header = records[0]
  need(header.type === 'header', 'Первая запись должна быть header.')
  if (header.schema !== ECO_REPLAY_SCHEMA) {
    throw new EcoReplayVersionError(`Неподдерживаемая версия файла: «${String(header.schema)}». Этот viewer читает «${ECO_REPLAY_SCHEMA}».`)
  }
  const policy = 'policy' in header ? parsePolicy(header.policy, header) : null
  exactKeys(header, policy ? [...HEADER_KEYS, 'policy'] : HEADER_KEYS, 'header')
  need(isRec(header.config) && isRec(header.map), 'header: нет config или map.')
  need(hashDomain(CONFIG_DOMAIN, utf8(canonicalString(header.config))) === header.config_hash, 'header.config_hash не совпадает с конфигурацией в файле.')
  const rawMap = header.map as Rec
  need(hashDomain(MAP_DOMAIN, utf8(canonicalString({ ...rawMap, map_hash: '' }))) === rawMap.map_hash && rawMap.map_hash === header.map_hash,
    'header.map_hash не совпадает с картой в файле.')
  need(sameList(header.cohort_fields, COHORT_FIELDS) && sameList(header.visited_fields, VISITED_FIELDS) && sameList(header.foreign_fields, FOREIGN_FIELDS)
    && sameList(header.sighting_fields, SIGHTING_FIELDS), 'header: раскладка строк не совпадает с этой версией viewer.')
  need(isInt(header.colonies) && header.colonies >= 2 && header.colonies <= 8 && typeof header.seed === 'string' && /^(0|[1-9][0-9]*)$/.test(header.seed), 'header: неверные colonies/seed.')
  need(isInt(header.frame_every) && header.frame_every >= 1 && isInt(header.ticks_requested), 'header: неверная частота кадров.')
  const colonyCount = header.colonies

  // ---------------------------------------------------------------- map
  need(Array.isArray(rawMap.nodes) && Array.isArray(rawMap.edges) && Array.isArray(rawMap.nests), 'map: нет nodes/edges/nests.')
  const nodes: EcoNode[] = (rawMap.nodes as Rec[]).map((node, index) => {
    need(isRec(node) && node.node_id === index && isInt(node.x) && isInt(node.y) && ['junction', 'patch', 'nest'].includes(node.kind as string)
      && isInt(node.nest_of) && Array.isArray(node.patch), `map.nodes[${index}]: неверный узел.`)
    const patch = (node.patch as Rec[])[0]
    return { id: index, kind: node.kind as EcoNode['kind'], nestOf: node.nest_of, x: node.x, y: node.y,
      capacity: patch && isInt(patch.capacity) ? patch.capacity : 0, zone: patch && typeof patch.zone === 'string' ? patch.zone : '' }
  })
  const edges: EcoEdge[] = (rawMap.edges as Rec[]).map((edge, index) => {
    need(isRec(edge) && isInt(edge.a) && isInt(edge.b) && isInt(edge.length) && isInt(edge.lane_cap) && typeof edge.chokepoint === 'boolean'
      && edge.a >= 0 && edge.b < nodes.length && edge.a < edge.b, `map.edges[${index}]: неверное ребро.`)
    return { a: edge.a, b: edge.b, length: edge.length, laneCap: edge.lane_cap, chokepoint: edge.chokepoint }
  })
  const nests = rawMap.nests as number[]
  need(nests.length === colonyCount && nests.every((id) => isInt(id) && id >= 0 && id < nodes.length && nodes[id].kind === 'nest'), 'map.nests: неверные гнёзда.')
  const patchNodes = nodes.filter((node) => node.kind === 'patch').map((node) => node.id)
  const mapParams = (header.config as Rec).map_params
  const economy = (header.config as Rec).economy
  const queenMaxHp = isRec(economy) && isInt(economy.queen_hp) ? economy.queen_hp : 240
  const worldSide = isRec(mapParams) && isInt(mapParams.world_side) ? mapParams.world_side : Math.max(...nodes.map((n) => Math.max(n.x, n.y)))

  // ---------------------------------------------------------------- events, frames, result
  const events: EcoEvent[] = []
  const frames: EcoFrame[] = []
  let eventsHead = ''
  let lastFrameTick = -1
  let lastEventTick = 0
  for (let r = 1; r < records.length - 1; r += 1) {
    const record = records[r]
    if (record.type === 'event') {
      exactKeys(record, EVENT_KEYS, `event ${r}`)
      need(EVENT_KINDS.includes(record.kind as EventKind) && ['tick', 'colony', 'other', 'node', 'a', 'b'].every((key) => isInt(record[key])), `event ${r}: неверное событие.`)
      const tick = record.tick as number
      need(tick >= lastEventTick && tick >= lastFrameTick, `event ${r}: события идут не в порядке времени.`)
      lastEventTick = tick
      const item: EcoEvent = { index: events.length, tick, kind: record.kind as EventKind, colony: record.colony as number, other: record.other as number,
        node: record.node as number, a: record.a as number, b: record.b as number }
      need(item.colony >= 0 && item.colony < colonyCount && item.other >= -1 && item.other < colonyCount && item.node >= -1 && item.node < nodes.length, `event ${r}: вне диапазона.`)
      eventsHead = hashDomain(EVENTS_DOMAIN, utf8(canonicalString({ prev: eventsHead, event: { tick, kind: item.kind, colony: item.colony, other: item.other, node: item.node, a: item.a, b: item.b } })))
      events.push(item)
    } else if (record.type === 'frame') {
      const frame = parseFrame(record, frames.length, colonyCount, nodes, patchNodes, lastFrameTick)
      lastFrameTick = frame.tick
      frames.push(frame)
    } else {
      throw new EcoReplayError(`Запись ${r + 1}: неизвестный тип «${String(record.type)}».`)
    }
  }
  need(frames.length > 0 && frames[0].tick === 0 && frames[0].reason === 'start', 'Первый кадр должен быть начальным состоянием.')

  const result = records[records.length - 1]
  need(result.type === 'result', 'Последняя запись должна быть result.')
  exactKeys(result, policy ? [...RESULT_KEYS, 'decision_count', 'decisions_hash'] : RESULT_KEYS, 'result')
  if (policy) {
    need(isInt(result.decision_count) && result.decision_count >= 0 && typeof result.decisions_hash === 'string' && /^[0-9a-f]{64}$/.test(result.decisions_hash), 'result: неверные decision_count/decisions_hash.')
    policy.decisionCount = result.decision_count as number
    policy.decisionsHash = result.decisions_hash as string
  }
  need(isInt(result.ticks_completed) && frames[frames.length - 1].reason === 'final' && frames[frames.length - 1].tick === result.ticks_completed, 'result: последний кадр должен быть финальным.')
  need(result.event_count === events.length && result.events_hash === eventsHead, 'result.events_hash не совпадает с записями событий.')
  need(result.frame_count === frames.length && result.record_count === records.length, 'result: число записей не совпадает.')
  need(result.final_state_hash === frames[frames.length - 1].stateHash, 'result.final_state_hash не совпадает с финальным кадром.')
  need(events.every((item) => item.tick < (result.ticks_completed as number)), 'Событие позже последнего завершённого тика.')
  const outcome = result.world_outcome as Rec
  need(isRec(outcome) && typeof outcome.finished === 'boolean' && typeof outcome.reason === 'string' && isInt(outcome.terminal_tick), 'result.world_outcome: неверный формат.')
  need(Array.isArray(result.colony_outcomes) && result.colony_outcomes.length === colonyCount, 'result.colony_outcomes: одна запись на колонию.')
  const colonyOutcomes = (result.colony_outcomes as Rec[]).map((item, index) => {
    need(isRec(item) && item.colony === index && STATUSES.includes(item.status as ColonyStatus) && typeof item.reason === 'string' && isInt(item.terminal_tick), 'result.colony_outcomes: неверная запись.')
    return { colony: index, status: item.status as ColonyStatus, reason: item.reason as string, terminalTick: item.terminal_tick as number }
  })

  return {
    fileName, seed: header.seed as string, fixture: String(header.fixture), roles: header.roles as Record<string, string>, colonyCount,
    ticksRequested: header.ticks_requested as number, ticksCompleted: result.ticks_completed as number, frameEvery: header.frame_every as number,
    configHash: header.config_hash as string, mapHash: header.map_hash as string, rulesVersion: String(header.rules_version),
    nodes, edges, nests, patchNodes, worldSide, queenMaxHp, decisionK: header.decision_k as number, frames, events, eventsHash: eventsHead, finalStateHash: result.final_state_hash as string,
    chainHead: previous, worldOutcome: { finished: outcome.finished as boolean, reason: outcome.reason as string, terminalTick: outcome.terminal_tick as number },
    colonyOutcomes, verified: { chain: true, configHash: true, mapHash: true, eventsHash: true }, policy,
  }
}

function parsePolicy(value: unknown, header: Rec): EcoPolicyInfo {
  need(isRec(value), 'header.policy: ожидается объект.')
  exactKeys(value, 'curriculum' in value ? ['schema', 'policy_seed', 'policies', 'curriculum'] : ['schema', 'policy_seed', 'policies'], 'header.policy')
  need(!('curriculum' in value) || (typeof value.curriculum === 'string' && CURRICULUM_PROFILES.includes(value.curriculum)), `header.policy: неизвестный curriculum «${String(value.curriculum)}».`)
  need(value.schema === POLICY_SCHEMA, `Неподдерживаемая версия политик: «${String(value.schema)}».`)
  need(typeof value.policy_seed === 'string' && /^(0|[1-9][0-9]*)$/.test(value.policy_seed) && Array.isArray(value.policies) && value.policies.length === header.colonies && header.fixture === 'policy',
    'header.policy: неверные policy_seed или список политик.')
  const policies = (value.policies as unknown[]).map((item, index) => {
    need(isRec(item), 'header.policy: неверная запись.')
    exactKeys(item, ['colony', 'kind', 'variant', 'stream'], 'header.policy.policy')
    need(item.colony === index && typeof item.kind === 'string' && POLICY_KINDS.includes(item.kind) && typeof item.variant === 'string' && typeof item.stream === 'string',
      `header.policy: неверная политика колонии ${index}.`)
    return { colony: index, kind: item.kind as string, variant: item.variant as string }
  })
  return { seed: value.policy_seed as string, policies, decisionCount: 0, decisionsHash: '', curriculum: 'curriculum' in value ? (value.curriculum as string) : 'none' }
}

function parseFrame(record: Rec, index: number, colonyCount: number, nodes: EcoNode[], patchNodes: number[], lastTick: number): EcoFrame {
  exactKeys(record, FRAME_KEYS, `frame ${index}`)
  need(isInt(record.tick) && record.tick > lastTick && FRAME_REASONS.includes(record.reason as FrameReason)
    && typeof record.state_hash === 'string' && record.state_hash.length === 64, `frame ${index}: неверный заголовок кадра.`)
  need(Array.isArray(record.patches) && record.patches.length === patchNodes.length && record.patches.every(isInt), `frame ${index}: неверные запасы площадок.`)
  need(Array.isArray(record.piles), `frame ${index}: неверные кучи.`)
  const piles = (record.piles as unknown[]).map((row) => { const [node, stock] = intRow(row, 2, `frame ${index}.piles`); return { node, stock } })
  need(Array.isArray(record.colonies) && record.colonies.length === colonyCount, `frame ${index}: одна строка на колонию.`)
  const colonies: ColonyFrame[] = (record.colonies as Rec[]).map((row, colony) => {
    need(isRec(row), `frame ${index}: неверная строка колонии.`)
    exactKeys(row, COLONY_KEYS, `frame ${index}.colony`)
    need(row.id === colony && STATUSES.includes(row.status as ColonyStatus) && COLONY_KEYS.filter((key) => key !== 'status').every((key) => isInt(row[key])), `frame ${index}: неверная колония ${colony}.`)
    return { id: colony, status: row.status as ColonyStatus, food: row.food as number, queenHp: row.queen_hp as number, workers: row.workers as number,
      soldiers: row.soldiers as number, broodWorkers: row.brood_workers as number, broodSoldiers: row.brood_soldiers as number, starvationDebt: row.starvation_debt as number }
  })
  need(Array.isArray(record.cohorts), `frame ${index}: неверные когорты.`)
  const seen = new Set<number>()
  const cohorts: CohortState[] = (record.cohorts as unknown[]).map((row) => {
    const [id, colony, caste, count, hp, load, kind, phase, where, a, b, depart, arrive, targetNode, targetColony] = intRow(row, COHORT_FIELDS.length, `frame ${index}.cohort`)
    need(!seen.has(id) && colony >= 0 && colony < colonyCount && (caste === 0 || caste === 1) && count >= 1 && hp >= 1 && load >= 0 && kind >= 0 && kind < KIND_NAMES.length
      && phase >= 0 && phase < PHASE_NAMES.length && (where === 0 || where === 1) && a >= 0 && a < nodes.length && (where === 0 ? b === -1 : b >= 0 && b < nodes.length && depart < arrive)
      && targetNode >= -1 && targetNode < nodes.length && targetColony >= -1 && targetColony < colonyCount, `frame ${index}: неверная когорта ${id}.`)
    seen.add(id)
    return { id, colony, caste: caste === 0 ? 'worker' : 'soldier', count, hp, load, kind: KIND_NAMES[kind], phase: PHASE_NAMES[phase], atNode: where === 0,
      from: a, to: b, depart, arrive, targetNode, targetColony }
  })
  need(Array.isArray(record.memory) && record.memory.length <= colonyCount, `frame ${index}: неверная память.`)
  const memory: MemoryFrame[] = (record.memory as Rec[]).map((row) => {
    need(isRec(row), `frame ${index}: неверная запись памяти.`)
    exactKeys(row, MEMORY_KEYS, `frame ${index}.memory`)
    need(isInt(row.colony) && row.colony >= 0 && row.colony < colonyCount && Array.isArray(row.visited) && Array.isArray(row.discovered)
      && Array.isArray(row.foreign) && Array.isArray(row.sightings), `frame ${index}: неверная память колонии.`)
    const visited = (row.visited as unknown[]).map((item) => {
      const [node, seenTick, lastStock, lastStockTick, queenSeen, danger] = intRow(item, VISITED_FIELDS.length, `frame ${index}.visited`)
      need(node >= 0 && node < nodes.length && (queenSeen === 0 || queenSeen === 1 || queenSeen === 2), `frame ${index}: неверный посещённый узел.`)
      return { node, seenTick, lastStock, lastStockTick, queenSeen: queenSeen as 0 | 1 | 2, danger }
    })
    const visitedIds = new Set(visited.map((item) => item.node))
    need(visitedIds.size === visited.length, `frame ${index}: повтор посещённого узла.`)
    const discovered = (row.discovered as unknown[]).map((item) => { need(isInt(item) && item >= 0 && item < nodes.length && !visitedIds.has(item), `frame ${index}: неверный известный узел.`); return item as number })
    const foreign = (row.foreign as unknown[]).map((item) => {
      const [colony, firstContact, lastSeen, workersBucket, soldiersBucket, nestNode, theyAttacked, iAttacked, killsMine, killsTheirs] = intRow(item, FOREIGN_FIELDS.length, `frame ${index}.foreign`)
      need(colony >= 0 && colony < colonyCount && colony !== row.colony && nestNode >= -1 && nestNode < nodes.length, `frame ${index}: неверная запись о колонии.`)
      return { colony, firstContact, lastSeen, workersBucket, soldiersBucket, nestNode, theyAttacked, iAttacked, killsMine, killsTheirs }
    })
    const sightings = (row.sightings as unknown[]).map((item) => {
      const [node, colony, tick, workersBucket, soldiersBucket, aggression] = intRow(item, SIGHTING_FIELDS.length, `frame ${index}.sighting`)
      need(visitedIds.has(node) && colony >= 0 && colony < colonyCount && (aggression === 0 || aggression === 1), `frame ${index}: неверное наблюдение.`)
      return { node, colony, tick, workersBucket, soldiersBucket, aggression: aggression === 1 }
    })
    return { colony: row.colony as number, visited, discovered, foreign, sightings }
  })
  return { index, tick: record.tick as number, reason: record.reason as FrameReason, stateHash: record.state_hash as string,
    patches: record.patches as number[], piles, colonies, cohorts, memory }
}

export async function parseEcoReplayFile(file: { name: string; arrayBuffer(): Promise<ArrayBuffer> }, onProgress?: (p: EcoParseProgress) => void): Promise<EcoReplay> {
  return parseEcoReplayBytes(new Uint8Array(await file.arrayBuffer()), file.name, onProgress)
}
