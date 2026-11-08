import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import EcoApp from './EcoApp.vue'
import Root from '../Root.vue'
import { learnedSampleBytes, policySampleBytes, rechain, sampleBytes, textLines, joinLines } from './testing'

function respondWithSample() {
  const bytes = sampleBytes()
  vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, status: 200, arrayBuffer: async () => bytes.slice().buffer })))
}

async function settle(wrapper: ReturnType<typeof mount>) {
  for (let i = 0; i < 6; i += 1) {
    await new Promise((resolve) => setTimeout(resolve, 30))
    await flushPromises()
  }
  await wrapper.vm.$nextTick()
}

function chooseFile(wrapper: ReturnType<typeof mount>, name: string, bytes: Uint8Array) {
  const input = wrapper.get('input[type="file"]').element as HTMLInputElement
  Object.defineProperty(input, 'files', { configurable: true, value: [{ name, size: bytes.byteLength, arrayBuffer: async () => bytes.slice().buffer }] })
  return wrapper.get('input[type="file"]').trigger('change')
}

describe('E1 viewer application', () => {
  beforeEach(() => { window.history.replaceState({}, '', '/') })
  afterEach(() => { vi.unstubAllGlobals() })

  it('loads the bundled real run and shows the 8 colonies, the map and the story', async () => {
    respondWithSample()
    const wrapper = mount(EcoApp, { attachTo: document.body })
    await settle(wrapper)
    expect(wrapper.text()).toContain('8 колоний · 96 узлов · 166 рёбер')
    expect(wrapper.text()).toContain('цепочка ✓')
    expect(wrapper.findAll('.eco-colony')).toHaveLength(8)
    expect(wrapper.findAll('.eco-node')).toHaveLength(96)
    expect(wrapper.findAll('.eco-node.kind-nest')).toHaveLength(8)
    expect(wrapper.find('[data-testid="tick"]').text()).toBe('0')
    expect(wrapper.text()).not.toContain('Файл не открыт')
    wrapper.unmount()
  })

  it('jumps to an event from the feed and highlights it, then shows the queen death and the orphan', async () => {
    respondWithSample()
    window.history.replaceState({}, '', '/?tick=2630')
    const wrapper = mount(EcoApp, { attachTo: document.body })
    await settle(wrapper)
    expect(wrapper.find('[data-testid="tick"]').text()).toBe('2630')
    const row = wrapper.findAll('.eco-event').find((item) => item.text().includes('Гибель матки'))!
    expect(row.text()).toContain('тик 2621')
    expect(wrapper.findAll('.eco-colony')[0].text()).toContain('сирота')
    // back to the start, then use the feed: enable future events and click the queen death
    await wrapper.findAll('.eco-controls button')[0].trigger('click')
    expect(wrapper.find('[data-testid="tick"]').text()).toBe('0')
    expect(wrapper.findAll('.eco-colony')[0].text()).toContain('жива')
    await wrapper.get('.eco-filter-row input[type="checkbox"]').setValue(true)
    let future = wrapper.findAll('.eco-event').find((item) => item.text().includes('Гибель матки'))
    for (let guard = 0; !future && guard < 10; guard += 1) {
      await wrapper.get('.eco-more button').trigger('click')
      future = wrapper.findAll('.eco-event').find((item) => item.text().includes('Гибель матки'))
    }
    expect(future).toBeDefined()
    await future!.trigger('click')
    expect(wrapper.find('[data-testid="tick"]').text()).toBe('2622')
    expect(wrapper.findAll('.eco-colony')[0].text()).toContain('сирота')
    expect(wrapper.find('.eco-focus-note').text()).toContain('погибла')
    expect(wrapper.find('.eco-focus').exists()).toBe(true)
    wrapper.unmount()
    window.history.replaceState({}, '', '/')
  })

  it('switches to a colony view that hides rivals the colony has never met', async () => {
    respondWithSample()
    window.history.replaceState({}, '', '/?tick=500')
    const wrapper = mount(EcoApp, { attachTo: document.body })
    await settle(wrapper)
    const hidden = wrapper.findAll('.eco-colony').filter((card) => card.text().includes('не известна'))
    expect(hidden).toHaveLength(0)
    const modeButtons = wrapper.findAll('.eco-mode button')
    await modeButtons[1].trigger('click')
    expect(wrapper.get('[data-testid="mode-note"]').text()).toContain('Экранный режим')
    expect(wrapper.findAll('.eco-node').length).toBeLessThan(96)
    const unknown = wrapper.findAll('.eco-colony').filter((card) => card.text().includes('не известна'))
    expect(unknown.length).toBeGreaterThan(0)
    for (const card of unknown) {
      expect(card.text()).toContain('скрыты')
      expect(card.text()).not.toMatch(/рабочие\s*\d/)
    }
    await modeButtons[0].trigger('click')
    expect(wrapper.findAll('.eco-node')).toHaveLength(96)
    wrapper.unmount()
    window.history.replaceState({}, '', '/')
  })

  it('opens the run of independent policies, names who decided and still hides rivals in the colony view', async () => {
    respondWithSample()
    const wrapper = mount(EcoApp, { attachTo: document.body })
    await settle(wrapper)
    expect(wrapper.find('[data-testid="policy-chip"]').exists()).toBe(false) // the bundled default is a fixture run
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, status: 200, arrayBuffer: async () => policySampleBytes().slice().buffer })))
    await wrapper.get('[data-testid="policy-sample"]').trigger('click')
    await settle(wrapper)
    const chip = wrapper.get('[data-testid="policy-chip"]').text()
    expect(chip).toContain('независимые политики')
    expect(chip).toContain('heuristic:aggressor ×3')
    expect(chip).toContain('random_valid ×1')
    expect(wrapper.text()).toContain('eco_policy_seed50012.jsonl')
    expect(wrapper.findAll('.eco-colony')).toHaveLength(8)
    expect(wrapper.findAll('.eco-colony')[6].text()).toContain('random_valid')
    // colony view of colony 0: other colonies' policies are fixture metadata of the viewer and are not shown for colonies it has not met
    await wrapper.findAll('.eco-mode button')[1].trigger('click')
    await wrapper.get('.eco-select select').setValue(0)
    const cards = wrapper.findAll('.eco-colony')
    expect(cards[0].text()).toContain('heuristic:explorer')
    cards.forEach((card, id) => { if (id !== 0) expect(card.text()).not.toMatch(/random_valid|heuristic:/) })
    wrapper.unmount()
  })

  it('opens the run of the PPO-trained network and names it as the policy of all eight colonies', async () => {
    respondWithSample()
    const wrapper = mount(EcoApp, { attachTo: document.body })
    await settle(wrapper)
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, status: 200, arrayBuffer: async () => learnedSampleBytes().slice().buffer })))
    await wrapper.get('[data-testid="learned-sample"]').trigger('click')
    await settle(wrapper)
    const chip = wrapper.get('[data-testid="policy-chip"]').text()
    expect(chip).toMatch(/learned:sample-[0-9a-f]{16} ×8/)
    expect(wrapper.text()).toContain('eco_learned_ppo_seed90000.jsonl')
    expect(wrapper.findAll('.eco-colony')).toHaveLength(8)
    wrapper.unmount()
  })

  it('marks a run played under a curriculum profile as a diagnostic mode, and a full-action-space run is not marked (FOR-50)', async () => {
    respondWithSample()
    const wrapper = mount(EcoApp, { attachTo: document.body })
    await settle(wrapper)
    await chooseFile(wrapper, 'plain.jsonl', policySampleBytes())
    await settle(wrapper)
    expect(wrapper.find('[data-testid="policy-chip"]').exists()).toBe(true)
    expect(wrapper.find('[data-testid="curriculum-chip"]').exists()).toBe(false)
    const nest = rechain(textLines(policySampleBytes()), (record) => (record.type === 'header' ? { ...record, policy: { ...record.policy, curriculum: 'nest-v1' } } : record))
    await chooseFile(wrapper, 'nest.jsonl', nest)
    await settle(wrapper)
    expect(wrapper.get('[data-testid="curriculum-chip"]').text()).toBe('curriculum: nest-v1')
    expect(wrapper.text()).toContain('nest.jsonl')
    wrapper.unmount()
  })

  it('keeps the current replay visible when a broken file is chosen', async () => {
    respondWithSample()
    const wrapper = mount(EcoApp, { attachTo: document.body })
    await settle(wrapper)
    const lines = textLines(sampleBytes())
    const tampered = [...lines]
    const frameLine = lines.findIndex((line, index) => index > 5 && line.includes('"type":"frame"'))
    tampered[frameLine] = tampered[frameLine].replace('"food":', '"fo0d":')
    await chooseFile(wrapper, 'broken.jsonl', joinLines(tampered))
    await settle(wrapper)
    expect(wrapper.text()).toContain('Файл не открыт')
    expect(wrapper.text()).toContain('цепочк')
    expect(wrapper.findAll('.eco-colony')).toHaveLength(8) // the previous replay is still on screen
    await chooseFile(wrapper, 'good.jsonl', sampleBytes())
    await settle(wrapper)
    expect(wrapper.text()).not.toContain('Файл не открыт')
    expect(wrapper.text()).toContain('good.jsonl')
    wrapper.unmount()
  })

  it('shows a readable message if the bundled example cannot be fetched', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: false, status: 404, arrayBuffer: async () => new ArrayBuffer(0) })))
    const wrapper = mount(EcoApp, { attachTo: document.body })
    await settle(wrapper)
    expect(wrapper.text()).toContain('Пример не загружен')
    expect(wrapper.text()).toContain('replay-export')
    wrapper.unmount()
  })

  it('root opens the E1 viewer by default and keeps the L0 viewer reachable', async () => {
    respondWithSample()
    window.location.hash = ''
    const wrapper = mount(Root, { attachTo: document.body })
    await settle(wrapper)
    expect(wrapper.text()).toContain('Ecosystem v1 · зритель replay')
    window.location.hash = '#/l0'
    window.dispatchEvent(new HashChangeEvent('hashchange'))
    await settle(wrapper)
    expect(wrapper.text()).toContain('ДЕМО · НЕ СИМУЛЯЦИЯ')
    wrapper.unmount()
    window.location.hash = ''
  })
})
