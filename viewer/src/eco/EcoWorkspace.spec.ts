import { mount } from '@vue/test-utils'
import { afterEach, describe, expect, it } from 'vitest'
import EcoWorkspace from './EcoWorkspace.vue'
import { sampleReplay } from './testing'
import type { EcoEvent, EcoReplay } from './types'

/**
 * Perspective regression (FOR-47 review): in "Вид колонии" nothing around the map may depend on facts the colony has not observed by the displayed tick —
 * not the live/orphan/extinct summary, the timeline markers and jumps, the feed and its counters, the future toggle, the fixture roles of rivals,
 * nor an omniscient highlight carried over from "Все сведения".
 */
const replay = sampleReplay()
const HIDDEN_KINDS = new Set(['queen_dead', 'extinct'])
// colony 0's queen is killed by colony 3 at tick 2621 and the colony goes extinct at 4400; colonies 1 and 5 never met colony 0.
const RIVAL = 0
const OBSERVERS = [1, 5]

function withoutRivalDeath(source: EcoReplay): EcoReplay {
  const events = source.events
    .filter((event) => !(event.colony === RIVAL && HIDDEN_KINDS.has(event.kind)))
    .map((event, index) => ({ ...event, index }))
  const frames = source.frames.map((frame) => (frame.tick < 2621 ? frame : {
    ...frame,
    colonies: frame.colonies.map((colony) => (colony.id === RIVAL ? { ...colony, status: 'alive' as const, queenHp: source.queenMaxHp, workers: colony.workers + 7, food: colony.food + 123 } : colony)),
  }))
  return { ...source, events, frames }
}

function open(source: EcoReplay, search: string) {
  window.history.replaceState({}, '', `/?${search}`)
  return mount(EcoWorkspace, { props: { replay: source }, attachTo: document.body })
}

describe('colony view does not leak global or future knowledge', () => {
  afterEach(() => { window.history.replaceState({}, '', '/') })

  it('sanity: the alteration really is invisible to the observers and really changes the omniscient view', () => {
    const alt = withoutRivalDeath(replay)
    expect(replay.events.filter((e) => e.colony === RIVAL && HIDDEN_KINDS.has(e.kind))).toHaveLength(2)
    expect(alt.events.length).toBe(replay.events.length - 2)
    for (const observer of OBSERVERS) {
      const memory = replay.frames[replay.frames.length - 1].memory.find((m) => m.colony === observer)!
      expect(memory.foreign.map((f) => f.colony)).not.toContain(RIVAL)
    }
    const a = open(replay, 'tick=2630')
    const b = open(alt, 'tick=2630')
    expect(a.html()).not.toBe(b.html())
    expect(a.text()).toContain('сирот')
    a.unmount()
    b.unmount()
  })

  for (const observer of OBSERVERS) {
    for (const tick of [1000, 2630, 4500]) {
      it(`colony ${observer} at tick ${tick}: altering an unseen rival's queen death / extinction leaves the whole UI byte-identical`, () => {
        const query = `view=colony&colony=${observer}&tick=${tick}`
        const real = open(replay, query)
        const altered = open(withoutRivalDeath(replay), query)
        expect(altered.html()).toBe(real.html())
        real.unmount()
        altered.unmount()
      })
    }
  }

  it('shows no summary of other colonies, no foreign roles, no future toggle and no hidden-colony status styling', () => {
    const wrapper = open(replay, 'view=colony&colony=5&tick=2630')
    const summary = wrapper.get('.eco-summary').text()
    expect(summary).toContain('колония 5')
    expect(summary).not.toMatch(/сирот|вымерло|живых/)
    expect(wrapper.find('.eco-filter-row input[type="checkbox"]').exists()).toBe(false)
    const [raiderColony, role] = Object.entries(replay.roles)[0]
    expect(role).toBeTruthy()
    expect(Number(raiderColony)).not.toBe(5)
    wrapper.findAll('.eco-colony').forEach((card, id) => {
      if (id !== 5) {
        expect(card.text()).not.toContain(role)
        expect(card.classes().some((c) => /^status-(alive|orphan|extinct)$/.test(c))).toBe(false)
      }
    })
    expect(wrapper.findAll('.eco-event').some((item) => item.text().includes('Гибель матки'))).toBe(false)
    expect(wrapper.findAll('.eco-moments button.kind-queen_dead')).toHaveLength(0)
    expect(wrapper.findAll('.eco-moments button.kind-extinct')).toHaveLength(0)
    // event counters in the chips only count what this colony has seen
    const total = wrapper.findAll('.eco-chips .chip small').reduce((sum, el) => sum + Number(el.text()), 0)
    const own = replay.events.filter((e) => e.tick < 2630 && (e.colony === 5 || ((e.kind === 'combat' || e.kind === 'queen_dead') && e.other === 5))).length
    expect(total).toBe(own)
    wrapper.unmount()
  })

  it('does not keep an omniscient highlight when switching to the colony view', async () => {
    const wrapper = open(replay, 'tick=0')
    await wrapper.get('.eco-filter-row input[type="checkbox"]').setValue(true)
    let row = wrapper.findAll('.eco-event').find((item) => item.text().includes('Гибель матки'))
    for (let guard = 0; !row && guard < 10; guard += 1) {
      await wrapper.get('.eco-more button').trigger('click')
      row = wrapper.findAll('.eco-event').find((item) => item.text().includes('Гибель матки'))
    }
    await row!.trigger('click')
    expect(wrapper.find('.eco-focus').exists()).toBe(true)
    expect(wrapper.find('.eco-focus-note').exists()).toBe(true)
    await wrapper.findAll('.eco-mode button')[1].trigger('click') // colony view (colony 3, the raider, by default)
    expect(wrapper.find('.eco-focus').exists()).toBe(false)
    expect(wrapper.find('.eco-focus-note').exists()).toBe(false)
    expect(wrapper.find('.eco-event.active').exists()).toBe(false)
    await wrapper.get('.eco-select select').setValue(5)
    expect(wrapper.find('.eco-focus').exists()).toBe(false)
    expect(wrapper.find('.eco-focus-note').exists()).toBe(false)
    expect(wrapper.findAll('.eco-event').some((item) => item.text().includes('Гибель матки'))).toBe(false)
    wrapper.unmount()
  })

  it('only offers moments the colony has already observed, and the jump buttons cannot reach the future', async () => {
    const queenDeath = replay.events.find((e) => e.kind === 'queen_dead' && e.colony === RIVAL) as EcoEvent
    // the killer (colony 3) did take part: the marker appears only after the displayed tick has passed the event
    const early = open(replay, `view=colony&colony=${queenDeath.other}&tick=${queenDeath.tick}`)
    expect(early.findAll('.eco-moments button.kind-queen_dead')).toHaveLength(0)
    early.unmount()
    const late = open(replay, `view=colony&colony=${queenDeath.other}&tick=${queenDeath.tick + 1}`)
    expect(late.findAll('.eco-moments button.kind-queen_dead')).toHaveLength(1)
    late.unmount()

    const wrapper = open(replay, `view=colony&colony=${queenDeath.other}&tick=0`)
    expect(wrapper.findAll('.eco-moments button')).toHaveLength(0)
    await wrapper.get('button[aria-label="Следующее ключевое событие"]').trigger('click')
    expect(wrapper.get('[data-testid="tick"]').text()).toBe('0') // nothing observed yet, nowhere to jump
    wrapper.unmount()

    // all-knowledge mode keeps the full future-history navigation
    const all = open(replay, 'tick=0')
    expect(all.findAll('.eco-moments button').length).toBeGreaterThan(3)
    await all.get('button[aria-label="Следующее ключевое событие"]').trigger('click')
    expect(Number(all.get('[data-testid="tick"]').text())).toBeGreaterThan(0)
    all.unmount()
  })
})
