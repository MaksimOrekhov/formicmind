import { flushPromises, mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import App from './App.vue'
import { utf8 } from './data/replayCrypto'

function chooseFile(input: HTMLInputElement, name: string, bytes: Uint8Array) {
  const file = {
    name,
    size: bytes.byteLength,
    arrayBuffer: async () => bytes.slice().buffer,
  } as File
  Object.defineProperty(input, 'files', { configurable: true, value: [file] })
}

describe('local replay import flow', () => {
  it('keeps demo or the previous replay visible until the next file fully validates', async () => {
    const wrapper = mount(App)
    const input = wrapper.get('input[type="file"]').element as HTMLInputElement
    expect(wrapper.text()).toContain('ДЕМО · НЕ СИМУЛЯЦИЯ')

    const validText = await import('../test-fixtures/raid_win.jsonl?raw').then((module) => module.default)
    chooseFile(input, 'raid_win.jsonl', utf8(validText))
    await wrapper.get('input[type="file"]').trigger('change')
    await new Promise((resolve) => setTimeout(resolve, 20))
    await flushPromises()
    expect(wrapper.text()).toContain('REAL REPLAY')
    expect(wrapper.text()).toContain('raid_win.jsonl')

    const invalidBytes = utf8(validText.replace('formicmind.replay', 'formicmind.invalid'))
    chooseFile(input, 'broken.jsonl', invalidBytes)
    await wrapper.get('input[type="file"]').trigger('change')
    await new Promise((resolve) => setTimeout(resolve, 20))
    await flushPromises()
    expect(wrapper.text()).toContain('REAL REPLAY')
    expect(wrapper.text()).toContain('Файл не открыт')
    wrapper.unmount()
  })
})
