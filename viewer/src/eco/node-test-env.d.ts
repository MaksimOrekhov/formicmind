// Minimal typings for the Node APIs used only by the Vitest helpers in testing.ts (the project does not depend on @types/node).
declare module 'node:fs' {
  export function readFileSync(path: string): Uint8Array
  export function readFileSync(path: string, encoding: 'utf-8'): string
}
declare module 'node:path' {
  export function resolve(...paths: string[]): string
}
declare const process: { cwd(): string }
