import type { ReplayTerminal, Side } from './replay'

export interface OutcomePresentation {
  title: string
  detail: string
  tone: 'win' | 'draw' | 'neutral'
}

function terminalDetail(terminal: ReplayTerminal) {
  const tickState = terminal.tickCompleted ? 'тик завершён' : 'тик прерван'
  return `Терминальный тик ${terminal.terminalTick} · ${terminal.stage} · ${tickState}.`
}

function winnerTitle(winner: Side | null) {
  return winner ? `Победа колонии ${winner}` : 'Победа'
}

export function presentOutcome(terminal: ReplayTerminal): OutcomePresentation {
  switch (terminal.outcome) {
    case 'win':
      return {
        title: winnerTitle(terminal.winner),
        detail: terminal.winner ? terminalDetail(terminal) : `Победитель не указан в записи. ${terminalDetail(terminal)}`,
        tone: terminal.winner ? 'win' : 'neutral',
      }
    case 'draw':
      return { title: 'Ничья', detail: terminalDetail(terminal), tone: 'draw' }
    case 'collapse':
      return {
        title: terminal.winner ? winnerTitle(terminal.winner) : 'Коллапс',
        detail: terminal.winner ? `Исход: коллапс. ${terminalDetail(terminal)}` : `Победитель не указан в записи. ${terminalDetail(terminal)}`,
        tone: terminal.winner ? 'win' : 'neutral',
      }
    case 'timeout':
      return {
        title: terminal.winner ? winnerTitle(terminal.winner) : 'Таймаут',
        detail: terminal.winner ? `Исход: таймаут. ${terminalDetail(terminal)}` : `Победитель не указан в записи. ${terminalDetail(terminal)}`,
        tone: terminal.winner ? 'win' : 'neutral',
      }
  }
}
