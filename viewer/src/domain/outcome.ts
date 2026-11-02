import type { ReplayTerminal, Side } from './replay'

export interface OutcomePresentation {
  title: string
  detail: string
  tone: 'win' | 'draw' | 'neutral'
}

function terminalDetail(terminal: ReplayTerminal) {
  const tickState = terminal.tickCompleted ? 'тик завершён' : 'тик прерван'
  const reason = terminal.reason === 'queen_death' ? 'гибель матки' : terminal.reason === 'economic_collapse' ? 'экономический коллапс' : 'таймаут'
  return `Терминальный тик ${terminal.terminalTick} · ${reason} · ${terminal.stage} · ${tickState}.`
}

function winnerTitle(winner: Side | null) {
  return winner ? `Победа колонии ${winner}` : 'Победа'
}

export function presentOutcome(terminal: ReplayTerminal): OutcomePresentation {
  switch (terminal.outcome) {
    case 'draw':
      return { title: 'Ничья', detail: terminalDetail(terminal), tone: 'draw' }
    case 'win':
      return {
        title: winnerTitle(terminal.winner),
        detail: terminal.winner ? terminalDetail(terminal) : `Победитель не указан в записи. ${terminalDetail(terminal)}`,
        tone: terminal.winner ? 'win' : 'neutral',
      }
  }
}
