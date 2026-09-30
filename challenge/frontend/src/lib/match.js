import { Chess } from 'chess.js'
import { publicPlayer, settingDifferences } from './providers.js'

export const MAX_PLIES = 300
export const REPORT_FORMAT = 'chess-model-benchmark/v2'

const TERMINATIONS = {
  CHECKMATE: 'Checkmate', STALEMATE: 'Stalemate', INSUFFICIENT_MATERIAL: 'Insufficient material',
  SEVENTYFIVE_MOVES: '75-move rule', FIVEFOLD_REPETITION: 'Fivefold repetition',
  FIFTY_MOVES: '50-move rule', THREEFOLD_REPETITION: 'Threefold repetition',
  MOVE_LIMIT: `Move limit (${MAX_PLIES} half-moves), adjudicated draw`,
}

export const otherModel = model => (model === 'A' ? 'B' : 'A')

// Game 1: A is White. Game 2 swaps colors so each model plays both sides.
export const modelFor = (gameIndex, color) => ((gameIndex === 0) === (color === 'w') ? 'A' : 'B')
export const colorOf = (gameIndex, model) => (modelFor(gameIndex, 'w') === model ? 'w' : 'b')
export const gameCount = mode => (mode === 'paired' ? 2 : 1)

export function displayResult(result) {
  return result === '1/2-1/2' ? '½–½' : result === '1-0' ? '1–0' : result === '0-1' ? '0–1' : '*'
}

export function terminationLabel(code) {
  if (!code) return 'Game over'
  return TERMINATIONS[code] || code.toLowerCase().replaceAll('_', ' ').replace(/^./, c => c.toUpperCase())
}

export function localOutcome(game) {
  if (game.isCheckmate()) return { result: game.turn() === 'w' ? '0-1' : '1-0', termination: 'CHECKMATE' }
  if (game.isStalemate()) return { result: '1/2-1/2', termination: 'STALEMATE' }
  if (game.isInsufficientMaterial()) return { result: '1/2-1/2', termination: 'INSUFFICIENT_MATERIAL' }
  if (game.isThreefoldRepetition()) return { result: '1/2-1/2', termination: 'THREEFOLD_REPETITION' }
  if (game.isDraw()) return { result: '1/2-1/2', termination: 'FIFTY_MOVES' }
  return null
}

// The server's result is authoritative; the local check and move limit are fallbacks.
export function resolveOutcome(serverResult, serverTermination, game, ply) {
  if (['1-0', '0-1', '1/2-1/2'].includes(serverResult)) return { result: serverResult, termination: serverTermination || null }
  const local = localOutcome(game)
  if (local) return local
  if (ply >= MAX_PLIES) return { result: '1/2-1/2', termination: 'MOVE_LIMIT' }
  return null
}

export function pointsFor(result, color) {
  if (result === '1/2-1/2') return 0.5
  return result === (color === 'w' ? '1-0' : '0-1') ? 1 : 0
}

export function matchScore(games) {
  return games.reduce((score, game) => {
    if (!game.outcome) return score
    score.A += pointsFor(game.outcome.result, colorOf(game.index, 'A'))
    score.B += pointsFor(game.outcome.result, colorOf(game.index, 'B'))
    return score
  }, { A: 0, B: 0 })
}

export function formatPoints(value) {
  return Number.isInteger(value) ? String(value) : `${Math.floor(value) || ''}½`
}

export function winnerOf(game) {
  if (!game?.outcome || game.outcome.result === '1/2-1/2') return null
  return modelFor(game.index, game.outcome.result === '1-0' ? 'w' : 'b')
}

export function totalTokens(usage) {
  if (!usage || typeof usage !== 'object') return null
  const { input_tokens: input, output_tokens: output } = usage
  return typeof input === 'number' && typeof output === 'number' ? input + output : null
}

export function applyUci(game, uci) {
  if (typeof uci !== 'string' || !/^[a-h][1-8][a-h][1-8][qrbn]?$/.test(uci)) return null
  try {
    return game.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] })
  } catch {
    return null
  }
}

export function fenAt(game, ply) {
  if (!ply) return new Chess().fen()
  return game.moves[ply - 1]?.fen_after || new Chess().fen()
}

function modelName(players, model) {
  return players[model].model.trim() || `Model ${model}`
}

export function pgnFor(game, players, date = new Date()) {
  const board = new Chess()
  const white = modelFor(game.index, 'w')
  const black = otherModel(white)
  const result = game.outcome?.result || '*'
  board.setHeader('Event', 'Chess Model Benchmark')
  board.setHeader('Site', 'Chess Model Challenge')
  board.setHeader('Date', date.toISOString().slice(0, 10).replaceAll('-', '.'))
  board.setHeader('Round', String(game.index + 1))
  board.setHeader('White', `Model ${white}: ${modelName(players, white)}`)
  board.setHeader('Black', `Model ${black}: ${modelName(players, black)}`)
  board.setHeader('Result', result)
  if (game.outcome) board.setHeader('Termination', terminationLabel(game.outcome.termination))
  for (const move of game.moves) {
    applyUci(board, move.uci)
    if (move.explanation) board.setComment(move.explanation.replace(/[{}]/g, ''))
  }
  return board.pgn()
}

// Failed requests are recorded per game; retried moves had an invalid first reply.
export function reliabilitySummary(games) {
  const summary = { A: { failed_requests: 0, retried_moves: 0 }, B: { failed_requests: 0, retried_moves: 0 } }
  for (const game of games) {
    for (const failure of game.failures || []) summary[failure.model].failed_requests += 1
    for (const move of game.moves) if (move.attempts > 1) summary[move.model || modelFor(game.index, move.color)].retried_moves += 1
  }
  return summary
}

export function buildReport({ mode, players, games, generatedAt = new Date() }) {
  const score = matchScore(games)
  return {
    format: REPORT_FORMAT,
    generated_at: generatedAt.toISOString(),
    mode,
    complete: games.length === gameCount(mode) && games.every(game => game.outcome),
    note: 'Assembled in the browser for review. Explanations are public move rationales, not private chain-of-thought.',
    models: { A: publicPlayer(players.A), B: publicPlayer(players.B) },
    score,
    reliability: reliabilitySummary(games),
    settings_match: settingDifferences(players).length === 0,
    setting_differences: settingDifferences(players),
    games: games.map(game => ({
      game: game.index + 1,
      white: modelFor(game.index, 'w'),
      black: modelFor(game.index, 'b'),
      result: game.outcome?.result || '*',
      termination: game.outcome ? terminationLabel(game.outcome.termination) : 'In progress',
      winner: winnerOf(game),
      failures: (game.failures || []).map(({ ply, model, color, kind, message }) => ({ ply, model, color, kind, message })),
      moves: game.moves.map(move => ({
        ply: move.ply, color: move.color, model: modelFor(game.index, move.color), san: move.san, uci: move.uci,
        fen_after: move.fen_after, explanation: move.explanation, attempts: move.attempts,
        elapsed_ms: move.elapsed_ms, usage: move.usage,
      })),
    })),
  }
}
