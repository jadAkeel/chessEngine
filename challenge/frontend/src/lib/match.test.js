import test from 'node:test'
import assert from 'node:assert/strict'
import { Chess } from 'chess.js'
import {
  applyUci, buildReport, colorOf, fenAt, formatPoints, matchScore, modelFor, pgnFor, resolveOutcome, winnerOf,
} from './match.js'
import { INITIAL_PLAYER, baseUrlProblem, endpointPreview, playerProblems, publicPlayer, requestPlayer } from './providers.js'
import { MoveError, redactKey, requestMove } from './api.js'

const players = {
  A: { ...INITIAL_PLAYER, model: 'alpha-model', api_key: 'sk-secret-A', base_url: 'https://api.example.com/v1' },
  B: { ...INITIAL_PLAYER, provider: 'anthropic', model: 'beta-model', api_key: 'sk-secret-B' },
}

test('colors swap between paired games and results are attributed to models', () => {
  assert.equal(modelFor(0, 'w'), 'A')
  assert.equal(modelFor(0, 'b'), 'B')
  assert.equal(modelFor(1, 'w'), 'B')
  assert.equal(modelFor(1, 'b'), 'A')
  assert.equal(colorOf(1, 'A'), 'b')
  const games = [
    { index: 0, moves: [], outcome: { result: '1-0', termination: 'CHECKMATE' } },
    { index: 1, moves: [], outcome: { result: '1-0', termination: 'CHECKMATE' } },
  ]
  // White won both games: A won game 1 as White, B won game 2 as White.
  assert.deepEqual(matchScore(games), { A: 1, B: 1 })
  assert.equal(winnerOf(games[0]), 'A')
  assert.equal(winnerOf(games[1]), 'B')
  const blackWins = [{ index: 1, moves: [], outcome: { result: '0-1' } }]
  assert.deepEqual(matchScore(blackWins), { A: 1, B: 0 })
  assert.deepEqual(matchScore([{ index: 0, outcome: { result: '1/2-1/2' } }, { index: 1, outcome: null }]), { A: 0.5, B: 0.5 })
})

test('points format with halves', () => {
  assert.equal(formatPoints(0), '0')
  assert.equal(formatPoints(0.5), '½')
  assert.equal(formatPoints(1.5), '1½')
  assert.equal(formatPoints(2), '2')
})

test('server outcome wins, then local detection, then the move-limit draw', () => {
  const game = new Chess()
  assert.deepEqual(resolveOutcome('1/2-1/2', 'THREEFOLD_REPETITION', game, 20), { result: '1/2-1/2', termination: 'THREEFOLD_REPETITION' })
  assert.equal(resolveOutcome(null, null, game, 1), null)
  for (const uci of ['f2f3', 'e7e5', 'g2g4', 'd8h4']) applyUci(game, uci)
  assert.deepEqual(resolveOutcome(null, null, game, 4), { result: '0-1', termination: 'CHECKMATE' })
  assert.equal(resolveOutcome(null, null, new Chess(), 300).termination, 'MOVE_LIMIT')
})

test('applyUci rejects malformed and illegal moves without changing the board', () => {
  const game = new Chess()
  const start = game.fen()
  assert.equal(applyUci(game, 'e2e5'), null)
  assert.equal(applyUci(game, 'bogus'), null)
  assert.equal(applyUci(game, 42), null)
  assert.equal(game.fen(), start)
  assert.equal(applyUci(game, 'e2e4').san, 'e4')
})

test('chess.js and the server agree on en passant FEN fields', () => {
  const game = new Chess()
  for (const uci of ['e2e4', 'a7a6', 'e4e5', 'd7d5']) applyUci(game, uci)
  assert.equal(game.fen(), 'rnbqkbnr/1pp1pppp/p7/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3')
  const noCapture = new Chess()
  applyUci(noCapture, 'e2e4')
  assert.equal(noCapture.fen(), 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1')
})

test('fenAt returns the start position and recorded positions', () => {
  const game = { moves: [{ fen_after: 'x' }] }
  assert.equal(fenAt(game, 0), new Chess().fen())
  assert.equal(fenAt(game, 1), 'x')
})

test('reports and PGN never contain API keys and label swapped colors', () => {
  const moves = [{ ply: 1, color: 'w', model: 'B', san: 'e4', uci: 'e2e4', fen_after: 'f', explanation: 'Center {x}', attempts: 1, elapsed_ms: 5, usage: {} }]
  const games = [
    { index: 0, moves: [], outcome: { result: '0-1', termination: 'CHECKMATE' } },
    { index: 1, moves, outcome: null },
  ]
  const report = buildReport({ mode: 'paired', players, games, generatedAt: new Date('2026-01-02T00:00:00Z') })
  const text = JSON.stringify(report)
  assert.ok(!text.includes('sk-secret'))
  assert.equal(report.games[1].white, 'B')
  assert.equal(report.games[1].moves[0].model, 'B')
  assert.equal(report.games[0].winner, 'B')
  assert.equal(report.complete, false)
  assert.deepEqual(report.score, { A: 0, B: 1 })
  const pgn = pgnFor(games[1], players, new Date('2026-01-02T00:00:00Z'))
  assert.ok(!pgn.includes('sk-secret'))
  assert.match(pgn, /\[White "Model B: beta-model"\]/)
  assert.match(pgn, /\[Black "Model A: alpha-model"\]/)
  assert.match(pgn, /1\. e4 \{Center x\}/)
})

test('provider errors redact the submitted key before display or export', async () => {
  const originalFetch = globalThis.fetch
  globalThis.fetch = async () => ({
    ok: false, status: 400,
    json: async () => ({ detail: `Provider rejected ${players.A.api_key}; echoed ${players.A.api_key}` }),
  })
  try {
    await assert.rejects(requestMove(players.A, []), error => {
      assert.ok(error instanceof MoveError)
      assert.equal(error.kind, 'config')
      assert.equal(error.message, 'Provider rejected [REDACTED API KEY]; echoed [REDACTED API KEY]')
      const report = buildReport({ mode: 'single', players, games: [{ index: 0, moves: [], outcome: null, failures: [{ ply: 1, model: 'A', color: 'w', kind: error.kind, message: error.message }] }] })
      assert.ok(!JSON.stringify(report).includes(players.A.api_key))
      return true
    })
    assert.equal(redactKey('ordinary provider error', players.A.api_key), 'ordinary provider error')
  } finally {
    globalThis.fetch = originalFetch
  }
})

test('request payload maps presets and keeps the key only in the request', () => {
  const request = requestPlayer(players.A)
  assert.equal(request.provider, 'openai_compatible')
  assert.equal(request.base_url, 'https://api.example.com/v1')
  assert.equal(request.api_key, 'sk-secret-A')
  assert.equal(requestPlayer({ ...players.A, provider: 'openrouter' }).base_url, 'https://openrouter.ai/api/v1')
  assert.equal(requestPlayer(players.B).provider, 'anthropic')
  assert.ok(!('api_key' in publicPlayer(players.A)))
  assert.equal(endpointPreview({ ...players.A, base_url: 'https://api.example.com/v1/' }), 'https://api.example.com/v1/chat/completions')
  assert.equal(endpointPreview({ ...players.B, provider: 'gemini', model: 'g' }), 'https://generativelanguage.googleapis.com/v1beta/models/g:generateContent')
})

test('setup validation explains bad custom URLs and fields', () => {
  assert.equal(baseUrlProblem('https://api.example.com/v1'), '')
  assert.match(baseUrlProblem('http://api.example.com'), /https/)
  assert.match(baseUrlProblem('https://u:p@api.example.com'), /credentials/)
  assert.match(baseUrlProblem('https://api.example.com:8443/v1'), /port/i)
  assert.match(baseUrlProblem('https://api.example.com/v1/chat/completions'), /base URL only/)
  assert.deepEqual(playerProblems(players.A), {})
  const problems = playerProblems({ ...INITIAL_PLAYER, max_tokens: 10 })
  assert.deepEqual(Object.keys(problems).sort(), ['api_key', 'base_url', 'max_tokens', 'model'])
})

test('reliability counts failed requests and retried moves per model across swapped games', async () => {
  const { reliabilitySummary } = await import('./match.js')
  const games = [
    { index: 0, moves: [{ ply: 1, color: 'w', model: 'A', attempts: 2 }], failures: [{ ply: 2, model: 'B', kind: 'provider' }] },
    { index: 1, moves: [{ ply: 1, color: 'w', attempts: 2 }], failures: [{ ply: 1, model: 'B' }, { ply: 2, model: 'A' }] },
  ]
  assert.deepEqual(reliabilitySummary(games), {
    A: { failed_requests: 1, retried_moves: 1 },
    B: { failed_requests: 2, retried_moves: 1 },
  })
})

test('setting differences are listed and included in the report', async () => {
  const { settingDifferences } = await import('./providers.js')
  assert.deepEqual(settingDifferences(players), [])
  const uneven = { A: players.A, B: { ...players.B, max_tokens: '4096', reasoning: 'high' } }
  assert.deepEqual(settingDifferences(uneven).map(item => item.field), ['reasoning', 'max_tokens'])
  const report = buildReport({ mode: 'single', players: uneven, games: [{ index: 0, moves: [], outcome: null, failures: [{ ply: 1, model: 'A', color: 'w', kind: 'rate', message: 'm', extra: 'x' }] }] })
  assert.equal(report.settings_match, false)
  assert.equal(report.games[0].failures[0].kind, 'rate')
  assert.ok(!('extra' in report.games[0].failures[0]))
  assert.equal(report.reliability.A.failed_requests, 1)
})
