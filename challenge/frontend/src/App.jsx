import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Chess } from 'chess.js'
import { Chessboard } from 'react-chessboard'
import {
  AlertTriangle, ArrowDownToLine, BrainCircuit, ChevronLeft, ChevronRight, Clock3, FileJson,
  Loader2, Pause, Play, RotateCcw, ShieldCheck, SkipForward, Swords, Trophy, Zap,
} from 'lucide-react'
import PlayerSetup from './components/PlayerSetup.jsx'
import SampleReplay from './components/SampleReplay.jsx'
import { INITIAL_PLAYER, playerProblems, providerInfo, settingDifferences } from './lib/providers.js'
import { MoveError, requestMove } from './lib/api.js'
import {
  MAX_PLIES, applyUci, buildReport, displayResult, fenAt, formatPoints, gameCount, matchScore,
  modelFor, pgnFor, reliabilitySummary, resolveOutcome, terminationLabel, totalTokens, winnerOf,
} from './lib/match.js'

const START_FEN = new Chess().fen()
const COLOR_NAME = { w: 'White', b: 'Black' }
const ERROR_TITLES = {
  setup: 'Finish setting up the models', auth: 'API key or model access rejected', rate: 'Provider rate limit reached',
  busy: 'Timed out or server busy', network: 'Cannot reach the challenge server', config: 'Request rejected',
  mismatch: 'Board mismatch', provider: 'The model did not produce a move',
}
const ERROR_HINTS = {
  auth: 'Check the key and that it can access this model. While paused you can paste a replacement key and retry.',
  rate: 'Wait a moment, then retry. Nothing was recorded for this turn.',
  busy: 'Retry the move. If it keeps timing out, reset and allow a longer timeout.',
  network: 'Check your connection, then retry. Nothing was recorded for this turn.',
  config: 'Check the base URL, model ID and advanced settings. Settings can be edited before the first move; afterwards reset the match.',
  mismatch: 'The move was not applied. Retry once; if it repeats, export what you have and reset the match.',
  provider: 'Retry the move. The model gets the same position and a fresh request.',
}

function prefersReducedMotion() {
  return typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
}

function downloadText(filename, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type }))
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 1000)
}

function kingSquare(fen, color) {
  const rows = new Chess(fen).board()
  for (const row of rows) for (const piece of row) if (piece?.type === 'k' && piece.color === color) return piece.square
  return null
}

function seconds(ms) {
  return typeof ms === 'number' ? `${(ms / 1000).toFixed(1)}s` : '—'
}

function moveLabel(move) {
  return `${Math.ceil(move.ply / 2)}${move.color === 'w' ? '.' : '…'} ${move.san}`
}

function useElementWidth() {
  const ref = useRef(null)
  const [width, setWidth] = useState(0)
  useEffect(() => {
    const element = ref.current
    if (!element) return undefined
    const measure = () => setWidth(Math.floor(element.clientWidth))
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(element)
    return () => observer.disconnect()
  }, [])
  return [ref, width]
}

function PlayerBar({ color, model, player, thinking, now, toMove, winner }) {
  const name = player.model.trim() || `Model ${model}`
  const elapsed = thinking ? Math.max(0, Math.floor((now - thinking.since) / 1000)) : 0
  return (
    <div className={`player-bar ${toMove ? 'to-move' : ''} ${thinking ? 'is-thinking' : ''}`}>
      <span className={`piece-badge ${color}`} aria-hidden="true">{color === 'w' ? '♔' : '♚'}</span>
      <span className={`model-chip chip-${model.toLowerCase()}`} aria-hidden="true">{model}</span>
      <div className="bar-title">
        <strong title={name}>{name}</strong>
        <span>{COLOR_NAME[color]} · Model {model} · {providerInfo(player.provider).label}</span>
      </div>
      {winner ? <span className="bar-state win"><Trophy size={13} aria-hidden="true" /> Winner</span>
        : thinking ? <span className="bar-state thinking"><Loader2 size={13} className="spin" aria-hidden="true" /> Thinking {elapsed}s<span className="muted"> / {thinking.timeout}s</span></span>
          : toMove ? <span className="bar-state">To move</span> : null}
    </div>
  )
}

export default function App() {
  const gameRef = useRef(new Chess())
  const requestRef = useRef(null)
  const requestIdRef = useRef(0)
  const runningRef = useRef(false)
  const verificationIdRef = useRef({ A: 0, B: 0 })
  const [players, setPlayers] = useState({ A: { ...INITIAL_PLAYER }, B: { ...INITIAL_PLAYER } })
  const [mode, setMode] = useState('paired')
  const [games, setGames] = useState([])
  const [running, setRunning] = useState(false)
  const [stepQueued, setStepQueued] = useState(false)
  const [thinking, setThinking] = useState(null)
  const [error, setError] = useState(null)
  const [review, setReview] = useState(null)
  const [verification, setVerification] = useState({ A: null, B: null })
  const [showErrors, setShowErrors] = useState(false)
  const [announcement, setAnnouncement] = useState('')
  const [now, setNow] = useState(() => Date.now())
  const [boardRef, boardWidth] = useElementWidth()

  runningRef.current = running
  const total = gameCount(mode)
  const current = games[games.length - 1] || null
  const matchOver = games.length === total && games.every(game => game.outcome)
  const between = !!current?.outcome && !matchOver
  const started = games.some(game => game.moves.length > 0)
  const locked = started || !!thinking
  const paused = started && !running && !thinking && !matchOver

  const stopRequest = useCallback(() => {
    requestIdRef.current += 1
    requestRef.current?.abort()
    requestRef.current = null
    setThinking(null)
  }, [])

  useEffect(() => () => {
    requestIdRef.current += 1
    requestRef.current?.abort()
  }, [])

  useEffect(() => {
    if (!thinking) return undefined
    setNow(Date.now())
    const timer = window.setInterval(() => setNow(Date.now()), 500)
    return () => window.clearInterval(timer)
  }, [thinking])

  const setupProblem = () => {
    for (const model of ['A', 'B']) {
      const problems = Object.values(playerProblems(players[model]))
      if (problems.length) return { model, message: `Model ${model}: ${problems[0]}` }
    }
    return null
  }

  const ensureReady = () => {
    const problem = setupProblem()
    if (!problem) return true
    setShowErrors(true)
    setError({ kind: 'setup', model: problem.model, message: problem.message })
    return false
  }

  const updatePlayer = (model, next) => {
    verificationIdRef.current[model] += 1
    setVerification(previous => ({ ...previous, [model]: null }))
    setPlayers(previous => ({ ...previous, [model]: next }))
    if (error?.kind === 'setup') setError(null)
  }

  const verifyPlayer = async model => {
    const problems = Object.values(playerProblems(players[model]))
    if (problems.length) {
      setShowErrors(true)
      setVerification(previous => ({ ...previous, [model]: { state: 'error', message: problems[0] } }))
      return
    }
    const id = ++verificationIdRef.current[model]
    setVerification(previous => ({ ...previous, [model]: { state: 'testing', message: 'Asking the provider for an opening move…' } }))
    try {
      const data = await requestMove(players[model], [])
      if (verificationIdRef.current[model] !== id) return
      const board = new Chess()
      const move = data.fen_before === board.fen() ? applyUci(board, data.move) : null
      if (!move || board.fen() !== data.fen_after) throw new Error('The provider answered, but the test move did not verify.')
      const retried = data.attempts > 1 ? ' after one format retry' : ''
      setVerification(previous => ({ ...previous, [model]: { state: 'ok', message: `Connected. Played ${move.san}${retried} in ${seconds(data.elapsed_ms)}.` } }))
    } catch (err) {
      if (verificationIdRef.current[model] === id) {
        setVerification(previous => ({ ...previous, [model]: { state: 'error', message: err.message || 'Connection test failed.' } }))
      }
    }
  }

  const makeMove = useCallback(async () => {
    const game = gameRef.current
    const active = games[games.length - 1]
    if (requestRef.current || !active || active.outcome || game.isGameOver() || game.history().length >= MAX_PLIES) return
    const color = game.turn()
    const model = modelFor(active.index, color)
    const history = game.history({ verbose: true }).map(move => move.lan)
    const fenBefore = game.fen()
    const controller = new AbortController()
    const requestId = ++requestIdRef.current
    requestRef.current = controller
    setThinking({ model, color, since: Date.now(), timeout: Number(players[model].timeout_seconds) })
    setError(null)
    try {
      const data = await requestMove(players[model], history, controller.signal)
      if (requestId !== requestIdRef.current) return
      if (data.fen_before !== fenBefore) throw new MoveError('The server rebuilt a different position from this board. The move was not applied.', 409)
      const move = applyUci(game, data.move)
      if (!move) throw new MoveError('The server returned a move this board rejects. The move was not applied.', 409)
      if (game.fen() !== data.fen_after) {
        game.undo()
        throw new MoveError('The position after the move differs from the server. The move was not applied.', 409)
      }
      const ply = history.length + 1
      const record = {
        ply, color, model, san: move.san, uci: move.lan, from: move.from, to: move.to, fen_after: data.fen_after,
        check: game.inCheck(), explanation: typeof data.explanation === 'string' ? data.explanation : '',
        usage: data.usage && typeof data.usage === 'object' ? data.usage : {},
        elapsed_ms: typeof data.elapsed_ms === 'number' ? data.elapsed_ms : null,
        attempts: Number.isInteger(data.attempts) ? data.attempts : 1,
      }
      verificationIdRef.current[model] += 1
      setVerification(previous => ({
        ...previous,
        [model]: { state: 'ok', message: `Connected. Played ${move.san} in game ${active.index + 1} in ${seconds(data.elapsed_ms)}.` },
      }))
      const outcome = resolveOutcome(data.result, data.termination, game, ply)
      setGames(previous => previous.map((item, index) => (
        index === previous.length - 1 ? { ...item, moves: [...item.moves, record], outcome } : item
      )))
      let message = `Game ${active.index + 1}, ${moveLabel(record)} by ${COLOR_NAME[color]}, Model ${model}.`
      if (outcome) {
        const winner = winnerOf({ index: active.index, outcome })
        message += ` Game over: ${displayResult(outcome.result)}, ${terminationLabel(outcome.termination)}${winner ? `. Model ${winner} wins` : ''}.`
        if (active.index + 1 >= total) setRunning(false)
      }
      setAnnouncement(message)
    } catch (err) {
      if (requestId !== requestIdRef.current || err.name === 'AbortError') return
      const kind = err instanceof MoveError ? err.kind : 'provider'
      const failure = { ply: history.length + 1, model, color, kind, message: err.message || 'Request failed.' }
      setGames(previous => previous.map((item, index) => (
        index === previous.length - 1 ? { ...item, failures: [...(item.failures || []), failure] } : item
      )))
      setError({ kind, model, color, message: err.message || 'Could not get a move from the provider.', resume: runningRef.current })
      setRunning(false)
    } finally {
      if (requestId === requestIdRef.current) {
        requestRef.current = null
        setThinking(null)
      }
    }
  }, [games, players, total])

  const advanceGame = useCallback(() => {
    gameRef.current = new Chess()
    setReview(null)
    setGames(previous => {
      const last = previous[previous.length - 1]
      if (!last?.outcome || previous.length >= total) return previous
      return [...previous, { index: previous.length, moves: [], outcome: null }]
    })
    setAnnouncement(`Game ${games.length + 1} begins. Colors are swapped.`)
  }, [games.length, total])

  useEffect(() => {
    if (thinking || error || (!running && !stepQueued)) return undefined
    if (matchOver) {
      setRunning(false)
      setStepQueued(false)
      return undefined
    }
    if (between) {
      if (stepQueued) {
        setStepQueued(false)
        advanceGame()
        return undefined
      }
      const timer = window.setTimeout(advanceGame, 2500)
      return () => window.clearTimeout(timer)
    }
    const timer = window.setTimeout(() => {
      setStepQueued(false)
      void makeMove()
    }, stepQueued ? 0 : 350)
    return () => window.clearTimeout(timer)
  }, [running, stepQueued, thinking, error, matchOver, between, makeMove, advanceGame])

  const beginIfNeeded = () => {
    if (!games.length) {
      gameRef.current = new Chess()
      setGames([{ index: 0, moves: [], outcome: null }])
    }
  }

  const start = () => {
    if (!ensureReady()) return
    beginIfNeeded()
    setError(null)
    setReview(null)
    if (between) advanceGame()
    setRunning(true)
  }

  const pause = () => {
    const wasThinking = !!requestRef.current
    setRunning(false)
    setStepQueued(false)
    stopRequest()
    setAnnouncement(wasThinking ? 'Paused. The pending request was discarded.' : 'Paused.')
  }

  const step = () => {
    if (!ensureReady()) return
    beginIfNeeded()
    setError(null)
    setReview(null)
    setStepQueued(true)
  }

  const retry = () => {
    const resume = error?.resume
    setError(null)
    if (resume) setRunning(true)
    else setStepQueued(true)
  }

  const reset = () => {
    if (started && !window.confirm('Reset the match? Moves and scores that you have not exported will be lost.')) return
    setRunning(false)
    setStepQueued(false)
    stopRequest()
    gameRef.current = new Chess()
    setGames([])
    setError(null)
    setReview(null)
    setShowErrors(false)
    setAnnouncement('Match reset.')
  }

  const exportPgn = () => {
    const text = games.filter(game => game.moves.length).map(game => pgnFor(game, players)).join('\n\n')
    if (text) downloadText('chess-model-benchmark.pgn', `${text}\n`, 'application/x-chess-pgn')
  }

  const exportReport = () => {
    downloadText('chess-model-benchmark.json', JSON.stringify(buildReport({ mode, players, games }), null, 2), 'application/json')
  }

  // Board view: the live game unless a past move is being reviewed.
  const viewGame = review ? games[review.game] : current
  const viewIndex = viewGame?.index ?? 0
  const viewPly = review ? review.ply : viewGame?.moves.length ?? 0
  const boardFen = viewGame ? fenAt(viewGame, viewPly) : START_FEN
  const viewMove = viewGame && viewPly ? viewGame.moves[viewPly - 1] : null
  const atEnd = !review || (viewGame && review.ply === viewGame.moves.length)
  const viewOutcome = atEnd ? viewGame?.outcome : null
  const viewWinner = viewOutcome ? winnerOf(viewGame) : null
  const sideToMove = boardFen.split(' ')[1]

  const squareStyles = useMemo(() => {
    const styles = {}
    if (viewMove) {
      styles[viewMove.from] = { background: 'rgba(240, 214, 110, 0.45)' }
      styles[viewMove.to] = { background: 'rgba(240, 214, 110, 0.6)' }
      if (viewMove.check) {
        const king = kingSquare(viewMove.fen_after, viewMove.color === 'w' ? 'b' : 'w')
        if (king) styles[king] = { background: 'radial-gradient(circle, rgba(239, 83, 80, 0.85) 0%, rgba(239, 83, 80, 0.35) 55%, transparent 75%)' }
      }
    }
    return styles
  }, [viewMove])

  const goToPly = useCallback((gameIndex, ply) => {
    const game = games[gameIndex]
    if (!game) return
    const bounded = Math.max(0, Math.min(ply, game.moves.length))
    const isLive = gameIndex === games.length - 1 && bounded === game.moves.length
    setReview(isLive ? null : { game: gameIndex, ply: bounded })
  }, [games])

  useEffect(() => {
    const onKey = event => {
      if (event.altKey || event.ctrlKey || event.metaKey) return
      const target = event.target
      if (target instanceof HTMLElement && (target.closest('input, select, textarea, [contenteditable="true"]'))) return
      if (!games.length) return
      const gameIndex = review ? review.game : games.length - 1
      const ply = review ? review.ply : games[gameIndex].moves.length
      const actions = {
        ArrowLeft: () => goToPly(gameIndex, ply - 1),
        ArrowRight: () => goToPly(gameIndex, ply + 1),
        Escape: () => setReview(null),
      }
      if (actions[event.key]) {
        event.preventDefault()
        actions[event.key]()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [games, review, goToPly])

  const score = matchScore(games)
  const differences = settingDifferences(players)
  const reliability = reliabilitySummary(games)
  const hasReliabilityData = games.some(game => game.failures?.length || game.moves.some(move => move.attempts > 1))
  const hasReportData = started || games.some(game => game.failures?.length)
  const selected = viewMove
  const liveMoveCount = current?.moves.length ?? 0

  let status
  if (thinking) status = `Model ${thinking.model} (${COLOR_NAME[thinking.color]}) is choosing a move…`
  else if (matchOver) {
    status = score.A === score.B ? `Match complete: tied ${formatPoints(score.A)}–${formatPoints(score.B)}.` : `Match complete: Model ${score.A > score.B ? 'A' : 'B'} wins ${formatPoints(Math.max(score.A, score.B))}–${formatPoints(Math.min(score.A, score.B))}.`
  } else if (between) status = running ? `Game ${current.index + 1} finished (${displayResult(current.outcome.result)}). Game ${current.index + 2} starts with colors swapped…` : `Game ${current.index + 1} finished. Start game ${current.index + 2} when ready.`
  else if (error) status = 'Stopped. Fix the problem below or retry.'
  else if (running) status = 'Match in progress'
  else if (started) status = `Paused after ${liveMoveCount} half-move${liveMoveCount === 1 ? '' : 's'}.`
  else if (games.length) status = 'Paused before the first move.'
  else status = 'Configure both models, then start the match.'

  const primary = running
    ? { label: 'Pause', icon: Pause, onClick: pause }
    : matchOver ? { label: 'New match', icon: RotateCcw, onClick: reset }
      : between ? { label: `Start game ${current.index + 2}`, icon: Play, onClick: start }
        : { label: games.length ? 'Resume' : 'Start match', icon: Play, onClick: start }
  const PrimaryIcon = primary.icon

  return (
    <div className="app-shell">
      <a className="skip-link" href="#match">Skip to the match</a>
      <header className="topbar">
        <span className="brand-mark" aria-hidden="true"><Swords size={18} strokeWidth={2.2} /></span>
        <div className="brand-text"><strong>Chess Model Challenge</strong><small>Paired benchmark for two AI models</small></div>
      </header>

      <div className="sr-only" aria-live="polite" aria-atomic="true">{announcement}</div>

      <main className="layout">
        <section className="welcome-card" aria-labelledby="welcome-title">
          <div className="welcome-intro">
            <p className="eyebrow">Put your models on the board</p>
            <h2 id="welcome-title">Compare AI models by how they play chess.</h2>
            <p>Bring two models, watch them choose legal moves, and compare their games, public rationale and token use.</p>
            <p className="fine-print">This measures chess performance under your chosen settings. It is not a measure of general intelligence.</p>
          </div>
          <ol className="setup-steps" aria-label="How to run a challenge">
            <li><strong>Connect two models</strong><span>Choose a provider or compatible custom endpoint and enter your own API keys.</span></li>
            <li><strong>Match their budgets</strong><span>Set reasoning, output tokens and timeout. Use two games to swap colors.</span></li>
            <li><strong>Start and compare</strong><span>Watch the match, review moves, and export PGN or a JSON report.</span></li>
          </ol>
          <p className="welcome-billing">Your provider bills API calls under your account. Connection checks also make a model request. Keep this tab open while a match runs.</p>
          <SampleReplay />
        </section>
        <section id="match" className="board-column" aria-labelledby="match-title" tabIndex={-1}>
          <div className="section-heading">
            <h1 id="match-title">Game {viewIndex + 1} of {total}</h1>
            <span className="muted">{modelFor(viewIndex, 'w') === 'A' ? 'A plays White · B plays Black' : 'B plays White · A plays Black'}</span>
            <span className={`match-indicator ${running ? 'is-live' : ''}`}>{running ? 'Live' : matchOver ? 'Finished' : thinking ? 'Thinking' : games.length ? 'Paused' : 'Ready'}</span>
          </div>

          <div className="arena-card">
            {review && (
              <div className="review-banner">
                <span>Reviewing game {review.game + 1}{viewMove ? `, after ${moveLabel(viewMove)}` : ', start position'}</span>
                <button type="button" className="text-button" onClick={() => setReview(null)}>Back to live</button>
              </div>
            )}
            <PlayerBar color="b" model={modelFor(viewIndex, 'b')} player={players[modelFor(viewIndex, 'b')]}
              thinking={!review && thinking?.color === 'b' ? thinking : null} now={now}
              toMove={!review && !viewOutcome && !!current && sideToMove === 'b'} winner={viewWinner === modelFor(viewIndex, 'b')} />
            <div className="board-frame" ref={boardRef} role="img" aria-label={`Chess board, ${viewMove ? `after ${moveLabel(viewMove)}` : 'start position'}. ${sideToMove === 'w' ? 'White' : 'Black'} to move. FEN ${boardFen}`}>
              <Chessboard
                id={1} position={boardFen} boardWidth={Math.max(240, boardWidth - 8)} arePiecesDraggable={false} areArrowsAllowed={false}
                animationDuration={prefersReducedMotion() ? 0 : 180} boardOrientation="white" showBoardNotation
                customSquareStyles={squareStyles}
                customDarkSquareStyle={{ backgroundColor: '#6d8f7e' }} customLightSquareStyle={{ backgroundColor: '#e6ede0' }}
                customBoardStyle={{ borderRadius: '4px', overflow: 'hidden' }}
              />
            </div>
            <PlayerBar color="w" model={modelFor(viewIndex, 'w')} player={players[modelFor(viewIndex, 'w')]}
              thinking={!review && thinking?.color === 'w' ? thinking : null} now={now}
              toMove={!review && !viewOutcome && !!current && sideToMove === 'w'} winner={viewWinner === modelFor(viewIndex, 'w')} />
            {viewOutcome && (
              <div className="result-strip">
                <strong>{displayResult(viewOutcome.result)}</strong>
                <span>{terminationLabel(viewOutcome.termination)} · {viewWinner ? `Model ${viewWinner} wins game ${viewIndex + 1}` : `Game ${viewIndex + 1} drawn`}</span>
              </div>
            )}
          </div>

          <div className="controls-card">
            <p className="status-text">{status}</p>
            <div className="control-buttons">
              <button type="button" className="button primary" onClick={primary.onClick}><PrimaryIcon size={16} aria-hidden="true" /> {primary.label}</button>
              <button type="button" className="button secondary" onClick={step} disabled={running || !!thinking || matchOver}>
                <SkipForward size={16} aria-hidden="true" /> {between ? 'Next game' : 'One move'}
              </button>
              <button type="button" className="button secondary" onClick={reset} disabled={!games.length && !thinking}>
                <RotateCcw size={16} aria-hidden="true" /> Reset
              </button>
            </div>
          </div>

          {error && (
            <div className="error-box" role="alert">
              <AlertTriangle size={18} aria-hidden="true" />
              <div>
                <strong>{ERROR_TITLES[error.kind] || ERROR_TITLES.provider}{error.kind !== 'setup' && error.model ? ` · Model ${error.model}` : ''}</strong>
                <p>{error.message}</p>
                {ERROR_HINTS[error.kind] && <p className="muted">{ERROR_HINTS[error.kind]}</p>}
                <div className="error-actions">
                  {error.kind !== 'setup' && <button type="button" className="button primary small" onClick={retry}>Retry move</button>}
                  {error.model && <a className="button secondary small" href={`#setup-${error.model}`}>Model {error.model} settings</a>}
                  <button type="button" className="text-button" onClick={() => setError(null)}>Dismiss</button>
                </div>
              </div>
            </div>
          )}
        </section>

        <div className={`side-column ${started ? 'match-started' : ''}`}>
          <section className="score-card" aria-labelledby="score-title">
            <div className="score-top">
              <h2 id="score-title" className="eyebrow">Benchmark score</h2>
              <div className="export-buttons">
                <button type="button" className="text-button" onClick={exportPgn} disabled={!started}><ArrowDownToLine size={14} aria-hidden="true" /> PGN</button>
                <button type="button" className="text-button" onClick={exportReport} disabled={!hasReportData}><FileJson size={14} aria-hidden="true" /> JSON report</button>
              </div>
            </div>
            <div className="scoreline" aria-label={`Model A ${formatPoints(score.A)}, Model B ${formatPoints(score.B)}`}>
              <div className="score-side"><span className="model-chip chip-a" aria-hidden="true">A</span><span className="score-name">{players.A.model.trim() || 'Model A'}</span><strong>{formatPoints(score.A)}</strong></div>
              <span className="score-sep" aria-hidden="true">–</span>
              <div className="score-side right"><strong>{formatPoints(score.B)}</strong><span className="score-name">{players.B.model.trim() || 'Model B'}</span><span className="model-chip chip-b" aria-hidden="true">B</span></div>
            </div>
            <ol className="game-list">
              {Array.from({ length: total }, (_, index) => {
                const game = games[index]
                const winner = winnerOf(game)
                return (
                  <li key={index} className={current?.index === index && !matchOver ? 'current' : ''}>
                    <span className="game-name">Game {index + 1}</span>
                    <span className="muted">{modelFor(index, 'w')} White · {modelFor(index, 'b')} Black</span>
                    <span className="game-result">
                      {game?.outcome ? `${displayResult(game.outcome.result)} · ${winner ? `Model ${winner} wins` : 'Draw'}`
                        : game?.moves.length ? `In progress · ${game.moves.length} half-move${game.moves.length === 1 ? '' : 's'}` : index === games.length ? 'Up next' : 'Not started'}
                    </span>
                  </li>
                )
              })}
            </ol>
            {hasReliabilityData && (
              <p className="reliability">
                Failed requests: A {reliability.A.failed_requests} · B {reliability.B.failed_requests}
                <span aria-hidden="true"> | </span>
                Moves needing a retry: A {reliability.A.retried_moves} · B {reliability.B.retried_moves}
              </p>
            )}
            {differences.length > 0 && started && <p className="settings-warning">Settings differ between the models, so this is not a like-for-like comparison.</p>}
            <p className="fine-print">{mode === 'paired' ? 'Each model plays both colors once. Win 1, draw ½.' : 'Single game: Model A plays White.'} One pair is a sample, not a ranking.</p>
          </section>

          <section className="setup-section" aria-labelledby="setup-title">
            <div className="section-heading compact">
              <h2 id="setup-title">Models</h2>
              <span className="muted">Settings follow the model when colors swap.</span>
            </div>
            <fieldset className="mode-card" disabled={locked}>
              <legend className="eyebrow">Format</legend>
              <div className="segmented">
                <label><input type="radio" name="mode" value="paired" checked={mode === 'paired'} onChange={() => setMode('paired')} /><span>Two games, colors swapped</span></label>
                <label><input type="radio" name="mode" value="single" checked={mode === 'single'} onChange={() => setMode('single')} /><span>Single game</span></label>
              </div>
            </fieldset>
            {differences.length > 0 && (
              <div className="settings-warning" role="note">
                <strong>Settings differ between the models.</strong> Results are fairer when both use the same budget:
                <ul>{differences.map(item => <li key={item.field}>{item.label}: A {item.A} · B {item.B}</li>)}</ul>
              </div>
            )}
            {['A', 'B'].map(model => (
              <div id={`setup-${model}`} key={model} className="setup-anchor">
                <PlayerSetup
                  model={model} player={players[model]} onChange={next => updatePlayer(model, next)}
                  onVerify={() => void verifyPlayer(model)} verification={verification[model]}
                  locked={locked} keyEditable={paused && !thinking} showErrors={showErrors}
                  colorNote={mode === 'paired' ? `${model === 'A' ? 'White' : 'Black'} in game 1 · ${model === 'A' ? 'Black' : 'White'} in game 2` : `Plays ${model === 'A' ? 'White' : 'Black'}`}
                />
              </div>
            ))}
            <p className="security-note"><ShieldCheck size={15} aria-hidden="true" /><span>Keys stay in this tab's memory and are sent to the challenge server for provider calls. They are not included in PGN or JSON exports. Reloading clears them.</span></p>
          </section>

          <section className="analysis-card" aria-labelledby="analysis-title">
            <div className="card-heading">
              <h2 id="analysis-title">Moves and rationale</h2>
              {games.length > 1 && (
                <div className="game-tabs" role="group" aria-label="Choose game">
                  {games.map(game => (
                    <button key={game.index} type="button" aria-pressed={viewIndex === game.index}
                      onClick={() => (game.index === games.length - 1 ? setReview(null) : goToPly(game.index, game.moves.length))}>Game {game.index + 1}</button>
                  ))}
                </div>
              )}
            </div>

            {selected ? (
              <div className="rationale">
                <div className="rationale-top">
                  <span className={`piece-badge small ${selected.color}`} aria-hidden="true">{selected.color === 'w' ? '♔' : '♚'}</span>
                  <strong>{moveLabel(selected)}</strong>
                  <span className="muted">{COLOR_NAME[selected.color]} · Model {selected.model}</span>
                </div>
                <p>{selected.explanation || 'No rationale supplied with this move.'}</p>
                <div className="rationale-meta">
                  <span><ShieldCheck size={13} aria-hidden="true" /> Legal, board verified{selected.attempts > 1 ? ' · needed 1 retry' : ''}</span>
                  <span><Clock3 size={13} aria-hidden="true" /> {seconds(selected.elapsed_ms)}</span>
                  <span><Zap size={13} aria-hidden="true" /> {totalTokens(selected.usage) ?? '—'} tokens{typeof selected.usage?.reasoning_tokens === 'number' ? ` (${selected.usage.reasoning_tokens} reasoning)` : ''}</span>
                </div>
                <p className="fine-print">Public rationale written by the model with its move, not private chain-of-thought.</p>
              </div>
            ) : (
              <div className="empty-state">
                <BrainCircuit size={24} aria-hidden="true" />
                <p>{thinking ? 'Waiting for the first move…' : 'Each move and the model\'s short public rationale will appear here.'}</p>
              </div>
            )}

            {viewGame?.moves.length > 0 && (
              <>
                <div className="move-nav">
                  <button type="button" className="button secondary small" aria-label="Previous move" onClick={() => goToPly(viewIndex, viewPly - 1)} disabled={viewPly === 0}><ChevronLeft size={16} aria-hidden="true" /></button>
                  <span className="muted">{viewPly} / {viewGame.moves.length}<span className="key-hint"> · ← → keys</span></span>
                  <button type="button" className="button secondary small" aria-label="Next move" onClick={() => goToPly(viewIndex, viewPly + 1)} disabled={viewPly >= viewGame.moves.length}><ChevronRight size={16} aria-hidden="true" /></button>
                </div>
                <ol className="move-list" aria-label={`Game ${viewIndex + 1} moves`}>
                  {Array.from({ length: Math.ceil(viewGame.moves.length / 2) }, (_, row) => (
                    <li key={row} className="move-row">
                      <span className="move-number" aria-hidden="true">{row + 1}.</span>
                      {[0, 1].map(side => {
                        const move = viewGame.moves[row * 2 + side]
                        if (!move) return <span key={side} />
                        const isSelected = selected?.ply === move.ply
                        return (
                          <button key={side} type="button" className={`move-choice ${isSelected ? 'selected' : ''}`} aria-current={isSelected ? 'step' : undefined}
                            aria-label={`${moveLabel(move)}, ${COLOR_NAME[move.color]}, Model ${move.model}`} onClick={() => goToPly(viewIndex, move.ply)}>
                            <strong>{move.san}</strong>
                            <small>{seconds(move.elapsed_ms)}{move.attempts > 1 ? ' · retry' : ''}</small>
                          </button>
                        )
                      })}
                    </li>
                  ))}
                </ol>
              </>
            )}
          </section>
        </div>
      </main>

      <footer className="footer">
        <span>The server rebuilds every position and sends both models the same board, FEN, history and legal moves.</span>
        <span>Provider calls may be billed.</span>
      </footer>
    </div>
  )
}
