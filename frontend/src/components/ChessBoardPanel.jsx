import React, { useEffect, useState, useRef } from 'react'
import { Chessboard } from 'react-chessboard'
import { Chess } from 'chess.js'
import { playMoveSoundFor } from '@/utils/sound'

const DEFAULT_API_BASE_URL = import.meta.env.PROD
  ? 'https://chessengine-2.onrender.com'
  : 'http://localhost:8000'
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL).replace(/\/$/, '')

export default function ChessBoardPanel() {
  const gameRef = useRef(new Chess())
  const [fen, setFen] = useState(gameRef.current.fen())
  const [thinking, setThinking] = useState(false)
  const [engineWarmupStatus, setEngineWarmupStatus] = useState('idle')
  const engineWarmupPromiseRef = useRef(null)

  const warmupEngineServer = async () => {
    if (engineWarmupStatus === 'ready') return true
    if (engineWarmupPromiseRef.current) return engineWarmupPromiseRef.current

    setEngineWarmupStatus('waking')
    const warmupPromise = fetch(`${API_BASE_URL}/health`, { cache: 'no-store' })
      .then(async (res) => {
        if (!res.ok) throw new Error(`Health check failed: ${res.status}`)
        const data = await res.json().catch(() => ({}))
        if (data.model === false) throw new Error('Model is not loaded')
        setEngineWarmupStatus('ready')
        return true
      })
      .catch((err) => {
        console.error('Engine warmup failed', err)
        setEngineWarmupStatus('error')
        return false
      })
      .finally(() => {
        engineWarmupPromiseRef.current = null
      })

    engineWarmupPromiseRef.current = warmupPromise
    return warmupPromise
  }

  useEffect(() => {
    void warmupEngineServer()
  }, [])

  const makeEngineMove = async (currentFen) => {
    setThinking(true)
    try {
      const serverReady = await warmupEngineServer()
      if (!serverReady) return

      const res = await fetch(`${API_BASE_URL}/fastmove`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen: currentFen, topk: 16, depth: 6, max_simulations: 96, adaptive: true })
      })
      const data = await res.json()
      const uci = data.move || data.moves?.[0]?.uci || data.best_move
      if (uci) {
        const from = uci.slice(0, 2)
        const to = uci.slice(2, 4)
        const move = gameRef.current.move({ from, to, promotion: 'q' })
        if (move) playMoveSoundFor(move, gameRef.current)
        setFen(gameRef.current.fen())
      }
    } catch (err) {
      console.error('Engine request failed', err)
    } finally {
      setThinking(false)
    }
  }

  const onDrop = (sourceSquare, targetSquare) => {
    const move = gameRef.current.move({ from: sourceSquare, to: targetSquare, promotion: 'q' })
    if (move === null) return false
    playMoveSoundFor(move, gameRef.current)
    setFen(gameRef.current.fen())
    makeEngineMove(gameRef.current.fen())
    return true
  }

  return (
    <div>
      <div style={{ marginBottom: 8 }}>
        {engineWarmupStatus === 'waking' ? 'Loading engine model...' : `Thinking: ${thinking ? 'Yes' : 'No'}`}
      </div>
      <Chessboard position={fen} onPieceDrop={onDrop} arePiecesDraggable={!thinking && engineWarmupStatus !== 'waking'} />
    </div>
  )
}
