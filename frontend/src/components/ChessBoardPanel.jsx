import React, { useState, useRef } from 'react'
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

  const makeEngineMove = async (currentFen) => {
    setThinking(true)
    try {
      const res = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen: currentFen, topk: 1 })
      })
      const data = await res.json()
      const uci = data.moves?.[0]?.uci || data.move || data.best_move
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
      <div style={{ marginBottom: 8 }}>Thinking: {thinking ? 'Yes' : 'No'}</div>
      <Chessboard position={fen} onPieceDrop={onDrop} />
    </div>
  )
}
