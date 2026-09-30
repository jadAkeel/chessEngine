import React, { useEffect, useRef, useState } from 'react'
import { Chess } from 'chess.js'
import { Chessboard } from 'react-chessboard'
import { ChevronLeft, ChevronRight, RotateCcw, Play, X } from 'lucide-react'

// Handwritten opening for onboarding only. No provider calls or benchmark records.
const MOVES = ['e4', 'c5', 'Nf3', 'd6', 'd4', 'cxd4', 'Nxd4', 'Nf6', 'Nc3', 'a6']
const POSITIONS = [new Chess().fen()]
const sample = new Chess()
for (const move of MOVES) {
  sample.move(move)
  POSITIONS.push(sample.fen())
}

export default function SampleReplay() {
  const [open, setOpen] = useState(false)
  const [ply, setPly] = useState(0)
  const [width, setWidth] = useState(0)
  const boardRef = useRef(null)

  useEffect(() => {
    if (!open || !boardRef.current) return undefined
    const element = boardRef.current
    const measure = () => setWidth(Math.floor(element.clientWidth))
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(element)
    return () => observer.disconnect()
  }, [open])

  return (
    <div className="sample-replay">
      <button type="button" className="button secondary" aria-expanded={open} aria-controls="sample-replay-panel" onClick={() => setOpen(value => !value)}>
        {open ? <X size={16} aria-hidden="true" /> : <Play size={16} aria-hidden="true" />}
        {open ? 'Close sample replay' : 'Watch sample replay — no key needed'}
      </button>
      {open && (
        <section id="sample-replay-panel" className="sample-panel" aria-labelledby="sample-title">
          <div className="sample-copy">
            <h3 id="sample-title">Scripted demonstration</h3>
            <p>This opening is a handwritten replay. No AI models are called and no API key is needed. It does not count toward your match score or report.</p>
            <p className="fine-print">Use the arrows to see a position change after each move. In a real match, each model receives the updated board before choosing its move.</p>
            <p className="sample-position" aria-live="polite">{ply === 0 ? 'Starting position' : `${Math.ceil(ply / 2)}${ply % 2 ? '.' : '…'} ${MOVES[ply - 1]}`} · {ply % 2 ? 'Black' : 'White'} to move</p>
            <div className="control-buttons">
              <button type="button" className="button secondary small" aria-label="Previous sample move" disabled={ply === 0} onClick={() => setPly(value => value - 1)}><ChevronLeft size={16} aria-hidden="true" /></button>
              <span className="muted">{ply} / {MOVES.length} half-moves</span>
              <button type="button" className="button secondary small" aria-label="Next sample move" disabled={ply === MOVES.length} onClick={() => setPly(value => value + 1)}><ChevronRight size={16} aria-hidden="true" /></button>
              <button type="button" className="text-button" disabled={ply === 0} onClick={() => setPly(0)}><RotateCcw size={14} aria-hidden="true" /> Restart sample</button>
            </div>
            {ply === MOVES.length && <p className="fine-print">End of this opening sample. No winner is determined.</p>}
          </div>
          <div className="sample-board" ref={boardRef} role="img" aria-label={`Scripted sample chess board. FEN ${POSITIONS[ply]}`}>
            {width > 0 && <Chessboard id="sample-board" position={POSITIONS[ply]} boardWidth={width} arePiecesDraggable={false} areArrowsAllowed={false}
              animationDuration={0} boardOrientation="white" showBoardNotation
              customDarkSquareStyle={{ backgroundColor: '#6d8f7e' }} customLightSquareStyle={{ backgroundColor: '#e6ede0' }}
              customBoardStyle={{ borderRadius: '4px', overflow: 'hidden' }} />}
          </div>
        </section>
      )}
    </div>
  )
}
