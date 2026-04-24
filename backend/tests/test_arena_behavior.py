from __future__ import annotations

import chess

from app.evaluation.arena import _build_arena_opening_positions, _select_move_with_fallback
from app.game.repetition import position_key
from app.infra.config import AppConfig, ArenaConfig


class _StubAnalysis:
    def __init__(self, visit_counts, policy, score=0.0):
        self.visit_counts = visit_counts
        self.policy = policy
        self.score = score


class _StubEngine:
    def __init__(self, analysis):
        self._analysis = analysis

    def analyze(self, board, add_noise, num_simulations, temperature):
        return self._analysis


def test_opening_book_contains_deeper_positions():
    boards = _build_arena_opening_positions()
    assert boards
    assert any(len(board.move_stack) >= 0 for board in boards)
    assert any(board.fen() != chess.STARTING_FEN and board.fullmove_number >= 4 for board in boards)


def test_select_move_prefers_non_repetition_when_hard_block_enabled():
    board = chess.Board()
    cycle = [
        'g1f3', 'g8f6', 'f3g1', 'f6g8',
        'g1f3', 'g8f6', 'f3g1', 'f6g8',
    ]
    for uci in cycle:
        board.push_uci(uci)

    repeat_move = chess.Move.from_uci('g1f3')
    fresh_move = chess.Move.from_uci('b1c3')
    assert repeat_move in board.legal_moves
    assert fresh_move in board.legal_moves

    cfg = AppConfig(arena=ArenaConfig(fallback_top_k=2, repetition_break_count=3, repetition_move_weight=0.3, hard_block_repetition=True, contempt_factor=0.05))
    seen_positions = {position_key(board): 2}
    board.push(repeat_move)
    seen_positions[position_key(board)] = 2
    board.pop()

    engine = _StubEngine(_StubAnalysis(
        visit_counts={repeat_move.uci(): 100, fresh_move.uci(): 90},
        policy={repeat_move.uci(): 0.7, fresh_move.uci(): 0.3},
        score=0.1,
    ))

    move, score, repetition_counts = _select_move_with_fallback(engine, board, cfg, seen_positions, is_candidate=True)

    assert move == fresh_move
    assert score == 0.1
    assert repetition_counts[repeat_move.uci()] >= 3
