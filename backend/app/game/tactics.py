"""Short tactical checks shared by search and emergency move selection."""
from __future__ import annotations

import time
from collections.abc import Iterable

import chess

_PIECE_VALUES = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
                 chess.ROOK: 500, chess.QUEEN: 900}


class TacticalSearchExpired(Exception):
    pass


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TacticalSearchExpired


def find_mate_in_one(board: chess.Board, deadline: float | None = None) -> chess.Move | None:
    for move in list(board.legal_moves):
        _check_deadline(deadline)
        if not board.gives_check(move):
            continue
        board.push(move)
        try:
            if board.is_checkmate():
                return move
        finally:
            board.pop()
    return None


def has_forcing_mate_in_two(board: chess.Board, deadline: float | None = None) -> bool:
    """Prove a mate in one, or a checking move with mate after every defence.

    Quiet first moves are intentionally outside this bounded tactical check.
    A defender's available draw refutes a forced mate.
    """
    if board.is_game_over(claim_draw=True):
        return False
    for move in list(board.legal_moves):
        _check_deadline(deadline)
        if not board.gives_check(move):
            continue
        board.push(move)
        try:
            if board.is_checkmate():
                return True
            if board.is_game_over(claim_draw=True):
                continue
            forced = True
            for reply in list(board.legal_moves):
                _check_deadline(deadline)
                board.push(reply)
                try:
                    if board.is_game_over(claim_draw=True) or find_mate_in_one(board, deadline) is None:
                        forced = False
                        break
                finally:
                    board.pop()
            if forced:
                return True
        finally:
            board.pop()
    return False


def immediate_exchange_loss(board: chess.Board, move: chess.Move, deadline: float | None = None) -> int:
    """Material lost to one legal capture, allowing the best legal recapture.

    This is a short-horizon safety estimate, not a proof that a sacrifice loses.
    Credit captures/promotions made by the candidate move itself.
    """
    mover = board.turn
    def material():
        return sum(value * (len(board.pieces(piece, mover)) - len(board.pieces(piece, not mover)))
                   for piece, value in _PIECE_VALUES.items())
    before = material()
    board.push(move)
    try:
        worst = material() - before
        for reply in list(board.generate_legal_captures()):
            _check_deadline(deadline)
            board.push(reply)
            try:
                best = material() - before
                for recapture in list(board.generate_legal_captures()):
                    _check_deadline(deadline)
                    if recapture.to_square != reply.to_square:
                        continue
                    board.push(recapture)
                    try:
                        best = max(best, material() - before)
                    finally:
                        board.pop()
                worst = min(worst, best)
            finally:
                board.pop()
        return max(0, -worst)
    finally:
        board.pop()


def select_safe_move(
    board: chess.Board,
    ranked_moves: Iterable[chess.Move],
    deadline: float | None = None,
) -> tuple[chess.Move | None, set[chess.Move]]:
    """Screen short mates and quiet moves that drop a minor piece or more.

    On budget exhaustion, return the first move not already rejected. If every
    quiet option drops material, prefer the smallest estimated loss. Checking
    sacrifices are left to MCTS; if every option allows mate, retain its ranking.
    """
    moves = list(ranked_moves)
    rejected: set[chess.Move] = set()
    material_candidates: list[tuple[int, int, chess.Move]] = []
    for move in moves:
        checking = board.gives_check(move)
        forced_loss = False
        board.push(move)
        try:
            if board.is_game_over(claim_draw=True):
                return move, rejected
            forced_loss = has_forcing_mate_in_two(board, deadline)
        except TacticalSearchExpired:
            return move, rejected
        finally:
            board.pop()
        if forced_loss:
            rejected.add(move)
            continue
        try:
            loss = 0 if checking else immediate_exchange_loss(board, move, deadline)
        except TacticalSearchExpired:
            return move, rejected
        if loss < 300:
            return move, rejected
        material_candidates.append((loss, len(material_candidates), move))
        rejected.add(move)
    if material_candidates:
        chosen = min(material_candidates)[2]
        return chosen, rejected - {chosen}
    return (moves[0] if moves else None), set()
