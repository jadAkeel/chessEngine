"""Short tactical checks shared by search and emergency move selection."""
from __future__ import annotations

import time
from collections.abc import Iterable, Mapping

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


def find_forcing_mate_in_two(board: chess.Board, deadline: float | None = None) -> chess.Move | None:
    """The side to move's mate in one, or a checking move with mate after every defence.

    Quiet first moves are intentionally outside this bounded tactical check.
    A defender's available draw refutes a forced mate.
    """
    if board.is_game_over(claim_draw=True):
        return None
    for move in list(board.legal_moves):
        _check_deadline(deadline)
        if not board.gives_check(move):
            continue
        board.push(move)
        try:
            if board.is_checkmate():
                return move
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
                return move
        finally:
            board.pop()
    return None


def has_forcing_mate_in_two(board: chess.Board, deadline: float | None = None) -> bool:
    """True when the side to move has a mate in one or a checking mate in two."""
    return find_forcing_mate_in_two(board, deadline) is not None


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


# A move that ends the game without mate is refused while the mover is ahead by this much.
DRAW_AVOIDANCE_ROOT_VALUE = 0.15
# Quiet material the screen will not let a move give away (beyond what was already hanging).
MATERIAL_VETO_CP = 300
# Search evidence that keeps a move the material screen would veto.
SEARCH_OVERRIDE_MIN_VISITS = 8
SEARCH_OVERRIDE_MARGIN = 0.2


def _ends_in_draw(board: chess.Board) -> bool:
    """After our move: the game is drawn, or the opponent (now to move) can claim a draw."""
    if board.is_stalemate() or board.is_insufficient_material():
        return True
    if board.halfmove_clock >= 100 or board.is_repetition(3):
        return True
    return board.can_claim_threefold_repetition()


def _loss_if_passing(board: chess.Board, deadline: float | None) -> int:
    """Material already hanging: what one capture would win if the mover could pass."""
    if board.is_check():
        return 0
    return immediate_exchange_loss(board, chess.Move.null(), deadline)


def _check_forces_mate(board: chess.Board, move: chess.Move, deadline: float | None) -> bool:
    board.push(move)
    try:
        replies = list(board.legal_moves)
        if not replies:
            return board.is_checkmate()
        for reply in replies:
            _check_deadline(deadline)
            board.push(reply)
            try:
                if find_mate_in_one(board, deadline) is None:
                    return False
            finally:
                board.pop()
        return True
    finally:
        board.pop()


def select_safe_move(
    board: chess.Board,
    ranked_moves: Iterable[chess.Move],
    deadline: float | None = None,
    *,
    root_value: float | None = None,
    move_values: Mapping[chess.Move, tuple[int, float]] | None = None,
) -> tuple[chess.Move | None, set[chess.Move]]:
    """Take the best-ranked move that passes three short proofs.

    1. A checkmate is always taken. Any other game-ending move (stalemate, dead
       material, a draw the opponent could claim) is refused while the mover is
       ahead (``root_value`` above +0.15), since it throws the win away.
    2. A move after which the opponent has a mate in one or a checking mate in
       two is refused.
    3. A move that gives away at least a minor piece to one capture-recapture is
       refused. The loss is counted beyond what was already hanging before the
       move, so a pre-existing threat does not veto every move that answers it
       otherwise; checks are tested too, unless they force mate.

    ``move_values`` maps moves to ``(visits, value for the mover)`` from the search.
    A move vetoed only by the material estimate is kept when the search backs it
    clearly: at least 8 visits and a value 0.2 above the move that would replace
    it (the 2-ply estimate cannot see a sacrifice the search already has).

    On budget exhaustion, return the first move not already rejected. If every
    option drops material, prefer the smallest excess loss; if every option
    allows mate, keep the original ranking.
    """
    moves = list(ranked_moves)
    rejected: set[chess.Move] = set()
    material_candidates: list[tuple[int, int, chess.Move]] = []
    vetoed_for_material: list[chess.Move] = []
    winning = root_value is not None and float(root_value) > DRAW_AVOIDANCE_ROOT_VALUE
    baseline: int | None = None
    for move in moves:
        forced_loss = False
        board.push(move)
        try:
            if board.is_checkmate():
                return move, rejected
            if _ends_in_draw(board):
                if winning:
                    rejected.add(move)
                    continue
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
            if baseline is None:
                baseline = _loss_if_passing(board, deadline)
            excess = immediate_exchange_loss(board, move, deadline) - baseline
            if excess >= MATERIAL_VETO_CP and board.gives_check(move) and _check_forces_mate(board, move, deadline):
                excess = 0
        except TacticalSearchExpired:
            return move, rejected
        if excess < MATERIAL_VETO_CP:
            chosen = _search_backed_alternative(move, vetoed_for_material, move_values) or move
            return chosen, rejected - {chosen}
        material_candidates.append((excess, len(material_candidates), move))
        vetoed_for_material.append(move)
        rejected.add(move)
    if material_candidates:
        chosen = min(material_candidates)[2]
        return chosen, rejected - {chosen}
    return (moves[0] if moves else None), set()


def _search_backed_alternative(
    replacement: chess.Move,
    vetoed: list[chess.Move],
    move_values: Mapping[chess.Move, tuple[int, float]] | None,
) -> chess.Move | None:
    if not move_values or not vetoed or replacement not in move_values:
        return None
    replacement_value = float(move_values[replacement][1])
    for move in vetoed:
        stats = move_values.get(move)
        if stats is None:
            continue
        visits, value = int(stats[0]), float(stats[1])
        if visits >= SEARCH_OVERRIDE_MIN_VISITS and value >= replacement_value + SEARCH_OVERRIDE_MARGIN:
            return move
    return None
