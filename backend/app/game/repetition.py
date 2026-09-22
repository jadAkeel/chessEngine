from __future__ import annotations

from collections.abc import Mapping
import hashlib

import chess

PositionKey = tuple[int, ...]
_NO_EP_SQUARE = -1


def _legal_ep_square(board: chess.Board) -> int | None:
    ep_square = getattr(board, "ep_square", None)
    if ep_square is None:
        return None

    has_legal_en_passant = getattr(board, "has_legal_en_passant", None)
    if callable(has_legal_en_passant):
        return int(ep_square) if bool(has_legal_en_passant()) else None

    is_en_passant = getattr(board, "is_en_passant", None)
    if callable(is_en_passant):
        for move in board.legal_moves:
            if bool(is_en_passant(move)):
                return int(ep_square)

    return None


def position_key(board: chess.Board) -> PositionKey:
    if not isinstance(board, chess.Board):
        raise TypeError("Expected board to be chess.Board")

    ep_square = _legal_ep_square(board)
    return (
        int(board.pieces_mask(chess.PAWN, chess.WHITE)),
        int(board.pieces_mask(chess.KNIGHT, chess.WHITE)),
        int(board.pieces_mask(chess.BISHOP, chess.WHITE)),
        int(board.pieces_mask(chess.ROOK, chess.WHITE)),
        int(board.pieces_mask(chess.QUEEN, chess.WHITE)),
        int(board.pieces_mask(chess.KING, chess.WHITE)),
        int(board.pieces_mask(chess.PAWN, chess.BLACK)),
        int(board.pieces_mask(chess.KNIGHT, chess.BLACK)),
        int(board.pieces_mask(chess.BISHOP, chess.BLACK)),
        int(board.pieces_mask(chess.ROOK, chess.BLACK)),
        int(board.pieces_mask(chess.QUEEN, chess.BLACK)),
        int(board.pieces_mask(chess.KING, chess.BLACK)),
        int(board.turn),
        int(board.clean_castling_rights()),
        _NO_EP_SQUARE if ep_square is None else int(ep_square),
    )


def position_key_token(key: PositionKey) -> str:
    payload = ",".join(str(int(part)) for part in key).encode("ascii")
    return hashlib.blake2b(payload, digest_size=12).hexdigest()


def build_seen_positions(board: chess.Board) -> dict[PositionKey, int]:
    if not isinstance(board, chess.Board):
        raise TypeError("Expected board to be chess.Board")

    if not board.move_stack:
        return {position_key(board): 1}

    root = board.root()
    replay = root.copy(stack=False)
    seen_positions: dict[PositionKey, int] = {position_key(replay): 1}

    for move in board.move_stack:
        replay.push(move)
        key = position_key(replay)
        seen_positions[key] = seen_positions.get(key, 0) + 1

    return seen_positions


def current_repetition_count(board: chess.Board, *, max_probe: int = 8) -> int:
    if not isinstance(board, chess.Board):
        raise TypeError("Expected board to be chess.Board")

    if not board.move_stack:
        return 1

    probe_limit = max(2, int(max_probe))
    for count in range(probe_limit, 1, -1):
        if board.is_repetition(count):
            return count

    return 1


def count_repetition_after_move(
    board: chess.Board,
    move: chess.Move,
    seen_positions: Mapping[PositionKey, int] | None = None,
) -> int:
    if not isinstance(board, chess.Board):
        raise TypeError("Expected board to be chess.Board")

    board.push(move)
    try:
        if seen_positions is not None:
            next_key = position_key(board)
            return int(seen_positions.get(next_key, 0) + 1)
        return int(current_repetition_count(board))
    finally:
        board.pop()


def filter_repetition_moves(
    policy_dict: Mapping[chess.Move, float],
    board: chess.Board,
    seen_positions: Mapping[PositionKey, int] | None = None,
    *,
    repeat_break_count: int,
    repeat_weight: float,
) -> tuple[dict[chess.Move, float], dict[str, int]]:
    if not policy_dict:
        return {}, {}

    repeat_break_count = max(2, int(repeat_break_count))
    repeat_weight = min(max(float(repeat_weight), 0.0), 1.0)

    repetition_counts = {
        move.uci(): count_repetition_after_move(board, move, seen_positions)
        for move in policy_dict
    }

    filtered: dict[chess.Move, float] = {}
    for move, prob in policy_dict.items():
        next_repeat = repetition_counts[move.uci()]
        adjusted = float(prob)

        if next_repeat >= repeat_break_count:
            adjusted *= repeat_weight
        elif next_repeat == repeat_break_count - 1:
            adjusted *= max(repeat_weight, 0.35)

        filtered[move] = adjusted

    total = float(sum(filtered.values()))
    if total <= 1e-12:
        total = float(sum(float(prob) for prob in policy_dict.values()))
        if total <= 1e-12:
            uniform = 1.0 / len(policy_dict)
            return ({move: uniform for move in policy_dict}, repetition_counts)
        return ({move: float(prob) / total for move, prob in policy_dict.items()}, repetition_counts)

    return ({move: float(prob) / total for move, prob in filtered.items()}, repetition_counts)
