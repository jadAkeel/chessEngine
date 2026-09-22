from __future__ import annotations

import chess
import numpy as np

BOARD_SIZE = 8
NUM_POLICY_PLANES = 73
NUM_MOVES = BOARD_SIZE * BOARD_SIZE * NUM_POLICY_PLANES

DIRECTIONS = [
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
]
MAX_DISTANCE = 7
KNIGHT_MOVES = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1),
]
# AlphaZero-style encoding reserves the promotion planes for underpromotions only.
# Queen promotions are encoded as normal one-step directional moves and are restored
# when decoding against a real board position.
UNDERPROMOTION_PIECES = [chess.ROOK, chess.BISHOP, chess.KNIGHT]


def square_to_coord(square: int) -> tuple[int, int]:
    return chess.square_rank(square), chess.square_file(square)


def coord_to_square(rank: int, file: int):
    if 0 <= rank < 8 and 0 <= file < 8:
        return chess.square(file, rank)
    return None


def _flip(rank: int, file: int) -> tuple[int, int]:
    return 7 - rank, 7 - file


def _to_oriented(rank: int, file: int, board: chess.Board) -> tuple[int, int]:
    return _flip(rank, file) if board.turn == chess.BLACK else (rank, file)


def _from_oriented(rank: int, file: int, board: chess.Board) -> tuple[int, int]:
    return _flip(rank, file) if board.turn == chess.BLACK else (rank, file)


def _is_last_rank(square: int) -> bool:
    rank = chess.square_rank(square)
    return rank in {0, 7}


def move_to_policy(move: chess.Move, board: chess.Board):
    from_rank, from_file = _to_oriented(*square_to_coord(move.from_square), board)
    to_rank, to_file = _to_oriented(*square_to_coord(move.to_square), board)

    dr = to_rank - from_rank
    df = to_file - from_file

    if move.promotion:
        direction = df
        if direction not in (-1, 0, 1):
            return None
        if move.promotion in UNDERPROMOTION_PIECES:
            promo_idx = UNDERPROMOTION_PIECES.index(move.promotion)
            plane = 64 + (direction + 1) * 3 + promo_idx
            return from_rank, from_file, plane
        # Queen promotions intentionally fall through to the regular directional
        # move planes used for one-step pawn pushes/captures.

    for i, (r, f) in enumerate(KNIGHT_MOVES):
        if (dr, df) == (r, f):
            return from_rank, from_file, 56 + i

    for dir_idx, (r, f) in enumerate(DIRECTIONS):
        for dist in range(1, MAX_DISTANCE + 1):
            if (dr, df) == (r * dist, f * dist):
                plane = dir_idx * 7 + (dist - 1)
                return from_rank, from_file, plane

    return None


def policy_to_move(rank: int, file: int, plane: int, board: chess.Board):
    oriented_from_rank, oriented_from_file = rank, file

    if plane < 56:
        dir_idx = plane // 7
        dist = (plane % 7) + 1
        dr, df = DIRECTIONS[dir_idx]
        oriented_to_rank = oriented_from_rank + dr * dist
        oriented_to_file = oriented_from_file + df * dist
        promotion = None
    elif plane < 64:
        dr, df = KNIGHT_MOVES[plane - 56]
        oriented_to_rank = oriented_from_rank + dr
        oriented_to_file = oriented_from_file + df
        promotion = None
    else:
        promo_plane = plane - 64
        direction = (promo_plane // 3) - 1
        promotion = UNDERPROMOTION_PIECES[promo_plane % 3]
        oriented_to_rank = oriented_from_rank + 1
        oriented_to_file = oriented_from_file + direction

    if not (0 <= oriented_to_rank < 8 and 0 <= oriented_to_file < 8):
        return None

    from_rank, from_file = _from_oriented(oriented_from_rank, oriented_from_file, board)
    to_rank, to_file = _from_oriented(oriented_to_rank, oriented_to_file, board)
    from_sq = coord_to_square(from_rank, from_file)
    to_sq = coord_to_square(to_rank, to_file)
    if from_sq is None or to_sq is None:
        return None

    # Queen promotions share the normal directional planes. Restore them when we
    # have the real board context and a pawn reaches the last rank.
    if promotion is None:
        piece = board.piece_at(from_sq)
        if piece is not None and piece.piece_type == chess.PAWN and _is_last_rank(to_sq):
            promotion = chess.QUEEN

    return chess.Move(from_sq, to_sq, promotion=promotion)


def move_to_index(move: chess.Move, board: chess.Board | None = None) -> int:
    if board is not None:
        policy_idx = move_to_policy(move, board)
        if policy_idx is not None:
            rank, file, plane = policy_idx
            return (rank * 8 + file) * NUM_POLICY_PLANES + plane

    # Absolute fallback used by augmentation utilities and generic indexing.
    from_rank, from_file = square_to_coord(move.from_square)
    pseudo_board = chess.Board.empty()
    pseudo_board.turn = chess.WHITE
    normalized = move_to_policy(chess.Move(move.from_square, move.to_square, promotion=move.promotion), pseudo_board)
    if normalized is None:
        raise ValueError(f"Cannot encode move: {move.uci()}")
    _, _, plane = normalized
    return (from_rank * 8 + from_file) * NUM_POLICY_PLANES + plane


def index_to_move(index: int, board: chess.Board | None = None):
    if not (0 <= index < NUM_MOVES):
        return None
    square_idx, plane = divmod(int(index), NUM_POLICY_PLANES)
    rank, file = divmod(square_idx, 8)
    if board is None:
        board = chess.Board.empty()
        board.turn = chess.WHITE
    return policy_to_move(rank, file, plane, board)


def create_empty_policy():
    return np.zeros((8, 8, NUM_POLICY_PLANES), dtype=np.float32)


def legal_moves_mask(board: chess.Board):
    mask = create_empty_policy()
    for move in board.legal_moves:
        res = move_to_policy(move, board)
        if res is None:
            continue
        r, f, p = res
        mask[r, f, p] = 1.0
    return mask


def policy_to_move_list(policy: np.ndarray, board: chess.Board):
    moves = []
    for r in range(8):
        for f in range(8):
            for p in range(NUM_POLICY_PLANES):
                prob = policy[r, f, p]
                if prob <= 0:
                    continue
                move = policy_to_move(r, f, p, board)
                if move and move in board.legal_moves:
                    moves.append((move, prob))
    return moves
