from __future__ import annotations

from dataclasses import dataclass

import chess

from app.infra.config import PrinciplePenaltiesConfig

CENTRAL_SQUARES = {chess.D4, chess.E4, chess.D5, chess.E5}
EXTENDED_CENTER = {
    chess.C3,
    chess.D3,
    chess.E3,
    chess.F3,
    chess.C4,
    chess.D4,
    chess.E4,
    chess.F4,
    chess.C5,
    chess.D5,
    chess.E5,
    chess.F5,
    chess.C6,
    chess.D6,
    chess.E6,
    chess.F6,
}
RIM_FILES = {0, 7}
RIM_RANKS = {0, 7}
KINGSIDE_FILES = {5, 6, 7}
QUEENSIDE_FILES = {0, 1, 2}
EARLY_SINGLE_STEP_PAWN_FILES = {2, 3, 4, 5}


@dataclass(frozen=True)
class PrinciplePenaltyResult:
    components: dict[str, float]
    reasons: dict[str, list[str]]

    @property
    def total(self) -> float:
        return float(sum(self.components.values()))


def principle_penalty_components(
    before: chess.Board,
    after: chess.Board,
    move: chess.Move,
    cfg: PrinciplePenaltiesConfig,
) -> PrinciplePenaltyResult:
    if not bool(getattr(cfg, "enabled", False)):
        return PrinciplePenaltyResult({}, {})

    mover = not after.turn
    components: dict[str, float] = {}
    reasons: dict[str, list[str]] = {}

    def add(name: str, scale: float, reason: str) -> None:
        weight = max(0.0, float(getattr(cfg, name, 0.0)))
        value = weight * float(scale)
        if value <= 0.0:
            return
        components[name] = components.get(name, 0.0) + value
        reasons.setdefault(name, []).append(reason)

    _king_safety(before, after, move, mover, add)
    _opening_development(before, after, move, mover, add)
    _center_control(before, after, move, mover, add)
    _tactics(after, mover, add)
    _pawn_structure(before, after, move, mover, add)
    _piece_activity(before, after, move, mover, add)
    _rook_activity(before, after, move, mover, add)
    _endgame(before, after, move, mover, add)

    cap = max(0.0, float(getattr(cfg, "max_total_per_move", 0.0)))
    total = float(sum(components.values()))
    if cap > 0.0 and total > cap:
        scale = cap / total
        components = {name: value * scale for name, value in components.items()}

    return PrinciplePenaltyResult(components, reasons)


def _king_safety(before: chess.Board, after: chess.Board, move: chess.Move, mover: chess.Color, add) -> None:
    piece = before.piece_at(move.from_square)
    if piece is None:
        return

    if piece.piece_type == chess.PAWN and _is_early_f_pawn_move(before, move, mover):
        add("king_safety", 0.8, "early_f_pawn_move_weakens_king")

    if piece.piece_type == chess.PAWN and _is_kingside_pawn_push(before, move, mover):
        if _own_king_castled_or_kingside(before, mover) and not before.is_capture(move) and not after.is_check():
            add("king_safety", 1.0, "kingside_pawn_push_after_castling")

    if piece.piece_type == chess.KING and before.fullmove_number <= 18 and not before.is_castling(move):
        if not before.is_check() and not after.is_check():
            add("king_safety", 0.9, "early_king_move")

    if before.fullmove_number >= 10 and _king_on_start_square(before, mover) and _has_any_castling_right(before, mover):
        if piece.piece_type not in {chess.KING, chess.ROOK} and not before.is_capture(move) and not after.is_check():
            add("king_safety", 0.7, "delayed_castling")

    if _king_on_start_square(before, mover) and _opens_center(before, move):
        if _is_opening_central_pawn_claim(before, move, mover):
            return
        add("king_safety", 0.9, "opened_center_with_uncastled_king")


def _opening_development(before: chess.Board, after: chess.Board, move: chess.Move, mover: chess.Color, add) -> None:
    piece = before.piece_at(move.from_square)
    phase_scale = _opening_phase_scale(before, mover)
    if piece is None or phase_scale <= 0.0:
        return

    if piece.piece_type == chess.QUEEN and _minor_pieces_on_home_squares(before, mover) >= 2:
        if not before.is_capture(move) and not after.is_check():
            add("opening_development", 1.0 * phase_scale, "early_queen_move")

    if piece.piece_type in {chess.KNIGHT, chess.BISHOP} and _piece_has_moved_before(before, move.from_square):
        if not before.is_capture(move) and not after.is_check():
            add("opening_development", 0.7 * phase_scale, "same_piece_moved_twice_opening")

    if before.is_capture(move) or after.is_check():
        return

    home_minors = _minor_pieces_on_home_squares(before, mover)
    if home_minors < 2:
        return

    if piece.piece_type == chess.PAWN:
        if _is_early_f_pawn_move(before, move, mover):
            add("opening_development", 1.2 * phase_scale, "early_f_pawn_move_blocks_knight")
        if _is_timid_opening_pawn_single_step(before, move, mover):
            scale = 0.65 if _opens_home_bishop_diagonal(before, move, mover) else 1.0
            add("opening_development", scale * phase_scale, "single_step_pawn_when_double_step_available")
            repeated_scale = min(1.2, 0.55 * _recent_own_timid_pawn_single_steps(before, mover))
            if repeated_scale > 0.0:
                add("opening_development", repeated_scale * phase_scale, "serial_single_step_pawn_moves_before_development")
        if _piece_has_moved_before(before, move.from_square):
            add("opening_development", 0.9 * phase_scale, "repeated_pawn_move_before_development")
        elif _flank_pawn_push(before, move, mover) and not (
            _opens_home_bishop_diagonal(before, move, mover) and _is_single_step_pawn_move(move)
        ):
            add("opening_development", 1.0 * phase_scale, "flank_pawn_push_before_minor_development")
        elif (
            _has_central_pawn_presence(before, mover)
            and not _is_central_pawn_advance(before, move, mover)
            and not _opens_home_bishop_diagonal(before, move, mover)
        ):
            add("opening_development", 0.7 * phase_scale, "extra_pawn_move_before_minor_development")
        return

    if piece.piece_type in {chess.KNIGHT, chess.BISHOP}:
        if _is_passive_development_square(move.to_square, mover):
            add("opening_development", 0.6 * phase_scale, "minor_piece_to_passive_square")
        return

    if piece.piece_type == chess.ROOK:
        add("opening_development", 0.9 * phase_scale, "rook_move_before_minor_development")
        return

    if before.fullmove_number >= 5 and not before.is_castling(move):
        add("opening_development", 0.6 * phase_scale, "delays_minor_piece_development")


def _center_control(before: chess.Board, after: chess.Board, move: chess.Move, mover: chess.Color, add) -> None:
    piece = before.piece_at(move.from_square)
    if piece is None:
        return

    if piece.piece_type == chess.PAWN and before.is_capture(move):
        from_file = chess.square_file(move.from_square)
        to_file = chess.square_file(move.to_square)
        toward_center = abs(to_file - 3.5) < abs(from_file - 3.5)
        if not toward_center:
            add("center_control", 0.5, "pawn_capture_away_from_center")

    if _flank_pawn_push(before, move, mover) and not _has_central_pawn_presence(after, mover):
        add("center_control", 0.5, "flank_play_without_center")


def _tactics(after: chess.Board, mover: chess.Color, add) -> None:
    opponent = not mover
    if _opponent_has_mate_in_one(after, opponent):
        add("tactics", 1.0, "allows_mate_in_one")
    if _opponent_has_royal_fork(after, opponent):
        add("tactics", 0.9, "allows_royal_fork")


def _pawn_structure(before: chess.Board, after: chess.Board, move: chess.Move, mover: chess.Color, add) -> None:
    piece = before.piece_at(move.from_square)
    if piece is None or piece.piece_type != chess.PAWN:
        return

    before_doubled = _doubled_pawns(before, mover)
    after_doubled = _doubled_pawns(after, mover)
    if after_doubled > before_doubled:
        add("pawn_structure", min(1.0, after_doubled - before_doubled), "creates_doubled_pawn")

    before_isolated = _isolated_pawns(before, mover)
    after_isolated = _isolated_pawns(after, mover)
    if after_isolated > before_isolated:
        add("pawn_structure", min(1.0, after_isolated - before_isolated), "creates_isolated_pawn")


def _piece_activity(before: chess.Board, after: chess.Board, move: chess.Move, mover: chess.Color, add) -> None:
    piece = before.piece_at(move.from_square)
    if piece is None:
        return

    if piece.piece_type == chess.KNIGHT and _is_rim_square(move.to_square) and not before.is_capture(move) and not after.is_check():
        add("piece_activity", 0.8, "knight_on_rim")

    if piece.piece_type in {chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN}:
        if _undoes_recent_own_piece_move(before, move, mover) and not before.is_capture(move) and not after.is_check():
            add("piece_activity", 0.7, "undoes_recent_piece_move")

    captured = before.piece_at(move.to_square)
    if piece.piece_type == chess.BISHOP and captured is not None and captured.piece_type == chess.KNIGHT:
        if not after.is_check() and _material_without_kings(before, mover) >= _material_without_kings(before, not mover):
            add("piece_activity", 0.4, "bishop_for_knight_without_clear_reason")


def _rook_activity(before: chess.Board, after: chess.Board, move: chess.Move, mover: chess.Color, add) -> None:
    piece = before.piece_at(move.from_square)
    if piece is not None and piece.color == mover and piece.piece_type == chess.ROOK:
        if not before.is_capture(move) and not after.is_check():
            home_minors = _minor_pieces_on_home_squares(before, mover)
            if before.fullmove_number <= 24 and home_minors >= 2:
                add("rook_activity", 0.8, "rook_move_with_minor_pieces_undeveloped")
            if _undoes_recent_own_piece_move(before, move, mover):
                add("rook_activity", 0.7, "rook_returns_to_previous_square")

    opponent = not mover
    enemy_rooks = after.pieces(chess.ROOK, opponent)
    danger_rank = 1 if mover == chess.WHITE else 6
    king = after.king(mover)
    for square in enemy_rooks:
        if chess.square_rank(square) == danger_rank:
            if king is not None and chess.square_distance(square, king) <= 4:
                add("rook_activity", 0.9, "allows_enemy_rook_on_seventh")
                break

    if _endgame_like(after) and _has_passed_pawn(after, mover):
        for rook in after.pieces(chess.ROOK, mover):
            if not _rook_behind_any_passed_pawn(after, rook, mover):
                add("rook_activity", 0.3, "rook_not_behind_passed_pawn")
                break


def _endgame(before: chess.Board, after: chess.Board, move: chess.Move, mover: chess.Color, add) -> None:
    if not _endgame_like(after):
        return
    piece = before.piece_at(move.from_square)
    if piece is None or piece.piece_type != chess.KING:
        return
    before_center = _distance_to_center(move.from_square)
    after_center = _distance_to_center(move.to_square)
    if after_center > before_center and not before.is_capture(move):
        add("endgame", 0.6, "passive_endgame_king")


def _is_kingside_pawn_push(board: chess.Board, move: chess.Move, mover: chess.Color) -> bool:
    piece = board.piece_at(move.from_square)
    return bool(piece and piece.color == mover and piece.piece_type == chess.PAWN and chess.square_file(move.from_square) in KINGSIDE_FILES)


def _is_early_f_pawn_move(board: chess.Board, move: chess.Move, mover: chess.Color) -> bool:
    piece = board.piece_at(move.from_square)
    if piece is None or piece.color != mover or piece.piece_type != chess.PAWN:
        return False
    if chess.square_file(move.from_square) != 5 or board.fullmove_number > 12:
        return False

    home_rank = 1 if mover == chess.WHITE else 6
    knight_square = chess.G1 if mover == chess.WHITE else chess.G8
    knight = board.piece_at(knight_square)
    return bool(
        chess.square_rank(move.from_square) == home_rank
        and knight is not None
        and knight.color == mover
        and knight.piece_type == chess.KNIGHT
        and _king_on_start_square(board, mover)
    )


def _flank_pawn_push(board: chess.Board, move: chess.Move, mover: chess.Color) -> bool:
    piece = board.piece_at(move.from_square)
    if piece is None or piece.color != mover or piece.piece_type != chess.PAWN:
        return False
    return chess.square_file(move.from_square) in KINGSIDE_FILES | QUEENSIDE_FILES


def _own_king_castled_or_kingside(board: chess.Board, color: chess.Color) -> bool:
    king = board.king(color)
    if king is None:
        return False
    home_rank = 0 if color == chess.WHITE else 7
    return chess.square_rank(king) == home_rank and chess.square_file(king) >= 5


def _king_on_start_square(board: chess.Board, color: chess.Color) -> bool:
    return board.king(color) == (chess.E1 if color == chess.WHITE else chess.E8)


def _has_any_castling_right(board: chess.Board, color: chess.Color) -> bool:
    return board.has_kingside_castling_rights(color) or board.has_queenside_castling_rights(color)


def _opens_center(board: chess.Board, move: chess.Move) -> bool:
    return move.from_square in CENTRAL_SQUARES or move.to_square in CENTRAL_SQUARES or (
        board.is_capture(move) and move.to_square in EXTENDED_CENTER
    )


def _minor_pieces_on_home_squares(board: chess.Board, color: chess.Color) -> int:
    squares = (chess.B1, chess.C1, chess.F1, chess.G1) if color == chess.WHITE else (chess.B8, chess.C8, chess.F8, chess.G8)
    return sum(1 for square in squares if (piece := board.piece_at(square)) and piece.color == color and piece.piece_type in {chess.KNIGHT, chess.BISHOP})


def _opening_phase_scale(board: chess.Board, color: chess.Color) -> float:
    if _endgame_like(board):
        return 0.0
    if _minor_pieces_on_home_squares(board, color) <= 1:
        return 0.0
    if board.fullmove_number > 12:
        return 0.0
    if board.fullmove_number <= 8:
        return 1.0
    return max(0.0, (13.0 - float(board.fullmove_number)) / 5.0)


def _piece_has_moved_before(board: chess.Board, square: int) -> bool:
    for move in board.move_stack:
        if move.to_square == square:
            return True
    return False


def _undoes_recent_own_piece_move(board: chess.Board, move: chess.Move, mover: chess.Color, plies: int = 8) -> bool:
    start = max(0, len(board.move_stack) - int(plies))
    last_move_index = len(board.move_stack) - 1
    for idx in range(len(board.move_stack) - 1, start - 1, -1):
        previous = board.move_stack[idx]
        previous_mover = not board.turn if (last_move_index - idx) % 2 == 0 else board.turn
        if previous_mover != mover:
            continue
        if previous.from_square == move.to_square and previous.to_square == move.from_square:
            return True
    return False


def _has_central_pawn_presence(board: chess.Board, color: chess.Color) -> bool:
    return any(square in CENTRAL_SQUARES for square in board.pieces(chess.PAWN, color))


def _is_central_pawn_advance(board: chess.Board, move: chess.Move, mover: chess.Color) -> bool:
    piece = board.piece_at(move.from_square)
    if piece is None or piece.color != mover or piece.piece_type != chess.PAWN:
        return False
    return move.to_square in CENTRAL_SQUARES and chess.square_file(move.from_square) in {3, 4}


def _is_opening_central_pawn_claim(board: chess.Board, move: chess.Move, mover: chess.Color) -> bool:
    piece = board.piece_at(move.from_square)
    if piece is None or piece.color != mover or piece.piece_type != chess.PAWN:
        return False
    if board.fullmove_number > 8:
        return False
    from_file = chess.square_file(move.from_square)
    from_rank = chess.square_rank(move.from_square)
    start_rank = 1 if mover == chess.WHITE else 6
    return bool(from_file in {3, 4} and from_rank == start_rank and move.to_square in CENTRAL_SQUARES)


def _is_single_step_pawn_move(move: chess.Move) -> bool:
    return abs(chess.square_rank(move.to_square) - chess.square_rank(move.from_square)) == 1


def _is_timid_opening_pawn_single_step(board: chess.Board, move: chess.Move, mover: chess.Color) -> bool:
    piece = board.piece_at(move.from_square)
    if piece is None or piece.color != mover or piece.piece_type != chess.PAWN:
        return False
    if not _is_home_pawn_single_step_shape(move, mover):
        return False
    if chess.square_file(move.from_square) not in EARLY_SINGLE_STEP_PAWN_FILES:
        return False

    double_move = _matching_double_step_move(move, mover)
    return double_move in board.legal_moves


def _recent_own_timid_pawn_single_steps(board: chess.Board, mover: chess.Color, plies: int = 8) -> int:
    count = 0
    start = max(0, len(board.move_stack) - int(plies))
    last_move_index = len(board.move_stack) - 1
    for idx in range(len(board.move_stack) - 1, start - 1, -1):
        previous = board.move_stack[idx]
        previous_mover = not board.turn if (last_move_index - idx) % 2 == 0 else board.turn
        if previous_mover != mover:
            continue
        if _is_home_pawn_single_step_shape(previous, mover):
            count += 1
    return count


def _is_home_pawn_single_step_shape(move: chess.Move, mover: chess.Color) -> bool:
    from_rank = chess.square_rank(move.from_square)
    to_rank = chess.square_rank(move.to_square)
    from_file = chess.square_file(move.from_square)
    to_file = chess.square_file(move.to_square)
    start_rank = 1 if mover == chess.WHITE else 6
    step = 1 if mover == chess.WHITE else -1
    return from_file == to_file and from_rank == start_rank and to_rank == start_rank + step


def _matching_double_step_move(move: chess.Move, mover: chess.Color) -> chess.Move:
    from_file = chess.square_file(move.from_square)
    start_rank = 1 if mover == chess.WHITE else 6
    step = 1 if mover == chess.WHITE else -1
    return chess.Move(move.from_square, chess.square(from_file, start_rank + 2 * step))


def _opens_home_bishop_diagonal(board: chess.Board, move: chess.Move, mover: chess.Color) -> bool:
    piece = board.piece_at(move.from_square)
    if piece is None or piece.color != mover or piece.piece_type != chess.PAWN:
        return False

    start_rank = 1 if mover == chess.WHITE else 6
    if chess.square_rank(move.from_square) != start_rank:
        return False

    home_rank = 0 if mover == chess.WHITE else 7
    file_idx = chess.square_file(move.from_square)
    bishop_files = {
        1: (2,),
        3: (2,),
        4: (5,),
        6: (5,),
    }.get(file_idx, ())
    for bishop_file in bishop_files:
        bishop = board.piece_at(chess.square(bishop_file, home_rank))
        if bishop and bishop.color == mover and bishop.piece_type == chess.BISHOP:
            return True
    return False


def _is_passive_development_square(square: int, mover: chess.Color) -> bool:
    if _is_rim_square(square):
        return True
    home_rank = 0 if mover == chess.WHITE else 7
    return chess.square_rank(square) == home_rank


def _opponent_has_mate_in_one(board: chess.Board, opponent: chess.Color) -> bool:
    if board.turn != opponent:
        return False
    for reply in board.legal_moves:
        board.push(reply)
        try:
            if board.is_checkmate():
                return True
        finally:
            board.pop()
    return False


def _opponent_has_royal_fork(board: chess.Board, opponent: chess.Color) -> bool:
    if board.turn != opponent:
        return False
    mover_king = board.king(not opponent)
    mover_queen = next(iter(board.pieces(chess.QUEEN, not opponent)), None)
    mover_rooks = set(board.pieces(chess.ROOK, not opponent))
    if mover_king is None:
        return False

    for reply in board.legal_moves:
        piece = board.piece_at(reply.from_square)
        if piece is None or piece.color != opponent or piece.piece_type != chess.KNIGHT:
            continue
        board.push(reply)
        try:
            moved_knight = board.piece_at(reply.to_square)
            if moved_knight is None:
                continue
            attacks_king = board.is_check()
            attacks_queen = mover_queen is not None and board.is_attacked_by(opponent, mover_queen)
            attacks_rook = any(board.is_attacked_by(opponent, rook) for rook in mover_rooks)
            if attacks_king and (attacks_queen or attacks_rook):
                return True
        finally:
            board.pop()
    return False


def _doubled_pawns(board: chess.Board, color: chess.Color) -> int:
    total = 0
    for file_idx in range(8):
        count = sum(1 for square in board.pieces(chess.PAWN, color) if chess.square_file(square) == file_idx)
        total += max(0, count - 1)
    return total


def _isolated_pawns(board: chess.Board, color: chess.Color) -> int:
    pawn_files = {chess.square_file(square) for square in board.pieces(chess.PAWN, color)}
    isolated = 0
    for square in board.pieces(chess.PAWN, color):
        file_idx = chess.square_file(square)
        if (file_idx - 1) not in pawn_files and (file_idx + 1) not in pawn_files:
            isolated += 1
    return isolated


def _is_rim_square(square: int) -> bool:
    return chess.square_file(square) in RIM_FILES or chess.square_rank(square) in RIM_RANKS


def _material_without_kings(board: chess.Board, color: chess.Color) -> int:
    values = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900}
    return sum(len(board.pieces(piece_type, color)) * value for piece_type, value in values.items())


def _endgame_like(board: chess.Board) -> bool:
    if board.pieces(chess.QUEEN, chess.WHITE) or board.pieces(chess.QUEEN, chess.BLACK):
        return False
    return _material_without_kings(board, chess.WHITE) + _material_without_kings(board, chess.BLACK) <= 2600


def _has_passed_pawn(board: chess.Board, color: chess.Color) -> bool:
    return any(_is_passed_pawn(board, square, color) for square in board.pieces(chess.PAWN, color))


def _is_passed_pawn(board: chess.Board, square: int, color: chess.Color) -> bool:
    file_idx = chess.square_file(square)
    rank = chess.square_rank(square)
    enemy = not color
    for enemy_pawn in board.pieces(chess.PAWN, enemy):
        enemy_file = chess.square_file(enemy_pawn)
        enemy_rank = chess.square_rank(enemy_pawn)
        if abs(enemy_file - file_idx) > 1:
            continue
        if color == chess.WHITE and enemy_rank > rank:
            return False
        if color == chess.BLACK and enemy_rank < rank:
            return False
    return True


def _rook_behind_any_passed_pawn(board: chess.Board, rook: int, color: chess.Color) -> bool:
    rook_file = chess.square_file(rook)
    rook_rank = chess.square_rank(rook)
    for pawn in board.pieces(chess.PAWN, color):
        if chess.square_file(pawn) != rook_file or not _is_passed_pawn(board, pawn, color):
            continue
        pawn_rank = chess.square_rank(pawn)
        if color == chess.WHITE and rook_rank < pawn_rank:
            return True
        if color == chess.BLACK and rook_rank > pawn_rank:
            return True
    return False


def _distance_to_center(square: int) -> float:
    file_idx = chess.square_file(square)
    rank = chess.square_rank(square)
    return min(abs(file_idx - 3) + abs(rank - 3), abs(file_idx - 4) + abs(rank - 3), abs(file_idx - 3) + abs(rank - 4), abs(file_idx - 4) + abs(rank - 4))
