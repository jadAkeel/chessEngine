import chess

MATE_SCORE = 100000

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

PAWN_TABLE = [
 0,0,0,0,0,0,0,0,
 5,10,10,-20,-20,10,10,5,
 5,-5,-10,0,0,-10,-5,5,
 0,0,0,20,20,0,0,0,
 5,5,10,25,25,10,5,5,
 10,10,20,30,30,20,10,10,
 50,50,50,50,50,50,50,50,
 0,0,0,0,0,0,0,0
]

KNIGHT_TABLE = [
 -50,-40,-30,-30,-30,-30,-40,-50,
 -40,-20,0,5,5,0,-20,-40,
 -30,5,10,15,15,10,5,-30,
 -30,0,15,20,20,15,0,-30,
 -30,5,15,20,20,15,5,-30,
 -30,0,10,15,15,10,0,-30,
 -40,-20,0,0,0,0,-20,-40,
 -50,-40,-30,-30,-30,-30,-40,-50
]

BISHOP_TABLE = [
 -20,-10,-10,-10,-10,-10,-10,-20,
 -10,5,0,0,0,0,5,-10,
 -10,10,10,10,10,10,10,-10,
 -10,0,10,10,10,10,0,-10,
 -10,5,5,10,10,5,5,-10,
 -10,0,5,10,10,5,0,-10,
 -10,0,0,0,0,0,0,-10,
 -20,-10,-10,-10,-10,-10,-10,-20
]

PIECE_TABLES = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
}

PASSED_PAWN_BONUS_BY_RELATIVE_RANK = [0, 8, 16, 28, 45, 80, 150, 0]


def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo_pair(rating_a, rating_b, score_a, k_factor=24):
    exp_a = expected_score(rating_a, rating_b)
    new_a = rating_a + k_factor * (score_a - exp_a)
    new_b = rating_b + k_factor * ((1 - score_a) - (1 - exp_a))
    return float(new_a), float(new_b)


def piece_square_bonus(piece, square):
    table = PIECE_TABLES.get(piece.piece_type)
    if table is None:
        return 0

    idx = square if piece.color == chess.WHITE else chess.square_mirror(square)
    val = table[idx]

    return val if piece.color == chess.WHITE else -val


def is_passed_pawn(board: chess.Board, square: int, color: chess.Color) -> bool:
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


def passed_pawn_bonus(board: chess.Board, square: int, color: chess.Color) -> int:
    if not is_passed_pawn(board, square, color):
        return 0
    rank = chess.square_rank(square)
    relative_rank = rank if color == chess.WHITE else 7 - rank
    bonus = PASSED_PAWN_BONUS_BY_RELATIVE_RANK[relative_rank]
    return bonus if color == chess.WHITE else -bonus


def mobility_bonus(board):
    return 2 * (board.legal_moves.count())


def evaluate_board(board: chess.Board):
    if board.is_checkmate():
        return -MATE_SCORE if board.turn else MATE_SCORE

    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = 0

    for square, piece in board.piece_map().items():
        value = PIECE_VALUES[piece.piece_type]

        if piece.color == chess.WHITE:
            score += value
        else:
            score -= value

        score += piece_square_bonus(piece, square)
        if piece.piece_type == chess.PAWN:
            score += passed_pawn_bonus(board, square, piece.color)

    # mobility
    turn = 1 if board.turn == chess.WHITE else -1
    score += mobility_bonus(board) * turn

    return score
