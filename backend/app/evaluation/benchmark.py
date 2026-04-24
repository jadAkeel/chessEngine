import math
import chess
from .metrics import evaluate_board

CAPTURE_BONUS = 10000


def order_moves(board):
    moves = list(board.legal_moves)

    def score(move):
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            if victim and attacker:
                return CAPTURE_BONUS + victim.piece_type * 10 - attacker.piece_type
        if board.gives_check(move):
            return 500
        return 0

    moves.sort(key=score, reverse=True)
    return moves


def quiescence(board, alpha, beta):
    stand_pat = evaluate_board(board)

    if stand_pat >= beta:
        return beta

    alpha = max(alpha, stand_pat)

    for move in board.legal_moves:
        if not board.is_capture(move):
            continue

        board.push(move)
        score = -quiescence(board, -beta, -alpha)
        board.pop()

        if score >= beta:
            return beta

        alpha = max(alpha, score)

    return alpha


def negamax(board, depth, alpha, beta, color):
    if depth == 0 or board.is_game_over():
        return color * quiescence(board, alpha, beta)

    best = -math.inf

    for move in order_moves(board):
        board.push(move)

        score = -negamax(board, depth - 1, -beta, -alpha, -color)

        board.pop()

        best = max(best, score)
        alpha = max(alpha, score)

        if alpha >= beta:
            break

    return best


def find_best_move(fen, depth=3):
    board = chess.Board(fen)

    best_move = None
    best_score = -math.inf

    for move in order_moves(board):
        board.push(move)

        score = -negamax(board, depth - 1, -math.inf, math.inf, -1)

        board.pop()

        if score > best_score:
            best_score = score
            best_move = move

    return (best_move.uci() if best_move else None, int(best_score))