"""The root tactical layer: proven short mates, draw avoidance while winning, and a
material screen that counts only what the move itself gives away."""
from unittest.mock import patch

import chess

from app.game.tactics import find_forcing_mate_in_two, find_mate_in_one, select_safe_move
from app.infra.config import AppConfig, MCTSConfig, ModelConfig
from app.mcts.search import MCTS

KASPAROV_MATE_IN_TWO = "Q2rq2r/1p1k1pp1/1Bbpn2p/5Bb1/P2N4/1P5P/5PP1/2R1R1K1 w - - 8 27"  # Qxb7+ Bxb7 Rc7#
WON_QUEEN_ENDING = "8/8/8/8/3k4/q7/8/1K6 b - - 17 63"  # match game 6: ...Kc3 and ...Kd3 stalemate
CARLSEN_MAEDER = "r1bQ2rk/pp1n3p/2p5/3p1N2/3P1q2/P1N5/1PP2PPP/1K2R2R w - - 15 28"  # Qd8 already en prise
ABANDONED_KNIGHT = "r1b2rk1/pp2nppp/4p3/3pP3/1q6/1P1B1Q2/P2N1PPP/R3K2R w KQ - 3 13"  # O-O drops Nd2


def _forces_mate(board: chess.Board, move: chess.Move) -> bool:
    board.push(move)
    try:
        if board.is_checkmate():
            return True
        for reply in list(board.legal_moves):
            board.push(reply)
            try:
                if find_mate_in_one(board) is None:
                    return False
            finally:
                board.pop()
        return True
    finally:
        board.pop()


def test_forcing_mate_in_two_is_found_and_is_a_real_mate():
    board = chess.Board(KASPAROV_MATE_IN_TWO)
    move = find_forcing_mate_in_two(board)
    assert move is not None and board.gives_check(move)
    assert _forces_mate(board, move)
    assert find_forcing_mate_in_two(chess.Board()) is None


def test_search_plays_a_proven_mate_in_two_without_the_network():
    """PUCT at 64 visits found 2 of 11 such GM mates: the first move is a low-prior sacrifice."""
    board = chess.Board(KASPAROV_MATE_IN_TWO)
    cfg = AppConfig(model=ModelConfig(channels=8, res_blocks=1), mcts=MCTSConfig(num_simulations=64))
    with patch("app.mcts.search.predict_boards") as predict:
        result = MCTS(None, cfg).search(board, num_simulations=64)
    predict.assert_not_called()
    assert _forces_mate(board, result["best_move"])
    assert result["root_value"] == 1.0


def test_winning_side_refuses_a_stalemating_move():
    board = chess.Board(WON_QUEEN_ENDING)
    stalemate, winning = chess.Move.from_uci("d4c3"), chess.Move.from_uci("d4c4")
    assert select_safe_move(board, [stalemate, winning], root_value=0.8)[0] == winning
    # Without a winning evaluation a game-ending move stays acceptable (a draw can be welcome).
    assert select_safe_move(board, [stalemate, winning], root_value=-0.5)[0] == stalemate
    assert select_safe_move(board, [stalemate, winning])[0] == stalemate


def test_checkmate_is_always_taken():
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")
    assert select_safe_move(board, [chess.Move.from_uci("d8h4")], root_value=0.9)[0] == chess.Move.from_uci("d8h4")


def test_material_already_en_prise_does_not_veto_every_other_move():
    """Qd8 is attacked before the move. Counting that loss against Re8 made the old screen
    play Qxg8+ (queen for a rook; checks were exempt)."""
    board = chess.Board(CARLSEN_MAEDER)
    re8, qxg8 = chess.Move.from_uci("e1e8"), chess.Move.from_uci("d8g8")
    assert select_safe_move(board, [re8, qxg8])[0] == re8


def test_a_move_that_abandons_a_defended_piece_is_still_vetoed():
    board = chess.Board(ABANDONED_KNIGHT)
    castle, defend = chess.Move.from_uci("e1g1"), chess.Move.from_uci("a1d1")
    assert select_safe_move(board, [castle, defend])[0] == defend


def test_clear_search_evidence_keeps_a_move_the_material_estimate_vetoes():
    board = chess.Board(ABANDONED_KNIGHT)
    castle, defend = chess.Move.from_uci("e1g1"), chess.Move.from_uci("a1d1")
    backed = {castle: (10, 0.55), defend: (6, 0.10)}
    assert select_safe_move(board, [castle, defend], move_values=backed)[0] == castle
    thin = {castle: (5, 0.55), defend: (6, 0.10)}  # too few visits to trust
    assert select_safe_move(board, [castle, defend], move_values=thin)[0] == defend


def test_a_checking_sacrifice_that_forces_mate_is_not_vetoed():
    board = chess.Board(KASPAROV_MATE_IN_TWO)
    sacrifice = find_forcing_mate_in_two(board)
    quiet = next(move for move in board.legal_moves if not board.gives_check(move) and not board.is_capture(move))
    assert select_safe_move(board, [sacrifice, quiet])[0] == sacrifice


def test_screen_restores_the_board():
    board = chess.Board(CARLSEN_MAEDER)
    before = board.fen(), list(board.move_stack)
    select_safe_move(board, list(board.legal_moves), root_value=0.3)
    assert (board.fen(), list(board.move_stack)) == before
