import chess

from app.api.main import (
    _adaptive_simulations,
    _fastmove_complexity,
    _find_mate_in_one,
    _move_allows_mate_in_one,
)


def test_find_mate_in_one_from_reported_game():
    board = chess.Board()
    for san in (
        "d4 e6 e3 d5 Bd3 Nf6 c3 c5 Bd2 Nc6 Nf3 Be7 c4 O-O b3 cxd4 "
        "exd4 dxc4 Bxc4 Nxd4 Nxd4 Qxd4 Nc3 Rd8 O-O Qxd2 Qxd2 Rxd2 "
        "g3 a6 Rfd1 Rxd1+ Nxd1 b5 Bd3 Bb7 f3 Bxf3 Nf2 Rd8 Bf1 Bc5 "
        "Re1 Bxf2+ Kxf2 Be4 a3 Ng4+ Kg1 Bf3 b4 Rd2 Bd3 Rxd3 Rc1 Rxa3"
    ).split():
        board.push_san(san)

    move = _find_mate_in_one(board)

    assert move is not None
    assert board.san(move) == "Rc8#"


def test_adaptive_simulations_keep_obvious_positions_fast():
    assert _adaptive_simulations(depth=6, complexity=0, max_simulations=180) == 0
    assert _adaptive_simulations(depth=6, complexity=1, max_simulations=180) == 0


def test_adaptive_simulations_raise_budget_for_complex_positions():
    assert _adaptive_simulations(depth=6, complexity=3, max_simulations=180) == 60
    assert _adaptive_simulations(depth=6, complexity=8, max_simulations=180) == 150
    assert _adaptive_simulations(depth=10, complexity=8, max_simulations=180) == 180


def test_complexity_marks_check_as_forcing_position():
    board = chess.Board("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")

    complexity, reasons = _fastmove_complexity(board, [])

    assert complexity >= 3
    assert "king_in_check" in reasons


def test_move_allows_mate_in_one_detects_blunder():
    board = chess.Board()
    board.push_san("f3")
    board.push_san("e5")

    assert _move_allows_mate_in_one(board, "g2g4") is True
    assert _move_allows_mate_in_one(board, "g2g3") is False
