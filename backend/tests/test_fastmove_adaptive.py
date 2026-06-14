import chess

from app.api.main import (
    _adaptive_simulations,
    _fastmove_complexity,
    _find_mate_in_one,
    _is_decisive_fast_choice,
    _is_light_adaptive_search,
    _move_allows_forced_mate_in_two,
    _move_allows_mate_in_one,
    _should_use_adaptive_search,
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
    assert _adaptive_simulations(depth=6, complexity=0, max_simulations=96) == 0
    assert _adaptive_simulations(depth=6, complexity=1, max_simulations=96) == 0
    assert _adaptive_simulations(depth=6, complexity=2, max_simulations=96) == 0


def test_adaptive_simulations_raise_budget_for_complex_positions():
    assert _adaptive_simulations(depth=6, complexity=3, max_simulations=96) == 28
    assert _adaptive_simulations(depth=6, complexity=8, max_simulations=96) == 72
    assert _adaptive_simulations(depth=10, complexity=8, max_simulations=96) == 96


def test_light_adaptive_search_uses_small_budget_for_close_forcing_choices():
    reasons = ["forcing_moves_available", "top_moves_very_close"]

    assert _should_use_adaptive_search(4, reasons, depth=6) is True
    assert _is_light_adaptive_search(4, reasons, depth=6) is True
    assert _adaptive_simulations(depth=6, complexity=4, max_simulations=96, light=True) == 8


def test_light_adaptive_search_keeps_full_budget_for_tactical_danger():
    reasons = ["many_forcing_moves", "top_moves_very_close"]

    assert _is_light_adaptive_search(5, reasons, depth=6) is False
    assert _is_light_adaptive_search(
        4,
        ["forcing_moves_available", "top_moves_very_close", "best_fast_move_allows_mate_two"],
        depth=6,
    ) is False
    assert _adaptive_simulations(depth=6, complexity=5, max_simulations=96, light=False) == 40


def test_adaptive_search_skips_quiet_close_policy_scores():
    assert _should_use_adaptive_search(3, ["top_moves_very_close"], depth=6) is False
    assert _should_use_adaptive_search(4, ["top_moves_close", "forcing_moves_available"], depth=6) is False
    assert _should_use_adaptive_search(2, ["many_forcing_moves"], depth=6) is False


def test_adaptive_search_runs_for_tactical_urgency():
    assert _should_use_adaptive_search(3, ["king_in_check"], depth=6) is True
    assert _should_use_adaptive_search(4, ["best_fast_move_allows_mate"], depth=6) is True
    assert _should_use_adaptive_search(6, ["many_forcing_moves", "top_moves_close"], depth=6) is True
    assert _should_use_adaptive_search(4, ["many_forcing_moves", "top_moves_close"], depth=6) is True
    assert _should_use_adaptive_search(
        4,
        ["many_forcing_moves", "top_moves_competitive", "high_value_capture"],
        depth=6,
    ) is True
    assert _should_use_adaptive_search(4, ["forcing_moves_available", "top_moves_very_close"], depth=6) is True


def test_decisive_fast_choice_skips_obvious_material_win():
    board = chess.Board("4k3/8/8/8/8/4q3/8/4RK2 w - - 0 1")
    candidates = [
        {"uci": "e1e3", "score": 900.0},
        {"uci": "f1g1", "score": 860.0},
    ]

    assert _is_decisive_fast_choice(board, candidates, ["many_forcing_moves", "top_moves_close"]) is True


def test_decisive_fast_choice_does_not_skip_small_unclear_capture():
    board = chess.Board("4k3/8/8/8/8/8/4p3/4RK2 w - - 0 1")
    candidates = [
        {"uci": "e1e2", "score": 250.0},
        {"uci": "f1g1", "score": 230.0},
    ]

    assert _is_decisive_fast_choice(board, candidates, ["many_forcing_moves", "top_moves_close"]) is False


def test_move_allows_forced_mate_in_two_from_reported_loss():
    board = chess.Board("8/5pk1/3p1n2/3Br3/5Qp1/6K1/8/7R b - - 5 36")

    assert _move_allows_forced_mate_in_two(board, "e5d5") is True
    assert _move_allows_forced_mate_in_two(board, "g7g8") is False


def test_complexity_marks_best_fast_move_allows_mate_two():
    board = chess.Board("8/5pk1/3p1n2/3Br3/5Qp1/6K1/8/7R b - - 5 36")
    candidates = [
        {"uci": "e5d5", "score": 900.0},
        {"uci": "g7g8", "score": 850.0},
    ]

    complexity, reasons = _fastmove_complexity(board, candidates)

    assert complexity >= 4
    assert "best_fast_move_allows_mate_two" in reasons


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
