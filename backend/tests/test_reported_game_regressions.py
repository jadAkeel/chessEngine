import chess

from app.api.main import (
    _fastmove_complexity,
    _is_king_exposed,
    _move_allows_promotion_threat,
    _safe_candidate_fallback,
    _should_use_adaptive_search,
)


def test_reported_loss_rejects_rook_capture_with_mate_two_risk():
    board = chess.Board("8/5pk1/3p1n2/3Br3/5Qp1/6K1/8/7R b - - 5 36")
    candidates = [
        {"uci": "e5d5", "score": 900.0},
        {"uci": "g7g8", "score": 850.0},
    ]

    fallback = _safe_candidate_fallback(board, candidates)

    assert fallback is not None
    assert fallback[0] == "g7g8"


def test_reported_loss_marks_king_exposure_before_final_tactic():
    board = chess.Board("8/5pk1/3p1n2/3Br3/5Qp1/6K1/8/7R b - - 5 36")

    assert _is_king_exposed(board, chess.BLACK) is True


def test_reported_loss_c5_allows_near_promotion_threat():
    board = chess.Board("B5k1/2p2p2/pp1p1np1/3P4/P2p1P2/6PK/1r6/7R b - - 0 24")

    assert _move_allows_promotion_threat(board, "c7c5") is True
    assert _move_allows_promotion_threat(board, "b2b3") is False


def test_reported_loss_complexity_marks_promotion_threat():
    board = chess.Board("B5k1/2p2p2/pp1p1np1/3P4/P2p1P2/6PK/1r6/7R b - - 0 24")
    candidates = [
        {"uci": "c7c5", "score": 900.0},
        {"uci": "b2b3", "score": 850.0},
    ]

    complexity, reasons = _fastmove_complexity(board, candidates)

    assert "best_fast_move_allows_promotion_threat" in reasons
    assert _should_use_adaptive_search(complexity, reasons, depth=6) is True


def test_hanging_queen_candidate_triggers_search_and_fallback():
    board = chess.Board("3qk3/8/8/8/2P5/8/8/4K3 b - - 0 1")
    candidates = [
        {"uci": "d8d5", "score": 1000.0},
        {"uci": "d8e7", "score": 100.0},
    ]

    complexity, reasons = _fastmove_complexity(board, candidates)
    fallback = _safe_candidate_fallback(board, candidates)

    assert "best_fast_move_allows_queen_capture" in reasons
    assert _should_use_adaptive_search(complexity, reasons, depth=6) is True
    assert fallback is not None
    assert fallback[0] == "d8e7"
