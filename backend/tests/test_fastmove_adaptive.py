import chess
import os
import pytest

from app.api.main import (
    _adaptive_simulations,
    _adaptive_simulation_steps,
    _fastmove_complexity,
    _find_mate_in_one,
    _has_later_simulation_step,
    _is_decisive_fast_choice,
    _is_light_adaptive_search,
    _load_model,
    _mcts_confident_enough,
    _move_safety_flags,
    _move_allows_forced_mate_in_two,
    _move_allows_mate_in_one,
    _move_allows_queen_capture,
    _move_allows_valuable_piece_capture,
    _score_fast_candidate,
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


def test_adaptive_simulations_start_from_minimum_for_quiet_positions():
    """Minimum probe is now 30 (not 12) for quiet / low-complexity positions."""
    assert _adaptive_simulations(depth=6, complexity=0, max_simulations=96) == 30
    assert _adaptive_simulations(depth=6, complexity=1, max_simulations=96) == 30
    assert _adaptive_simulations(depth=6, complexity=2, max_simulations=96) == 30
    assert _adaptive_simulation_steps(depth=6, complexity=0, max_simulations=96) == [30]


def test_adaptive_simulations_raise_budget_for_complex_positions():
    assert _adaptive_simulations(depth=6, complexity=3, max_simulations=96) == 30
    assert _adaptive_simulations(depth=6, complexity=4, max_simulations=96) == 64
    assert _adaptive_simulations(depth=6, complexity=6, max_simulations=96) == 76
    assert _adaptive_simulations(depth=6, complexity=8, max_simulations=96) == 96
    assert _adaptive_simulations(depth=10, complexity=8, max_simulations=96) == 96
    assert _adaptive_simulation_steps(depth=6, complexity=3, max_simulations=96) == [30]
    # With RUNGS=[30,64,96], target=96 yields progressive steps
    assert _adaptive_simulation_steps(depth=6, complexity=8, max_simulations=96) == [30, 64, 96]


def test_progressive_simulation_steps_targets():
    """Verify the progressive rung ladder matches the spec:
    target 30 -> [30], 64 -> [30,64], 76 -> [30,64,76], 96 -> [30,64,96].
    """
    steps_30 = _adaptive_simulation_steps(depth=6, complexity=2, max_simulations=96)
    assert steps_30 == [30], f"Expected [30] got {steps_30}"

    steps_64 = _adaptive_simulation_steps(depth=6, complexity=4, max_simulations=96)
    assert steps_64 == [30, 64], f"Expected [30,64] got {steps_64}"

    steps_76 = _adaptive_simulation_steps(depth=6, complexity=6, max_simulations=96)
    assert steps_76 == [30, 64, 76], f"Expected [30,64,76] got {steps_76}"

    steps_96 = _adaptive_simulation_steps(depth=6, complexity=8, max_simulations=96)
    assert steps_96 == [30, 64, 96], f"Expected [30,64,96] got {steps_96}"

    # With higher max_simulations and depth=10, target can exceed 96
    steps_deep = _adaptive_simulation_steps(depth=10, complexity=8, max_simulations=180)
    assert steps_deep == [30, 64, 96, 154], f"Expected [30,64,96,154] got {steps_deep}"


def test_unsafe_mcts_probe_can_retry_higher_budget():
    """Updated to use current minimum step of 30 (instead of legacy 12)."""
    assert _has_later_simulation_step(30, [30, 64, 96]) is True
    assert _has_later_simulation_step(64, [30, 64, 96]) is True
    assert _has_later_simulation_step(96, [30, 64, 96]) is False
    assert _has_later_simulation_step(30, [30]) is False


def test_light_adaptive_search_uses_small_budget_for_close_forcing_choices():
    reasons = ["forcing_moves_available", "top_moves_very_close"]

    assert _should_use_adaptive_search(4, reasons, depth=6) is True
    assert _is_light_adaptive_search(4, reasons, depth=6) is True
    assert _adaptive_simulations(depth=6, complexity=4, max_simulations=96, light=True) == 30
    assert _adaptive_simulation_steps(depth=6, complexity=4, max_simulations=96, light=True) == [30]


def test_light_adaptive_search_runs_for_maybe_tactical_positions():
    assert _should_use_adaptive_search(1, ["top_moves_competitive"], depth=6) is False
    assert _is_light_adaptive_search(1, ["top_moves_competitive"], depth=6) is True
    assert _is_light_adaptive_search(2, ["best_fast_move_allows_minor_capture"], depth=6) is True


def test_light_adaptive_search_keeps_full_budget_for_tactical_danger():
    reasons = ["many_forcing_moves", "top_moves_very_close"]
    assert _is_light_adaptive_search(5, reasons, depth=6) is False
    assert _is_light_adaptive_search(
        4,
        ["forcing_moves_available", "top_moves_very_close", "best_fast_move_allows_mate_two"],
        depth=6,
    ) is False
    assert _adaptive_simulations(depth=6, complexity=5, max_simulations=96, light=False) == 64


def test_full_adaptive_search_does_not_confidence_stop_at_first_probe():
    """With the new minimum of 30, neither light nor full mode should be confident at 12."""
    root_debug = [
        {"uci": "e2e4", "visits": 6},
        {"uci": "d2d4", "visits": 1},
    ]

    assert _mcts_confident_enough(
        fast_move="e2e4", mcts_move="e2e4", root_debug=root_debug,
        safety_flags={}, light=False, current_simulations=12,
    ) is False
    assert _mcts_confident_enough(
        fast_move="e2e4", mcts_move="e2e4", root_debug=root_debug,
        safety_flags={}, light=True, current_simulations=12,
    ) is False


def test_light_mode_confident_at_thirty_simulations():
    """Light mode can stop at 30 when MCTS agrees with fast move and no safety risk."""
    root_debug = [
        {"uci": "e2e4", "visits": 16},
        {"uci": "d2d4", "visits": 4},
    ]

    assert _mcts_confident_enough(
        fast_move="e2e4", mcts_move="e2e4", root_debug=root_debug,
        safety_flags={}, light=True, current_simulations=30,
    ) is True

    assert _mcts_confident_enough(
        fast_move="e2e4", mcts_move="e2e4", root_debug=root_debug,
        safety_flags={}, light=True, current_simulations=25,
    ) is False


def test_light_mode_confidence_requires_no_safety_risk():
    """Even at >= 30 sims, light mode should not stop if safety flags are present."""
    root_debug = [
        {"uci": "e2e4", "visits": 16},
        {"uci": "d2d4", "visits": 4},
    ]

    assert _mcts_confident_enough(
        fast_move="e2e4", mcts_move="e2e4", root_debug=root_debug,
        safety_flags={"mate_one": True, "mate_two": False, "promotion_threat": False,
                       "queen_capture": False, "valuable_piece_capture": False},
        light=True, current_simulations=30,
    ) is False



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
        4, ["many_forcing_moves", "top_moves_competitive", "high_value_capture"], depth=6,
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


def test_queen_move_in_front_of_pawn_is_marked_unsafe():
    board = chess.Board("3qk3/8/8/8/2P5/8/8/4K3 b - - 0 1")
    assert _move_allows_queen_capture(board, "d8d5") is True
    assert _move_allows_queen_capture(board, "d8e7") is False
    assert _move_allows_valuable_piece_capture(board, "d8d5") is True


def test_fast_score_penalizes_hanging_queen():
    board = chess.Board("3qk3/8/8/8/2P5/8/8/4K3 b - - 0 1")
    hanging_queen = chess.Move.from_uci("d8d5")
    safe_queen = chess.Move.from_uci("d8e7")
    assert _score_fast_candidate(board, hanging_queen, 0.90) < _score_fast_candidate(board, safe_queen, 0.10)


def test_fast_score_prefers_development_over_extra_quiet_pawn_push():
    board = chess.Board("rn1qk2r/1bp1n1bp/pp1p1pp1/4p3/1PBPP1P1/2N1BN1P/P1P2P2/R2Q1RK1 b kq - 0 8")
    extra_pawn = chess.Move.from_uci("c7c6")
    develop_knight = chess.Move.from_uci("b8c6")
    assert _score_fast_candidate(board, develop_knight, 0.30) > _score_fast_candidate(board, extra_pawn, 0.36)


def test_fast_score_prefers_profitable_rook_capture_from_reported_game():
    board = chess.Board()
    for san in (
        "e4 e5 Nf3 b6 d4 exd4 Nxd4 Bb7 Bf4 Bb4+ c3 Bc5 b4 Be7 "
        "e5 d5 Bd3 Nd7 c4 dxc4 Bxc4 Bxg2 Rg1 Bxb4+ Nd2 Bh3 Nf5 "
        "Bxf5 Rb1 Qe7 Qb3 O-O-O Qxb4 Nxe5 Qxe7 Nd3+ Bxd3 Nxe7 "
        "Bg5 Rhe8 Nf3 Bxd3 Rc1 f6 Rg4 fxg5 Kd2 h6 a4 Bc4+ "
        "Kc3 Be2 Rc4 Bxf3 Kb2 Nf5 Rxc7+ Kb8 a5 Rd2+ Kb1 Rxf2 "
        "axb6 axb6 R1c4 Rxh2 R7c6"
    ).split():
        board.push_san(san)

    capture_rook = chess.Move.from_uci("f3c6")
    quiet_king = chess.Move.from_uci("b8b7")

    assert board.san(capture_rook) == "Bxc6"
    assert board.san(quiet_king) == "Kb7"
    assert _score_fast_candidate(board, capture_rook, 0.10) > _score_fast_candidate(board, quiet_king, 0.10)
    assert _move_safety_flags(board, capture_rook.uci())["valuable_piece_capture"] is False
    assert _is_decisive_fast_choice(
        board,
        [{"uci": capture_rook.uci(), "score": 818.0}, {"uci": quiet_king.uci(), "score": 158.0}],
        ["many_forcing_moves", "best_fast_move_allows_minor_capture", "high_value_capture"],
    ) is True


def test_rook_and_minor_piece_hanging_moves_are_marked_unsafe():
    rook_board = chess.Board("3rk3/8/8/8/2P5/8/8/4K3 b - - 0 1")
    minor_board = chess.Board("4k3/8/5n2/8/2P5/8/8/4K3 b - - 0 1")
    assert _move_allows_valuable_piece_capture(rook_board, "d8d5") is True
    assert _move_allows_valuable_piece_capture(rook_board, "d8d7") is False
    assert _move_allows_valuable_piece_capture(minor_board, "f6d5") is True


# --- Missing-checkpoint guard tests ------------------------------------------

class _MockConfig:
    class _System:
        device = "cpu"
        checkpoint_path = ""
    system = _System()


class _MockModel:
    def to(self, device):
        return self
    def eval(self):
        pass


def test_load_model_raises_on_missing_checkpoint(monkeypatch, tmp_path):
    """Test that _load_model raises RuntimeError if checkpoint missing
    and ALLOW_RANDOM_WEIGHTS is not set."""
    missing = tmp_path / "nonexistent.pt"

    cfg = _MockConfig()
    cfg.system.checkpoint_path = str(missing)

    monkeypatch.setattr("app.api.main.load_config", lambda: cfg)
    monkeypatch.setattr("app.api.main.validate_config", lambda _: None)
    monkeypatch.setattr("app.api.main.select_device", lambda _: "cpu")
    monkeypatch.setattr("app.api.main.configure_torch_runtime", lambda *a, **kw: None)
    monkeypatch.setattr("app.api.main.ChessNet", lambda _cfg: _MockModel())
    monkeypatch.delenv("ALLOW_RANDOM_WEIGHTS", raising=False)

    with pytest.raises(RuntimeError, match="Checkpoint not found"):
        _load_model()


def test_load_model_allow_random_weights(monkeypatch, tmp_path):
    """Test that _load_model proceeds with random weights when
    ALLOW_RANDOM_WEIGHTS=1 and checkpoint is missing."""
    missing = tmp_path / "nonexistent.pt"

    cfg = _MockConfig()
    cfg.system.checkpoint_path = str(missing)

    monkeypatch.setattr("app.api.main.load_config", lambda: cfg)
    monkeypatch.setattr("app.api.main.validate_config", lambda _: None)
    monkeypatch.setattr("app.api.main.select_device", lambda _: "cpu")
    monkeypatch.setattr("app.api.main.configure_torch_runtime", lambda *a, **kw: None)
    monkeypatch.setattr("app.api.main.ChessNet", lambda _cfg: _MockModel())
    monkeypatch.setenv("ALLOW_RANDOM_WEIGHTS", "1")

    model, device = _load_model()
    assert device == "cpu"
    assert model is not None


def test_load_model_allow_random_weights_variants(monkeypatch, tmp_path):
    """Test ALLOW_RANDOM_WEIGHTS accepted values: 1, true, yes."""
    missing = tmp_path / "nonexistent.pt"

    cfg = _MockConfig()
    cfg.system.checkpoint_path = str(missing)

    monkeypatch.setattr("app.api.main.load_config", lambda: cfg)
    monkeypatch.setattr("app.api.main.validate_config", lambda _: None)
    monkeypatch.setattr("app.api.main.select_device", lambda _: "cpu")
    monkeypatch.setattr("app.api.main.configure_torch_runtime", lambda *a, **kw: None)
    monkeypatch.setattr("app.api.main.ChessNet", lambda _cfg: _MockModel())

    for env_val in ("true", "yes"):
        monkeypatch.setenv("ALLOW_RANDOM_WEIGHTS", env_val)
        model, device = _load_model()
        assert device == "cpu"
        assert model is not None

    monkeypatch.setenv("ALLOW_RANDOM_WEIGHTS", "0")
    with pytest.raises(RuntimeError, match="Checkpoint not found"):
        _load_model()
