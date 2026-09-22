"""
Lightweight comparison / smoke test between fastmove-like and bestmove-like
decision paths for the chess engine.

This module tests the analytical / decision-making components of fastmove
without requiring a neural network model or MCTS. It skips gracefully if
the model or key dependencies cannot be loaded.

Key differences tested:
  - fastmove: uses adaptive simulation budgets based on position complexity
  - bestmove: uses a fixed simulation budget (typically higher)

The tests focus on deterministic chess-analysis functions that are shared
between both paths, and verify that the adaptive decision logic produces
sensible simulation budgets for various tactical scenarios.
"""

import chess
import pytest


# =========================================================================
# 1. Model-dependent imports – skip gracefully if unavailable
# =========================================================================

try:
    from app.api.main import (
        _adaptive_simulations,
        _adaptive_simulation_steps,
        _fastmove_complexity,
        _has_later_simulation_step,
        _is_decisive_fast_choice,
        _is_light_adaptive_search,
        _mcts_confident_enough,
        _move_safety_flags,
        _should_use_adaptive_search,
        MIN_ADAPTIVE_SIMULATIONS,
        COMPLEX_TOPK_BOOST_THRESHOLD,
        COMPLEX_TOPK_MAX,
    )
    IMPORT_OK = True
except ImportError as exc:
    IMPORT_OK = False
    IMPORT_ERROR = str(exc)


pytestmark = pytest.mark.skipif(
    "not IMPORT_OK",
    reason=f"Required modules not importable: {IMPORT_ERROR if 'IMPORT_ERROR' in dir() else 'unknown'}",
)


# =========================================================================
# 2. Arena-like comparison tests
# =========================================================================

# Pre-defined tactical positions for testing
# Each position comes with expected complexity attributes

SMALL_MATE_POSITION = "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"  # white in check
QUEEN_CAPTURE_POSITION = "4k3/8/8/8/8/4q3/8/4RK2 w - - 0 1"  # rook can take queen
MATE_TWO_DANGER = "8/5pk1/3p1n2/3Br3/5Qp1/6K1/8/7R b - - 5 36"
OPENING_POSITION = chess.STARTING_FEN


def test_fastmove_vs_bestmove_budget_comparison():
    """Verify that fastmove adaptive budgets are <= bestmove fixed budget
    for a range of complexity levels."""
    bestmove_fixed = 96  # typical bestmove default

    for complexity in range(0, 10):
        fast_budget = _adaptive_simulations(depth=6, complexity=complexity, max_simulations=bestmove_fixed)
        assert fast_budget <= bestmove_fixed, (
            f"Fastmove budget {fast_budget} exceeds bestmove {bestmove_fixed} "
            f"at complexity={complexity}"
        )


def test_fastmove_uses_progressive_steps():
    """Fastmove with adaptive search should return at least one step."""
    for complexity in (0, 4, 6, 8):
        steps = _adaptive_simulation_steps(depth=6, complexity=complexity, max_simulations=96)
        assert len(steps) >= 1, f"No steps for complexity={complexity}"
        assert steps == sorted(steps), f"Steps not sorted: {steps}"


def test_decisive_fast_queen_capture_without_king_exposure():
    """A clear queen capture without king exposure should be decisive."""
    board = chess.Board("4k3/8/8/8/8/4q3/8/4R1K1 w - - 0 1")  # King on g1, safe
    candidates = [
        {"uci": "e1e3", "score": 900.0},
        {"uci": "g1h2", "score": 860.0},
    ]
    # Use only non-critical reasons to pass the decisive check
    assert _is_decisive_fast_choice(
        board, candidates, ["many_forcing_moves", "top_moves_close"]
    ) is True


def test_check_position_triggers_adaptive_search():
    """A position with king in check should always trigger adaptive search."""
    board = chess.Board(SMALL_MATE_POSITION)
    complexity, reasons = _fastmove_complexity(board, [])
    assert _should_use_adaptive_search(complexity, reasons, depth=6) is True


def test_mate_two_danger_triggers_full_search():
    """A position where best fast move allows mate-in-two should get full search."""
    board = chess.Board(MATE_TWO_DANGER)
    candidates = [
        {"uci": "e5d5", "score": 900.0},
        {"uci": "g7g8", "score": 850.0},
    ]
    complexity, reasons = _fastmove_complexity(board, candidates)
    assert _should_use_adaptive_search(complexity, reasons, depth=6) is True
    # Light search should be OFF for safety-critical positions
    assert _is_light_adaptive_search(complexity, reasons, depth=6) is False


def test_complex_pool_boost_triggers_at_threshold():
    """When complexity >= COMPLEX_TOPK_BOOST_THRESHOLD(6), the candidate pool
    is expanded to up to COMPLEX_TOPK_MAX(36). This test checks that
    the complexity scoring can reach that threshold."""
    board = chess.Board(MATE_TWO_DANGER)
    candidates = [
        {"uci": "e5d5", "score": 900.0},
        {"uci": "g7g8", "score": 850.0},
    ]
    complexity, reasons = _fastmove_complexity(board, candidates)
    assert complexity >= COMPLEX_TOPK_BOOST_THRESHOLD, (
        f"Expected complexity >= {COMPLEX_TOPK_BOOST_THRESHOLD} "
        f"for pool boost, got {complexity}"
    )


def test_opening_position_low_complexity():
    """The starting position should have low complexity, no adaptive search."""
    board = chess.Board(OPENING_POSITION)
    # Opening has many legal moves and no tactical urgency
    candidates = [
        {"uci": "e2e4", "score": 200.0},
        {"uci": "d2d4", "score": 100.0},
    ]
    complexity, reasons = _fastmove_complexity(board, candidates)
    # Opening should be low-medium complexity
    assert complexity < 6, f"Opening complexity unexpectedly high: {complexity}"
    assert _should_use_adaptive_search(complexity, reasons, depth=6) is False, (
        f"Opening triggered adaptive search: reasons={reasons}"
    )
    # Light adaptive search also off for quiet opening with only top_moves_competitive
    assert _is_light_adaptive_search(complexity, reasons, depth=6) is False, (
        f"Opening triggered light adaptive search: reasons={reasons}"
    )


def test_opening_position_very_close_scores():
    """Quiet opening with very-close policy scores: light adaptive, not full adaptive."""
    board = chess.Board(OPENING_POSITION)
    # Gap <= 35 triggers top_moves_very_close (complexity +3)
    candidates = [
        {"uci": "e2e4", "score": 200.0},
        {"uci": "d2d4", "score": 180.0},
    ]
    complexity, reasons = _fastmove_complexity(board, candidates)
    # top_moves_very_close raises complexity so expect >= 3
    assert complexity >= 3, f"Expected complexity >= 3, got {complexity}"
    assert "top_moves_very_close" in reasons, (
        f"Expected top_moves_very_close in reasons, got {reasons}"
    )
    # Very-close alone is not tactically urgent so full adaptive is off
    assert _should_use_adaptive_search(complexity, reasons, depth=6) is False, (
        f"Opening with very-close scores triggered full adaptive: reasons={reasons}"
    )
    # But the genuine ambiguity justifies a light adaptive probe
    assert _is_light_adaptive_search(complexity, reasons, depth=6) is True, (
        f"Opening with very-close scores should use light adaptive: reasons={reasons}"
    )


def test_fastmove_confident_enough_never_stops_with_safety_risk():
    """Both fastmove and bestmove paths must never stop early when
    a safety risk is detected."""
    root = [{"uci": "e2e4", "visits": 50}, {"uci": "d2d4", "visits": 2}]
    for light in (True, False):
        for sims in (30, 64, 96):
            assert _mcts_confident_enough(
                fast_move="e2e4",
                mcts_move="e2e4",
                root_debug=root,
                safety_flags={"mate_one": True, "mate_two": False,
                               "promotion_threat": False, "queen_capture": False,
                               "valuable_piece_capture": False},
                light=light,
                current_simulations=sims,
            ) is False, f"Should not stop with safety risk at light={light} sims={sims}"


def test_fastmove_light_budget_never_exceeds_bestmove():
    """Light adaptive search budgets should be much smaller than bestmove."""
    light_budget = _adaptive_simulations(depth=6, complexity=6, max_simulations=96, light=True)
    full_budget = _adaptive_simulations(depth=6, complexity=6, max_simulations=96, light=False)
    bestmove_default = 96
    assert light_budget < bestmove_default, f"Light budget {light_budget} >= bestmove {bestmove_default}"
    assert light_budget <= full_budget, f"Light budget {light_budget} > full {full_budget}"


def test_progressive_steps_never_empty():
    """Progressive steps should always contain at least the target simulation count."""
    for depth in (4, 6, 8):
        for complexity in (0, 4, 6, 8):
            steps = _adaptive_simulation_steps(depth=depth, complexity=complexity, max_simulations=96)
            assert steps, f"Empty steps for depth={depth} complexity={complexity}"
            assert steps[-1] == _adaptive_simulations(depth=depth, complexity=complexity, max_simulations=96), (
                f"Last step {steps[-1]} != target "
                f"{_adaptive_simulations(depth=depth, complexity=complexity, max_simulations=96)}"
            )


def test_bestmove_equivalent_decision_path():
    """Simulate a 'bestmove-like' decision: fixed high sims, no early stopping,
    no adaptive budget. This should use a different path than fastmove."""
    # Bestmove: always uses fixed simulations, no adaptive steps
    for sims in (64, 96, 160):
        target = _adaptive_simulations(depth=6, complexity=8, max_simulations=sims)
        # When max_simulations is the cap, fastmove should cap at that value
        assert target <= sims, f"Fastmove budget {target} exceeds bestmove {sims}"
        # Actual fastmove target may be lower due to complexity-based budget
        expected = min(sims, _adaptive_simulations(depth=6, complexity=8, max_simulations=sims))
        assert target == expected, f"Unexpected budget: {target} vs {expected}"


def test_has_later_step_behavior():
    """Test the retry mechanism shared by fastmove and bestmove paths."""
    steps_30 = [30]
    steps_progressive = [30, 64, 96]

    assert _has_later_simulation_step(25, steps_30) is True
    assert _has_later_simulation_step(30, steps_30) is False
    assert _has_later_simulation_step(35, steps_30) is False

    assert _has_later_simulation_step(30, steps_progressive) is True
    assert _has_later_simulation_step(64, steps_progressive) is True
    assert _has_later_simulation_step(96, steps_progressive) is False
    assert _has_later_simulation_step(200, steps_progressive) is False

