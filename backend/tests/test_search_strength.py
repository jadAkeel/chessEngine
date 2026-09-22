"""Regressions for the search-strength repairs: virtual loss, FPU, penalty caching,
terminal detection, tree reuse, root repetition filtering, cumulative ladders and
clock-based time management."""
from unittest.mock import patch

import chess
import numpy as np
import pytest

from app.game.move_encoding import NUM_MOVES, move_to_index
from app.game.repetition import build_seen_positions
from app.infra.config import AppConfig, MCTSConfig, ModelConfig, PrinciplePenaltiesConfig
from app.mcts.node import Node
from app.mcts.search import MCTS


def _cfg(**mcts_overrides) -> AppConfig:
    return AppConfig(
        model=ModelConfig(input_planes=20, channels=8, res_blocks=1, value_dropout=0.0),
        mcts=MCTSConfig(num_simulations=8, inference_batch_size=16, classical_value_alpha=0.0, **mcts_overrides),
    )


def _fake_predict(favoured: dict[str, str]):
    """Network stub: strongly prefer ``favoured[fen_without_clocks]`` when it is legal, value 0."""

    def predict(model, boards, cfg=None, device=None):
        boards = list(boards)
        logits = np.zeros((len(boards), NUM_MOVES), dtype=np.float32)
        for row, board in enumerate(boards):
            uci = favoured.get(" ".join(board.fen().split()[:4]))
            if uci:
                move = chess.Move.from_uci(uci)
                if move in board.legal_moves:
                    logits[row, move_to_index(move, board)] = 8.0
        return logits, np.zeros((len(boards),), dtype=np.float32)

    return predict


# --------------------------------------------------------------------------- selection


def test_virtual_loss_is_diluted_by_real_visits_instead_of_flattening_the_root():
    """One pending rollout must not erase a well-visited child's advantage (the old
    ``score - 1.0`` penalty forced every batch to fan out over distinct children)."""
    search = MCTS(None, _cfg())
    root = Node(0.0)
    root.visit_count = 40
    good, other = chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")
    root.children = {good: Node(0.5, root), other: Node(0.5, root)}
    root.children[good].visit_count = root.children[other].visit_count = 20
    root.children[good].value_sum = -10.0  # +0.5 for the mover
    root.children[other].value_sum = 0.0
    root.children[good].add_virtual_visit(1)
    with patch.object(search, "_ensure_child_penalties"):
        assert search._select_child(root, chess.Board())[0] == good


def test_unvisited_children_inherit_parent_value_minus_fpu_reduction():
    """In a losing parent an unvisited child no longer looks like a neutral 0.0 and
    outranks a child whose value is already known to be better than the parent."""
    search = MCTS(None, _cfg(fpu_reduction=0.25))
    root = Node(0.0)
    root.visit_count, root.value_sum = 10, -6.0  # parent q = -0.6
    known, fresh = chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")
    root.children = {known: Node(0.05, root), fresh: Node(0.05, root)}
    root.children[known].visit_count, root.children[known].value_sum = 5, 2.5  # -0.5 for the mover
    with patch.object(search, "_ensure_child_penalties"):
        assert search._select_child(root, chess.Board())[0] == known


def test_edge_penalties_are_computed_once_per_node_and_cached_on_children():
    search = MCTS(None, _cfg())
    board = chess.Board()
    root = Node(0.0)
    root.children = {move: Node(1.0 / 20, root) for move in board.legal_moves}
    seen = build_seen_positions(board)
    with patch.object(search, "_move_penalty_components", wraps=search._move_penalty_components) as compute:
        search._select_child(root, board, seen)
        search._select_child(root, board, seen)
        search._select_child(root, board, seen)
    assert compute.call_count == len(root.children)
    assert all(child.penalty_components is not None for child in root.children.values())
    assert root.penalties_ready


def test_penalties_are_recomputed_when_a_reused_node_surfaces_to_the_root():
    cfg = AppConfig(
        model=ModelConfig(input_planes=20, channels=8, res_blocks=1, value_dropout=0.0),
        mcts=MCTSConfig(num_simulations=8, inference_batch_size=16, classical_value_alpha=0.0),
        principle_penalties=PrinciplePenaltiesConfig(enabled=True),
    )
    search = MCTS(None, cfg)
    board = chess.Board("rnb1k1nr/4q1b1/1ppppp2/7p/P1PPP1PN/2NBB1P1/P6P/R2QR1K1 b - - 0 16")
    node = Node(0.0)
    node.children = {move: Node(0.05, node) for move in board.legal_moves}
    seen = build_seen_positions(board)
    search._ensure_child_penalties(node, board, seen, depth=5)   # deep: tactical only
    deep = node.children[chess.Move.from_uci("e8f8")].penalty_components
    assert node.penalty_tier == 2
    assert not any(name.startswith("principle.") for name in deep)
    search._ensure_child_penalties(node, board, seen, depth=5)   # same tier: no recompute
    search._ensure_child_penalties(node, board, seen, depth=0)   # surfaced to root: recompute
    root_level = node.children[chess.Move.from_uci("e8f8")].penalty_components
    assert node.penalty_tier == 0
    assert root_level["principle.king_safety"] > 0.0
    assert root_level["principle.tactics"] > 0.0
    search._ensure_child_penalties(node, board, seen, depth=3)   # deeper again: keep the richer set
    assert node.penalty_tier == 0


# --------------------------------------------------------------------------- terminal handling


def test_only_an_actual_third_occurrence_is_a_terminal_draw():
    """``is_game_over(claim_draw=True)`` also fires when the mover merely *could* repeat;
    that position is not a draw for the search and must still get a move."""
    board = chess.Board()
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"]:
        board.push_uci(uci)
    search = MCTS(None, _cfg())
    assert board.can_claim_threefold_repetition()
    assert not search._is_terminal_board(board)
    board.push_uci("f6g8")
    assert search._is_terminal_board(board)
    assert search._terminal_value(board) == 0.0


def test_incremental_repetition_map_matches_full_board_check():
    board = chess.Board()
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"]:
        board.push_uci(uci)
    search = MCTS(None, _cfg())
    seen = build_seen_positions(board)
    assert search._terminal_state(board, seen) == 0.0
    assert search._terminal_state(chess.Board(), build_seen_positions(chess.Board())) is None
    mate = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert search._terminal_state(mate, {}) == -1.0


def test_stagnation_shrinks_leaf_value_toward_draw_for_both_sides():
    search = MCTS(None, _cfg())
    fresh = chess.Board()
    stale = chess.Board(chess.STARTING_FEN.replace(" 0 1", " 85 60"))
    assert search._blend_value(fresh, 0.8) == pytest.approx(0.8)
    assert 0.0 < search._blend_value(stale, 0.8) < 0.8
    assert search._blend_value(stale, -0.8) == pytest.approx(-search._blend_value(stale, 0.8))


# --------------------------------------------------------------------------- tree reuse


def test_same_position_search_continues_the_retained_tree():
    search = MCTS(None, _cfg())
    board = chess.Board()
    with patch("app.mcts.search.predict_boards", side_effect=_fake_predict({})):
        first = search.search(board, num_simulations=8)
        assert first["retained_visits"] == 0
        assert search.retained_visits(board) == 8
        second = search.search(board, num_simulations=8)
    assert second["retained_visits"] == 8
    assert sum(second["visit_counts"].values()) == 16


def test_tree_is_reused_after_the_played_moves_and_dropped_otherwise():
    search = MCTS(None, _cfg())
    start = chess.Board()
    after_e4 = chess.Board()
    after_e4.push_uci("e2e4")
    favoured = {
        " ".join(start.fen().split()[:4]): "e2e4",
        " ".join(after_e4.fen().split()[:4]): "e7e5",
    }
    with patch("app.mcts.search.predict_boards", side_effect=_fake_predict(favoured)):
        search.search(start, num_simulations=48)
        continued = chess.Board()
        continued.push_uci("e2e4")
        continued.push_uci("e7e5")
        retained = search.retained_visits(continued)
        assert retained > 0
        result = search.search(continued, num_simulations=8)
        assert result["retained_visits"] == retained
        assert result["best_move"] in continued.legal_moves

        elsewhere = chess.Board()
        elsewhere.push_uci("d2d4")
        elsewhere.push_uci("d7d5")
        # Same ply count but a different history: nothing may be reused.
        assert search.retained_visits(elsewhere) == 0
        assert search.search(elsewhere, num_simulations=4)["retained_visits"] == 0


def test_tree_reuse_can_be_disabled():
    search = MCTS(None, _cfg(reuse_tree=False))
    board = chess.Board()
    with patch("app.mcts.search.predict_boards", side_effect=_fake_predict({})):
        search.search(board, num_simulations=8)
        assert search.retained_visits(board) == 0
        assert search.search(board, num_simulations=8)["retained_visits"] == 0


def test_reused_root_gets_fresh_dirichlet_noise_in_selfplay_mode():
    search = MCTS(None, _cfg(dirichlet_eps=0.5, dirichlet_alpha=0.3))
    board = chess.Board()
    with patch("app.mcts.search.predict_boards", side_effect=_fake_predict({})):
        search.search(board, num_simulations=4, add_noise=False)
        before = {move: child.prior for move, child in search._root.children.items()}
        search.search(board, num_simulations=4, add_noise=True)
        after = {move: child.prior for move, child in search._root.children.items()}
    assert before != after
    assert sum(after.values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- root policy


def test_root_repetition_filter_is_skipped_when_the_mover_is_worse_in_play_mode():
    search = MCTS(None, _cfg())
    board = chess.Board()
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"]:
        board.push_uci(uci)
    seen = build_seen_positions(board)
    repeat, other = chess.Move.from_uci("g1f3"), chess.Move.from_uci("e2e4")
    policy = {repeat: 0.7, other: 0.3}
    with patch.object(search, "_root_edge_penalty", return_value=0.0):
        losing, _ = search._adjust_root_policy(board, policy, seen, root_value=-0.5, training_mode=False)
        winning, _ = search._adjust_root_policy(board, policy, seen, root_value=0.3, training_mode=False)
        selfplay, _ = search._adjust_root_policy(board, policy, seen, root_value=-0.5, training_mode=True)
    assert losing[repeat] == pytest.approx(0.7)
    assert winning[repeat] < winning[other]
    assert selfplay[repeat] < selfplay[other]


def test_oscillation_penalty_targets_the_movers_own_previous_move_and_spares_checks():
    """The old check compared against the opponent's last move and could never fire."""
    search = MCTS(None, _cfg())
    board = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    board.push_uci("a1a7")
    board.push_uci("e8d8")
    quiet_return = chess.Move.from_uci("a7a1")
    assert search._oscillation_penalty(board, quiet_return) > 0.0
    assert search._oscillation_penalty(board, chess.Move.from_uci("a7b7")) == 0.0
    checking = chess.Board("2k5/3R4/8/8/8/8/8/4K3 w - - 0 1")
    checking.push_uci("d7a7")
    checking.push_uci("c8d8")
    returning_with_check = chess.Move.from_uci("a7d7")
    assert checking.gives_check(returning_with_check)
    assert search._oscillation_penalty(checking, returning_with_check) == 0.0


# --------------------------------------------------------------------------- API / bot integration


def test_api_search_is_cumulative_across_ladder_rungs():
    from app.api import main as api

    class _FakeMCTS:
        def __init__(self):
            self.requested = []

        def retained_visits(self, board):
            return 30

        def search(self, board, num_simulations, **kwargs):
            self.requested.append((num_simulations, kwargs.get("time_limit_sec")))
            move = chess.Move.from_uci("e2e4")
            return {"best_move": move, "visit_counts": {move: 64}, "root_diagnostics": [], "completed_simulations": num_simulations}

    fake = _FakeMCTS()
    with patch.object(api, "_get_mcts", return_value=fake):
        api._best_move_with_mcts(object(), chess.Board(), "cpu", 64, cumulative=True, time_limit_sec=2.0)
        api._best_move_with_mcts(object(), chess.Board(), "cpu", 64)
    assert fake.requested == [(34, 2.0), (64, None)]


def test_bot_spends_its_clock_on_quiet_positions():
    from app.cli import lichess_bot as bot_module
    from app.cli.lichess_bot import BotConfig, LichessBot

    class _Model:
        def predict(self, board, device=None):
            return object(), 0.0

    class _Engine:
        model = _Model()
        device = "cpu"

    candidates = [
        {"uci": "e2e4", "score": 1000.0, "prob": 0.8},
        {"uci": "d2d4", "score": 700.0, "prob": 0.2},
    ]
    bot = LichessBot(bot_cfg=BotConfig(token="test"), engine=_Engine())
    with (
        patch.object(bot_module, "_legal_moves_with_probs", return_value=candidates),
        patch.object(bot_module, "_fast_policy_move", return_value=("e2e4", "e4", candidates)),
        patch.object(bot_module, "_fastmove_complexity", return_value=(0, [])),
        patch.object(bot_module, "_is_decisive_fast_choice", return_value=False),
        patch.object(bot_module, "_should_use_adaptive_search", return_value=False),
        patch.object(bot_module, "_is_light_adaptive_search", return_value=False),
        patch.object(
            bot_module, "_best_move_with_mcts",
            return_value=("d2d4", "d4", [{"uci": "d2d4", "visits": 128}]),
        ) as mock_mcts,
    ):
        move = bot._compute_move_sync(chess.Board(), 128, 5.0, "normal")
    assert move == "d2d4"
    assert mock_mcts.call_count >= 1
    last = mock_mcts.call_args_list[-1]
    assert last.args[3] == 128
    assert last.kwargs["cumulative"] is True
    assert 0.0 < last.kwargs["time_limit_sec"] <= 4.5


def test_bot_plays_policy_move_only_when_the_clock_is_nearly_gone():
    from app.cli import lichess_bot as bot_module
    from app.cli.lichess_bot import BotConfig, LichessBot

    class _Model:
        def predict(self, board, device=None):
            return object(), 0.0

    class _Engine:
        model = _Model()
        device = "cpu"

    candidates = [{"uci": "e2e4", "score": 1000.0, "prob": 0.9}, {"uci": "d2d4", "score": 500.0, "prob": 0.1}]
    bot = LichessBot(bot_cfg=BotConfig(token="test"), engine=_Engine())
    with (
        patch.object(bot_module, "_legal_moves_with_probs", return_value=candidates),
        patch.object(bot_module, "_fast_policy_move", return_value=("e2e4", "e4", candidates)),
        patch.object(bot_module, "_fastmove_complexity", return_value=(0, [])),
        patch.object(bot_module, "_is_decisive_fast_choice", return_value=False),
        patch.object(bot_module, "_should_use_adaptive_search", return_value=False),
        patch.object(bot_module, "_is_light_adaptive_search", return_value=False),
        patch.object(bot_module, "_best_move_with_mcts") as mock_mcts,
    ):
        assert bot._compute_move_sync(chess.Board(), 16, 0.2, "normal") == "e2e4"
    mock_mcts.assert_not_called()


def test_clock_allocation_spreads_time_and_lets_the_deadline_govern_simulations():
    from app.cli.lichess_bot import calculate_dynamic_thinking

    board = chess.Board()
    sims, allocated, urgency = calculate_dynamic_thinking(board, 180.0, 2.0, max_sims=256)
    assert urgency == "normal"
    assert sims == 256
    assert allocated == pytest.approx(180.0 / 49 + 1.6, rel=0.01)

    late = chess.Board()
    late.fullmove_number = 60
    _, late_alloc, _ = calculate_dynamic_thinking(late, 40.0, 0.0, max_sims=256)
    assert late_alloc == pytest.approx(2.0, rel=0.01)  # never below a 20-move horizon

    sims_low, alloc_low, _ = calculate_dynamic_thinking(board, 4.0, 0.0, max_sims=256)
    assert sims_low == 16
    assert alloc_low <= 0.6

    only = chess.Board("k7/8/1K6/8/8/8/8/7R b - - 0 1")
    assert len(list(only.legal_moves)) == 1
    assert calculate_dynamic_thinking(only, 100.0, 0.0)[2] == "forced_move"
