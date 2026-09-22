from dataclasses import replace
from unittest.mock import patch

import chess
import numpy as np
import pytest

from app.game.move_encoding import NUM_MOVES
from app.game.repetition import build_seen_positions, position_key
from app.infra.config import AppConfig
from app.mcts.search import MCTS
from app.mcts.node import Node
from app.game.tactics import find_mate_in_one, has_forcing_mate_in_two, select_safe_move


def test_even_exchange_is_not_a_hanging_piece():
    search = MCTS(None, AppConfig())
    board = chess.Board("4k3/8/8/8/2p5/8/3PP3/4K3 w - - 0 1")
    assert search._piece_tactical_penalty(board, chess.Move.from_uci("d2d3")) == 0.0


def test_principle_computation_is_reused_but_repetition_is_not():
    search = MCTS(None, AppConfig())
    board = chess.Board()
    move = chess.Move.from_uci("g1f3")
    seen = build_seen_positions(board)
    after = board.copy()
    after.push(move)
    repeated = {**seen, position_key(after): 2}
    from app.game.principles import principle_penalty_components
    with patch("app.mcts.search.principle_penalty_components", wraps=principle_penalty_components) as evaluate:
        first = search._move_penalty_components(board, move, seen)
        second = search._move_penalty_components(board, move, repeated)
    assert evaluate.call_count == 1
    assert second["repetition"] > first["repetition"]


def test_terminal_search_does_not_invent_a_move_or_evaluate_network():
    board = chess.Board()
    for move in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
        board.push_uci(move)
    with patch("app.mcts.search.predict_boards") as predict:
        result = MCTS(None, AppConfig()).search(board, num_simulations=4)
    assert result["best_move"] is None
    assert result["root_value"] == 0.0
    predict.assert_not_called()


def test_root_guard_rejects_reported_forced_mate_and_restores_history():
    board = chess.Board("8/5pk1/3p1n2/3Br3/5Qp1/6K1/8/7R b - - 5 36")
    original = board.fen(), list(board.move_stack)
    losing = chess.Move.from_uci('e5d5')
    escape = chess.Move.from_uci('g7g8')
    move, rejected = select_safe_move(board, [losing, escape])
    assert move == escape
    assert losing in rejected
    assert (board.fen(), board.move_stack) == original


def test_root_guard_keeps_checkmate_and_draw():
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")
    mate = chess.Move.from_uci('d8h4')
    assert select_safe_move(board, [mate])[0] == mate
    draw = chess.Board('7k/8/6K1/8/8/8/8/8 w - - 0 1')
    assert not has_forcing_mate_in_two(draw)


def test_expired_tactical_check_does_not_mutate_board():
    board = chess.Board()
    move = chess.Move.from_uci('e2e4')
    original = board.fen()
    assert select_safe_move(board, [move], deadline=0)[0] == move
    assert board.fen() == original
    assert board.move_stack == []


def test_mate_in_one_bypasses_network():
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")
    with patch('app.mcts.search.predict_boards') as predict:
        result = MCTS(None, AppConfig()).search(board, num_simulations=256)
    assert result['best_move'].uci() == 'd8h4'
    assert result['root_value'] == 1
    predict.assert_not_called()


def test_search_deadline_returns_available_root_policy():
    board = chess.Board()
    search = MCTS(None, AppConfig())
    clock = [0.0]
    def root_only(node, board, add_noise):
        node.children = {chess.Move.from_uci('e2e4'): Node(0.9), chess.Move.from_uci('d2d4'): Node(0.1)}
        clock[0] = 2.0
        return 0.25
    # The deadline expires during root evaluation; no detached rollouts follow.
    with patch.object(search, '_expand_node', side_effect=root_only), patch('app.mcts.search.time.monotonic', side_effect=lambda: clock[0]):
        result = search.search(board, num_simulations=256, time_limit_sec=1)
    assert result['best_move'].uci() == 'e2e4'
    assert sum(result['visit_counts'].values()) == 0
    assert result['root_value'] == 0.25


def test_engine_cache_distinguishes_same_fen_with_different_history():
    from app.core.engine import Engine
    board = chess.Board()
    for uci in ['g1f3', 'g8f6', 'f3g1', 'f6g8']:
        board.push_uci(uci)
    reconstructed = chess.Board(board.fen())
    assert Engine._cache_key(board, 32, 0.05) != Engine._cache_key(reconstructed, 32, 0.05)


def test_checkpoint_does_not_override_explicit_search_settings(tmp_path):
    import torch
    from app.core.engine import Engine
    from app.infra.config import ModelConfig
    from app.model.network import ChessNet
    from app.model.checkpoint import save_checkpoint
    old = replace(AppConfig(), model=ModelConfig(channels=8, res_blocks=1))
    path = tmp_path / 'model.pth'
    save_checkpoint(path, ChessNet(old), old)
    requested = replace(old, mcts=replace(old.mcts, num_simulations=73, c_puct=2.0))
    engine = Engine(model_path=str(path), cfg=requested, device='cpu')
    assert engine.cfg == requested
    assert engine.mcts.c_puct == 2.0


def test_root_visit_tie_prefers_value_for_mover_not_opponent():
    board = chess.Board()
    good, bad = chess.Move.from_uci('e2e4'), chess.Move.from_uci('d2d4')
    root = Node(0)
    root.children = {good: Node(0.5), bad: Node(0.5)}
    root.children[good].visit_count = root.children[bad].visit_count = 1
    root.children[good].value_sum = -0.8
    root.children[bad].value_sum = 0.8
    search = MCTS(None, AppConfig())
    with patch.object(search, '_move_penalty', return_value=0):
        assert search._select_root_move(board, root, {good: 0.5, bad: 0.5}) == good


def test_live_game_cannot_ignore_hanging_queen_by_moving_a_rook():
    from app.infra.config import load_config
    search = MCTS(None, load_config('config/default.yaml'))
    board = chess.Board('r1b2rk1/pp3ppp/3q2n1/3pp3/P7/1P1BR2Q/5PPP/R5K1 w - - 0 19')
    ignoring = search._move_penalty_components(board, chess.Move.from_uci('a1e1'))
    saving = search._move_penalty_components(board, chess.Move.from_uci('h3g3'))
    assert ignoring['tactical'] >= 0.6
    assert saving['tactical'] < ignoring['tactical']


def test_live_game_castling_cannot_abandon_knight_when_it_can_be_defended():
    from app.game.tactics import immediate_exchange_loss
    board = chess.Board('r1b2rk1/pp2nppp/4p3/3pP3/1q6/1P1B1Q2/P2N1PPP/R3K2R w KQ - 3 13')
    castle, defend = chess.Move.from_uci('e1g1'), chess.Move.from_uci('a1d1')
    assert immediate_exchange_loss(board, castle) == 320
    assert immediate_exchange_loss(board, defend) == 0
    assert select_safe_move(board, [castle, defend])[0] == defend


def test_root_exchange_screen_credits_captures_and_legal_recaptures():
    from app.game.tactics import immediate_exchange_loss
    equal = chess.Board('4k3/8/8/8/2p5/8/3PP3/4K3 w - - 0 1')
    assert immediate_exchange_loss(equal, chess.Move.from_uci('d2d3')) == 0
    win = chess.Board('4k3/8/8/8/3q4/8/3R4/4K3 w - - 0 1')
    assert immediate_exchange_loss(win, chess.Move.from_uci('d2d4')) == 0


def test_position_penalty_includes_en_passant_capture():
    board = chess.Board('4k3/8/8/8/3p4/8/4P3/4K3 w - - 0 1')
    search = MCTS(None, AppConfig())
    assert search._move_penalty_components(board, chess.Move.from_uci('e2e4'))['tactical'] > 0


def test_all_material_losing_candidates_keep_the_least_costly_option():
    board = chess.Board()
    moves = [chess.Move.from_uci(uci) for uci in ['e2e4', 'd2d4', 'c2c4']]
    with patch('app.game.tactics.has_forcing_mate_in_two', return_value=False), patch(
        'app.game.tactics.immediate_exchange_loss', side_effect=[900, 320, 500]
    ):
        chosen, rejected = select_safe_move(board, moves)
    assert chosen == moves[1]
    assert chosen not in rejected
    assert board.move_stack == []
