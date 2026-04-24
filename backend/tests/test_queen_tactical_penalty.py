import unittest

import chess

from app.infra.config import AppConfig, ArenaConfig, MCTSConfig, ModelConfig, ReplayConfig, SelfPlayConfig, SystemConfig, TrainingConfig
from app.mcts.node import Node
from app.mcts.search import MCTS


class QueenTacticalPenaltyTests(unittest.TestCase):
    def _cfg(self, **mcts_overrides) -> AppConfig:
        return AppConfig(
            model=ModelConfig(input_planes=20, channels=32, res_blocks=1, value_dropout=0.0),
            training=TrainingConfig(),
            replay=ReplayConfig(),
            mcts=MCTSConfig(
                num_simulations=2,
                inference_batch_size=1,
                classical_value_alpha=0.0,
                queen_blunder_penalty=0.60,
                queen_hanging_penalty=0.30,
                queen_sac_compensation_threshold=500,
                queen_check_discount=0.75,
                **mcts_overrides,
            ),
            selfplay=SelfPlayConfig(max_game_length=20),
            arena=ArenaConfig(games=2),
            system=SystemConfig(device="cpu", checkpoint_path="models/best_model.pth", default_bestmove_simulations=2),
        )

    def test_penalizes_hanging_queen_move(self):
        cfg = self._cfg()
        mcts = MCTS(model=None, cfg=cfg, device="cpu")
        board = chess.Board("3rk3/8/8/8/8/8/8/3QK3 w - - 0 1")

        hanging_move = chess.Move.from_uci("d1d7")
        safe_move = chess.Move.from_uci("d1e2")

        self.assertIn(hanging_move, board.legal_moves)
        self.assertIn(safe_move, board.legal_moves)

        hanging_penalty = mcts._queen_tactical_penalty(board, hanging_move)
        safe_penalty = mcts._queen_tactical_penalty(board, safe_move)

        self.assertGreater(hanging_penalty, 0.05)
        self.assertLess(safe_penalty, hanging_penalty)

    def test_select_child_prefers_safe_move_when_scores_equal(self):
        cfg = self._cfg()
        mcts = MCTS(model=None, cfg=cfg, device="cpu")
        board = chess.Board("3rk3/8/8/8/8/8/8/3QK3 w - - 0 1")

        hanging_move = chess.Move.from_uci("d1d7")
        safe_move = chess.Move.from_uci("d1e2")

        root = Node(prior=0.0)
        root.visit_count = 4
        root.children[hanging_move] = Node(prior=0.5, parent=root)
        root.children[safe_move] = Node(prior=0.5, parent=root)

        selected_move, _ = mcts._select_child(root, board)
        self.assertEqual(selected_move, safe_move)

    def test_forcing_check_reduces_penalty(self):
        cfg = self._cfg()
        mcts = MCTS(model=None, cfg=cfg, device="cpu")
        board = chess.Board("3rk3/8/8/8/8/8/8/3QK3 w - - 0 1")

        quiet_hanging = chess.Move.from_uci("d1d7")
        checking_hanging = chess.Move.from_uci("d1h5")

        self.assertIn(checking_hanging, board.legal_moves)
        quiet_penalty = mcts._queen_tactical_penalty(board, quiet_hanging)
        check_penalty = mcts._queen_tactical_penalty(board, checking_hanging)

        self.assertGreater(quiet_penalty, check_penalty)


class GeneralPieceTacticalPenaltyTests(unittest.TestCase):
    def _cfg(self, **mcts_overrides) -> AppConfig:
        return AppConfig(
            model=ModelConfig(input_planes=20, channels=32, res_blocks=1, value_dropout=0.0),
            training=TrainingConfig(),
            replay=ReplayConfig(),
            mcts=MCTSConfig(
                num_simulations=2,
                inference_batch_size=1,
                classical_value_alpha=0.0,
                queen_blunder_penalty=0.60,
                queen_hanging_penalty=0.30,
                queen_sac_compensation_threshold=500,
                queen_check_discount=0.75,
                **mcts_overrides,
            ),
            selfplay=SelfPlayConfig(max_game_length=20),
            arena=ArenaConfig(games=2),
            system=SystemConfig(device="cpu", checkpoint_path="models/best_model.pth", default_bestmove_simulations=2),
        )

    def test_general_piece_penalty_catches_hanging_moves(self):
        cfg = self._cfg()
        mcts = MCTS(model=None, cfg=cfg, device="cpu")

        cases = [
            ("rook", "3rk3/8/8/8/8/8/8/3RK3 w - - 0 1", "d1d7", "d1a1"),
            ("knight", "4kr2/8/8/8/8/8/3N4/4K3 w - - 0 1", "d2f3", "d2b3"),
            ("bishop", "4k3/6r1/8/8/8/8/8/2B1K3 w - - 0 1", "c1g5", "c1e3"),
            ("pawn", "3rk3/8/8/8/8/8/3PP3/4K3 w - - 0 1", "d2d4", "e2e3"),
        ]

        for piece_name, fen, hanging_uci, safe_uci in cases:
            with self.subTest(piece=piece_name):
                board = chess.Board(fen)
                hanging_move = chess.Move.from_uci(hanging_uci)
                safe_move = chess.Move.from_uci(safe_uci)

                self.assertIn(hanging_move, board.legal_moves)
                self.assertIn(safe_move, board.legal_moves)

                hanging_penalty = mcts._piece_tactical_penalty(board, hanging_move)
                safe_penalty = mcts._piece_tactical_penalty(board, safe_move)

                self.assertGreater(hanging_penalty, 0.01)
                self.assertLess(safe_penalty, hanging_penalty)


if __name__ == "__main__":
    unittest.main()
