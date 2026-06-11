import unittest

import chess

from app.evaluation.metrics import evaluate_board
from app.game.repetition import build_seen_positions, filter_repetition_moves
from app.infra.config import AppConfig, ArenaConfig, MCTSConfig, ModelConfig, ReplayConfig, SelfPlayConfig, SystemConfig, TrainingConfig
from app.mcts.search import MCTS
from app.model.network import ChessNet


class RepetitionAndValueTests(unittest.TestCase):
    def _cfg(self, **mcts_overrides) -> AppConfig:
        return AppConfig(
            model=ModelConfig(input_planes=20, channels=32, res_blocks=1, value_dropout=0.0),
            training=TrainingConfig(),
            replay=ReplayConfig(),
            mcts=MCTSConfig(num_simulations=2, inference_batch_size=1, **mcts_overrides),
            selfplay=SelfPlayConfig(max_game_length=20, repetition_penalty=0.45, repetition_break_count=3, repetition_move_weight=0.05),
            arena=ArenaConfig(games=2),
            system=SystemConfig(device="cpu", checkpoint_path="models/best_model.pth", default_bestmove_simulations=2),
        )

    def test_classical_blend_uses_side_to_move_perspective(self):
        cfg = self._cfg(classical_value_alpha=1.0)
        model = ChessNet(cfg)
        mcts = MCTS(model=model, cfg=cfg, device="cpu")

        # Black is completely winning materially. From black-to-move perspective,
        # the blended value should be strongly positive.
        board = chess.Board("4k3/8/8/8/8/8/4q3/4K3 b - - 0 1")
        blended = mcts._blend_value(board, nn_value=0.0)
        self.assertGreater(blended, 0.5)

    def test_classical_eval_rewards_advanced_black_passed_pawn(self):
        distant = chess.Board("4k3/8/2p5/8/8/8/8/4K3 b - - 0 1")
        near_promotion = chess.Board("4k3/8/8/8/8/8/2p5/4K3 b - - 0 1")

        self.assertLess(evaluate_board(near_promotion), evaluate_board(distant) - 100)

    def test_claimable_threefold_is_treated_as_terminal_draw(self):
        cfg = self._cfg()
        model = ChessNet(cfg)
        mcts = MCTS(model=model, cfg=cfg, device="cpu")

        board = chess.Board()
        for uci in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"]:
            board.push_uci(uci)

        self.assertFalse(board.is_game_over(claim_draw=False))
        self.assertTrue(board.can_claim_threefold_repetition())
        self.assertTrue(mcts._is_terminal_board(board))
        self.assertEqual(mcts._terminal_value(board), 0.0)

    def test_root_repetition_filter_downweights_looping_move(self):
        board = chess.Board()
        for uci in ["g1f3", "g8f6", "f3g1", "f6g8"]:
            board.push_uci(uci)

        seen_positions = build_seen_positions(board)
        policy = {
            chess.Move.from_uci("g1f3"): 0.7,
            chess.Move.from_uci("e2e4"): 0.3,
        }
        filtered, repetition_counts = filter_repetition_moves(
            policy,
            board,
            seen_positions,
            repeat_break_count=3,
            repeat_weight=0.05,
        )

        self.assertEqual(repetition_counts["g1f3"], 2)
        self.assertGreater(filtered[chess.Move.from_uci("e2e4")], filtered[chess.Move.from_uci("g1f3")])

    def test_position_key_ignores_pseudo_ep_and_keeps_legal_ep(self):
        from app.game.repetition import position_key

        pseudo_ep = chess.Board('rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 2')
        pseudo_ep_without = chess.Board('rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 2')
        legal_ep = chess.Board('rnbqkbnr/ppppp1pp/8/4Pp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3')
        legal_ep_without = chess.Board('rnbqkbnr/ppppp1pp/8/4Pp2/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3')

        self.assertEqual(position_key(pseudo_ep), position_key(pseudo_ep_without))
        self.assertNotEqual(position_key(legal_ep), position_key(legal_ep_without))


if __name__ == "__main__":
    unittest.main()
