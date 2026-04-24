import unittest

import chess

from app.infra.config import AppConfig, ModelConfig, TrainingConfig, ReplayConfig, MCTSConfig, SelfPlayConfig, ArenaConfig, SystemConfig
from app.mcts.search import MCTS
from app.model.network import ChessNet


class MCTSSmokeTests(unittest.TestCase):
    def test_search_returns_legal_move(self):
        cfg = AppConfig(
            model=ModelConfig(input_planes=20, channels=32, res_blocks=1, value_dropout=0.0),
            training=TrainingConfig(),
            replay=ReplayConfig(),
            mcts=MCTSConfig(num_simulations=2, inference_batch_size=1),
            selfplay=SelfPlayConfig(max_game_length=20),
            arena=ArenaConfig(games=2),
            system=SystemConfig(device='cpu', checkpoint_path='models/best_model.pth', default_bestmove_simulations=2),
        )
        model = ChessNet(cfg)
        board = chess.Board()
        result = MCTS(model=model, cfg=cfg, device='cpu').search(board, num_simulations=2)
        self.assertIn(result['best_move'], board.legal_moves)
        self.assertGreater(len(result['visit_counts']), 0)


if __name__ == '__main__':
    unittest.main()
