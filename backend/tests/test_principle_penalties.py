import unittest

import chess

from app.game.principles import principle_penalty_components
from app.infra.config import (
    AppConfig,
    ArenaConfig,
    MCTSConfig,
    ModelConfig,
    PrinciplePenaltiesConfig,
    ReplayConfig,
    SelfPlayConfig,
    SystemConfig,
    TrainingConfig,
    validate,
)
from app.mcts.search import MCTS


class PrinciplePenaltyTests(unittest.TestCase):
    def _principles(self, **overrides):
        return PrinciplePenaltiesConfig(enabled=True, **overrides)

    def _cfg(self, **principle_overrides) -> AppConfig:
        return AppConfig(
            model=ModelConfig(input_planes=20, channels=32, res_blocks=1, value_dropout=0.0),
            training=TrainingConfig(),
            replay=ReplayConfig(),
            principle_penalties=PrinciplePenaltiesConfig(enabled=True, **principle_overrides),
            mcts=MCTSConfig(num_simulations=2, inference_batch_size=1, classical_value_alpha=0.0),
            selfplay=SelfPlayConfig(max_game_length=20),
            arena=ArenaConfig(games=2),
            system=SystemConfig(device="cpu", checkpoint_path="models/best_model.pth", default_bestmove_simulations=2),
        )

    def _components_after(self, fen: str, uci: str, cfg: PrinciplePenaltiesConfig | None = None):
        before = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        self.assertIn(move, before.legal_moves)
        return self._components_for_board(before, move, cfg)

    def _components_for_board(
        self,
        before: chess.Board,
        move: chess.Move,
        cfg: PrinciplePenaltiesConfig | None = None,
    ):
        self.assertIn(move, before.legal_moves)
        after = before.copy(stack=True)
        after.push(move)
        return principle_penalty_components(before, after, move, cfg or self._principles()).components

    def _board_after_san(self, moves: tuple[str, ...]) -> chess.Board:
        board = chess.Board()
        for san in moves:
            board.push_san(san)
        return board

    def test_disabled_principles_return_no_penalty(self):
        components = self._components_after(
            "rnb1k1nr/4q1b1/1ppppp2/7p/P1PPP1PN/2NBB1P1/P6P/R2QR1K1 b - - 0 16",
            "e8f8",
            PrinciplePenaltiesConfig(enabled=False),
        )
        self.assertEqual(components, {})

    def test_king_move_allows_knight_fork_from_first_game(self):
        components = self._components_after(
            "rnb1k1nr/4q1b1/1ppppp2/7p/P1PPP1PN/2NBB1P1/P6P/R2QR1K1 b - - 0 16",
            "e8f8",
        )
        self.assertGreater(components.get("king_safety", 0.0), 0.0)
        self.assertGreater(components.get("tactics", 0.0), 0.0)

    def test_kingside_pawn_push_after_castling_from_second_game(self):
        components = self._components_after(
            "r2r2k1/6b1/1pn1b1pp/p2N1p2/2P4N/3Q4/PP3PPP/R2R2K1 b - - 1 18",
            "g6g5",
        )
        self.assertGreater(components.get("king_safety", 0.0), 0.0)

    def test_allows_mate_in_one_from_second_game(self):
        components = self._components_after(
            "r2r3k/6b1/1pn2N2/7Q/p1P5/8/PP3PPP/R2R2K1 b - - 0 23",
            "g7h6",
        )
        self.assertGreater(components.get("tactics", 0.0), 0.0)

    def test_passive_endgame_king_is_penalized(self):
        components = self._components_after(
            "8/8/8/4k3/8/8/8/4K3 b - - 0 40",
            "e5e6",
        )
        self.assertGreater(components.get("endgame", 0.0), 0.0)

    def test_opening_central_pawn_move_is_not_development_penalty(self):
        components = self._components_for_board(chess.Board(), chess.Move.from_uci("e2e4"))

        self.assertEqual(components.get("opening_development", 0.0), 0.0)

    def test_repeated_opening_pawn_move_is_penalized_before_development(self):
        board = chess.Board()
        for uci in ("e2e4", "a7a6"):
            board.push(chess.Move.from_uci(uci))

        components = self._components_for_board(board, chess.Move.from_uci("e4e5"))

        self.assertGreater(components.get("opening_development", 0.0), 0.0)

    def test_developing_knight_avoids_penalty_extra_pawn_move_gets_penalized(self):
        board = chess.Board()
        for uci in ("e2e4", "e7e5"):
            board.push(chess.Move.from_uci(uci))

        knight_components = self._components_for_board(board, chess.Move.from_uci("g1f3"))
        pawn_components = self._components_for_board(board, chess.Move.from_uci("a2a3"))

        self.assertEqual(knight_components.get("opening_development", 0.0), 0.0)
        self.assertGreater(pawn_components.get("opening_development", 0.0), 0.0)

    def test_early_f_pawn_move_is_penalized_more_than_knight_development(self):
        board = self._board_after_san(("e4", "e6", "Bc4", "d6", "Nf3"))

        f_pawn_components = self._components_for_board(board, chess.Move.from_uci("f7f6"))
        knight_components = self._components_for_board(board, chess.Move.from_uci("g8f6"))

        self.assertGreater(f_pawn_components.get("king_safety", 0.0), 0.0)
        self.assertGreater(f_pawn_components.get("opening_development", 0.0), 0.0)
        self.assertGreater(sum(f_pawn_components.values()), sum(knight_components.values()) + 0.05)

    def test_single_step_center_pawn_is_penalized_when_double_step_is_available(self):
        board = self._board_after_san(("e4",))

        timid_components = self._components_for_board(board, chess.Move.from_uci("e7e6"))
        assertive_components = self._components_for_board(board, chess.Move.from_uci("e7e5"))

        self.assertGreater(timid_components.get("opening_development", 0.0), 0.0)
        self.assertEqual(assertive_components.get("king_safety", 0.0), 0.0)
        self.assertGreater(sum(timid_components.values()), sum(assertive_components.values()) + 0.02)

    def test_serial_single_step_center_pawns_increase_opening_penalty(self):
        first_board = self._board_after_san(("e4",))
        first_components = self._components_for_board(first_board, chess.Move.from_uci("e7e6"))

        repeated_board = self._board_after_san(("e4", "e6", "Bc4"))
        repeated_components = self._components_for_board(repeated_board, chess.Move.from_uci("d7d6"))

        self.assertGreater(
            repeated_components.get("opening_development", 0.0),
            first_components.get("opening_development", 0.0),
        )

    def test_flank_pawn_push_before_minor_development_is_penalized(self):
        board = self._board_after_san(("e4", "e6", "Bc4", "d6", "Nf3"))

        components = self._components_for_board(board, chess.Move.from_uci("b7b5"))

        self.assertGreater(components.get("opening_development", 0.0), 0.0)
        self.assertGreater(components.get("center_control", 0.0), 0.0)

    def test_rook_retreat_after_early_lift_is_penalized(self):
        board = self._board_after_san(
            (
                "e4",
                "e6",
                "Bc4",
                "d6",
                "Nf3",
                "f6",
                "d3",
                "c6",
                "Bf4",
                "b5",
                "Bb3",
                "a5",
                "Nc3",
                "a4",
                "Bxe6",
                "Bxe6",
                "O-O",
                "g5",
                "Be3",
                "h5",
                "Re1",
                "h4",
                "b3",
                "a3",
                "Nd4",
                "Bf7",
                "g4",
                "hxg3",
                "fxg3",
                "Rh5",
                "g4",
            )
        )

        components = self._components_for_board(board, chess.Move.from_uci("h5h8"))

        self.assertGreater(components.get("piece_activity", 0.0), 0.0)
        self.assertGreater(components.get("rook_activity", 0.0), 0.0)

    def test_bishop_diagonal_pawn_move_is_allowed_after_center_claim(self):
        board = chess.Board()
        for uci in ("e2e4", "e7e5"):
            board.push(chess.Move.from_uci(uci))

        components = self._components_for_board(board, chess.Move.from_uci("g2g3"))

        self.assertEqual(components.get("opening_development", 0.0), 0.0)

    def test_opening_development_penalty_fades_out_after_opening(self):
        late_opening_like = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 13")
        endgame = chess.Board("8/8/8/4k3/8/8/8/4K3 b - - 0 6")

        late_components = self._components_for_board(late_opening_like, chess.Move.from_uci("a2a3"))
        endgame_components = self._components_for_board(endgame, chess.Move.from_uci("e5e6"))

        self.assertEqual(late_components.get("opening_development", 0.0), 0.0)
        self.assertEqual(endgame_components.get("opening_development", 0.0), 0.0)

    def test_mcts_includes_principle_component_when_enabled(self):
        cfg = self._cfg()
        mcts = MCTS(model=None, cfg=cfg, device="cpu")
        board = chess.Board("rnb1k1nr/4q1b1/1ppppp2/7p/P1PPP1PN/2NBB1P1/P6P/R2QR1K1 b - - 0 16")
        move = chess.Move.from_uci("e8f8")

        components = mcts._move_penalty_components(board, move)

        self.assertGreater(components.get("principle.king_safety", 0.0), 0.0)
        self.assertGreater(components.get("principle.tactics", 0.0), 0.0)

    def test_principle_validation_rejects_negative_and_large_values(self):
        with self.assertRaisesRegex(ValueError, "principle_penalties.king_safety must be >= 0"):
            validate(self._cfg(king_safety=-0.01))

        with self.assertRaisesRegex(ValueError, "principle_penalties.max_total_per_move is suspiciously large"):
            validate(self._cfg(max_total_per_move=0.75))


if __name__ == "__main__":
    unittest.main()
