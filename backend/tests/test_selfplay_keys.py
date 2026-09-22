import unittest

import chess

from app.selfplay.generator import _position_key


class PositionKeyTests(unittest.TestCase):
    def test_ignores_move_counters(self):
        board_a = chess.Board()
        board_b = chess.Board()
        board_b.halfmove_clock = 77
        board_b.fullmove_number = 32
        self.assertEqual(_position_key(board_a), _position_key(board_b))

    def test_includes_castling_rights(self):
        board_a = chess.Board()
        board_b = chess.Board()
        board_b.castling_rights = chess.BB_EMPTY
        self.assertNotEqual(_position_key(board_a), _position_key(board_b))

    def test_ignores_pseudo_en_passant_square_without_legal_capture(self):
        board_with_pseudo_ep = chess.Board('rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 2')
        board_without_ep = chess.Board('rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 2')
        self.assertEqual(_position_key(board_with_pseudo_ep), _position_key(board_without_ep))

    def test_keeps_legal_en_passant_square(self):
        board_with_legal_ep = chess.Board('rnbqkbnr/ppppp1pp/8/4Pp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3')
        board_without_ep = chess.Board('rnbqkbnr/ppppp1pp/8/4Pp2/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3')
        self.assertNotEqual(_position_key(board_with_legal_ep), _position_key(board_without_ep))


if __name__ == '__main__':
    unittest.main()
