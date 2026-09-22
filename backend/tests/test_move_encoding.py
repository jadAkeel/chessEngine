import unittest

import chess

from app.game.move_encoding import index_to_move, move_to_index


class MoveEncodingTests(unittest.TestCase):
    def assertRoundTrip(self, fen: str, uci: str) -> None:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        idx = move_to_index(move, board)
        decoded = index_to_move(idx, board)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.uci(), uci)

    def test_white_queen_promotion_roundtrip(self):
        self.assertRoundTrip('8/4P3/8/8/8/8/8/k6K w - - 0 1', 'e7e8q')

    def test_black_queen_promotion_roundtrip(self):
        self.assertRoundTrip('8/8/8/8/8/8/4p3/4K2k b - - 0 1', 'e2e1q')

    def test_black_underpromotion_roundtrip(self):
        self.assertRoundTrip('8/8/8/8/8/8/4p3/4K2k b - - 0 1', 'e2e1n')

    def test_black_regular_move_roundtrip(self):
        self.assertRoundTrip('8/8/8/8/8/4p3/8/4K2k b - - 0 1', 'e3e2')


if __name__ == '__main__':
    unittest.main()
