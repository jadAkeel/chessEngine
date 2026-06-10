import unittest

from app.training.train_external import _best_val_loss_from_history


class TrainExternalHistoryTests(unittest.TestCase):
    def test_best_val_loss_uses_previous_history(self):
        history = {"val_loss": [2.55, "2.48", None, "bad", float("nan"), 2.60]}

        self.assertEqual(_best_val_loss_from_history(history), 2.48)

    def test_best_val_loss_defaults_to_infinity_without_history(self):
        self.assertEqual(_best_val_loss_from_history({}), float("inf"))


if __name__ == "__main__":
    unittest.main()
