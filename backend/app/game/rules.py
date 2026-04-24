
import chess

def terminal_value(board: chess.Board):
    outcome = board.outcome(claim_draw=False)

    if outcome is None:
        return None  # game not finished

    if outcome.winner is None:
        return 0  # draw

    return 1 if outcome.winner == chess.WHITE else -1