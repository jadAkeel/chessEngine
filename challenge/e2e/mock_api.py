"""Browser QA server: the real challenge API and built UI with scripted providers.

Only the outbound provider call is replaced; request validation, SSRF checks on the
configured URL, board rebuilding, legality, and results use the production code.
Never deploy this file. Run from the repository root after `npm run build`:

    python challenge/e2e/mock_api.py  # serves http://127.0.0.1:8011

Model IDs select behavior (combine words, e.g. "strong-slow"):
  strong        mates in one when possible, otherwise plays a quick attacking line
  weak          weakens its king so games end in a few moves
  slow          waits 4 s before answering (pause/reset while thinking)
  flaky         every third call for that model fails with HTTP 503
  badkey        always fails with a rejected-key error
  illegal-once  first answer per turn is illegal, the correction is legal
  truncated     stops at the output-token limit before a move
"""
from __future__ import annotations

import asyncio
import os
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("CHALLENGE_STATIC_DIR", str(ROOT / "frontend" / "dist"))
os.environ.setdefault("CHALLENGE_RATE_LIMIT_PER_MINUTE", "600")
sys.path.insert(0, str(ROOT / "backend"))

import chess  # noqa: E402
import api  # noqa: E402

PLANS = {
    ("strong", chess.WHITE): ["e2e4", "d2d4", "d1h5"],
    ("strong", chess.BLACK): ["e7e5", "d8h4"],
    ("weak", chess.WHITE): ["f2f3", "g2g4"],
    ("weak", chess.BLACK): ["f7f6", "g7g5"],
}
calls: dict[str, int] = {}


def _choose(model: str, board: chess.Board) -> tuple[chess.Move, str]:
    legal = list(board.legal_moves)
    for move in legal:
        board.push(move)
        mate = board.is_checkmate()
        board.pop()
        if mate and "strong" in model:
            return move, "Delivers checkmate."
    style = "strong" if "strong" in model else "weak" if "weak" in model else None
    for uci in PLANS.get((style, board.turn), []):
        move = chess.Move.from_uci(uci)
        if move in legal:
            return move, f"Follows the {style} plan with {board.san(move)}."
    move = random.Random(len(board.move_stack)).choice(legal)
    return move, f"Develops with {board.san(move)}."


async def fake_ask(_session, player, board, retry):
    api._request_details(player, board, retry)  # keep URL and model validation in the path
    model = player.model.lower()
    call = calls[model] = calls.get(model, 0) + 1
    await asyncio.sleep(4 if "slow" in model else 0.15)
    if "badkey" in model:
        raise api.ProviderFailure("Provider rejected the API key or model access", 401)
    if "flaky" in model and call % 3 == 0:
        raise api.ProviderFailure("Provider is unavailable (HTTP 503); retry shortly", 503)
    if "illegal-once" in model and not retry:
        return '{"move":"e1e8","explanation":"oops"}', {"input_tokens": 900, "output_tokens": 5}, None
    move, why = _choose(model, board)
    usage = {"input_tokens": 850 + 4 * len(board.move_stack), "output_tokens": 40, "reasoning_tokens": 12}
    if "truncated" in model:
        return '{"move":', usage, "length"
    return f'Thinking done. {{"move":"{move.uci()}","explanation":"{why}"}}', usage, None


api._ask_provider = fake_ask
app = api.app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8011")))
