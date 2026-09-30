"""Compare checkpoints on positions from the bot's own recent Lichess games.

Reports, per model: policy entropy and top-1 probability over legal moves,
mean |value|, and the value's MSE against the final game result. The last
model given is the candidate; it passes the gate when its value MSE is no
worse than the baseline's (first model) and its values are not compressed.

    python scripts/compare_models_on_bot_games.py \\
        "13 months=trained_output/broadcast_13m/checkpoints/external_best_model.pth" \\
        "mix=trained_output/mix/checkpoints/external_best_model.pth"
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import urllib.request
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.game.board_encoding import encode_board  # noqa: E402
from app.game.move_encoding import move_to_index  # noqa: E402
from app.infra.config import load_config  # noqa: E402
from app.model.checkpoint import load_checkpoint  # noqa: E402
from app.model.network import ChessNet  # noqa: E402


def fetch_positions(user: str, max_games: int) -> list[tuple[chess.Board, float]]:
    """Every 4th position from ply 10 on, with the result from the side to move."""
    url = f"https://lichess.org/api/games/user/{user}?max={max_games}&rated=true&moves=true&pgnInJson=true"
    request = urllib.request.Request(url, headers={"Accept": "application/x-ndjson"})
    with urllib.request.urlopen(request, timeout=60) as response:
        games = [json.loads(line) for line in response.read().decode().splitlines() if line.strip()]
    positions = []
    for game in games:
        pgn = chess.pgn.read_game(io.StringIO(game["pgn"]))
        result = pgn.headers.get("Result")
        if result not in ("1-0", "0-1", "1/2-1/2"):
            continue
        white = 1.0 if result == "1-0" else (-1.0 if result == "0-1" else 0.0)
        board = pgn.board()
        for ply, move in enumerate(pgn.mainline_moves()):
            if ply >= 10 and ply % 4 == 0 and not board.is_game_over():
                positions.append((board.copy(), white if board.turn == chess.WHITE else -white))
            board.push(move)
    print(f"games {len(games)} positions {len(positions)}")
    return positions


def evaluate(path: str, cfg, states: torch.Tensor, legal: list[list[int]], outcome: np.ndarray) -> dict:
    model = ChessNet(cfg)
    load_checkpoint(path, model=model, device="cpu")
    model.eval()
    logits, values = [], []
    with torch.no_grad():
        for start in range(0, len(states), 64):
            batch_logits, batch_values = model(states[start:start + 64])
            logits.append(batch_logits)
            values.append(batch_values.reshape(-1))
    logits_np = torch.cat(logits).numpy()
    values_np = torch.cat(values).numpy()
    entropy, top1_prob, top1 = [], [], []
    for row, indices in enumerate(legal):
        z = logits_np[row, indices]
        p = np.exp(z - z.max())
        p /= p.sum()
        entropy.append(float(-(p * np.log(p + 1e-12)).sum()))
        top1_prob.append(float(p.max()))
        top1.append(indices[int(p.argmax())])
    return {
        "entropy": float(np.mean(entropy)),
        "top1_prob": float(np.mean(top1_prob)),
        "top1": top1,
        "values": values_np,
        "value_abs": float(np.mean(np.abs(values_np))),
        "value_mse": float(np.mean((values_np - outcome) ** 2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", help="name=checkpoint path; first = baseline, last = candidate")
    parser.add_argument("--user", default="chessengineboot")
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--mse-tolerance", type=float, default=0.0, help="Allowed value MSE above the baseline")
    parser.add_argument("--min-value-abs-ratio", type=float, default=0.9, help="Candidate |value| / baseline |value|")
    args = parser.parse_args()

    models = dict(item.split("=", 1) for item in args.models)
    if len(models) < 2:
        parser.error("give at least a baseline and a candidate")
    torch.set_num_threads(4)
    random.seed(0)
    # get_current_config() is the small default; the checkpoints are 160x24.
    cfg = load_config(args.config)
    positions = fetch_positions(args.user, args.games)
    states = torch.stack([encode_board(board, cfg) for board, _ in positions])
    legal = [[move_to_index(move, board) for move in board.legal_moves] for board, _ in positions]
    outcome = np.array([result for _, result in positions])

    results = {name: evaluate(path, cfg, states, legal, outcome) for name, path in models.items()}
    names = list(results)
    print(f"{'model':22s} {'entropy':>8s} {'top1 prob':>9s} {'|value|':>8s} {'value MSE vs result':>20s}")
    for name in names:
        r = results[name]
        print(f"{name:22s} {r['entropy']:8.3f} {r['top1_prob']:9.3f} {r['value_abs']:8.3f} {r['value_mse']:20.4f}")
    baseline, candidate = results[names[0]], results[names[-1]]
    for other in names[1:]:
        a, b = results[names[0]], results[other]
        agree = np.mean([x == y for x, y in zip(a["top1"], b["top1"])])
        corr = np.corrcoef(a["values"], b["values"])[0, 1]
        print(f"{names[0]} vs {other}: same first move {agree:.1%}, value corr {corr:.3f}")

    mse_ok = candidate["value_mse"] <= baseline["value_mse"] + args.mse_tolerance
    abs_ok = candidate["value_abs"] >= args.min_value_abs_ratio * baseline["value_abs"]
    verdict = "PASS" if mse_ok and abs_ok else "FAIL"
    print(
        f"[GATE] {verdict}: value MSE {candidate['value_mse']:.4f} vs {baseline['value_mse']:.4f} "
        f"({'ok' if mse_ok else 'worse'}), |value| {candidate['value_abs']:.3f} vs {baseline['value_abs']:.3f} "
        f"({'ok' if abs_ok else 'compressed'})"
    )
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
