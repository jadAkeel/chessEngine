from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chess.pgn
import numpy as np
import zstandard as zstd

from app.game.board_encoding import encode_board
from app.game.move_encoding import NUM_MOVES, move_to_index
from app.infra.config import load_config


# =========================================
# ARG PARSER
# =========================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import Lichess PGN → SHARDED dataset (WITH LOGS)")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-games", type=int, default=10000)
    parser.add_argument("--max-samples", type=int, default=1000000)
    parser.add_argument("--shard-size", type=int, default=100000)
    parser.add_argument("--min-fullmove", type=int, default=5)
    parser.add_argument("--skip-draws", action="store_true")
    return parser


# =========================================
# HELPERS
# =========================================

def _value_from_result(result: str, white_to_move: bool) -> float:
    if result == "1-0":
        white_value = 1.0
    elif result == "0-1":
        white_value = -1.0
    else:
        white_value = 0.0
    return float(white_value if white_to_move else -white_value)


def _open_text_stream(path: Path):
    if path.suffix == ".zst":
        fh = open(path, "rb")
        dctx = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(fh)
        text = io.TextIOWrapper(reader, encoding="utf-8")
        return fh, reader, text
    fh = open(path, "r", encoding="utf-8")
    return fh, None, fh


def _save_shard(output_dir, shard_id, states, policy_indices, values, cfg):
    shard_path = output_dir / f"shard_{shard_id}.npz"

    np.savez_compressed(
        shard_path,
        states=np.stack(states).astype(np.float16, copy=False),
        policy_indices=np.asarray(policy_indices, dtype=np.int32),
        values=np.asarray(values, dtype=np.float32),
        input_planes=np.asarray([int(cfg.model.input_planes)], dtype=np.int16),
        policy_size=np.asarray([int(NUM_MOVES)], dtype=np.int32),
    )

    print(f"[SHARD SAVE] {shard_path.name} | samples={len(states)}")


# =========================================
# MAIN
# =========================================

def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_size = int(args.shard_size)

    states = []
    policy_indices = []
    values = []

    # =========================================
# AUTO SHARD ID (NO OVERWRITE)
# =========================================

    existing_shards = sorted(output_dir.glob("shard_*.npz"))

    if existing_shards:
        try:
            last_id = max(int(p.stem.split("_")[1]) for p in existing_shards)
            shard_id = last_id + 1
            print(f"[RESUME] Found {len(existing_shards)} shards, continuing from shard_{shard_id}")
        except Exception:
            print("[WARN] Failed to parse shard IDs, starting from 0")
            shard_id = 0
    else:
        shard_id = 0
        print("[INIT] No existing shards found, starting from shard_0")

    total_samples = 0
    total_games = 0

    start_time = time.time()

    print("\n[START] Importing PGN → shards")
    print(f"[INPUT] {input_path}")
    print(f"[OUTPUT DIR] {output_dir}")
    print(f"[CONFIG] shard_size={shard_size}, max_samples={args.max_samples}\n")

    fh, reader, text_stream = _open_text_stream(input_path)

    try:
        while total_games < int(args.max_games) and total_samples < int(args.max_samples):
            game = chess.pgn.read_game(text_stream)

            if game is None:
                print("[END] No more games")
                break

            total_games += 1

            result = (game.headers.get("Result") or "").strip()

            if result not in {"1-0", "0-1", "1/2-1/2"}:
                continue

            if args.skip_draws and result == "1/2-1/2":
                continue

            board = game.board()

            game_samples = 0

            for move in game.mainline_moves():
                if move not in board.legal_moves:
                    print("[WARN] illegal move detected, skipping game")
                    break

                if board.fullmove_number >= int(args.min_fullmove):
                    if not board.is_game_over(claim_draw=True) and not board.is_repetition():

                        state = encode_board(board, cfg).cpu().numpy().astype(np.float16, copy=False)
                        policy_idx = int(move_to_index(move, board))
                        value = _value_from_result(result, bool(board.turn))

                        states.append(state)
                        policy_indices.append(policy_idx)
                        values.append(value)

                        total_samples += 1
                        game_samples += 1

                        # 🔥 shard save
                        if len(states) >= shard_size:
                            _save_shard(output_dir, shard_id, states, policy_indices, values, cfg)

                            states, policy_indices, values = [], [], []
                            shard_id += 1

                        # 🔥 progress
                        if total_samples % 50000 == 0:
                            elapsed = time.time() - start_time
                            speed = total_samples / max(elapsed, 1e-6)
                            print(f"[PROGRESS] samples={total_samples} | games={total_games} | speed={speed:.1f} samples/sec")

                        if total_samples >= int(args.max_samples):
                            break

                board.push(move)

            # 🔥 game log
            if total_games % 100 == 0:
                print(f"[GAME] #{total_games} → samples from game={game_samples}")

        # 🔥 last shard
        if states:
            _save_shard(output_dir, shard_id, states, policy_indices, values, cfg)

        elapsed = time.time() - start_time

        print("\n[FINAL]")
        print(f"games={total_games}")
        print(f"samples={total_samples}")
        print(f"shards={shard_id + 1}")
        print(f"time={elapsed:.1f}s")
        print(f"speed={total_samples / max(elapsed,1e-6):.1f} samples/sec")

    finally:
        try:
            text_stream.close()
        except Exception:
            pass
        try:
            if reader:
                reader.close()
        except Exception:
            pass
        fh.close()


if __name__ == "__main__":
    main()