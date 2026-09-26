from __future__ import annotations

import argparse
import csv
import gzip
import io
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chess
import numpy as np
import zstandard as zstd

from app.game.board_encoding import encode_board
from app.game.move_encoding import NUM_MOVES, move_to_index
from app.infra.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Lichess Puzzles CSV(.zst) into training shards")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--input", type=str, required=True, help="Path to lichess_db_puzzle.csv or .csv.zst")
    parser.add_argument("--output-dir", type=str, required=True, help="Output folder for shards")
    parser.add_argument("--min-rating", type=int, default=1400, help="Minimum puzzle rating (default: 1400)")
    parser.add_argument("--max-rating", type=int, default=2600, help="Maximum puzzle rating (default: 2600)")
    parser.add_argument("--min-popularity", type=int, default=70, help="Minimum puzzle popularity (0-100, default: 70)")
    parser.add_argument("--max-samples", type=int, default=1000000, help="Max tactical training samples to extract")
    parser.add_argument("--shard-size", type=int, default=100000, help="Samples per shard file")
    parser.add_argument("--required-themes", type=str, default="", help="Comma-separated themes required (optional)")
    return parser


def _open_csv_stream(path: Path):
    if path.suffix == ".zst" or path.name.endswith(".csv.zst"):
        fh = open(path, "rb")
        dctx = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(fh)
        text = io.TextIOWrapper(reader, encoding="utf-8")
        return fh, reader, text
    elif path.suffix == ".gz":
        fh = gzip.open(path, "rt", encoding="utf-8")
        return fh, None, fh
    else:
        fh = open(path, "r", encoding="utf-8")
        return fh, None, fh


def _save_shard(output_dir: Path, shard_id: int, states: list, policy_indices: list, values: list, cfg) -> None:
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


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input puzzle file not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_themes = [t.strip().lower() for t in args.required_themes.split(",") if t.strip()]

    # Find highest existing shard ID to avoid overwrite
    existing_shards = sorted(output_dir.glob("shard_*.npz"))
    if existing_shards:
        try:
            last_id = max(int(p.stem.split("_")[1]) for p in existing_shards)
            shard_id = last_id + 1
            print(f"[RESUME] Found {len(existing_shards)} shards, starting from shard_{shard_id}")
        except Exception:
            shard_id = 0
    else:
        shard_id = 0

    states = []
    policy_indices = []
    values = []

    total_samples = 0
    total_puzzles = 0
    skipped_rating = 0
    skipped_popularity = 0
    skipped_themes = 0

    start_time = time.time()
    print(f"[START] Converting Lichess Puzzles: {input_path}")
    print(f"[CONFIG] min_rating={args.min_rating}, max_rating={args.max_rating}, min_pop={args.min_popularity}")
    print(f"[TARGET] max_samples={args.max_samples}, shard_size={args.shard_size}")

    fh, reader, text_stream = _open_csv_stream(input_path)

    try:
        csv_reader = csv.reader(text_stream)
        # Check header
        first_row = next(csv_reader, None)
        if first_row is None:
            print("[EMPTY] Puzzle file is empty")
            return

        # Header format: PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags
        has_named_header = "puzzleid" in first_row[0].lower() or "fen" in first_row[1].lower()
        if not has_named_header:
            # First row is actual data
            row_iterator = [first_row]
        else:
            row_iterator = []

        import itertools
        for row in itertools.chain(row_iterator, csv_reader):
            if not row or len(row) < 4:
                continue

            try:
                puzzle_id = row[0].strip()
                fen = row[1].strip()
                moves_str = row[2].strip()
                rating = int(row[3].strip())
                popularity = int(row[5].strip()) if len(row) > 5 and row[5].strip() else 100
                themes = row[7].strip().lower() if len(row) > 7 else ""
            except (ValueError, IndexError):
                continue

            if rating < args.min_rating or rating > args.max_rating:
                skipped_rating += 1
                continue

            if popularity < args.min_popularity:
                skipped_popularity += 1
                continue

            if required_themes:
                if not any(theme in themes for theme in required_themes):
                    skipped_themes += 1
                    continue

            total_puzzles += 1

            # Play moves on board
            try:
                board = chess.Board(fen)
                moves = moves_str.split()
                if not moves:
                    continue

                # Move 0 is played by opponent to set up the puzzle
                first_move = chess.Move.from_uci(moves[0])
                if first_move not in board.legal_moves:
                    continue
                board.push(first_move)

                # Now alternate: player tactical moves (indices 1, 3, 5...), opponent responses (indices 2, 4...)
                for move_idx in range(1, len(moves), 2):
                    sol_move = chess.Move.from_uci(moves[move_idx])
                    if sol_move not in board.legal_moves:
                        break

                    # Encode current board state for tactical solution
                    state = encode_board(board, cfg).cpu().numpy().astype(np.float16, copy=False)
                    policy_idx = int(move_to_index(sol_move, board))
                    # Tactical winning move target value is +1.0 for the player to move
                    target_val = 1.0

                    states.append(state)
                    policy_indices.append(policy_idx)
                    values.append(target_val)
                    total_samples += 1

                    if len(states) >= args.shard_size:
                        _save_shard(output_dir, shard_id, states, policy_indices, values, cfg)
                        states, policy_indices, values = [], [], []
                        shard_id += 1

                    if total_samples >= args.max_samples:
                        break

                    # Apply player solution move
                    board.push(sol_move)

                    # Apply opponent response if present
                    if move_idx + 1 < len(moves):
                        opp_move = chess.Move.from_uci(moves[move_idx + 1])
                        if opp_move not in board.legal_moves:
                            break
                        board.push(opp_move)

                if total_samples >= args.max_samples:
                    break

                if total_puzzles % 25000 == 0:
                    elapsed = time.time() - start_time
                    speed = total_samples / max(elapsed, 1e-6)
                    print(
                        f"[PROGRESS] puzzles={total_puzzles} | samples={total_samples} | "
                        f"speed={speed:.1f} samples/s | shard={shard_id}"
                    )

            except Exception as exc:
                continue

        # Save any remaining samples in buffer
        if states:
            _save_shard(output_dir, shard_id, states, policy_indices, values, cfg)

        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"[DONE] Processed {total_puzzles} puzzles -> {total_samples} samples")
        print(f"Elapsed time: {elapsed:.2f}s ({total_samples / max(elapsed, 1e-6):.1f} samples/s)")
        print(f"Skipped: rating={skipped_rating}, popularity={skipped_popularity}, themes={skipped_themes}")
        print("=" * 60)

    finally:
        if reader is not None:
            reader.close()
        fh.close()


if __name__ == "__main__":
    main()
