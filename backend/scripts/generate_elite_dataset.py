from __future__ import annotations

"""Generate a large, verified chess training dataset from Lichess Elite archives.

Designed to run either locally or inside a Kaggle kernel:

* streams monthly PGN archives straight out of their ZIP (no full extraction),
* encodes positions with the project's own ``encode_board`` / ``move_to_index``
  so shards stay byte-compatible with ``app.training.external_samples``,
* writes deterministic ``shard_NNNNN.npz`` files plus a ``manifest.json``
  describing counts, hashes, shapes, dtypes, sources, months and filters,
* resumes safely after interruption without duplicating shard ids or samples,
* deletes each monthly archive once consumed so Kaggle disk limits hold.

Unlike ``import_lichess_samples.py`` this script has no hidden game cap:
``--max-games`` defaults to 0 (unlimited) and the run is bounded by
``--target-samples`` instead.
"""

import argparse
import hashlib
import io
import json
import multiprocessing as mp
import os
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chess
import chess.pgn
import numpy as np

from app.game.move_encoding import NUM_MOVES
from app.infra.config import load_config

GENERATOR_VERSION = "elite-v1"
MANIFEST_NAME = "manifest.json"
BASE_URL = "https://database.nikonoel.fr/lichess_elite_{month}.zip"

REJECTION_KEYS = (
    "bad_result",
    "low_elo",
    "skipped_draw",
    "illegal_move",
    "out_of_range_policy",
    "non_finite_value",
    "non_finite_state",
    "empty_state",
    "terminal_position",
)


# =========================================
# ARG PARSER
# =========================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate verified NPZ shards from Lichess Elite monthly archives",
    )
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--work-dir", type=str, default="", help="Scratch dir for archives (default: <output-dir>/_work)")
    parser.add_argument("--start-month", type=str, default="2025-11", help="Newest month to consume, YYYY-MM")
    parser.add_argument("--months", type=int, default=12, help="How many months to walk backwards through")
    parser.add_argument("--target-samples", type=int, default=20_000_000, help="Stop once this many samples exist")
    parser.add_argument("--shard-size", type=int, default=125_000)
    parser.add_argument("--max-games", type=int, default=0, help="0 = unlimited (no hidden cap)")
    parser.add_argument("--min-fullmove", type=int, default=5)
    parser.add_argument("--max-fullmove", type=int, default=0)
    parser.add_argument("--min-elo", type=int, default=0)
    parser.add_argument("--skip-draws", action="store_true")
    parser.add_argument("--workers", type=int, default=0, help="0 = cpu_count()")
    parser.add_argument("--chunk-games", type=int, default=64, help="Games per worker task")
    parser.add_argument("--keep-archives", action="store_true", help="Do not delete downloaded ZIPs")
    return parser


# =========================================
# MONTHS
# =========================================

def iter_months(start_month: str, count: int) -> list[str]:
    year, month = (int(part) for part in start_month.split("-"))
    months = []
    for _ in range(int(count)):
        months.append(f"{year:04d}-{month:02d}")
        month -= 1
        if month == 0:
            year -= 1
            month = 12
    return months


def download_month(month: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = BASE_URL.format(month=month)
    dest = dest_dir / f"lichess_elite_{month}.zip"

    if dest.exists() and dest.stat().st_size > 0:
        print(f"[ARCHIVE] reuse {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)", flush=True)
        return dest

    tmp = dest.with_name(dest.name + ".part")
    print(f"[DOWNLOAD] {url}", flush=True)
    start = time.time()
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as handle:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            handle.write(block)
    os.replace(tmp, dest)
    print(
        f"[DOWNLOAD] done {dest.name} ({dest.stat().st_size / 1e6:.1f} MB in {time.time() - start:.1f}s)",
        flush=True,
    )
    return dest


# =========================================
# PGN STREAMING
# =========================================

def iter_raw_games(text_stream):
    """Yield complete PGN game blocks without loading the whole file."""
    buf: list[str] = []
    seen_movetext = False
    for line in text_stream:
        if seen_movetext and line.startswith("[Event "):
            yield "".join(buf)
            buf = [line]
            seen_movetext = False
            continue
        buf.append(line)
        stripped = line.strip()
        if stripped and not stripped.startswith("["):
            seen_movetext = True
    if buf and seen_movetext:
        yield "".join(buf)


def iter_month_games(archive: Path):
    with zipfile.ZipFile(archive) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".pgn")]
        if not names:
            raise FileNotFoundError(f"No .pgn entry inside {archive}")
        for name in sorted(names):
            with zf.open(name, "r") as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
                yield from iter_raw_games(text)


# =========================================
# ENCODING WORKERS
# =========================================

_WORKER: dict = {}


def _worker_init(config_path: str, options: dict) -> None:
    from app.game.board_encoding import encode_board

    _WORKER["cfg"] = load_config(config_path)
    _WORKER["encode_board"] = encode_board
    _WORKER["opt"] = options


def _value_from_result(result: str, white_to_move: bool) -> float:
    if result == "1-0":
        white_value = 1.0
    elif result == "0-1":
        white_value = -1.0
    else:
        white_value = 0.0
    return float(white_value if white_to_move else -white_value)


def encode_games(chunk: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    from app.game.move_encoding import move_to_index

    cfg = _WORKER["cfg"]
    encode_board = _WORKER["encode_board"]
    opt = _WORKER["opt"]

    planes = int(cfg.model.input_planes)
    min_fullmove = int(opt["min_fullmove"])
    max_fullmove = int(opt["max_fullmove"])
    min_elo = int(opt["min_elo"])
    skip_draws = bool(opt["skip_draws"])

    states: list[np.ndarray] = []
    policy_indices: list[int] = []
    values: list[float] = []
    stats = {key: 0 for key in REJECTION_KEYS}
    stats["games_used"] = 0

    for text in chunk:
        game = chess.pgn.read_game(io.StringIO(text))
        if game is None:
            stats["bad_result"] += 1
            continue

        result = (game.headers.get("Result") or "").strip()
        if result not in {"1-0", "0-1", "1/2-1/2"}:
            stats["bad_result"] += 1
            continue
        if skip_draws and result == "1/2-1/2":
            stats["skipped_draw"] += 1
            continue

        if min_elo > 0:
            try:
                w_elo = int((game.headers.get("WhiteElo") or "0").strip())
                b_elo = int((game.headers.get("BlackElo") or "0").strip())
            except (TypeError, ValueError):
                stats["low_elo"] += 1
                continue
            if w_elo < min_elo or b_elo < min_elo:
                stats["low_elo"] += 1
                continue

        board = game.board()
        used = False

        for move in game.mainline_moves():
            if move not in board.legal_moves:
                stats["illegal_move"] += 1
                break

            in_range = board.fullmove_number >= min_fullmove and (
                max_fullmove <= 0 or board.fullmove_number <= max_fullmove
            )
            if in_range:
                if board.is_game_over(claim_draw=True) or board.is_repetition():
                    stats["terminal_position"] += 1
                else:
                    state = encode_board(board, cfg).cpu().numpy().astype(np.float16, copy=False)
                    policy_idx = int(move_to_index(move, board))
                    value = _value_from_result(result, bool(board.turn))

                    if policy_idx < 0 or policy_idx >= NUM_MOVES:
                        stats["out_of_range_policy"] += 1
                    elif not np.isfinite(value) or abs(value) > 1.0:
                        stats["non_finite_value"] += 1
                    elif not np.all(np.isfinite(state)):
                        stats["non_finite_state"] += 1
                    elif not np.any(state):
                        stats["empty_state"] += 1
                    else:
                        states.append(state)
                        policy_indices.append(policy_idx)
                        values.append(value)
                        used = True

            board.push(move)

        if used:
            stats["games_used"] += 1

    if states:
        states_arr = np.stack(states).astype(np.float16, copy=False)
    else:
        states_arr = np.zeros((0, planes, 8, 8), dtype=np.float16)

    return (
        states_arr,
        np.asarray(policy_indices, dtype=np.int32),
        np.asarray(values, dtype=np.float32),
        stats,
    )


# =========================================
# MANIFEST
# =========================================

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def new_manifest(cfg, args) -> dict:
    return {
        "generator_version": GENERATOR_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "updated_utc": None,
        "source": "lichess-elite",
        "source_url_template": BASE_URL,
        "state_shape": [int(cfg.model.input_planes), 8, 8],
        "policy_size": int(NUM_MOVES),
        "dtypes": {"states": "float16", "policy_indices": "int32", "values": "float32"},
        "filters": {
            "min_fullmove": int(args.min_fullmove),
            "max_fullmove": int(args.max_fullmove),
            "min_elo": int(args.min_elo),
            "skip_draws": bool(args.skip_draws),
            "shard_size": int(args.shard_size),
            "max_games": int(args.max_games),
        },
        "total_samples": 0,
        "shards": [],
        "months": {},
        "rejected": {key: 0 for key in REJECTION_KEYS},
    }


def load_manifest(output_dir: Path, cfg, args) -> dict:
    path = output_dir / MANIFEST_NAME
    if not path.exists():
        return new_manifest(cfg, args)
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    print(
        f"[RESUME] manifest found: total_samples={manifest.get('total_samples', 0)} "
        f"shards={len(manifest.get('shards', []))}",
        flush=True,
    )
    return manifest


def save_manifest(output_dir: Path, manifest: dict) -> None:
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    path = output_dir / MANIFEST_NAME
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    os.replace(tmp, path)


# =========================================
# SHARD BUFFER
# =========================================

class ShardBuffer:
    """Fixed-capacity buffer that flushes full shards to disk."""

    def __init__(self, output_dir: Path, manifest: dict, cfg, shard_size: int):
        self.output_dir = output_dir
        self.manifest = manifest
        self.cfg = cfg
        self.capacity = int(shard_size)
        self.planes = int(cfg.model.input_planes)

        self.states = np.zeros((self.capacity, self.planes, 8, 8), dtype=np.float16)
        self.policy_indices = np.zeros((self.capacity,), dtype=np.int32)
        self.values = np.zeros((self.capacity,), dtype=np.float32)
        self.count = 0

        existing = [int(entry["id"]) for entry in manifest.get("shards", [])]
        self.next_id = (max(existing) + 1) if existing else 0

    def add(self, states: np.ndarray, policy_indices: np.ndarray, values: np.ndarray, month: str) -> None:
        offset = 0
        total = len(states)
        while offset < total:
            room = self.capacity - self.count
            take = min(room, total - offset)
            self.states[self.count:self.count + take] = states[offset:offset + take]
            self.policy_indices[self.count:self.count + take] = policy_indices[offset:offset + take]
            self.values[self.count:self.count + take] = values[offset:offset + take]
            self.count += take
            offset += take
            if self.count >= self.capacity:
                self.flush(month)

    def flush(self, month: str) -> int:
        if self.count == 0:
            return 0

        shard_id = self.next_id
        name = f"shard_{shard_id:05d}.npz"
        final = self.output_dir / name
        tmp = self.output_dir / f"{name}.building"

        np.savez_compressed(
            tmp,
            states=self.states[: self.count],
            policy_indices=self.policy_indices[: self.count],
            values=self.values[: self.count],
            input_planes=np.asarray([self.planes], dtype=np.int16),
            policy_size=np.asarray([int(NUM_MOVES)], dtype=np.int32),
        )
        # np.savez_compressed appends .npz when the target lacks that suffix.
        produced = tmp if tmp.exists() else tmp.with_name(tmp.name + ".npz")
        os.replace(produced, final)

        samples = int(self.count)
        self.manifest["shards"].append(
            {
                "id": shard_id,
                "file": name,
                "samples": samples,
                "sha256": _sha256(final),
                "bytes": final.stat().st_size,
                "month": month,
            }
        )
        self.manifest["total_samples"] = int(self.manifest["total_samples"]) + samples
        self.next_id += 1
        self.count = 0

        print(
            f"[SHARD] {name} samples={samples} total={self.manifest['total_samples']}",
            flush=True,
        )
        return samples


# =========================================
# MAIN
# =========================================

def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir) if args.work_dir else output_dir / "_work"

    manifest = load_manifest(output_dir, cfg, args)
    buffer = ShardBuffer(output_dir, manifest, cfg, args.shard_size)

    target = int(args.target_samples)
    workers = int(args.workers) or (os.cpu_count() or 2)
    options = {
        "min_fullmove": args.min_fullmove,
        "max_fullmove": args.max_fullmove,
        "min_elo": args.min_elo,
        "skip_draws": args.skip_draws,
    }

    months = iter_months(args.start_month, args.months)
    print(f"[PLAN] months={months}", flush=True)
    print(f"[PLAN] target_samples={target} shard_size={args.shard_size} workers={workers}", flush=True)

    if manifest["total_samples"] >= target:
        print(f"[DONE] target already met: {manifest['total_samples']} >= {target}", flush=True)
        return

    start_time = time.time()
    pool = mp.Pool(workers, initializer=_worker_init, initargs=(args.config, options))

    def drain(batch: list[list[str]], month: str, state: dict) -> int:
        """Encode a batch of game chunks and return how many games were consumed."""
        games = 0
        for (states, policies, values, stats), sent in zip(
            pool.imap(encode_games, batch, chunksize=1), batch
        ):
            games += len(sent)
            state["games_used"] += int(stats.pop("games_used", 0))
            for key in REJECTION_KEYS:
                manifest["rejected"][key] += int(stats.get(key, 0))
            if len(states):
                buffer.add(states, policies, values, month)
        return games

    try:
        for month in months:
            if manifest["total_samples"] >= target:
                break

            state = manifest["months"].setdefault(
                month,
                {
                    "url": BASE_URL.format(month=month),
                    "games_consumed": 0,
                    "games_used": 0,
                    "samples": 0,
                    "completed": False,
                },
            )
            if state.get("completed"):
                print(f"[SKIP] {month} already completed ({state['samples']} samples)", flush=True)
                continue

            skip_games = int(state.get("games_consumed", 0))
            if args.max_games and skip_games >= int(args.max_games):
                print(f"[SKIP] {month}: already consumed {skip_games} games (max-games reached)", flush=True)
                continue
            if skip_games:
                print(f"[RESUME] {month}: skipping first {skip_games} games", flush=True)

            archive = download_month(month, work_dir)
            month_start_total = int(manifest["total_samples"]) - int(state.get("samples", 0))

            game_iter = iter_month_games(archive)
            for _ in range(skip_games):
                if next(game_iter, None) is None:
                    break
            consumed = skip_games

            chunk: list[str] = []
            batch: list[list[str]] = []
            month_done = True

            for text in game_iter:
                chunk.append(text)
                if len(chunk) < int(args.chunk_games):
                    continue
                batch.append(chunk)
                chunk = []

                if len(batch) < workers * 2:
                    continue

                shards_before = len(manifest["shards"])
                consumed += drain(batch, month, state)
                batch = []

                if len(manifest["shards"]) > shards_before:
                    # Checkpoint only at shard boundaries so resume is exact.
                    state["games_consumed"] = consumed
                    state["samples"] = manifest["total_samples"] - month_start_total
                    save_manifest(output_dir, manifest)
                    elapsed = time.time() - start_time
                    print(
                        f"[PROGRESS] month={month} games={consumed} "
                        f"total={manifest['total_samples']} "
                        f"rate={(manifest['total_samples']) / max(elapsed, 1e-6):.0f}/s",
                        flush=True,
                    )

                if manifest["total_samples"] >= target:
                    month_done = False
                    break
                if args.max_games and consumed >= int(args.max_games):
                    month_done = False
                    break

            if month_done:
                if chunk:
                    batch.append(chunk)
                if batch:
                    consumed += drain(batch, month, state)

            # Release the ZIP handle before deleting the archive, otherwise the
            # file stays locked (Windows) or keeps occupying disk (Linux).
            game_iter.close()

            # Close the month: flush the tail so shards never span two months.
            buffer.flush(month)
            state["games_consumed"] = consumed
            state["samples"] = manifest["total_samples"] - month_start_total
            state["completed"] = bool(month_done)
            save_manifest(output_dir, manifest)

            print(
                f"[MONTH DONE] {month} games={consumed} used={state['games_used']} "
                f"samples={state['samples']} completed={state['completed']}",
                flush=True,
            )

            if not args.keep_archives:
                try:
                    archive.unlink()
                    print(f"[CLEAN] removed {archive.name}", flush=True)
                except OSError as exc:
                    print(f"[WARN] could not remove {archive}: {exc}", flush=True)

    finally:
        pool.close()
        pool.join()
        save_manifest(output_dir, manifest)

    elapsed = time.time() - start_time
    print("\n[FINAL]")
    print(f"total_samples={manifest['total_samples']}")
    print(f"shards={len(manifest['shards'])}")
    print(f"target_met={manifest['total_samples'] >= target}")
    print(f"elapsed={elapsed:.1f}s rate={manifest['total_samples'] / max(elapsed, 1e-6):.0f}/s")


if __name__ == "__main__":
    main()
