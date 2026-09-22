from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LICHESS_PUZZLES_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"

GM_COLLECTIONS = {
    "Carlsen": "https://www.pgnmentor.com/players/Carlsen.zip",
    "Kasparov": "https://www.pgnmentor.com/players/Kasparov.zip",
    "Fischer": "https://www.pgnmentor.com/players/Fischer.zip",
    "Tal": "https://www.pgnmentor.com/players/Tal.zip",
    "Anand": "https://www.pgnmentor.com/players/Anand.zip",
    "Karpov": "https://www.pgnmentor.com/players/Karpov.zip",
    "Caruana": "https://www.pgnmentor.com/players/Caruana.zip",
    "Nakamura": "https://www.pgnmentor.com/players/Nakamura.zip",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and prepare high-strength chess training datasets")
    parser.add_argument(
        "--type",
        choices=["puzzles", "gm-games", "all"],
        default="all",
        help="Which dataset to download and prepare (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "data" / "prepared_shards"),
        help="Output directory for generated .npz shards",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default=str(ROOT / "data" / "downloads"),
        help="Temporary download folder",
    )
    parser.add_argument(
        "--max-puzzle-samples",
        type=int,
        default=500000,
        help="Maximum tactical puzzle samples to process (default: 500,000)",
    )
    parser.add_argument(
        "--max-gm-samples",
        type=int,
        default=500000,
        help="Maximum GM game samples to process (default: 500,000)",
    )
    parser.add_argument(
        "--min-puzzle-rating",
        type=int,
        default=1500,
        help="Minimum puzzle rating to include (default: 1500)",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=100000,
        help="Number of samples per .npz shard",
    )
    parser.add_argument(
        "--keep-downloads",
        action="store_true",
        help="Keep downloaded raw files after processing",
    )
    return parser


def _download_file(url: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"[CACHE] Already downloaded: {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.1f} MB)")
        return

    print(f"[DOWNLOAD] Starting download: {url}")
    headers = {"User-Agent": "ChessEngineTrainingDownloader/1.0"}
    req = urllib.request.Request(url, headers=headers)

    start_time = time.time()
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        block_size = 1024 * 1024  # 1 MB

        while True:
            chunk = response.read(block_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = (downloaded / total_size) * 100
                speed = downloaded / (1024 * 1024 * max(time.time() - start_time, 1e-6))
                print(
                    f"\r[DOWNLOAD] {dest_path.name}: {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB "
                    f"({pct:.1f}%) @ {speed:.1f} MB/s",
                    end="",
                    flush=True,
                )
            else:
                print(f"\r[DOWNLOAD] {dest_path.name}: {downloaded / (1024*1024):.1f} MB", end="", flush=True)

    print(f"\n[DOWNLOAD COMPLETE] Saved to {dest_path}")


def process_puzzles(args, download_dir: Path, output_dir: Path) -> None:
    puzzle_file = download_dir / "lichess_db_puzzle.csv.zst"
    _download_file(LICHESS_PUZZLES_URL, puzzle_file)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "import_lichess_puzzles.py"),
        "--input", str(puzzle_file),
        "--output-dir", str(output_dir),
        "--min-rating", str(args.min_puzzle_rating),
        "--max-samples", str(args.max_puzzle_samples),
        "--shard-size", str(args.shard_size),
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    print(f"\n[EXEC] Running puzzle converter: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)


def process_gm_games(args, download_dir: Path, output_dir: Path) -> None:
    gm_raw_dir = download_dir / "gm_pgns"
    gm_raw_dir.mkdir(parents=True, exist_ok=True)

    combined_pgn_path = download_dir / "all_grandmasters.pgn"
    with open(combined_pgn_path, "w", encoding="utf-8") as combined_file:
        for name, url in GM_COLLECTIONS.items():
            zip_dest = download_dir / f"{name}.zip"
            try:
                _download_file(url, zip_dest)
                with zipfile.ZipFile(zip_dest, "r") as zf:
                    for filename in zf.namelist():
                        if filename.endswith(".pgn"):
                            zf.extract(filename, gm_raw_dir)
                            with open(gm_raw_dir / filename, "r", encoding="utf-8", errors="replace") as pgn_f:
                                combined_file.write(pgn_f.read())
                                combined_file.write("\n\n")
            except Exception as exc:
                print(f"[WARN] Failed to process GM {name}: {exc}")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "import_lichess_samples.py"),
        "--input", str(combined_pgn_path),
        "--output-dir", str(output_dir),
        "--max-samples", str(args.max_gm_samples),
        "--shard-size", str(args.shard_size),
        "--min-fullmove", "6",
        "--min-elo", "2200",
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    print(f"\n[EXEC] Running GM PGN converter: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)


def main() -> None:
    args = build_parser().parse_args()
    download_dir = Path(args.download_dir)
    output_dir = Path(args.output_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"CHESS DATASET DOWNLOAD & PREPARATION PIPELINE")
    print(f"Target dataset: {args.type}")
    print(f"Output shards: {output_dir}")
    print("=" * 60)

    if args.type in {"puzzles", "all"}:
        print("\n--- PROCESSING TACTICAL PUZZLES ---")
        process_puzzles(args, download_dir, output_dir)

    if args.type in {"gm-games", "all"}:
        print("\n--- PROCESSING GRANDMASTER GAMES ---")
        process_gm_games(args, download_dir, output_dir)

    if not args.keep_downloads and download_dir.exists():
        print("\n[CLEANUP] Cleaning temporary downloads folder...")
        shutil.rmtree(download_dir, ignore_errors=True)

    shards = list(output_dir.glob("shard_*.npz"))
    print("\n" + "=" * 60)
    print(f"[FINISHED] Total generated shards in {output_dir}: {len(shards)}")
    total_mb = sum(p.stat().st_size for p in shards) / (1024 * 1024)
    print(f"[TOTAL SIZE] {total_mb:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
