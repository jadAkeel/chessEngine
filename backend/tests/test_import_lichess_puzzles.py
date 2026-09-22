from __future__ import annotations

import csv
import io
from pathlib import Path
import tempfile

import chess
import numpy as np
import pytest

from app.game.move_encoding import NUM_MOVES
from app.infra.config import load_config
from scripts.import_lichess_puzzles import _save_shard, build_parser


def test_import_lichess_puzzles_parser():
    parser = build_parser()
    args = parser.parse_args(["--input", "dummy.csv", "--output-dir", "dummy_out", "--min-rating", "1500"])
    assert args.input == "dummy.csv"
    assert args.output_dir == "dummy_out"
    assert args.min_rating == 1500


def test_save_and_load_puzzle_shard(tmp_path: Path):
    cfg = load_config()
    states = [np.zeros((int(cfg.model.input_planes), 8, 8), dtype=np.float16)]
    policy_indices = [42]
    values = [1.0]

    _save_shard(tmp_path, 0, states, policy_indices, values, cfg)
    shard_file = tmp_path / "shard_0.npz"
    assert shard_file.exists()

    data = np.load(shard_file)
    assert data["states"].shape == (1, int(cfg.model.input_planes), 8, 8)
    assert data["policy_indices"][0] == 42
    assert data["values"][0] == 1.0
    assert data["policy_size"][0] == NUM_MOVES


def test_import_lichess_samples_min_elo():
    from scripts.import_lichess_samples import build_parser as build_samples_parser
    parser = build_samples_parser()
    args = parser.parse_args(["--input", "dummy.pgn", "--output-dir", "dummy_out", "--min-elo", "2100"])
    assert args.min_elo == 2100


def test_upload_to_kaggle_packaging(tmp_path: Path):
    from scripts.upload_to_kaggle import create_metadata, create_zip_archive
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    dummy_shard = shards_dir / "shard_0.npz"
    np.savez(dummy_shard, data=np.array([1, 2, 3]))

    meta = create_metadata(shards_dir, "testuser", "test-slug", "Test Title")
    assert meta.exists()

    zip_dest = tmp_path / "upload.zip"
    create_zip_archive(shards_dir, zip_dest)
    assert zip_dest.exists()
    assert zip_dest.stat().st_size > 0
