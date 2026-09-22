from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from app.game.move_encoding import NUM_MOVES
from app.infra.config import load_config
from scripts.generate_elite_dataset import (
    REJECTION_KEYS,
    ShardBuffer,
    _worker_init,
    build_parser,
    encode_games,
    iter_months,
    iter_raw_games,
    load_manifest,
    new_manifest,
    save_manifest,
)
from scripts.verify_dataset import verify


TWO_GAMES = """[Event "Rated Blitz game"]
[Site "https://lichess.org/abcd1234"]
[White "alpha"]
[Black "beta"]
[Result "1-0"]
[WhiteElo "2450"]
[BlackElo "2430"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 1-0

[Event "Rated Blitz game"]
[Site "https://lichess.org/efgh5678"]
[White "gamma"]
[Black "delta"]
[Result "0-1"]
[WhiteElo "2500"]
[BlackElo "2520"]

1. d4 Nf6 2. c4 e6 3. Nc3 Bb4 4. e3 O-O 5. Bd3 d5 6. Nf3 c5 7. O-O Nc6 8. a3 Bxc3 0-1
"""


def _args(**overrides):
    argv = ["--output-dir", "out"]
    for key, value in overrides.items():
        argv.extend([f"--{key.replace('_', '-')}", str(value)])
    return build_parser().parse_args(argv)


# =========================================
# PARSER / PLANNING
# =========================================

def test_parser_has_no_hidden_game_cap():
    args = _args()
    assert args.max_games == 0, "max_games must default to unlimited, not a hidden cap"
    assert args.target_samples == 20_000_000
    assert 100_000 <= args.shard_size <= 250_000


def test_iter_months_walks_backwards_over_year_boundary():
    assert iter_months("2025-02", 4) == ["2025-02", "2025-01", "2024-12", "2024-11"]
    assert len(iter_months("2025-11", 12)) == 12


# =========================================
# PGN STREAMING
# =========================================

def test_iter_raw_games_splits_on_game_boundaries():
    games = list(iter_raw_games(io.StringIO(TWO_GAMES)))
    assert len(games) == 2
    assert games[0].count("[Event ") == 1
    assert "abcd1234" in games[0]
    assert "efgh5678" in games[1]
    assert games[1].rstrip().endswith("0-1")


def test_iter_raw_games_handles_single_game():
    single = TWO_GAMES.split("[Event", 2)[1]
    games = list(iter_raw_games(io.StringIO("[Event" + single)))
    assert len(games) == 1


# =========================================
# ENCODING
# =========================================

def test_encode_games_produces_valid_samples():
    cfg = load_config()
    _worker_init(
        "config/default.yaml",
        {"min_fullmove": 5, "max_fullmove": 0, "min_elo": 0, "skip_draws": False},
    )
    states, policies, values, stats = encode_games(list(iter_raw_games(io.StringIO(TWO_GAMES))))

    planes = int(cfg.model.input_planes)
    assert states.shape[1:] == (planes, 8, 8)
    assert states.dtype == np.float16
    assert policies.dtype == np.int32
    assert values.dtype == np.float32
    assert len(states) == len(policies) == len(values) > 0

    assert int(policies.min()) >= 0 and int(policies.max()) < NUM_MOVES
    assert np.isfinite(values).all()
    assert float(np.abs(values).max()) <= 1.0
    assert np.isfinite(states).all()
    assert states.any(axis=(1, 2, 3)).all(), "no all-zero states"
    assert stats["games_used"] == 2


def test_encode_games_value_targets_are_not_all_positive():
    """Guards against the '+1 for everything' value-supervision failure."""
    _worker_init(
        "config/default.yaml",
        {"min_fullmove": 5, "max_fullmove": 0, "min_elo": 0, "skip_draws": False},
    )
    _, _, values, _ = encode_games(list(iter_raw_games(io.StringIO(TWO_GAMES))))
    assert set(np.unique(values)).issubset({-1.0, 0.0, 1.0})
    assert (values > 0).any() and (values < 0).any()


def test_encode_games_min_elo_filter_rejects_games():
    _worker_init(
        "config/default.yaml",
        {"min_fullmove": 5, "max_fullmove": 0, "min_elo": 4000, "skip_draws": False},
    )
    states, _, _, stats = encode_games(list(iter_raw_games(io.StringIO(TWO_GAMES))))
    assert len(states) == 0
    assert stats["low_elo"] == 2


# =========================================
# SHARDING / MANIFEST / RESUME
# =========================================

def _fill(buffer: ShardBuffer, count: int, month: str, planes: int, start: int = 0) -> None:
    states = np.ones((count, planes, 8, 8), dtype=np.float16)
    policies = np.arange(start, start + count, dtype=np.int32) % NUM_MOVES
    values = np.zeros((count,), dtype=np.float32)
    buffer.add(states, policies, values, month)


def test_shard_buffer_flushes_and_records_manifest(tmp_path: Path):
    cfg = load_config()
    planes = int(cfg.model.input_planes)
    manifest = new_manifest(cfg, _args(shard_size=10))
    buffer = ShardBuffer(tmp_path, manifest, cfg, shard_size=10)

    _fill(buffer, 25, "2025-11", planes)
    assert len(manifest["shards"]) == 2
    assert manifest["total_samples"] == 20

    buffer.flush("2025-11")
    assert len(manifest["shards"]) == 3
    assert manifest["total_samples"] == 25

    ids = [entry["id"] for entry in manifest["shards"]]
    assert ids == [0, 1, 2]
    for entry in manifest["shards"]:
        path = tmp_path / entry["file"]
        assert path.exists()
        assert not path.name.endswith(".building")
        data = np.load(path, allow_pickle=False)
        assert len(data["states"]) == entry["samples"]
        assert int(data["policy_size"][0]) == NUM_MOVES
        assert int(data["input_planes"][0]) == planes

    assert sum(entry["samples"] for entry in manifest["shards"]) == manifest["total_samples"]
    assert not list(tmp_path.glob("*.building*"))


def test_shard_buffer_resume_does_not_reuse_ids(tmp_path: Path):
    cfg = load_config()
    planes = int(cfg.model.input_planes)
    manifest = new_manifest(cfg, _args(shard_size=10))

    first = ShardBuffer(tmp_path, manifest, cfg, shard_size=10)
    _fill(first, 20, "2025-11", planes)
    save_manifest(tmp_path, manifest)
    assert [e["id"] for e in manifest["shards"]] == [0, 1]

    reloaded = load_manifest(tmp_path, cfg, _args(shard_size=10))
    second = ShardBuffer(tmp_path, reloaded, cfg, shard_size=10)
    assert second.next_id == 2

    _fill(second, 10, "2025-10", planes)
    ids = [entry["id"] for entry in reloaded["shards"]]
    assert ids == [0, 1, 2]
    assert len(ids) == len(set(ids))
    assert reloaded["total_samples"] == 30


def test_manifest_round_trip_records_required_metadata(tmp_path: Path):
    cfg = load_config()
    manifest = new_manifest(cfg, _args())
    save_manifest(tmp_path, manifest)

    with (tmp_path / "manifest.json").open(encoding="utf-8") as handle:
        stored = json.load(handle)

    assert stored["generator_version"]
    assert stored["source"] == "lichess-elite"
    assert stored["state_shape"] == [int(cfg.model.input_planes), 8, 8]
    assert stored["policy_size"] == NUM_MOVES
    assert stored["dtypes"] == {
        "states": "float16",
        "policy_indices": "int32",
        "values": "float32",
    }
    assert set(stored["rejected"]) == set(REJECTION_KEYS)
    assert "min_fullmove" in stored["filters"]
    assert stored["updated_utc"]


# =========================================
# VERIFICATION
# =========================================

def _build_dataset(tmp_path: Path, cfg, samples: int = 40, shard_size: int = 10) -> dict:
    manifest = new_manifest(cfg, _args(shard_size=shard_size))
    buffer = ShardBuffer(tmp_path, manifest, cfg, shard_size=shard_size)
    planes = int(cfg.model.input_planes)

    rng = np.random.default_rng(0)
    states = rng.integers(0, 2, size=(samples, planes, 8, 8)).astype(np.float16)
    states[:, 0, 0, 0] = 1.0  # guarantee no all-zero state
    policies = rng.integers(0, NUM_MOVES, size=samples).astype(np.int32)
    values = rng.choice([-1.0, 0.0, 1.0], size=samples).astype(np.float32)

    buffer.add(states, policies, values, "2025-11")
    buffer.flush("2025-11")
    save_manifest(tmp_path, manifest)
    return manifest


def test_verify_passes_on_clean_dataset(tmp_path: Path):
    cfg = load_config()
    _build_dataset(tmp_path, cfg)
    ok, report = verify(tmp_path, cfg, min_samples=40, split_sample_size=40, skip_hashes=False)
    assert ok, report["failures"]
    assert report["total_samples"] == 40
    assert report["months"] == {"2025-11": 40}


def test_verify_enforces_minimum_sample_count(tmp_path: Path):
    cfg = load_config()
    _build_dataset(tmp_path, cfg)
    ok, report = verify(tmp_path, cfg, min_samples=20_000_000, split_sample_size=40, skip_hashes=True)
    assert not ok
    assert any("< required" in failure for failure in report["failures"])


def test_verify_detects_tampered_shard(tmp_path: Path):
    cfg = load_config()
    manifest = _build_dataset(tmp_path, cfg)

    victim = tmp_path / manifest["shards"][0]["file"]
    planes = int(cfg.model.input_planes)
    np.savez_compressed(
        victim,
        states=np.ones((10, planes, 8, 8), dtype=np.float16),
        policy_indices=np.zeros((10,), dtype=np.int32),
        values=np.zeros((10,), dtype=np.float32),
        input_planes=np.asarray([planes], dtype=np.int16),
        policy_size=np.asarray([NUM_MOVES], dtype=np.int32),
    )

    ok, report = verify(tmp_path, cfg, min_samples=40, split_sample_size=40, skip_hashes=False)
    assert not ok
    assert any("sha256 mismatch" in failure for failure in report["failures"])


def test_verify_detects_out_of_range_policy(tmp_path: Path):
    cfg = load_config()
    planes = int(cfg.model.input_planes)
    manifest = new_manifest(cfg, _args(shard_size=10))
    buffer = ShardBuffer(tmp_path, manifest, cfg, shard_size=10)

    states = np.ones((10, planes, 8, 8), dtype=np.float16)
    policies = np.full((10,), NUM_MOVES + 5, dtype=np.int32)
    values = np.zeros((10,), dtype=np.float32)
    buffer.add(states, policies, values, "2025-11")
    buffer.flush("2025-11")
    save_manifest(tmp_path, manifest)

    ok, report = verify(tmp_path, cfg, min_samples=1, split_sample_size=10, skip_hashes=True)
    assert not ok
    assert any("policy index out of range" in failure for failure in report["failures"])


def test_verify_detects_missing_shard_file(tmp_path: Path):
    cfg = load_config()
    manifest = _build_dataset(tmp_path, cfg)
    (tmp_path / manifest["shards"][1]["file"]).unlink()

    ok, report = verify(tmp_path, cfg, min_samples=1, split_sample_size=40, skip_hashes=True)
    assert not ok
    assert any("missing shard file" in failure for failure in report["failures"])


def test_verify_requires_manifest(tmp_path: Path):
    cfg = load_config()
    ok, report = verify(tmp_path, cfg, min_samples=1, split_sample_size=10, skip_hashes=True)
    assert not ok
    assert any("manifest" in failure for failure in report["failures"])
