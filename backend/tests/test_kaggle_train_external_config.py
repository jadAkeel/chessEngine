from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import yaml

from app.infra.config import load_config
from scripts.kaggle_train_external import _write_config


def _args(tmp_path: Path, **overrides) -> argparse.Namespace:
    defaults = dict(
        samples_path=str(tmp_path / "shards"),
        max_samples=20_000_000,
        min_fullmove=0,
        max_fullmove=0,
        shuffle=True,
        validation_split=0.1,
        seed=42,
        dedup=True,
        filter_invalid=True,
        drop_zero_states=True,
        save_dir=str(tmp_path / "checkpoints"),
        benchmark_games=8,
        buffer_size=3_000_000,
        batch_size=128,
        epochs=2,
        train_steps_per_iter=10_000,
        config_out=str(tmp_path / "external_training_kaggle.yaml"),
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_config_ties_replay_capacity_to_buffer_size(tmp_path: Path):
    """ReplayBuffer preallocates replay.capacity, so it must follow buffer_size."""
    path = _write_config(_args(tmp_path, buffer_size=3_000_000))
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    assert data["training"]["buffer_size"] == 3_000_000
    assert data["replay"]["capacity"] == 3_000_000, (
        "replay.capacity must track buffer_size or the buffer reserves the "
        "default 5M states regardless of the requested size"
    )


@pytest.mark.parametrize("size", [1_000_000, 2_000_000, 3_000_000, 5_000_000])
def test_replay_capacity_follows_any_buffer_size(tmp_path: Path, size: int):
    path = _write_config(_args(tmp_path, buffer_size=size))
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assert data["replay"]["capacity"] == size
    assert data["training"]["buffer_size"] == size


def test_written_config_loads_and_applies_capacity(tmp_path: Path):
    path = _write_config(_args(tmp_path, buffer_size=250_000))
    cfg = load_config(str(path))

    assert int(cfg.replay.capacity) == 250_000
    assert int(cfg.training.buffer_size) == 250_000
    assert int(cfg.training.train_steps_per_iter) == 10_000


def test_written_config_records_training_plan(tmp_path: Path):
    path = _write_config(
        _args(tmp_path, buffer_size=3_000_000, train_steps_per_iter=10_000, max_samples=20_000_000)
    )
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    assert data["external"]["max_samples"] == 20_000_000
    assert data["external"]["validation_split"] == 0.1
    assert data["external"]["dedup"] is True
    assert data["training"]["train_steps_per_iter"] == 10_000
