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


def test_best_model_only_dataset_resumes_instead_of_training_from_scratch(tmp_path: Path):
    """v3 attaches only the v2 iteration-9 best model; it must become the base model."""
    from scripts.kaggle_train_external import _copy_checkpoint_inputs

    source = tmp_path / "input" / "chess-elite-checkpoints"
    source.mkdir(parents=True)
    (source / "external_best_model.pth").write_bytes(b"iter9")
    args = _args(tmp_path, base_model=None, checkpoint_input_dir=str(source))

    base_model = _copy_checkpoint_inputs(args)

    assert base_model == str(Path(args.save_dir) / "external_best_model.pth")
    assert Path(base_model).read_bytes() == b"iter9"


def test_latest_checkpoint_still_preferred_over_best(tmp_path: Path):
    from scripts.kaggle_train_external import _copy_checkpoint_inputs

    source = tmp_path / "input" / "ckpts"
    source.mkdir(parents=True)
    (source / "external_best_model.pth").write_bytes(b"best")
    (source / "external_latest_checkpoint.pth").write_bytes(b"latest")
    args = _args(tmp_path, base_model=None, checkpoint_input_dir=str(source))

    assert Path(_copy_checkpoint_inputs(args)).name == "external_latest_checkpoint.pth"


def _run_main_with_failing_upload(monkeypatch, tmp_path: Path, autosave: str) -> list[int]:
    import sys

    import scripts.kaggle_train_external as kte

    trained = []
    monkeypatch.setattr(kte, "_train_one_iteration", lambda args, cfg, base, env, it: trained.append(it))
    monkeypatch.setattr(kte, "_autosave_local", lambda *a, **k: None)

    def failing_upload(*args, **kwargs):
        raise RuntimeError("Kaggle dataset autosave failed")

    monkeypatch.setattr(kte, "_autosave_kaggle", failing_upload)
    monkeypatch.setattr(sys, "argv", [
        "kaggle_train_external.py", "--iterations", "3", "--device", "cpu",
        "--samples-path", str(tmp_path / "shards"), "--checkpoint-input-dir", "",
        "--save-dir", str(tmp_path / "ckpt"), "--config-out", str(tmp_path / "cfg.yaml"),
        "--autosave", autosave, "--kaggle-dataset-id", "jadakil/chess-elite-checkpoints",
    ])
    kte.main()
    return trained


def test_failed_kaggle_upload_does_not_abort_training_when_local_copy_exists(monkeypatch, tmp_path: Path):
    assert _run_main_with_failing_upload(monkeypatch, tmp_path, "both") == [1, 2, 3]


def test_failed_kaggle_upload_still_fatal_in_kaggle_only_mode(monkeypatch, tmp_path: Path):
    with pytest.raises(RuntimeError, match="autosave failed"):
        _run_main_with_failing_upload(monkeypatch, tmp_path, "kaggle")
