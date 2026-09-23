from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.prepare_kaggle_training import build_training_kernel

ROOT = Path(__file__).resolve().parents[1]
KERNEL_SCRIPT = ROOT / "kaggle" / "elite_train" / "elite_training.py"


def _load_kernel_module():
    spec = importlib.util.spec_from_file_location("elite_train_kernel", KERNEL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_manifest(directory: Path, total: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(
        json.dumps({"total_samples": total, "shards": []}), encoding="utf-8"
    )


# =========================================
# KERNEL METADATA
# =========================================

def test_training_kernel_requests_gpu_and_attaches_all_sources(tmp_path: Path):
    kernel_dir = build_training_kernel(
        tmp_path, "jadakil", "chess-elite-training", "Title",
        "chess-engine-code", "chess-elite-20m", "external-model-checkpoints",
    )
    metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))

    assert metadata["enable_gpu"] is True, "training must run on GPU"
    assert metadata["dataset_sources"] == [
        "jadakil/chess-engine-code",
        "jadakil/chess-elite-20m",
        "jadakil/external-model-checkpoints",
    ]
    assert (kernel_dir / metadata["code_file"]).exists()


# =========================================
# DATASET SIZE GUARD
# =========================================

def test_rejects_dataset_below_threshold(tmp_path: Path):
    """The old 450K shard set must never be mistaken for the 20M dataset."""
    kernel = _load_kernel_module()
    _write_manifest(tmp_path / "input" / "old-shards", 450_000)

    with pytest.raises(SystemExit) as excinfo:
        kernel.assert_dataset_is_large_enough(tmp_path / "input", minimum=20_000_000)
    assert "450000" in str(excinfo.value)


def test_accepts_dataset_at_threshold(tmp_path: Path):
    kernel = _load_kernel_module()
    _write_manifest(tmp_path / "input" / "elite", 21_500_000)

    total = kernel.assert_dataset_is_large_enough(tmp_path / "input", minimum=20_000_000)
    assert total == 21_500_000


def test_picks_largest_manifest_when_several_attached(tmp_path: Path):
    """Checkpoint and code datasets may also carry manifests."""
    kernel = _load_kernel_module()
    _write_manifest(tmp_path / "input" / "old-shards", 450_000)
    _write_manifest(tmp_path / "input" / "elite", 21_500_000)

    total = kernel.assert_dataset_is_large_enough(tmp_path / "input", minimum=20_000_000)
    assert total == 21_500_000


def test_raises_when_no_manifest_attached(tmp_path: Path):
    kernel = _load_kernel_module()
    (tmp_path / "input").mkdir()
    with pytest.raises(FileNotFoundError):
        kernel.assert_dataset_is_large_enough(tmp_path / "input", minimum=20_000_000)


def test_ignores_corrupt_manifest(tmp_path: Path):
    kernel = _load_kernel_module()
    broken = tmp_path / "input" / "broken"
    broken.mkdir(parents=True)
    (broken / "manifest.json").write_text("{not json", encoding="utf-8")
    _write_manifest(tmp_path / "input" / "elite", 20_000_000)

    total = kernel.assert_dataset_is_large_enough(tmp_path / "input", minimum=20_000_000)
    assert total == 20_000_000


# =========================================
# CODE DISCOVERY
# =========================================

@pytest.mark.parametrize(
    "relative",
    ["chess-engine-code", "datasets/jadakil/chess-engine-code"],
)
def test_training_kernel_finds_code_at_any_depth(tmp_path: Path, relative: str):
    kernel = _load_kernel_module()
    base = tmp_path / "input" / relative
    (base / "app" / "game").mkdir(parents=True)
    (base / "app" / "game" / "board_encoding.py").write_text("x", encoding="utf-8")
    (base / "scripts").mkdir()

    assert kernel.find_code_root(tmp_path / "input") == base


def test_training_kernel_keeps_checkpoints_local_until_arena():
    """Autosaving to the checkpoint dataset would publish weights before the Arena gate."""
    source = KERNEL_SCRIPT.read_text(encoding="utf-8")
    assert '"--autosave", "local"' in source
    assert "ensure_dependencies()" in source.split("def main")[1], "deps must install before training"


def test_refuses_to_train_from_scratch(tmp_path: Path):
    kernel = _load_kernel_module()
    (tmp_path / "input" / "chess-elite-21m").mkdir(parents=True)
    with pytest.raises(SystemExit) as excinfo:
        kernel.assert_checkpoint_attached(tmp_path / "input")
    assert "refusing to train from scratch" in str(excinfo.value)


@pytest.mark.parametrize("name", ["external_latest_checkpoint.pth", "external_best_model.pth"])
def test_finds_attached_checkpoint_at_any_depth(tmp_path: Path, name: str):
    kernel = _load_kernel_module()
    ckpt = tmp_path / "input" / "datasets" / "jadakil" / "external-model-checkpoints" / name
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"weights")
    assert kernel.assert_checkpoint_attached(tmp_path / "input") == ckpt


def test_prefers_latest_over_best_checkpoint(tmp_path: Path):
    kernel = _load_kernel_module()
    base = tmp_path / "input" / "external-model-checkpoints"
    base.mkdir(parents=True)
    (base / "external_best_model.pth").write_bytes(b"best")
    (base / "external_latest_checkpoint.pth").write_bytes(b"latest")
    assert kernel.assert_checkpoint_attached(tmp_path / "input").name == "external_latest_checkpoint.pth"
