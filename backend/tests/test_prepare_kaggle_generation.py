from __future__ import annotations

import importlib.util
import zipfile
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from scripts.prepare_kaggle_generation import build_code_payload, build_kernel_payload

ROOT = Path(__file__).resolve().parents[1]
KERNEL_SCRIPT = ROOT / "kaggle" / "elite_generate" / "elite_dataset_generation.py"


def _load_kernel_module():
    spec = importlib.util.spec_from_file_location("elite_kernel", KERNEL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =========================================
# CODE PAYLOAD
# =========================================

def _payload_names(code_dir: Path) -> list[str]:
    with zipfile.ZipFile(code_dir / "chess_engine_code.zip") as zf:
        return zf.namelist()


def test_code_payload_contains_generator_and_encoders(tmp_path: Path):
    code_dir = build_code_payload(tmp_path, "jadakil", "chess-engine-code", "Title")
    names = _payload_names(code_dir)

    for expected in (
        "scripts/generate_elite_dataset.py",
        "scripts/verify_dataset.py",
        "app/game/board_encoding.py",
        "app/game/move_encoding.py",
        "app/infra/config.py",
        "config/default.yaml",
        "requirements2_kaggle.txt",
    ):
        assert expected in names, f"missing {expected} from payload"

    metadata = json.loads((code_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"] == "jadakil/chess-engine-code"


def test_code_payload_is_a_single_uploadable_file(tmp_path: Path):
    """kaggle datasets create skips bare folders, so the payload must be a file."""
    code_dir = build_code_payload(tmp_path, "jadakil", "chess-engine-code", "Title")
    entries = sorted(p.name for p in code_dir.iterdir())
    assert entries == ["chess_engine_code.zip", "dataset-metadata.json"]
    assert not any(p.is_dir() for p in code_dir.iterdir())


def test_code_payload_excludes_heavy_artifacts(tmp_path: Path):
    code_dir = build_code_payload(tmp_path, "jadakil", "chess-engine-code", "Title")
    names = _payload_names(code_dir)
    for suffix in (".npz", ".pth", ".zip", ".pyc"):
        assert not [n for n in names if n.endswith(suffix)], f"payload must not ship {suffix}"
    assert not [n for n in names if "__pycache__" in n]


def test_code_payload_matches_local_sources(tmp_path: Path):
    """The kernel must run reviewed local code, not a stale GitHub checkout."""
    code_dir = build_code_payload(tmp_path, "jadakil", "chess-engine-code", "Title")
    with zipfile.ZipFile(code_dir / "chess_engine_code.zip") as zf:
        for relative in (
            "scripts/generate_elite_dataset.py",
            "app/game/board_encoding.py",
            "app/game/move_encoding.py",
        ):
            assert zf.read(relative) == (ROOT / relative).read_bytes()


# =========================================
# KERNEL PAYLOAD
# =========================================

def test_kernel_metadata_enables_internet_and_attaches_code(tmp_path: Path):
    kernel_dir = build_kernel_payload(
        tmp_path, "jadakil", "chess-elite-dataset-generation", "Title", "chess-engine-code", ""
    )
    metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))

    assert metadata["id"] == "jadakil/chess-elite-dataset-generation"
    assert metadata["enable_internet"] is True, "generation must download Lichess archives"
    assert metadata["kernel_type"] == "script"
    assert metadata["dataset_sources"] == ["jadakil/chess-engine-code"]
    assert (kernel_dir / metadata["code_file"]).exists()


def test_kernel_metadata_can_attach_resume_source(tmp_path: Path):
    kernel_dir = build_kernel_payload(
        tmp_path,
        "jadakil",
        "chess-elite-dataset-generation",
        "Title",
        "chess-engine-code",
        "jadakil/previous-output",
    )
    metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["dataset_sources"] == ["jadakil/chess-engine-code", "jadakil/previous-output"]


# =========================================
# KERNEL RUNTIME HELPERS
# =========================================

def test_kernel_extracts_zipped_payload(tmp_path: Path):
    """Mirrors Kaggle: the attached dataset holds chess_engine_code.zip."""
    kernel = _load_kernel_module()
    code_dir = build_code_payload(tmp_path / "build", "jadakil", "chess-engine-code", "Title")

    fake_input = tmp_path / "input" / "chess-engine-code"
    fake_input.mkdir(parents=True)
    shutil.copy2(code_dir / "chess_engine_code.zip", fake_input / "chess_engine_code.zip")

    found = kernel.find_code_root(tmp_path / "input", extract_to=tmp_path / "working" / "code")
    assert (found / "app" / "game" / "board_encoding.py").exists()
    assert (found / "scripts" / "generate_elite_dataset.py").exists()
    assert (found / "config" / "default.yaml").exists()


def _fake_extracted(base: Path) -> Path:
    (base / "app" / "game").mkdir(parents=True)
    (base / "app" / "game" / "board_encoding.py").write_text("x", encoding="utf-8")
    (base / "scripts").mkdir()
    return base


def test_kernel_finds_already_extracted_code(tmp_path: Path):
    kernel = _load_kernel_module()
    extracted = _fake_extracted(tmp_path / "input" / "chess-engine-code")
    assert kernel.find_code_root(tmp_path / "input") == extracted


@pytest.mark.parametrize(
    "relative",
    [
        "chess-engine-code",
        "datasets/jadakil/chess-engine-code",
        "some/deeply/nested/mount/chess-engine-code",
    ],
)
def test_kernel_finds_code_at_any_mount_depth(tmp_path: Path, relative: str):
    """Kaggle nests dataset mounts, e.g. /kaggle/input/datasets/<user>/<slug>."""
    kernel = _load_kernel_module()
    extracted = _fake_extracted(tmp_path / "input" / relative)
    assert kernel.find_code_root(tmp_path / "input") == extracted


def test_kernel_raises_when_code_missing(tmp_path: Path):
    kernel = _load_kernel_module()
    empty = tmp_path / "input"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        kernel.find_code_root(empty)


def test_kernel_restores_previous_shards_for_resume(tmp_path: Path):
    kernel = _load_kernel_module()

    previous = tmp_path / "input" / "previous-output"
    previous.mkdir(parents=True)
    np.savez_compressed(previous / "shard_00000.npz", states=np.zeros((1, 20, 8, 8), dtype=np.float16))
    (previous / "manifest.json").write_text(json.dumps({"total_samples": 1}), encoding="utf-8")

    output = tmp_path / "working" / "prepared_shards"
    kernel.restore_previous_output(output, tmp_path / "input")

    assert (output / "manifest.json").exists()
    assert (output / "shard_00000.npz").exists()


def test_kernel_restore_is_noop_without_previous_output(tmp_path: Path):
    kernel = _load_kernel_module()
    root = tmp_path / "input"
    (root / "chess-engine-code").mkdir(parents=True)

    output = tmp_path / "working" / "prepared_shards"
    kernel.restore_previous_output(output, root)
    assert not output.exists()
