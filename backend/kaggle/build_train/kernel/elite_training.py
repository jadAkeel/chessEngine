"""Kaggle kernel: resume external training on the generated Lichess Elite dataset.

Runs from the packaged code dataset rather than cloning GitHub, so the kernel
executes the exact code reviewed locally.

Expected attached datasets:
  * the packaged backend code    (chess_engine_code.zip)
  * the generated 20M+ shards    (shard_*.npz + manifest.json)
  * the checkpoint dataset       (external_latest_checkpoint.pth / external_best_model.pth)

Checkpoint and sample discovery are handled by scripts/kaggle_train_external.py,
which rglobs /kaggle/input and prefers external_latest_checkpoint.pth.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

CODE_DIR_NAME = os.environ.get("CHESS_CODE_DIR", "chess-engine-code")
INPUT_ROOT = Path(os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input"))
WORKING_ROOT = Path(os.environ.get("KAGGLE_WORKING_ROOT", "/kaggle/working"))

ITERATIONS = os.environ.get("TRAIN_ITERATIONS", "10")
TRAIN_STEPS_PER_ITER = os.environ.get("TRAIN_STEPS_PER_ITER", "10000")
BUFFER_SIZE = os.environ.get("TRAIN_BUFFER_SIZE", "3000000")
MAX_SAMPLES = os.environ.get("TRAIN_MAX_SAMPLES", "20000000")
BATCH_SIZE = os.environ.get("TRAIN_BATCH_SIZE", "128")
DEVICE = os.environ.get("TRAIN_DEVICE", "cuda")
MIN_DATASET_SAMPLES = int(os.environ.get("TRAIN_MIN_DATASET_SAMPLES", "20000000"))


def _is_code_root(base: Path) -> bool:
    return (base / "app" / "game" / "board_encoding.py").exists() and (base / "scripts").exists()


def find_code_root(root: Path | None = None, extract_to: Path | None = None) -> Path:
    """Locate the packaged backend code, whatever depth Kaggle mounts it at."""
    root = INPUT_ROOT if root is None else Path(root)

    preferred = root / CODE_DIR_NAME
    if preferred.is_dir() and _is_code_root(preferred):
        return preferred

    for marker in sorted(root.rglob("board_encoding.py")):
        base = marker.parent.parent.parent
        if _is_code_root(base):
            return base

    for archive in sorted(root.rglob("chess_engine_code.zip")):
        target = Path(extract_to) if extract_to else WORKING_ROOT / "code"
        target.mkdir(parents=True, exist_ok=True)
        print(f"[CODE] extracting {archive} -> {target}", flush=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
        if _is_code_root(target):
            return target

    raise FileNotFoundError("Could not locate packaged backend code under /kaggle/input.")


def assert_dataset_is_large_enough(root: Path | None = None, minimum: int | None = None) -> int:
    """Refuse to train on a dataset that is not the verified 20M one.

    Guards against silently training on the older 450K shard set.
    """
    root = INPUT_ROOT if root is None else Path(root)
    minimum = MIN_DATASET_SAMPLES if minimum is None else int(minimum)

    best_total = 0
    best_source = None
    for manifest_path in sorted(root.rglob("manifest.json")):
        try:
            with manifest_path.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        total = int(manifest.get("total_samples", 0))
        if total > best_total:
            best_total, best_source = total, manifest_path

    if best_source is None:
        raise FileNotFoundError(
            "No manifest.json found under /kaggle/input; attach the generated dataset."
        )

    print(f"[DATASET] manifest={best_source} total_samples={best_total}", flush=True)
    if best_total < minimum:
        raise SystemExit(
            f"Attached dataset has {best_total} samples, below the required {minimum}. "
            "Attach the verified 20M dataset, not the older shard set."
        )
    return best_total


def assert_checkpoint_attached(root: Path | None = None) -> Path:
    """Refuse to train from scratch: the run must resume the existing model.

    kaggle_train_external.py only warns and starts from random weights when no
    checkpoint is found, which would silently discard the trained model.
    """
    root = INPUT_ROOT if root is None else Path(root)
    for name in ("external_latest_checkpoint.pth", "external_best_model.pth"):
        found = sorted(root.rglob(name))
        if found:
            for path in found:
                print(f"[CHECKPOINT] found {path} ({path.stat().st_size} bytes)", flush=True)
            return found[0]
    raise SystemExit(
        "No external_latest_checkpoint.pth or external_best_model.pth under /kaggle/input; "
        "refusing to train from scratch. Attach the checkpoint dataset."
    )


def ensure_dependencies() -> None:
    """Install what the Kaggle image lacks (it ships without python-chess).

    Must run before kaggle_train_external.py, whose interpreter probe requires
    ``import chess`` to succeed.
    """
    import subprocess

    required = {"chess": "chess", "yaml": "PyYAML", "zstandard": "zstandard"}
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if not missing:
        print("[DEPS] all present", flush=True)
        return
    print(f"[DEPS] installing {missing}", flush=True)
    result = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *missing])
    if result.returncode != 0:
        raise SystemExit(f"pip install failed for {missing}")
    print("[DEPS] installed", flush=True)


def run_script(script: Path, argv: list[str], cwd: Path) -> None:
    """Run a project script in this interpreter (avoids interpreter mismatch)."""
    import runpy

    print(f"[RUN] {script.name} {' '.join(argv)}", flush=True)
    old_argv, old_cwd = sys.argv, os.getcwd()
    sys.argv = [str(script), *argv]
    os.chdir(cwd)
    if str(cwd) not in sys.path:
        sys.path.insert(0, str(cwd))
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise SystemExit(f"{script.name} exited with {exc.code}") from exc
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)


def main() -> None:
    code_root = find_code_root()
    print(f"[CODE] {code_root}", flush=True)

    local_code = WORKING_ROOT / "code"
    if not local_code.exists():
        shutil.copytree(code_root, local_code)
    print(f"[CODE] working copy at {local_code}", flush=True)

    total_samples = assert_dataset_is_large_enough()
    checkpoint = assert_checkpoint_attached()
    print(f"[PLAN] resuming from {checkpoint}", flush=True)
    ensure_dependencies()

    print(
        f"[PLAN] iterations={ITERATIONS} steps_per_iter={TRAIN_STEPS_PER_ITER} "
        f"buffer={BUFFER_SIZE} max_samples={MAX_SAMPLES} device={DEVICE}",
        flush=True,
    )
    print(f"[PLAN] dataset_samples={total_samples}", flush=True)

    run_script(
        local_code / "scripts" / "kaggle_train_external.py",
        [
            "--iterations", ITERATIONS,
            "--train-steps-per-iter", TRAIN_STEPS_PER_ITER,
            "--buffer-size", BUFFER_SIZE,
            "--max-samples", MAX_SAMPLES,
            "--batch-size", BATCH_SIZE,
            "--device", DEVICE,
            "--install-requirements",
            # Keep checkpoints in the kernel output. Versioning the checkpoint
            # dataset here would publish unevaluated weights before the Arena gate.
            "--autosave", "local",
        ],
        cwd=local_code,
    )

    save_dir = WORKING_ROOT / "checkpoints"
    print("\n[KERNEL DONE]", flush=True)
    for name in ("external_latest_checkpoint.pth", "external_best_model.pth", "external_history.json"):
        path = save_dir / name
        state = f"{path.stat().st_size} bytes" if path.exists() else "MISSING"
        print(f"  {name}: {state}", flush=True)


if __name__ == "__main__":
    main()
