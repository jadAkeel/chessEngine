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

# The user asked for ~14 h of training; a Kaggle session is capped at 12 h, so it
# runs as two ~7 h sessions (the second continues from the first's latest). The
# budget stops cleanly before an iteration that would overrun it, so ITERATIONS
# is only an upper bound.
ITERATIONS = os.environ.get("TRAIN_ITERATIONS", "40")
TIME_BUDGET_HOURS = os.environ.get("TRAIN_TIME_BUDGET_HOURS", "6.8")
# x2 epochs = 20k steps per 3M-sample buffer. Measured on 2x T4 with the O(batch)
# sampler: 0.074 s/step (0.216 before), so ~25 min of training per ~8 min refill.
TRAIN_STEPS_PER_ITER = os.environ.get("TRAIN_STEPS_PER_ITER", "10000")
BUFFER_SIZE = os.environ.get("TRAIN_BUFFER_SIZE", "3000000")
MAX_SAMPLES = os.environ.get("TRAIN_MAX_SAMPLES", "50000000")
BATCH_SIZE = os.environ.get("TRAIN_BATCH_SIZE", "128")
DEVICE = os.environ.get("TRAIN_DEVICE", "cuda")
# fp16 AMP overflowed once trunk activations reached ~18k (v2 iter 10 and v3
# went NaN), so v4 trains in fp32 with a lower learning rate.
LR = os.environ.get("TRAIN_LR", "0.0002")
USE_AMP = os.environ.get("TRAIN_USE_AMP", "0") == "1"
# "latest" continues from the checkpoint dataset's newest iteration; "best" starts
# from its best model (session 1 of the 14 h run: mrj v1 iter 6, as the user chose).
START_FROM = os.environ.get("TRAIN_START_FROM", "best")
# "both" = zip in the kernel output + a new version of the checkpoint dataset
# after every iteration. Old versions are kept so any iteration can be picked.
AUTOSAVE = os.environ.get("TRAIN_AUTOSAVE", "both")
# v3 resumes from the v2 iteration-9 model and versions this dataset, leaving
# the original external-model-checkpoints untouched.
CHECKPOINT_DATASET_ID = os.environ.get("TRAIN_CHECKPOINT_DATASET", "jadakil/chess-elite-checkpoints")
MIN_DATASET_SAMPLES = int(os.environ.get("TRAIN_MIN_DATASET_SAMPLES", "20000000"))


def _is_code_root(base: Path) -> bool:
    return (base / "app" / "game" / "board_encoding.py").exists() and (base / "scripts").exists()


def _code_root_rank(path: Path) -> tuple[int, int, str]:
    return (0 if CODE_DIR_NAME in path.parts else 1, len(path.parts), str(path))


def find_code_root(root: Path | None = None, extract_to: Path | None = None) -> Path:
    """Locate the packaged backend code, whatever depth Kaggle mounts it at."""
    root = INPUT_ROOT if root is None else Path(root)

    preferred = root / CODE_DIR_NAME
    if preferred.is_dir() and _is_code_root(preferred):
        return preferred

    # Prefer the code dataset: an attached kernel output (e.g. the generation
    # kernel) carries its own, possibly stale, copy under .../code. v4 ran that
    # stale copy because it sorts first alphabetically.
    bases = [m.parent.parent.parent for m in root.rglob("board_encoding.py")]
    for base in sorted(bases, key=_code_root_rank):
        if _is_code_root(base):
            return base

    for archive in sorted(root.rglob("chess_engine_code.zip"), key=_code_root_rank):
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


def combine_shard_dirs(root: Path | None = None, out_dir: Path | None = None) -> tuple[Path, int]:
    """Expose every attached shard set (one per month) as one flat folder.

    The trainer reads a single folder, and each generated set names its shards
    shard_00000.npz..., so they are symlinked under a per-set prefix. Returns
    the folder and the summed manifest sample count.
    """
    root = INPUT_ROOT if root is None else Path(root)
    out_dir = WORKING_ROOT / "combined_shards" if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for index, manifest_path in enumerate(sorted(root.rglob("manifest.json"))):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        shards = sorted(manifest_path.parent.glob("shard_*.npz"))
        if not shards:
            continue
        months = "_".join(sorted(manifest.get("months", {}))) or f"set{index}"
        for shard in shards:
            link = out_dir / f"{months}_{shard.name}"
            if not link.exists():
                try:
                    link.symlink_to(shard)
                except OSError:  # Windows without symlink privilege (local tests)
                    os.link(shard, link)
        total += int(manifest.get("total_samples", 0))
        print(f"[DATASET] + {months}: {len(shards)} shards, {manifest.get('total_samples')} samples", flush=True)

    if not total:
        raise FileNotFoundError("No generated shard sets (manifest.json + shard_*.npz) under /kaggle/input")
    print(f"[DATASET] combined {total} samples in {out_dir}", flush=True)
    return out_dir, total


def find_best_checkpoint(root: Path | None = None) -> Path | None:
    """The best model to start from; the dataset's latest may be a worse later iteration."""
    root = INPUT_ROOT if root is None else Path(root)
    found = sorted(root.rglob("external_best_model.pth"))
    return found[0] if found else None


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

    assert_dataset_is_large_enough()
    samples_dir, total_samples = combine_shard_dirs()
    assert_checkpoint_attached()
    if START_FROM == "best":
        checkpoint = find_best_checkpoint()
        print(f"[PLAN] starting from best model {checkpoint}", flush=True)
    else:
        # No --base-model: the wrapper resumes from the dataset's latest checkpoint.
        checkpoint = None
        print("[PLAN] starting from the latest checkpoint in the checkpoint dataset", flush=True)
    ensure_dependencies()

    print(
        f"[PLAN] iterations={ITERATIONS} steps_per_iter={TRAIN_STEPS_PER_ITER} "
        f"buffer={BUFFER_SIZE} max_samples={MAX_SAMPLES} device={DEVICE} "
        f"lr={LR} amp={USE_AMP} budget={TIME_BUDGET_HOURS}h autosave={AUTOSAVE} -> {CHECKPOINT_DATASET_ID}",
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
            "--autosave", AUTOSAVE,
            "--autosave-every", "1",
            "--kaggle-dataset-id", CHECKPOINT_DATASET_ID,
            # Old dataset versions are deliberately kept so every iteration can be picked.
            "--samples-path", str(samples_dir),
            "--lr", LR,
            "--time-budget-hours", TIME_BUDGET_HOURS,
            *([] if USE_AMP else ["--no-amp"]),
            *(["--base-model", str(checkpoint)] if checkpoint else []),
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
