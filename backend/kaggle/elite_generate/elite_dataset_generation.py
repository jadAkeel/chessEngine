"""Kaggle kernel: generate and verify the Lichess Elite training dataset.

Runs entirely from a packaged code dataset (``/kaggle/input/<code-slug>``) rather
than cloning GitHub, so the kernel always executes the exact code that was
reviewed locally.

Resume across sessions: attach this kernel's previous output as an extra input
dataset. Any ``manifest.json`` found under ``/kaggle/input`` is copied into the
working directory first, and generation continues from there.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

CODE_DIR_NAME = os.environ.get("CHESS_CODE_DIR", "chess-engine-code")
INPUT_ROOT = Path(os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input"))
WORKING_ROOT = Path(os.environ.get("KAGGLE_WORKING_ROOT", "/kaggle/working"))
OUTPUT_DIR = WORKING_ROOT / "prepared_shards"
WORK_DIR = WORKING_ROOT / "_archives"

START_MONTH = os.environ.get("ELITE_START_MONTH", "2025-11")
MONTHS = int(os.environ.get("ELITE_MONTHS", "4"))
TARGET_SAMPLES = int(os.environ.get("ELITE_TARGET_SAMPLES", "21500000"))
SHARD_SIZE = int(os.environ.get("ELITE_SHARD_SIZE", "125000"))
MIN_SAMPLES = int(os.environ.get("ELITE_MIN_SAMPLES", "20000000"))
MAX_GAMES = os.environ.get("ELITE_MAX_GAMES", "0")  # 0 = unlimited


def describe_input_tree(root: Path, limit: int = 60) -> None:
    """Print what is actually mounted, so attachment problems are diagnosable."""
    print(f"[INPUT] listing {root} (exists={root.exists()})", flush=True)
    if not root.exists():
        return
    shown = 0
    for path in sorted(root.rglob("*")):
        if shown >= limit:
            print(f"[INPUT]   ... (truncated at {limit} entries)", flush=True)
            break
        kind = "d" if path.is_dir() else "f"
        size = "" if path.is_dir() else f" {path.stat().st_size}B"
        print(f"[INPUT]   {kind} {path.relative_to(root)}{size}", flush=True)
        shown += 1


def _is_code_root(base: Path) -> bool:
    return (base / "app" / "game" / "board_encoding.py").exists() and (base / "scripts").exists()


def find_code_root(root: Path | None = None, extract_to: Path | None = None) -> Path:
    """Locate the packaged backend code under /kaggle/input.

    Accepts either an already-extracted tree or the ``chess_engine_code.zip``
    archive produced by scripts/prepare_kaggle_generation.py.
    """
    root = INPUT_ROOT if root is None else Path(root)
    describe_input_tree(root)

    preferred = root / CODE_DIR_NAME
    if preferred.is_dir() and _is_code_root(preferred):
        return preferred

    # Kaggle nests mounts unpredictably (e.g. /kaggle/input/datasets/<user>/<slug>),
    # so anchor on the marker file instead of assuming a directory depth.
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

    raise FileNotFoundError(
        "Could not locate packaged backend code under /kaggle/input. "
        "Attach the code dataset produced by scripts/prepare_kaggle_generation.py."
    )


def restore_previous_output(output_dir: Path, root: Path | None = None) -> None:
    """Copy a previous run's shards into the working dir so generation resumes."""
    root = INPUT_ROOT if root is None else Path(root)
    if not root.exists():
        return

    for manifest in sorted(root.rglob("manifest.json")):
        source = manifest.parent
        if not list(source.glob("shard_*.npz")):
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        if (output_dir / "manifest.json").exists():
            print(f"[RESUME] working dir already populated, ignoring {source}", flush=True)
            return
        print(f"[RESUME] restoring previous shards from {source}", flush=True)
        copied = 0
        for item in sorted(source.iterdir()):
            if item.is_file():
                shutil.copy2(item, output_dir / item.name)
                copied += 1
        print(f"[RESUME] restored {copied} files", flush=True)
        return


def ensure_dependencies() -> None:
    """Install anything the generator needs that the image lacks."""
    required = {"chess": "chess", "numpy": "numpy", "yaml": "PyYAML", "torch": "torch"}
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
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", *missing],
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"pip install failed for {missing}")
    for module in required:
        __import__(module)
    print("[DEPS] installed", flush=True)


def run_script(script: Path, argv: list[str], cwd: Path) -> None:
    """Run a project script in this interpreter.

    A subprocess would pick up a different interpreter than the one holding the
    image's site-packages, so the script is executed in-process instead.
    """
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

    # The packaged tree is read-only; copy it so Python can write bytecode.
    local_code = WORKING_ROOT / "code"
    if not local_code.exists():
        shutil.copytree(code_root, local_code)
    print(f"[CODE] working copy at {local_code}", flush=True)

    restore_previous_output(OUTPUT_DIR)

    workers = os.cpu_count() or 4
    print(f"[ENV] python={sys.executable} cpus={workers}", flush=True)
    ensure_dependencies()

    run_script(
        local_code / "scripts" / "generate_elite_dataset.py",
        [
            "--output-dir", str(OUTPUT_DIR),
            "--work-dir", str(WORK_DIR),
            "--start-month", START_MONTH,
            "--months", str(MONTHS),
            "--target-samples", str(TARGET_SAMPLES),
            "--shard-size", str(SHARD_SIZE),
            "--max-games", MAX_GAMES,
            "--workers", str(workers),
        ],
        cwd=local_code,
    )

    # Archives are deleted per-month by the generator; drop the scratch dir too.
    shutil.rmtree(WORK_DIR, ignore_errors=True)

    run_script(
        local_code / "scripts" / "verify_dataset.py",
        [
            "--shards-dir", str(OUTPUT_DIR),
            "--min-samples", str(MIN_SAMPLES),
        ],
        cwd=local_code,
    )

    shards = sorted(OUTPUT_DIR.glob("shard_*.npz"))
    total_bytes = sum(p.stat().st_size for p in shards)
    print("\n[KERNEL DONE]", flush=True)
    print(f"shard_files={len(shards)}", flush=True)
    print(f"output_bytes={total_bytes} ({total_bytes / 1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
