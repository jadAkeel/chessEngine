from __future__ import annotations

"""Build the push-ready Kaggle payloads for 20M-sample dataset generation.

Produces two directories under ``--build-dir``:

``code/``    a Kaggle *dataset* holding the exact local backend code the kernel
             runs. Packaging the code (instead of cloning GitHub ``main``) keeps
             the kernel pinned to reviewed local sources.
``kernel/``  a Kaggle *kernel* that runs the generator and the verifier.

Neither step contacts Kaggle; run the printed CLI commands once the Kaggle CLI
is authenticated.
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CODE_INCLUDES = ("app", "config", "scripts")
# kaggle_train_external.py --install-requirements reads this from the backend root.
CODE_FILE_INCLUDES = ("requirements2_kaggle.txt",)
CODE_ARCHIVE_STEM = "chess_engine_code"
CODE_EXCLUDES = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "*.pyo", ".pytest_cache", "*.npz", "*.pth", "*.zip", "*.log",
)
KERNEL_SOURCE = ROOT / "kaggle" / "elite_generate" / "elite_dataset_generation.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Kaggle code dataset + generation kernel")
    parser.add_argument("--username", type=str, default="jadakil")
    parser.add_argument("--build-dir", type=str, default=str(ROOT / "kaggle" / "build"))
    parser.add_argument("--code-slug", type=str, default="chess-engine-code")
    parser.add_argument("--kernel-slug", type=str, default="chess-elite-dataset-generation")
    parser.add_argument("--code-title", type=str, default="Chess Engine Code (dataset generation)")
    parser.add_argument("--kernel-title", type=str, default="Chess Elite Dataset Generation")
    parser.add_argument("--resume-from", type=str, default="", help="Optional previous output dataset, e.g. user/slug")
    parser.add_argument(
        "--start-month", type=str, default="",
        help="Month to generate, e.g. 2025-09 (baked into the kernel; Kaggle kernels take no env vars). "
        "Use a separate --kernel-slug per month: a kernel's output is replaced on every run.",
    )
    return parser


def build_code_payload(build_dir: Path, username: str, slug: str, title: str) -> Path:
    """Stage the backend code as a single zip, the shape Kaggle datasets accept.

    ``kaggle datasets create`` skips bare folders, so the tree is archived into
    one file that the kernel extracts itself. That keeps the payload independent
    of Kaggle's archive auto-extraction behaviour.
    """
    code_dir = build_dir / "code"
    if code_dir.exists():
        shutil.rmtree(code_dir)
    code_dir.mkdir(parents=True)

    staging = build_dir / "_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    for name in CODE_INCLUDES:
        source = ROOT / name
        if not source.exists():
            raise FileNotFoundError(f"Missing expected code directory: {source}")
        shutil.copytree(source, staging / name, ignore=CODE_EXCLUDES)
    for name in CODE_FILE_INCLUDES:
        source = ROOT / name
        if not source.exists():
            raise FileNotFoundError(f"Missing expected code file: {source}")
        shutil.copy2(source, staging / name)

    archive = shutil.make_archive(str(code_dir / CODE_ARCHIVE_STEM), "zip", root_dir=staging)
    shutil.rmtree(staging)

    metadata = {
        "title": title,
        "id": f"{username}/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (code_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    size = Path(archive).stat().st_size
    print(f"[CODE] {code_dir} ({Path(archive).name}, {size / 1e6:.2f} MB)")
    return code_dir


def _bake_start_month(script: Path, month: str) -> None:
    if not re.fullmatch(r"\d{4}-\d{2}", month):
        raise ValueError(f"--start-month must look like 2025-09, got {month!r}")
    source = script.read_text(encoding="utf-8")
    pattern = r'START_MONTH = os\.environ\.get\("ELITE_START_MONTH", "[0-9-]+"\)'
    baked, count = re.subn(pattern, f'START_MONTH = os.environ.get("ELITE_START_MONTH", "{month}")', source)
    if count != 1:
        raise RuntimeError("Could not find the START_MONTH default in the kernel script")
    script.write_text(baked, encoding="utf-8")
    print(f"[KERNEL] start_month={month}")


def build_kernel_payload(
    build_dir: Path,
    username: str,
    kernel_slug: str,
    kernel_title: str,
    code_slug: str,
    resume_from: str,
    start_month: str = "",
) -> Path:
    kernel_dir = build_dir / "kernel"
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    kernel_dir.mkdir(parents=True)

    if not KERNEL_SOURCE.exists():
        raise FileNotFoundError(f"Missing kernel script: {KERNEL_SOURCE}")
    shutil.copy2(KERNEL_SOURCE, kernel_dir / KERNEL_SOURCE.name)
    if start_month:
        _bake_start_month(kernel_dir / KERNEL_SOURCE.name, start_month)

    sources = [f"{username}/{code_slug}"]
    if resume_from:
        sources.append(resume_from)

    metadata = {
        "id": f"{username}/{kernel_slug}",
        "title": kernel_title,
        "code_file": KERNEL_SOURCE.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": sources,
        "competition_sources": [],
        "kernel_sources": [],
    }
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"[KERNEL] {kernel_dir}")
    print(f"[KERNEL] dataset_sources={sources}")
    return kernel_dir


def main() -> None:
    args = build_parser().parse_args()
    build_dir = Path(args.build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    code_dir = build_code_payload(build_dir, args.username, args.code_slug, args.code_title)
    kernel_dir = build_kernel_payload(
        build_dir,
        args.username,
        args.kernel_slug,
        args.kernel_title,
        args.code_slug,
        args.resume_from,
        args.start_month,
    )

    print("\n" + "=" * 68)
    print("NEXT STEPS (requires an authenticated Kaggle CLI)")
    print("=" * 68)
    print("1) Publish/refresh the code dataset:")
    print(f"   kaggle datasets create -p {code_dir}")
    print(f"   kaggle datasets version -p {code_dir} -m \"update code\"   # if it exists")
    print("\n2) Push and run the generation kernel:")
    print(f"   kaggle kernels push -p {kernel_dir}")
    print(f"   kaggle kernels status {args.username}/{args.kernel_slug}")
    print(f"   kaggle kernels output {args.username}/{args.kernel_slug} -p <local-dir>")
    print("\n3) Verify locally, then publish the dataset:")
    print("   python scripts/verify_dataset.py --shards-dir <local-dir> --min-samples 20000000")
    print("=" * 68)


if __name__ == "__main__":
    main()
