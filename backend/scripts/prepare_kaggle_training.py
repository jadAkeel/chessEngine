from __future__ import annotations

"""Build the push-ready Kaggle GPU training kernel.

Reuses the code dataset published by scripts/prepare_kaggle_generation.py and
attaches the generated dataset plus the checkpoint dataset, so training resumes
from the newest available checkpoint.
"""

import argparse
import re
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KERNEL_SOURCE = ROOT / "kaggle" / "elite_train" / "elite_training.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the Kaggle GPU training kernel")
    parser.add_argument("--username", type=str, default="jadakil")
    parser.add_argument("--build-dir", type=str, default=str(ROOT / "kaggle" / "build_train"))
    parser.add_argument("--kernel-slug", type=str, default="chess-elite-training")
    parser.add_argument("--kernel-title", type=str, default="Chess Elite Training")
    parser.add_argument("--code-slug", type=str, default="chess-engine-code")
    parser.add_argument(
        "--dataset-slug", type=str, default="",
        help="Generated shard dataset slug, e.g. chess-elite-21m; omit when all months come from --kernel-source",
    )
    parser.add_argument("--checkpoint-slug", type=str, default="chess-elite-checkpoints")
    parser.add_argument(
        "--kernel-source", action="append", default=[],
        help="Kernel whose output is attached as extra shards, e.g. chess-elite-dataset-generation (repeatable)",
    )
    return parser


def _bake_checkpoint_dataset(script: Path, dataset_id: str) -> None:
    """Point autosave at the pushing account's checkpoint dataset (kernels take no env vars)."""
    source = script.read_text(encoding="utf-8")
    pattern = r'CHECKPOINT_DATASET_ID = os\.environ\.get\("TRAIN_CHECKPOINT_DATASET", "[^"]+"\)'
    baked, count = re.subn(
        pattern, f'CHECKPOINT_DATASET_ID = os.environ.get("TRAIN_CHECKPOINT_DATASET", "{dataset_id}")', source
    )
    if count != 1:
        raise RuntimeError("Could not find the CHECKPOINT_DATASET_ID default in the kernel script")
    script.write_text(baked, encoding="utf-8")


def build_training_kernel(
    build_dir: Path,
    username: str,
    kernel_slug: str,
    kernel_title: str,
    code_slug: str,
    dataset_slug: str,
    checkpoint_slug: str,
    kernel_sources: list[str] | None = None,
) -> Path:
    kernel_dir = build_dir / "kernel"
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    kernel_dir.mkdir(parents=True)

    if not KERNEL_SOURCE.exists():
        raise FileNotFoundError(f"Missing kernel script: {KERNEL_SOURCE}")
    shutil.copy2(KERNEL_SOURCE, kernel_dir / KERNEL_SOURCE.name)
    _bake_checkpoint_dataset(kernel_dir / KERNEL_SOURCE.name, f"{username}/{checkpoint_slug}")

    sources = [
        f"{username}/{code_slug}",
        *([f"{username}/{dataset_slug}"] if dataset_slug else []),
        f"{username}/{checkpoint_slug}",
    ]
    if not dataset_slug and not kernel_sources:
        raise ValueError("Attach shards with --dataset-slug or at least one --kernel-source")

    metadata = {
        "id": f"{username}/{kernel_slug}",
        "title": kernel_title,
        "code_file": KERNEL_SOURCE.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": sources,
        "competition_sources": [],
        # A generation kernel's output mounts like a dataset, so a new month
        # can train without a download and re-upload round trip.
        "kernel_sources": [f"{username}/{slug}" for slug in (kernel_sources or [])],
    }
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"[KERNEL] {kernel_dir}")
    print(f"[KERNEL] gpu=True internet=True")
    for source in sources + metadata["kernel_sources"]:
        print(f"[KERNEL]   attached: {source}")
    return kernel_dir


def main() -> None:
    args = build_parser().parse_args()
    build_dir = Path(args.build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    kernel_dir = build_training_kernel(
        build_dir,
        args.username,
        args.kernel_slug,
        args.kernel_title,
        args.code_slug,
        args.dataset_slug,
        args.checkpoint_slug,
        args.kernel_source,
    )

    print("\n" + "=" * 68)
    print("NEXT STEPS")
    print("=" * 68)
    print(f"   kaggle kernels push -p {kernel_dir}")
    print(f"   kaggle kernels status {args.username}/{args.kernel_slug}")
    print(f"   kaggle kernels output {args.username}/{args.kernel_slug} -p <local-dir>")
    print("=" * 68)


if __name__ == "__main__":
    main()
