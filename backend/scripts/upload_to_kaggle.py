from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package and upload chess shards to Kaggle Datasets")
    parser.add_argument(
        "--shards-dir",
        type=str,
        default=str(ROOT / "data" / "prepared_shards"),
        help="Folder containing generated shard_*.npz files",
    )
    parser.add_argument(
        "--dataset-slug",
        type=str,
        default="chess-tactical-training-shards",
        help="Kaggle dataset URL slug",
    )
    parser.add_argument(
        "--dataset-title",
        type=str,
        default="Chess Tactical and Grandmaster Training Shards",
        help="Title of the Kaggle dataset",
    )
    parser.add_argument(
        "--username",
        type=str,
        default="jadakil",
        help="Kaggle username",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make dataset public (default is private)",
    )
    parser.add_argument(
        "--zip-output",
        type=str,
        default=str(ROOT / "data" / "kaggle_upload_shards.zip"),
        help="Destination path for packaged ZIP file",
    )
    return parser


def check_kaggle_cli() -> bool:
    kaggle_bin = shutil.which("kaggle")
    if not kaggle_bin:
        return False
    try:
        res = subprocess.run([kaggle_bin, "datasets", "list", "--mine"], capture_output=True, text=True)
        return res.returncode == 0
    except Exception:
        return False


def create_metadata(shards_dir: Path, username: str, slug: str, title: str) -> Path:
    metadata_path = shards_dir / "dataset-metadata.json"
    metadata = {
        "title": title,
        "id": f"{username}/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[METADATA] Created {metadata_path} for dataset ID: {username}/{slug}")
    return metadata_path


def create_zip_archive(shards_dir: Path, zip_dest: Path) -> None:
    zip_dest.parent.mkdir(parents=True, exist_ok=True)
    shards = list(shards_dir.glob("*.npz"))
    if not shards:
        raise FileNotFoundError(f"No .npz shard files found in {shards_dir}")

    print(f"[PACKAGING] Creating ZIP archive with {len(shards)} shards...")
    with zipfile.ZipFile(zip_dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        metadata_file = shards_dir / "dataset-metadata.json"
        if metadata_file.exists():
            zf.write(metadata_file, arcname="dataset-metadata.json")
        for shard in shards:
            zf.write(shard, arcname=shard.name)

    size_mb = zip_dest.stat().st_size / (1024 * 1024)
    print(f"[ZIP READY] Created {zip_dest} ({size_mb:.2f} MB)")


def main() -> None:
    args = build_parser().parse_args()
    shards_dir = Path(args.shards_dir)

    if not shards_dir.exists():
        raise FileNotFoundError(f"Shards directory does not exist: {shards_dir}")

    shards = list(shards_dir.glob("*.npz"))
    if not shards:
        print(f"[ERROR] No shard_*.npz files found in {shards_dir}")
        print("Run scripts/download_and_prepare_dataset.py first to generate shards.")
        return

    print("=" * 60)
    print("KAGGLE DATASET PACKAGING & UPLOAD TOOL")
    print(f"Shards directory: {shards_dir} ({len(shards)} shards)")
    print(f"Target Kaggle ID: {args.username}/{args.dataset_slug}")
    print("=" * 60)

    # 1. Create dataset-metadata.json
    create_metadata(shards_dir, args.username, args.dataset_slug, args.dataset_title)

    # 2. Create standalone ZIP file for easy manual drag-and-drop
    zip_path = Path(args.zip_output)
    create_zip_archive(shards_dir, zip_path)

    # 3. Check if Kaggle CLI is authenticated
    cli_ready = check_kaggle_cli()

    if cli_ready:
        print("\n[KAGGLE CLI] Detected authenticated Kaggle CLI!")
        print("[KAGGLE CLI] Uploading dataset automatically...")
        cmd = ["kaggle", "datasets", "create", "-p", str(shards_dir), "--dir-mode", "zip"]
        if args.public:
            cmd.append("--public")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print("\n" + "=" * 60)
            print("[SUCCESS] Dataset created successfully on Kaggle!")
            print(f"View online: https://www.kaggle.com/datasets/{args.username}/{args.dataset_slug}")
            print("=" * 60)
            return
        else:
            if "already exists" in res.stderr or "already exists" in res.stdout:
                print("[INFO] Dataset already exists on Kaggle. Creating a new version...")
                v_cmd = [
                    "kaggle", "datasets", "version",
                    "-p", str(shards_dir),
                    "-m", "Add new high-strength training shards",
                    "--dir-mode", "zip",
                ]
                v_res = subprocess.run(v_cmd, capture_output=True, text=True)
                if v_res.returncode == 0:
                    print("\n" + "=" * 60)
                    print("[SUCCESS] Dataset updated with a new version successfully!")
                    print(f"View online: https://www.kaggle.com/datasets/{args.username}/{args.dataset_slug}")
                    print("=" * 60)
                    return
                else:
                    print(f"[WARN] Version update output: {v_res.stdout}\n{v_res.stderr}")
            else:
                print(f"[WARN] Kaggle upload output: {res.stdout}\n{res.stderr}")

    # If Kaggle CLI is not configured or failed, show clear, easy instructions
    print("\n" + "=" * 60)
    print("HOW TO UPLOAD TO KAGGLE (2 SIMPLE OPTIONS):")
    print("=" * 60)
    print("OPTION 1: Web Interface (Easiest - 1 minute)")
    print("  1. Go to https://www.kaggle.com/datasets")
    print("  2. Click 'New Dataset' in the top right.")
    print(f"  3. Set Dataset Title: '{args.dataset_title}'")
    print(f"  4. Drag and drop the packaged ZIP file:")
    print(f"     -> {zip_path.resolve()}")
    print("  5. Click 'Create'!")
    print("")
    print("OPTION 2: One-Click CLI")
    print("  1. On Kaggle.com -> Account Settings -> Scroll to 'API' -> 'Create New Token'.")
    print("  2. Save kaggle.json to C:\\Users\\10User\\.kaggle\\kaggle.json")
    print("  3. Run this script again to auto-upload!")
    print("=" * 60)


if __name__ == "__main__":
    main()
