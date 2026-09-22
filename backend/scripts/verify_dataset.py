from __future__ import annotations

"""Verify a generated shard directory against its manifest.

Checks, in order:

* manifest present, shard ids unique, every listed shard on disk,
* per-shard sha256 and sample count match the manifest,
* array shapes/dtypes match the model contract (N, input_planes, 8, 8),
* policy indices inside [0, NUM_MOVES), values finite and within [-1, 1],
* states finite and non-empty,
* deterministic train/validation split ratio is close to the configured one,
* total sample count meets ``--min-samples``.

Exit code is 0 only when every check passes.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from app.game.move_encoding import NUM_MOVES
from app.infra.config import load_config
from app.training.external_samples import is_validation_position

MANIFEST_NAME = "manifest.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify generated NPZ shards against manifest.json")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--shards-dir", type=str, required=True)
    parser.add_argument("--min-samples", type=int, default=20_000_000)
    parser.add_argument("--split-sample-size", type=int, default=20_000, help="Positions sampled for the split ratio check")
    parser.add_argument("--skip-hashes", action="store_true", help="Skip sha256 re-hashing (faster, weaker)")
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(shards_dir: Path, cfg, *, min_samples: int, split_sample_size: int, skip_hashes: bool) -> tuple[bool, dict]:
    failures: list[str] = []
    manifest_path = shards_dir / MANIFEST_NAME
    if not manifest_path.exists():
        return False, {"failures": [f"missing {manifest_path}"], "total_samples": 0}

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    expected_planes = int(cfg.model.input_planes)
    entries = manifest.get("shards", [])

    ids = [int(entry["id"]) for entry in entries]
    if len(ids) != len(set(ids)):
        failures.append("duplicate shard ids in manifest")

    on_disk = {p.name for p in shards_dir.glob("shard_*.npz")}
    listed = {entry["file"] for entry in entries}
    for orphan in sorted(on_disk - listed):
        failures.append(f"shard on disk but not in manifest: {orphan}")

    total = 0
    counted_months: dict[str, int] = {}
    split_states: list[np.ndarray] = []
    per_shard_stride = max(1, len(entries)) if entries else 1
    want_per_shard = max(1, split_sample_size // per_shard_stride)

    for entry in entries:
        name = entry["file"]
        path = shards_dir / name
        if not path.exists():
            failures.append(f"missing shard file: {name}")
            continue

        if not skip_hashes:
            actual = sha256_file(path)
            if actual != entry.get("sha256"):
                failures.append(f"sha256 mismatch: {name}")

        data = np.load(path, mmap_mode="r", allow_pickle=False)
        states = data["states"]
        policy_indices = np.asarray(data["policy_indices"])
        values = np.asarray(data["values"])

        n = len(states)
        if n != int(entry["samples"]):
            failures.append(f"sample count mismatch in {name}: manifest={entry['samples']} actual={n}")

        if states.ndim != 4 or states.shape[1:] != (expected_planes, 8, 8):
            failures.append(f"bad state shape in {name}: {states.shape}")
        if states.dtype != np.float16:
            failures.append(f"bad states dtype in {name}: {states.dtype}")
        if policy_indices.dtype != np.int32:
            failures.append(f"bad policy_indices dtype in {name}: {policy_indices.dtype}")
        if values.dtype != np.float32:
            failures.append(f"bad values dtype in {name}: {values.dtype}")
        if len(policy_indices) != n or len(values) != n:
            failures.append(f"array length mismatch in {name}")

        if "input_planes" in data and int(np.asarray(data["input_planes"]).reshape(-1)[0]) != expected_planes:
            failures.append(f"input_planes mismatch in {name}")
        if "policy_size" in data and int(np.asarray(data["policy_size"]).reshape(-1)[0]) != NUM_MOVES:
            failures.append(f"policy_size mismatch in {name}")

        if n:
            if int(policy_indices.min()) < 0 or int(policy_indices.max()) >= NUM_MOVES:
                failures.append(f"policy index out of range in {name}")
            finite_values = np.isfinite(values)
            if not bool(finite_values.all()):
                failures.append(f"non-finite values in {name}")
            elif float(np.abs(values).max()) > 1.0:
                failures.append(f"value magnitude > 1 in {name}")

            block = np.asarray(states, dtype=np.float16)
            if not bool(np.isfinite(block).all()):
                failures.append(f"non-finite states in {name}")
            empty = int((~block.any(axis=(1, 2, 3))).sum())
            if empty:
                failures.append(f"{empty} all-zero states in {name}")

            step = max(1, n // want_per_shard)
            split_states.extend(np.ascontiguousarray(states[i]) for i in range(0, n, step))

        total += n
        month = entry.get("month", "unknown")
        counted_months[month] = counted_months.get(month, 0) + n

    if total != int(manifest.get("total_samples", -1)):
        failures.append(f"manifest total_samples={manifest.get('total_samples')} but shards sum to {total}")

    split_ratio = None
    if split_states:
        flags = [bool(is_validation_position(state, cfg)) for state in split_states]
        split_ratio = sum(flags) / len(flags)
        target = float(cfg.external.validation_split)
        # Tolerance follows the binomial standard error so the check stays
        # meaningful on millions of samples without firing on tiny fixtures.
        standard_error = (target * (1.0 - target) / len(flags)) ** 0.5
        tolerance = max(0.02, 3.0 * standard_error)
        if abs(split_ratio - target) > tolerance:
            failures.append(
                f"validation split ratio {split_ratio:.4f} outside {target} +/- {tolerance:.4f}"
            )

        # Determinism: the same state must always land on the same side.
        repeat = [bool(is_validation_position(state, cfg)) for state in split_states[:200]]
        if repeat != flags[:200]:
            failures.append("validation split is not deterministic")

    if total < int(min_samples):
        failures.append(f"total samples {total} < required {min_samples}")

    report = {
        "total_samples": total,
        "shards": len(entries),
        "months": counted_months,
        "validation_split_ratio": split_ratio,
        "generator_version": manifest.get("generator_version"),
        "failures": failures,
    }
    return not failures, report


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    shards_dir = Path(args.shards_dir)

    ok, report = verify(
        shards_dir,
        cfg,
        min_samples=args.min_samples,
        split_sample_size=args.split_sample_size,
        skip_hashes=args.skip_hashes,
    )

    print("=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)
    print(f"dir              : {shards_dir}")
    print(f"generator_version: {report.get('generator_version')}")
    print(f"shards           : {report['shards']}")
    print(f"total_samples    : {report['total_samples']}")
    print(f"required         : {args.min_samples}")
    if report.get("validation_split_ratio") is not None:
        print(f"val split ratio  : {report['validation_split_ratio']:.4f} (target {cfg.external.validation_split})")
    print("samples by month :")
    for month, count in sorted(report["months"].items()):
        print(f"  {month}: {count}")

    if report["failures"]:
        print("\nFAILURES:")
        for failure in report["failures"]:
            print(f"  - {failure}")
        print("\nRESULT: FAILED")
        sys.exit(1)

    print("\nRESULT: PASSED")


if __name__ == "__main__":
    main()
