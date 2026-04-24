from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from app.game.move_encoding import NUM_MOVES
from app.infra.config import AppConfig
from app.training.replay_buffer import PackedPolicy


# =========================================
# DATA STRUCT
# =========================================

@dataclass(frozen=True)
class ExternalSampleLoadResult:
    samples: list[tuple[np.ndarray, PackedPolicy, float]]
    stats: dict[str, int]


# =========================================
# HASH (DEDUP)
# =========================================

def _sample_hash(state: np.ndarray, move_index: int, value: float) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.ascontiguousarray(state, dtype=np.float16).tobytes())
    digest.update(struct.pack('<I', int(move_index)))
    digest.update(struct.pack('<f', float(value)))
    return digest.digest()


# =========================================
# POLICY BUILDER
# =========================================

def _build_policy(idx: int, *, soft_policy: bool = False, rng: np.random.Generator | None = None) -> PackedPolicy:
    if not soft_policy:
        return PackedPolicy(
            indices=np.array([idx], dtype=np.uint16),
            probs=np.array([1.0], dtype=np.float16),
        )

    rng = rng or np.random.default_rng()
    indices = [idx]
    probs = [0.7]

    while len(indices) < 4:
        candidate = int(rng.integers(0, NUM_MOVES))
        if candidate == idx or candidate in indices:
            continue
        indices.append(candidate)
        probs.append(0.1)

    return PackedPolicy(
        indices=np.array(indices, dtype=np.uint16),
        probs=np.array(probs, dtype=np.float16),
    )


# =========================================
# LOAD SINGLE FILE WITH FULL LOGS
# =========================================

def load_external_samples_with_stats(
    path: str | Path,
    cfg: AppConfig,
    *,
    max_samples: int = 0,
) -> ExternalSampleLoadResult:

    path = Path(path)
    print(f"\n[LOAD] file={path.name}")

    data = np.load(path, mmap_mode='r', allow_pickle=False)

    states = data['states']
    policy_indices = np.asarray(data['policy_indices'], dtype=np.int32)
    values = np.asarray(data['values'], dtype=np.float32)

    total_raw = len(states)
    print(f"[LOAD] raw samples={total_raw}")

    external_cfg = getattr(cfg, 'external', None)
    dedup_enabled = bool(getattr(external_cfg, 'dedup', True))
    filter_invalid = bool(getattr(external_cfg, 'filter_invalid', True))

    seen_hashes: set[bytes] = set()

    stats = {
        "accepted": 0,
        "dup": 0,
        "bad_policy": 0,
        "bad_value": 0,
        "bad_state": 0,
    }

    samples: list[tuple[np.ndarray, PackedPolicy, float]] = []

    limit = total_raw if max_samples <= 0 else min(total_raw, max_samples)

    for i in range(limit):
        idx = int(policy_indices[i])
        value = float(values[i])
        state = np.ascontiguousarray(states[i], dtype=np.float16)

        # ===== FILTER =====
        if filter_invalid:
            if idx < 0 or idx >= NUM_MOVES:
                stats["bad_policy"] += 1
                continue

            if not np.isfinite(value):
                stats["bad_value"] += 1
                continue

            if not np.all(np.isfinite(state)) or not np.any(state):
                stats["bad_state"] += 1
                continue

        # ===== DEDUP =====
        if dedup_enabled:
            h = _sample_hash(state, idx, value)
            if h in seen_hashes:
                stats["dup"] += 1
                continue
            seen_hashes.add(h)

        policy = _build_policy(idx)
        samples.append((state, policy, value))
        stats["accepted"] += 1

        # 🔥 PROGRESS LOG
        if i % 20000 == 0 and i > 0:
            print(f"[PROGRESS] {i}/{limit} | accepted={stats['accepted']}")

    print(
        f"[SUMMARY] accepted={stats['accepted']} | "
        f"dup={stats['dup']} | "
        f"bad={stats['bad_policy'] + stats['bad_value'] + stats['bad_state']}"
    )

    return ExternalSampleLoadResult(samples=samples, stats=stats)


# =========================================
# SHARD STREAMING (FINAL VERSION)
# =========================================

def load_external_samples_sharded(
    folder: str | Path,
    cfg: AppConfig,
    *,
    max_samples: int = 0,
) -> Iterator[tuple[np.ndarray, PackedPolicy, float]]:

    folder = Path(folder)
    shard_files = sorted(folder.glob("*.npz"))

    if not shard_files:
        raise FileNotFoundError(f"No shards found in {folder}")

    print(f"\n[SHARDS] total={len(shard_files)}")

    total_streamed = 0

    for shard_id, shard_path in enumerate(shard_files):
        print(f"\n[SHARD] ===== {shard_id+1}/{len(shard_files)} → {shard_path.name} =====")

        result = load_external_samples_with_stats(
            shard_path,
            cfg,
            max_samples=0,  # full shard
        )

        print(f"[SHARD DONE] accepted={result.stats['accepted']}")

        for sample in result.samples:
            yield sample
            total_streamed += 1

            if total_streamed % 50000 == 0:
                print(f"[STREAM] total streamed={total_streamed}")

            if max_samples and total_streamed >= max_samples:
                print(f"[STOP] reached max_samples={max_samples}")
                return

    print(f"\n[FINAL] total streamed={total_streamed}")