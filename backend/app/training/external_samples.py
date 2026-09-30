from __future__ import annotations

import hashlib
import struct
import random  # 🔥 أضفناها
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator

import numpy as np

from app.game.board_encoding import FULLMOVE_NUMBER_PLANE
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


def is_validation_position(state: np.ndarray, cfg: AppConfig) -> bool:
    """Stable position groups, independent of labels, clocks and file order.

    Mirror equivalents stay together because training can mirror a position.
    Existing shards have no game IDs, so this is a position split, not a game split.
    """
    position = np.ascontiguousarray(state[:18], dtype=np.float16)
    mirrored = np.flip(position, axis=2).copy()
    mirrored[[13, 14, 15, 16]] = mirrored[[14, 13, 16, 15]]
    canonical = min(position.tobytes(), mirrored.tobytes())
    digest = hashlib.blake2b(digest_size=8, person=b'chess-split-v1')
    digest.update(str(cfg.external.seed).encode('ascii'))
    digest.update(canonical)
    return int.from_bytes(digest.digest(), 'big') / 2**64 < cfg.external.validation_split


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


def _multi_move_policy(indices: np.ndarray, probs: np.ndarray, fallback_idx: int) -> PackedPolicy:
    """Policy over several scored moves; padded slots have index -1."""
    keep = (indices >= 0) & (indices < NUM_MOVES) & np.isfinite(probs) & (probs > 0)
    if int(keep.sum()) < 2:
        return _build_policy(fallback_idx)
    kept = probs[keep].astype(np.float64)
    return PackedPolicy(
        indices=indices[keep].astype(np.uint16),
        probs=(kept / kept.sum()).astype(np.float16),
    )


def _decoded_fullmove_number(state: np.ndarray, cfg: AppConfig) -> int:
    max_fullmove = max(1, int(getattr(cfg.system, "max_fullmove", 1)))
    encoded = float(state[FULLMOVE_NUMBER_PLANE, 0, 0])
    return int(round(encoded * max_fullmove))


def _sample_in_fullmove_range(state: np.ndarray, cfg: AppConfig) -> bool:
    external_cfg = getattr(cfg, "external", None)
    min_fullmove = int(getattr(external_cfg, "min_fullmove", 0))
    max_fullmove = int(getattr(external_cfg, "max_fullmove", 0))
    if min_fullmove <= 0 and max_fullmove <= 0:
        return True

    fullmove_number = _decoded_fullmove_number(state, cfg)
    if min_fullmove > 0 and fullmove_number < min_fullmove:
        return False
    if max_fullmove > 0 and fullmove_number > max_fullmove:
        return False
    return True


# =========================================
# LOAD SINGLE FILE WITH FULL LOGS
# =========================================

def load_external_samples_with_stats(
    path: str | Path,
    cfg: AppConfig,
    *,
    max_samples: int = 0,
    partition: str | None = None,
    rng: random.Random | None = None,
    seen_hashes: set[bytes] | None = None,
) -> ExternalSampleLoadResult:

    path = Path(path)
    if partition not in {None, 'train', 'validation'}:
        raise ValueError("partition must be train, validation, or None")
    if partition is not None and not 0.0 < cfg.external.validation_split < 1.0:
        raise ValueError("validation_split must be between zero and one for partitioned loading")
    rng = rng or random
    print(f"\n[LOAD] file={path.name}")

    data = np.load(path, mmap_mode='r', allow_pickle=False)

    states = data['states']
    policy_indices = np.asarray(data['policy_indices'], dtype=np.int32)
    values = np.asarray(data['values'], dtype=np.float32)

    expected_planes = int(cfg.model.input_planes)
    if states.ndim != 4 or states.shape[1:] != (expected_planes, 8, 8):
        raise ValueError(
            f"External sample states must have shape (N, {expected_planes}, 8, 8); "
            f"got {states.shape}"
        )
    if 'input_planes' in data and int(np.asarray(data['input_planes']).reshape(-1)[0]) != expected_planes:
        actual_planes = int(np.asarray(data['input_planes']).reshape(-1)[0])
        raise ValueError(
            f"External sample input_planes mismatch: expected {expected_planes}, got {actual_planes}"
        )
    if 'policy_size' in data and int(np.asarray(data['policy_size']).reshape(-1)[0]) != NUM_MOVES:
        actual_policy_size = int(np.asarray(data['policy_size']).reshape(-1)[0])
        raise ValueError(
            f"External sample policy_size mismatch: expected {NUM_MOVES}, got {actual_policy_size}"
        )
    if len(policy_indices) != len(states) or len(values) != len(states):
        raise ValueError(
            "External sample arrays must have matching first dimension: "
            f"states={len(states)} policy_indices={len(policy_indices)} values={len(values)}"
        )
    # Optional multi-move targets (e.g. several engine-scored moves per position).
    topk_indices = topk_probs = None
    if 'policy_topk_indices' in data and 'policy_topk_probs' in data:
        topk_indices = np.asarray(data['policy_topk_indices'], dtype=np.int32)
        topk_probs = np.asarray(data['policy_topk_probs'], dtype=np.float32)
        if topk_indices.shape != topk_probs.shape or topk_indices.shape[0] != len(states):
            raise ValueError(
                f"policy_topk arrays must share shape (N, K); got {topk_indices.shape} and {topk_probs.shape}"
            )

    total_raw = len(states)
    print(f"[LOAD] raw samples={total_raw}")

    external_cfg = getattr(cfg, 'external', None)
    dedup_enabled = bool(getattr(external_cfg, 'dedup', True))
    filter_invalid = bool(getattr(external_cfg, 'filter_invalid', True))
    drop_zero_states = bool(getattr(external_cfg, 'drop_zero_states', True))
    shuffle_enabled = bool(getattr(external_cfg, 'shuffle', True))

    seen_hashes = seen_hashes if seen_hashes is not None else set()

    stats = {
        "accepted": 0,
        "dup": 0,
        "bad_policy": 0,
        "bad_value": 0,
        "bad_state": 0,
        "skipped_fullmove": 0,
        "other_partition": 0,
    }

    samples: list[tuple[np.ndarray, PackedPolicy, float]] = []

    limit = total_raw if max_samples <= 0 or partition is not None else min(total_raw, max_samples)

    # 🔥 Shuffle indices داخل الشارد
    if shuffle_enabled and 0 < limit < total_raw:
        indices = rng.sample(range(total_raw), limit)
    else:
        indices = list(range(limit))
        if shuffle_enabled:
            rng.shuffle(indices)

    for i, idx_i in enumerate(indices):
        idx = int(policy_indices[idx_i])
        value = float(values[idx_i])
        state = np.ascontiguousarray(states[idx_i], dtype=np.float16)

        # ===== FILTER =====
        if filter_invalid:
            if idx < 0 or idx >= NUM_MOVES:
                stats["bad_policy"] += 1
                continue

            if not np.isfinite(value):
                stats["bad_value"] += 1
                continue

            if not np.all(np.isfinite(state)) or (drop_zero_states and not np.any(state)):
                stats["bad_state"] += 1
                continue

        if not _sample_in_fullmove_range(state, cfg):
            stats["skipped_fullmove"] += 1
            continue

        if partition is not None and is_validation_position(state, cfg) != (partition == 'validation'):
            stats["other_partition"] += 1
            continue

        # ===== DEDUP =====
        if dedup_enabled:
            h = _sample_hash(state, idx, value)
            if h in seen_hashes:
                stats["dup"] += 1
                continue
            seen_hashes.add(h)

        if topk_indices is None:
            policy = _build_policy(idx)
        else:
            policy = _multi_move_policy(topk_indices[idx_i], topk_probs[idx_i], idx)
        samples.append((state, policy, value))
        stats["accepted"] += 1
        if max_samples > 0 and len(samples) >= max_samples:
            break

        if i % 20000 == 0 and i > 0:
            print(f"[PROGRESS] {i}/{limit} | accepted={stats['accepted']}")

    print(
        f"[SUMMARY] accepted={stats['accepted']} | "
        f"dup={stats['dup']} | "
        f"bad={stats['bad_policy'] + stats['bad_value'] + stats['bad_state']}"
    )

    return ExternalSampleLoadResult(samples=samples, stats=stats)


def load_external_samples(
    path: str | Path,
    cfg: AppConfig,
    *,
    max_samples: int = 0,
) -> Iterator[tuple[np.ndarray, PackedPolicy, float]]:
    path = Path(path)
    if path.is_dir():
        yield from load_external_samples_sharded(path, cfg, max_samples=max_samples)
        return

    stable_cfg = replace(cfg, external=replace(cfg.external, shuffle=False))
    result = load_external_samples_with_stats(path, stable_cfg, max_samples=max_samples)
    yield from result.samples


# =========================================
# SHARD STREAMING (FINAL VERSION)
# =========================================

def load_external_samples_sharded(
    folder: str | Path,
    cfg: AppConfig,
    *,
    max_samples: int = 0,
    partition: str | None = None,
    shuffle_seed: int | None = None,
) -> Iterator[tuple[np.ndarray, PackedPolicy, float]]:

    folder = Path(folder)
    shard_files = [folder] if folder.is_file() else sorted(folder.glob("*.npz"))
    rng = random.Random(cfg.external.seed if shuffle_seed is None else shuffle_seed)

    # 🔥 Shuffle الشاردات
    external_cfg = getattr(cfg, 'external', None)
    if bool(getattr(external_cfg, 'shuffle', True)):
        rng.shuffle(shard_files)

    if not shard_files:
        raise FileNotFoundError(f"No shards found in {folder}")

    print(f"\n[SHARDS] total={len(shard_files)}")

    total_streamed = 0
    seen_files: set[bytes] = set()
    seen_samples: set[bytes] = set()

    for shard_id, shard_path in enumerate(shard_files):
        if bool(getattr(external_cfg, 'dedup', True)):
            with shard_path.open('rb') as handle:
                digest = hashlib.file_digest(handle, 'sha256').digest()
            if digest in seen_files:
                print(f"[DUPLICATE SHARD] {shard_path.name}")
                continue
            seen_files.add(digest)
        print(f"\n[SHARD] ===== {shard_id+1}/{len(shard_files)} -> {shard_path.name} =====")

        remaining = int(max_samples - total_streamed) if max_samples else 0
        result = load_external_samples_with_stats(
            shard_path,
            cfg,
            max_samples=remaining,
            partition=partition,
            rng=rng,
            seen_hashes=seen_samples,
        )

        print(f"[SHARD DONE] accepted={result.stats['accepted']}")

        # 🔥 Shuffle داخل الشارد
        samples = result.samples
        if bool(getattr(external_cfg, 'shuffle', True)):
            rng.shuffle(samples)

        for sample in samples:
            yield sample
            total_streamed += 1

            if total_streamed % 50000 == 0:
                print(f"[STREAM] total streamed={total_streamed}")

            if max_samples and total_streamed >= max_samples:
                print(f"[STOP] reached max_samples={max_samples}")
                return

    print(f"\n[FINAL] total streamed={total_streamed}")
