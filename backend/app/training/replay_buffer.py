from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import struct
from typing import Any

import numpy as np
import torch

from app.infra.config import AppConfig, ReplayConfig, get_current_config

DEFAULT_POLICY_SIZE = 4672
MANIFEST_FORMAT = 'ring_replay_buffer_manifest_v1'
DEFAULT_SAVE_SHARD_SIZE = 4096
_POLICY_EPS = 1e-12
_EMPTY_UINT16 = np.zeros((0,), dtype=np.uint16)
_EMPTY_FLOAT16 = np.zeros((0,), dtype=np.float16)
_EMPTY_FLOAT32 = np.zeros((0,), dtype=np.float32)
_EMPTY_INT64 = np.zeros((0,), dtype=np.int64)


@dataclass(frozen=True)
class PackedPolicy:
    indices: np.ndarray
    probs: np.ndarray

    def to_dense(self, num_actions: int = DEFAULT_POLICY_SIZE) -> np.ndarray:
        dense = np.zeros(num_actions, dtype=np.float32)
        if self.indices.size:
            dense[self.indices.astype(np.int64, copy=False)] = self.probs.astype(np.float32, copy=False)
        return dense


@dataclass(frozen=True)
class SparsePolicyBatch:
    indices: np.ndarray
    probs: np.ndarray
    lengths: np.ndarray
    num_actions: int

    @property
    def batch_size(self) -> int:
        return int(self.lengths.shape[0])


EMPTY_PACKED_POLICY = PackedPolicy(_EMPTY_UINT16, _EMPTY_FLOAT16)


def _normalized_policy_entropy_from_packed(policy: PackedPolicy) -> float:
    probs = np.asarray(policy.probs, dtype=np.float32).reshape(-1)
    probs = probs[np.isfinite(probs)]
    probs = probs[probs > _POLICY_EPS]
    if probs.size == 0:
        return 0.0
    total = float(probs.sum())
    if total <= _POLICY_EPS:
        return 0.0
    probs = probs / total
    h = -float(np.sum(probs * np.log(probs + 1e-12)))
    max_h = math.log(int(probs.size)) if int(probs.size) > 1 else 1.0
    if max_h <= 0.0:
        return 0.0
    return float(h / max_h)


def _clamp01(x: float) -> float:
    return float(min(max(float(x), 0.0), 1.0))


class _ReplayView(Sequence):
    def __init__(self, owner: 'ReplayBuffer', kind: str):
        self._owner = owner
        self._kind = kind

    @property
    def maxlen(self) -> int:
        return self._owner.capacity

    def __len__(self) -> int:
        return len(self._owner)

    def __iter__(self) -> Iterator[Any]:
        if self._kind == 'buffer':
            for idx in range(len(self._owner)):
                yield self._owner._get_sample(idx)
        else:
            for idx in range(len(self._owner)):
                yield self._owner._get_priority(idx)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self._kind == 'buffer':
            return self._owner._get_sample(index)
        return self._owner._get_priority(index)


def _resolve_manifest_shard_dir(manifest_data: dict[str, Any], manifest_path: str | Path) -> Path:
    manifest = Path(manifest_path)
    shard_dir_name = manifest_data.get('shard_dir') or manifest.with_suffix(manifest.suffix + '.d').name
    return manifest.parent / shard_dir_name


def _inspect_manifest_shards(
    manifest_data: dict[str, Any],
    manifest_path: str | Path,
) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]:
    shard_dir = _resolve_manifest_shard_dir(manifest_data, manifest_path)
    present: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for shard in manifest_data.get('shards', []):
        shard_file = shard_dir / str(shard.get('file'))
        if shard_file.exists():
            present.append(shard)
        else:
            missing.append(shard)

    return shard_dir, present, missing


def _rewrite_manifest_without_missing_shards(
    manifest_path: str | Path,
    manifest_data: dict[str, Any],
    present_shards: list[dict[str, Any]],
) -> None:
    manifest_path = Path(manifest_path)
    backup_path = manifest_path.with_name(manifest_path.name + '.bak')

    repaired = dict(manifest_data)
    repaired['shards'] = list(present_shards)

    if present_shards:
        repaired['last_saved_uid'] = int(max(int(s.get('max_uid', 0)) for s in present_shards))
        repaired['min_active_uid'] = int(min(int(s.get('min_uid', 0)) for s in present_shards))
        repaired['max_active_uid'] = int(max(int(s.get('max_uid', 0)) for s in present_shards))
        repaired['size'] = int(sum(int(s.get('count', 0)) for s in present_shards))
    else:
        repaired['last_saved_uid'] = 0
        repaired['min_active_uid'] = 0
        repaired['max_active_uid'] = 0
        repaired['size'] = 0

    torch.save(manifest_data, backup_path)
    torch.save(repaired, manifest_path)


def _pack_sparse_policy_arrays(indices: Any, probs: Any, policy_size: int) -> PackedPolicy:
    raw_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    raw_probs = np.asarray(probs, dtype=np.float32).reshape(-1)

    if raw_indices.size == 0 or raw_probs.size == 0:
        return EMPTY_PACKED_POLICY

    size = min(raw_indices.size, raw_probs.size)
    raw_indices = raw_indices[:size]
    raw_probs = raw_probs[:size]

    valid = np.isfinite(raw_probs) & (raw_probs > _POLICY_EPS) & (raw_indices >= 0)
    if policy_size > 0:
        valid &= raw_indices < int(policy_size)
    if not np.any(valid):
        return EMPTY_PACKED_POLICY

    clean_indices = raw_indices[valid]
    clean_probs = raw_probs[valid]

    order = np.argsort(clean_indices, kind='stable')
    clean_indices = clean_indices[order]
    clean_probs = clean_probs[order]

    if clean_indices.size > 1:
        unique_indices, first_pos = np.unique(clean_indices, return_index=True)
        if unique_indices.size != clean_indices.size:
            clean_probs = np.add.reduceat(clean_probs, first_pos)
            clean_indices = unique_indices

    total = float(clean_probs.sum())
    if not np.isfinite(total) or total <= _POLICY_EPS:
        return EMPTY_PACKED_POLICY
    clean_probs = clean_probs / total

    return PackedPolicy(
        indices=np.ascontiguousarray(clean_indices.astype(np.uint16, copy=False)),
        probs=np.ascontiguousarray(clean_probs.astype(np.float16, copy=False)),
    )


class ReplayBuffer:
    VERSION = 4
    FORMAT = 'ring_replay_buffer_compact_v2'

    def __init__(self, cfg: AppConfig | ReplayConfig | None = None):
        self.cfg = cfg or get_current_config()
        self.replay_cfg = self.cfg.replay if hasattr(self.cfg, 'replay') else self.cfg
        self.training_cfg = self.cfg.training if hasattr(self.cfg, 'training') else get_current_config().training
        self.capacity = max(1, int(self.replay_cfg.capacity))
        self.state_shape = (
            int(getattr(self.cfg.model, 'input_planes', 20)) if hasattr(self.cfg, 'model') else 20,
            8,
            8,
        )
        self.policy_size = int(DEFAULT_POLICY_SIZE)
        self.states = np.zeros((self.capacity, *self.state_shape), dtype=np.float16)
        self.policies: list[PackedPolicy | None] = [None] * self.capacity
        self.values = np.zeros(self.capacity, dtype=np.float32)
        self._priorities = np.zeros(self.capacity, dtype=np.float32)
        self.uids = np.zeros(self.capacity, dtype=np.int64)
        self.size = 0
        self.pos = 0
        self.next_uid = 1

        # Exact sample-level dedup: state + policy + value
        self.seen_hashes: set[bytes] = set()

        # State-level control
        self.state_hash_to_physicals: dict[bytes, set[int]] = defaultdict(set)
        self.min_keep_per_state = 4
        self.max_keep_per_state = 24

        self._last_saved_uid = 0
        self._saved_shards: list[dict[str, Any]] = []
        self._approx_max_priority = 1.0
        self._ordered_physical_cache: np.ndarray | None = None
        self._outcome_partition_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    @property
    def buffer(self) -> _ReplayView:
        return _ReplayView(self, 'buffer')

    @property
    def priorities(self) -> _ReplayView:
        return _ReplayView(self, 'priorities')

    def __len__(self) -> int:
        return int(self.size)

    def _invalidate_sampling_caches(self) -> None:
        self._ordered_physical_cache = None
        self._outcome_partition_cache = None

    def _pack_state(self, state: Any) -> np.ndarray:
        array = state.detach().cpu().numpy() if isinstance(state, torch.Tensor) else np.asarray(state)
        packed = np.ascontiguousarray(array, dtype=np.float16)
        if self.size == 0 and self.pos == 0 and packed.shape != self.state_shape:
            self.state_shape = tuple(int(dim) for dim in packed.shape)
            self.states = np.zeros((self.capacity, *self.state_shape), dtype=np.float16)
        if packed.shape != self.state_shape:
            raise ValueError(f'Expected state shape {self.state_shape}, got {packed.shape}')
        return packed

    def _pack_policy(self, policy: Any) -> PackedPolicy:
        if isinstance(policy, PackedPolicy):
            raw_indices = np.asarray(policy.indices).reshape(-1)
            if self.size == 0 and self.pos == 0 and raw_indices.size:
                self.policy_size = max(self.policy_size, int(raw_indices.max()) + 1)

            indices = np.ascontiguousarray(raw_indices, dtype=np.uint16)
            probs = np.ascontiguousarray(np.asarray(policy.probs).reshape(-1), dtype=np.float16)
            if indices.size == probs.size:
                if indices.size == 0:
                    return EMPTY_PACKED_POLICY
                indices_are_sorted = bool(np.all(indices[1:] > indices[:-1])) if indices.size > 1 else True
                probs32 = probs.astype(np.float32, copy=False)
                probs_valid = bool(np.all(np.isfinite(probs32)) and np.all(probs32 >= 0.0))
                total = float(probs32.sum()) if probs_valid else 0.0
                if indices_are_sorted and int(indices[-1]) < int(self.policy_size) and probs_valid and 0.99 <= total <= 1.01:
                    return PackedPolicy(indices=indices, probs=probs)

            return _pack_sparse_policy_arrays(policy.indices, policy.probs, self.policy_size)

        dense = policy.detach().cpu().numpy() if isinstance(policy, torch.Tensor) else np.asarray(policy)
        dense = np.ascontiguousarray(dense, dtype=np.float32).reshape(-1)
        if self.size == 0 and self.pos == 0 and dense.size != self.policy_size:
            self.policy_size = int(dense.size)
        if dense.size != self.policy_size:
            raise ValueError(f'Expected policy size {self.policy_size}, got {dense.size}')
        dense = np.where(np.isfinite(dense), dense, 0.0).astype(np.float32, copy=False)
        dense = np.clip(dense, 0.0, None)
        indices = np.flatnonzero(dense > _POLICY_EPS)
        if indices.size == 0:
            return EMPTY_PACKED_POLICY
        return _pack_sparse_policy_arrays(indices, dense[indices], self.policy_size)

    def _hash_packed_sample(self, state: np.ndarray, policy: PackedPolicy, value: float) -> bytes:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(np.ascontiguousarray(state, dtype=np.float16).tobytes())
        digest.update(np.ascontiguousarray(policy.indices, dtype=np.uint16).tobytes())
        digest.update(np.ascontiguousarray(policy.probs, dtype=np.float16).tobytes())
        digest.update(struct.pack('<f', float(value)))
        return digest.digest()

    def _hash_state_only(self, state: np.ndarray) -> bytes:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(np.ascontiguousarray(state, dtype=np.float16).tobytes())
        return digest.digest()

    def _next_priority(self) -> float:
        return float(max(self._approx_max_priority, 1.0))

    def _logical_to_physical(self, logical_index: int) -> int:
        if logical_index < 0 or logical_index >= self.size:
            raise IndexError(logical_index)
        if self.size < self.capacity:
            return logical_index
        return (self.pos + logical_index) % self.capacity

    def _ordered_physical_indices_array(self) -> np.ndarray:
        if self._ordered_physical_cache is not None:
            return self._ordered_physical_cache
        if self.size == 0:
            ordered = np.zeros((0,), dtype=np.int64)
        elif self.size < self.capacity:
            ordered = np.arange(self.size, dtype=np.int64)
        else:
            ordered = np.concatenate([
                np.arange(self.pos, self.capacity, dtype=np.int64),
                np.arange(0, self.pos, dtype=np.int64),
            ])
        self._ordered_physical_cache = ordered
        return ordered

    def _ordered_physical_indices(self) -> list[int]:
        return self._ordered_physical_indices_array().tolist()

    def _ordered_entries(self) -> list[tuple[np.ndarray, PackedPolicy, float, float, int]]:
        entries: list[tuple[np.ndarray, PackedPolicy, float, float, int]] = []
        for physical_index in self._ordered_physical_indices_array().tolist():
            packed_policy = self.policies[physical_index] or EMPTY_PACKED_POLICY
            entries.append((
                self.states[physical_index].copy(),
                PackedPolicy(packed_policy.indices.copy(), packed_policy.probs.copy()),
                float(self.values[physical_index]),
                float(self._priorities[physical_index]),
                int(self.uids[physical_index]),
            ))
        return entries

    def _get_sample(self, logical_index: int) -> tuple[Any, Any, float]:
        physical_index = self._logical_to_physical(logical_index)
        state = torch.from_numpy(self.states[physical_index].astype(np.float32, copy=True))
        packed_policy = self.policies[physical_index] or EMPTY_PACKED_POLICY
        return state, packed_policy.to_dense(self.policy_size), float(self.values[physical_index])

    def _get_priority(self, logical_index: int) -> float:
        value = float(self._priorities[self._logical_to_physical(logical_index)])
        return float(round(value, 6))

    def _get_value(self, logical_index: int) -> float:
        return float(self.values[self._logical_to_physical(logical_index)])

    def _set_sample(self, physical_index: int, state: Any, policy: Any, value: Any, priority: float, uid: int | None = None) -> None:
        packed_state = self._pack_state(state)
        packed_policy = self._pack_policy(policy)
        priority_value = float(max(priority, float(self.replay_cfg.eps)))
        self.states[physical_index] = packed_state
        self.policies[physical_index] = packed_policy
        self.values[physical_index] = np.float32(value)
        self._priorities[physical_index] = np.float32(priority_value)
        self.uids[physical_index] = int(uid if uid is not None else self.next_uid)
        self._approx_max_priority = max(self._approx_max_priority, priority_value)

    def _sample_importance_score(
        self,
        packed_policy: PackedPolicy,
        value: float,
        priority: float,
    ) -> float:
        abs_value = abs(float(value))
        entropy = _normalized_policy_entropy_from_packed(packed_policy)
        entropy_bonus = 1.0 - entropy
        priority_norm = _clamp01(float(priority) / 10.0)
        return float(
            3.0 * abs_value +
            1.0 * entropy_bonus +
            0.25 * priority_norm
        )

    def _remove_physical_index_from_state_pool(self, physical_index: int) -> None:
        packed_policy = self.policies[physical_index]
        if packed_policy is None:
            return
        state_hash = self._hash_state_only(self.states[physical_index])
        bucket = self.state_hash_to_physicals.get(state_hash)
        if not bucket:
            return
        bucket.discard(int(physical_index))
        if not bucket:
            self.state_hash_to_physicals.pop(state_hash, None)

    def _register_physical_index_in_state_pool(self, physical_index: int) -> None:
        packed_policy = self.policies[physical_index]
        if packed_policy is None:
            return
        state_hash = self._hash_state_only(self.states[physical_index])
        self.state_hash_to_physicals[state_hash].add(int(physical_index))

    def _choose_worst_physical_for_state(self, state_hash: bytes) -> int | None:
        candidates = self.state_hash_to_physicals.get(state_hash)
        if not candidates:
            return None

        worst_idx = None
        worst_score = None
        for physical_index in candidates:
            packed_policy = self.policies[physical_index] or EMPTY_PACKED_POLICY
            score = self._sample_importance_score(
                packed_policy=packed_policy,
                value=float(self.values[physical_index]),
                priority=float(self._priorities[physical_index]),
            )
            if worst_idx is None or score < worst_score:
                worst_idx = int(physical_index)
                worst_score = float(score)
        return worst_idx

    def _can_accept_more_for_state(
        self,
        state_hash: bytes,
        packed_policy: PackedPolicy,
        value: float,
        priority: float,
    ) -> tuple[bool, int | None]:
        bucket = self.state_hash_to_physicals.get(state_hash)
        count = len(bucket) if bucket is not None else 0

        if count < self.min_keep_per_state:
            return True, None

        new_score = self._sample_importance_score(
            packed_policy=packed_policy,
            value=float(value),
            priority=float(priority),
        )

        if count < self.max_keep_per_state:
            if new_score >= 0.75:
                return True, None
            return False, None

        worst_idx = self._choose_worst_physical_for_state(state_hash)
        if worst_idx is None:
            return True, None

        worst_policy = self.policies[worst_idx] or EMPTY_PACKED_POLICY
        worst_score = self._sample_importance_score(
            packed_policy=worst_policy,
            value=float(self.values[worst_idx]),
            priority=float(self._priorities[worst_idx]),
        )

        if new_score > worst_score:
            return True, worst_idx

        return False, None

    def rebuild_seen_hashes(self) -> None:
        self.seen_hashes = set()
        self.state_hash_to_physicals = defaultdict(set)
        for physical_index in self._ordered_physical_indices_array().tolist():
            packed_policy = self.policies[physical_index]
            if packed_policy is not None:
                self.seen_hashes.add(self._hash_packed_sample(self.states[physical_index], packed_policy, float(self.values[physical_index])))
                self.state_hash_to_physicals[self._hash_state_only(self.states[physical_index])].add(int(physical_index))

    def add(self, state: Any, policy: Any, value: Any) -> None:
        packed_state = self._pack_state(state)
        packed_policy = self._pack_policy(policy)
        value_f = float(value)

        sample_hash = self._hash_packed_sample(packed_state, packed_policy, value_f)
        if sample_hash in self.seen_hashes:
            return

        state_hash = self._hash_state_only(packed_state)
        proposed_priority = self._next_priority()

        accept, replace_physical = self._can_accept_more_for_state(
            state_hash=state_hash,
            packed_policy=packed_policy,
            value=value_f,
            priority=proposed_priority,
        )
        if not accept:
            return

        if replace_physical is not None:
            old_policy = self.policies[replace_physical]
            if old_policy is not None:
                self.seen_hashes.discard(
                    self._hash_packed_sample(
                        self.states[replace_physical],
                        old_policy,
                        float(self.values[replace_physical]),
                    )
                )
            self._remove_physical_index_from_state_pool(replace_physical)

            self._set_sample(
                replace_physical,
                packed_state,
                packed_policy,
                value_f,
                priority=proposed_priority,
                uid=self.next_uid,
            )
            self.seen_hashes.add(sample_hash)
            self._register_physical_index_in_state_pool(replace_physical)
            self.next_uid += 1
            self._invalidate_sampling_caches()
            return

        physical_index = self.pos
        if self.size == self.capacity:
            old_policy = self.policies[physical_index]
            if old_policy is not None:
                self.seen_hashes.discard(self._hash_packed_sample(self.states[physical_index], old_policy, float(self.values[physical_index])))
            self._remove_physical_index_from_state_pool(physical_index)
        else:
            self.size += 1

        self.seen_hashes.add(sample_hash)
        self._set_sample(physical_index, packed_state, packed_policy, value_f, priority=proposed_priority, uid=self.next_uid)
        self._register_physical_index_in_state_pool(physical_index)
        self.next_uid += 1
        self.pos = (self.pos + 1) % self.capacity
        self._invalidate_sampling_caches()

    def save_game(self, game_data: Iterable[tuple[Any, Any, float]]) -> None:
        for state, policy, value in game_data:
            self.add(state, policy, value)

    def _sample_from_physical_indices(self, physical_indices: np.ndarray, k: int, beta: float):
        pool = np.asarray(physical_indices, dtype=np.int64).reshape(-1)
        if k <= 0 or pool.size == 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
        k = min(k, int(pool.size))
        if not bool(getattr(self.replay_cfg, 'prioritized', True)):
            chosen_pos = np.random.choice(pool.size, size=k, replace=False)
            return pool[chosen_pos], np.ones((k,), dtype=np.float32)

        priorities = np.clip(self._priorities[pool].astype(np.float32, copy=False), float(self.replay_cfg.eps), 10.0)
        scaled = np.power(priorities + float(self.replay_cfg.eps), float(self.replay_cfg.alpha), dtype=np.float32)
        total = float(scaled.sum())
        probs = scaled / total if total > _POLICY_EPS else np.ones_like(scaled, dtype=np.float32) / float(len(scaled))

        policy_mix = float(getattr(self.replay_cfg, 'policy_mix', 0.0))
        if policy_mix > 0.0:
            uniform = np.full_like(probs, 1.0 / float(len(probs)))
            probs = (1.0 - policy_mix) * probs + policy_mix * uniform
            probs = probs / max(float(probs.sum()), _POLICY_EPS)

        chosen_pos = np.random.choice(pool.size, size=k, replace=False, p=probs)
        sampled_probs = probs[chosen_pos]
        weights = np.power(np.maximum(float(pool.size) * sampled_probs, float(self.replay_cfg.eps)), -beta)
        weights /= max(float(weights.max()), 1e-8)
        return pool[chosen_pos], weights.astype(np.float32, copy=False)

    def _outcome_partitions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._outcome_partition_cache is not None:
            return self._outcome_partition_cache
        active_physical = self._ordered_physical_indices_array()
        if active_physical.size == 0:
            self._outcome_partition_cache = (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
            )
            return self._outcome_partition_cache

        values = self.values[active_physical].astype(np.float32, copy=False)
        threshold = float(self.replay_cfg.draw_value_threshold)
        positive = active_physical[values > threshold]
        negative = active_physical[values < -threshold]
        draws = active_physical[np.abs(values) <= threshold]
        self._outcome_partition_cache = (positive, negative, draws)
        return self._outcome_partition_cache

    def _sample_balanced_batch(self, batch_size: int, beta: float):
        active_physical = self._ordered_physical_indices_array()
        positive, negative, draws = self._outcome_partitions()

        max_draw_fraction = min(max(float(getattr(self.replay_cfg, 'max_draw_fraction', 1.0)), 0.0), 1.0)
        max_draws = min(int(draws.size), int(round(batch_size * max_draw_fraction)))
        pos_target = min(int(positive.size), batch_size // 2)
        neg_target = min(int(negative.size), batch_size // 2)
        draws_target = min(max_draws, batch_size - pos_target - neg_target)
        remaining = batch_size - (pos_target + neg_target + draws_target)

        while remaining > 0:
            progressed = False
            if int(positive.size) > pos_target:
                pos_target += 1
                remaining -= 1
                progressed = True
            if remaining > 0 and int(negative.size) > neg_target:
                neg_target += 1
                remaining -= 1
                progressed = True
            if remaining > 0 and int(draws.size) > draws_target and draws_target < max_draws:
                draws_target += 1
                remaining -= 1
                progressed = True
            if not progressed:
                break

        pos_choice, pos_weights = self._sample_from_physical_indices(positive, pos_target, beta)
        neg_choice, neg_weights = self._sample_from_physical_indices(negative, neg_target, beta)
        draw_choice, draw_weights = self._sample_from_physical_indices(draws, draws_target, beta)

        chosen = np.concatenate([pos_choice, neg_choice, draw_choice]).astype(np.int64, copy=False)
        weights = np.concatenate([pos_weights, neg_weights, draw_weights]).astype(np.float32, copy=False)

        if chosen.size < batch_size:
            remaining_mask = np.isin(active_physical, chosen, assume_unique=False, invert=True)
            fallback_pool = active_physical[remaining_mask]
            if fallback_pool.size == 0:
                fallback_pool = active_physical
            extra_choice, extra_weights = self._sample_from_physical_indices(fallback_pool, batch_size - int(chosen.size), beta)
            chosen = np.concatenate([chosen, extra_choice]).astype(np.int64, copy=False)
            weights = np.concatenate([weights, extra_weights]).astype(np.float32, copy=False)

        if chosen.size > 1:
            order = np.random.permutation(chosen.size)
            chosen = chosen[order]
            weights = weights[order]
        return chosen, weights

    def _build_sparse_policy_batch(self, physical_indices: np.ndarray) -> SparsePolicyBatch:
        batch_physical = np.asarray(physical_indices, dtype=np.int64).reshape(-1)
        batch_size = int(batch_physical.size)
        if batch_size == 0:
            return SparsePolicyBatch(
                indices=np.zeros((0, 0), dtype=np.uint16),
                probs=np.zeros((0, 0), dtype=np.float16),
                lengths=np.zeros((0,), dtype=np.int32),
                num_actions=int(self.policy_size),
            )

        batch_policies = [self.policies[int(physical_index)] or EMPTY_PACKED_POLICY for physical_index in batch_physical.tolist()]
        lengths = np.fromiter((int(policy.indices.size) for policy in batch_policies), dtype=np.int32, count=batch_size)
        max_len = int(lengths.max()) if batch_size else 0
        indices = np.zeros((batch_size, max_len), dtype=np.uint16)
        probs = np.zeros((batch_size, max_len), dtype=np.float16)
        for row, policy in enumerate(batch_policies):
            length = int(lengths[row])
            if length <= 0:
                continue
            indices[row, :length] = policy.indices
            probs[row, :length] = policy.probs
        return SparsePolicyBatch(indices=indices, probs=probs, lengths=lengths, num_actions=int(self.policy_size))

    def sample_batch(self, batch_size: int | None = None, beta: float = 1.0):
        n = len(self)
        if n == 0:
            return [], [], [], [], []
        batch_size = min(batch_size or int(self.training_cfg.batch_size), n)
        active_physical = self._ordered_physical_indices_array()

        if bool(getattr(self.replay_cfg, 'balance_outcomes', False)):
            chosen, weights = self._sample_balanced_batch(batch_size, beta)
        else:
            recent_fraction = float(getattr(self.replay_cfg, 'recent_sample_fraction', 0.0))
            recent_window_size = int(getattr(self.replay_cfg, 'recent_window_size', 0) or 0)
            use_recent_mix = recent_fraction > 0.0 and recent_window_size > 0 and n > 1
            if not use_recent_mix:
                chosen, weights = self._sample_from_physical_indices(active_physical, batch_size, beta)
            else:
                recent_physical = active_physical[max(0, n - recent_window_size):]
                recent_batch = max(1, min(int(round(batch_size * recent_fraction)), int(recent_physical.size)))
                recent_chosen, recent_weights = self._sample_from_physical_indices(recent_physical, recent_batch, beta)
                remaining = batch_size - int(recent_chosen.size)
                other_mask = np.isin(active_physical, recent_chosen, assume_unique=False, invert=True)
                other_pool = active_physical[other_mask]
                other_chosen, other_weights = self._sample_from_physical_indices(other_pool if other_pool.size else active_physical, remaining, beta)
                chosen = np.concatenate([recent_chosen, other_chosen]).astype(np.int64, copy=False)
                weights = np.concatenate([recent_weights, other_weights]).astype(np.float32, copy=False)
                if chosen.size > 1:
                    order = np.random.permutation(chosen.size)
                    chosen = chosen[order]
                    weights = weights[order]

        states = np.ascontiguousarray(self.states[chosen], dtype=np.float16)
        policies = self._build_sparse_policy_batch(chosen)
        values = self.values[chosen].astype(np.float32, copy=True)
        return states, policies, values, chosen.astype(np.int64, copy=False), weights.astype(np.float32, copy=False)

    def update_priorities(self, indices, priorities) -> None:
        if indices is None or priorities is None:
            return
        physical_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        new_priorities = np.asarray(priorities, dtype=np.float32).reshape(-1)
        if physical_indices.size == 0 or new_priorities.size == 0:
            return
        count = min(physical_indices.size, new_priorities.size)
        physical_indices = physical_indices[:count]
        new_priorities = new_priorities[:count]
        valid = (physical_indices >= 0) & (physical_indices < self.capacity) & (self.uids[physical_indices] > 0)
        if not np.any(valid):
            return
        clipped = np.clip(new_priorities[valid], float(self.replay_cfg.eps), 10.0).astype(np.float32, copy=False)
        self._priorities[physical_indices[valid]] = clipped
        if clipped.size:
            self._approx_max_priority = max(self._approx_max_priority, float(clipped.max()))

    def _ordered_active_data(self):
        physical_indices = self._ordered_physical_indices_array()
        if physical_indices.size == 0:
            return np.zeros((0, *self.state_shape), dtype=np.float16), [], np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        states = np.ascontiguousarray(self.states[physical_indices], dtype=np.float16)
        policies = [self.policies[int(idx)] or EMPTY_PACKED_POLICY for idx in physical_indices.tolist()]
        values = self.values[physical_indices].astype(np.float32, copy=True)
        priorities = self._priorities[physical_indices].astype(np.float32, copy=True)
        uids = self.uids[physical_indices].astype(np.int64, copy=True)
        return states, policies, values, priorities, uids

    @classmethod
    def _state_dict_from_entries(cls, entries: list[tuple[np.ndarray, PackedPolicy, float, float, int]], capacity: int, state_shape: tuple[int, ...], policy_size: int, next_uid: int, pos: int) -> dict[str, Any]:
        if entries:
            states = np.ascontiguousarray(np.stack([entry[0] for entry in entries], axis=0), dtype=np.float16)
            policies = [entry[1] for entry in entries]
            values = np.asarray([entry[2] for entry in entries], dtype=np.float32)
            priorities = np.asarray([entry[3] for entry in entries], dtype=np.float32)
            uids = np.asarray([entry[4] for entry in entries], dtype=np.int64)
        else:
            states = np.zeros((0, *state_shape), dtype=np.float16)
            policies = []
            values = np.zeros((0,), dtype=np.float32)
            priorities = np.zeros((0,), dtype=np.float32)
            uids = np.zeros((0,), dtype=np.int64)
        lengths = np.array([int(policy.indices.size) for policy in policies], dtype=np.int32)
        if policies and int(lengths.sum()) > 0:
            indices = np.concatenate([policy.indices for policy in policies if policy.indices.size], axis=0).astype(np.uint16, copy=False)
            probs = np.concatenate([policy.probs for policy in policies if policy.probs.size], axis=0).astype(np.float16, copy=False)
        else:
            indices = np.zeros((0,), dtype=np.uint16)
            probs = np.zeros((0,), dtype=np.float16)
        return {
            'version': cls.VERSION,
            'format': cls.FORMAT,
            'capacity': int(max(1, capacity)),
            'size': int(len(entries)),
            'pos': int(pos),
            'next_uid': int(next_uid),
            'state_shape': tuple(state_shape),
            'state_dtype': 'float16',
            'policy_num_actions': int(policy_size),
            'states': states,
            'policy_lengths': lengths,
            'policy_indices': indices,
            'policy_probs': probs,
            'values': values,
            'priorities': priorities,
            'uids': uids,
        }

    def _state_dict_from_physical_indices(self, physical_indices: np.ndarray, *, capacity: int, pos: int, next_uid: int | None = None) -> dict[str, Any]:
        ordered = np.asarray(physical_indices, dtype=np.int64).reshape(-1)
        if ordered.size == 0:
            return self._state_dict_from_entries(
                [],
                capacity=capacity,
                state_shape=self.state_shape,
                policy_size=self.policy_size,
                next_uid=int(self.next_uid if next_uid is None else next_uid),
                pos=pos,
            )

        states = np.ascontiguousarray(self.states[ordered], dtype=np.float16)
        values = self.values[ordered].astype(np.float32, copy=True)
        priorities = self._priorities[ordered].astype(np.float32, copy=True)
        uids = self.uids[ordered].astype(np.int64, copy=True)
        policies = [self.policies[int(idx)] or EMPTY_PACKED_POLICY for idx in ordered.tolist()]
        lengths = np.fromiter((int(policy.indices.size) for policy in policies), dtype=np.int32, count=int(ordered.size))
        total_nnz = int(lengths.sum())
        if total_nnz > 0:
            indices = np.empty((total_nnz,), dtype=np.uint16)
            probs = np.empty((total_nnz,), dtype=np.float16)
            offset = 0
            for policy in policies:
                length = int(policy.indices.size)
                if length <= 0:
                    continue
                next_offset = offset + length
                indices[offset:next_offset] = policy.indices
                probs[offset:next_offset] = policy.probs
                offset = next_offset
        else:
            indices = np.zeros((0,), dtype=np.uint16)
            probs = np.zeros((0,), dtype=np.float16)

        return {
            'version': self.VERSION,
            'format': self.FORMAT,
            'capacity': int(max(1, capacity)),
            'size': int(ordered.size),
            'pos': int(pos),
            'next_uid': int(self.next_uid if next_uid is None else next_uid),
            'state_shape': tuple(self.state_shape),
            'state_dtype': 'float16',
            'policy_num_actions': int(self.policy_size),
            'states': states,
            'policy_lengths': lengths,
            'policy_indices': indices,
            'policy_probs': probs,
            'values': values,
            'priorities': priorities,
            'uids': uids,
        }

    def to_state_dict(self) -> dict[str, Any]:
        active_physical = self._ordered_physical_indices_array()
        return self._state_dict_from_physical_indices(
            active_physical,
            capacity=self.capacity,
            pos=int(self.size % self.capacity),
            next_uid=self.next_uid,
        )

    def _manifest_shard_dir(self, target: Path) -> Path:
        return target.with_suffix(target.suffix + '.d')

    def _write_atomic(self, payload: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + '.tmp')
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shard_dir = self._manifest_shard_dir(target)
        shard_dir.mkdir(parents=True, exist_ok=True)

        if not self._saved_shards and target.exists():
            try:
                existing = torch.load(target, map_location='cpu', weights_only=False)
                if isinstance(existing, dict) and existing.get('format') == MANIFEST_FORMAT:
                    self._saved_shards = list(existing.get('shards', []))
                    self._last_saved_uid = int(existing.get('last_saved_uid', self._last_saved_uid))
            except Exception:
                pass

        active_physical = self._ordered_physical_indices_array()
        if active_physical.size:
            active_uids = self.uids[active_physical].astype(np.int64, copy=False)
            min_active_uid = int(active_uids[0])
            max_active_uid = int(active_uids[-1])
        else:
            active_uids = np.zeros((0,), dtype=np.int64)
            min_active_uid = int(self.next_uid)
            max_active_uid = int(self._last_saved_uid)

        pending_start = int(np.searchsorted(active_uids, self._last_saved_uid + 1, side='left')) if active_uids.size else 0
        pending_physical = active_physical[pending_start:]
        shard_sample_limit = int(getattr(self.replay_cfg, 'save_shard_size', DEFAULT_SAVE_SHARD_SIZE) or DEFAULT_SAVE_SHARD_SIZE)
        shard_sample_limit = max(1, shard_sample_limit)

        for offset in range(0, int(pending_physical.size), shard_sample_limit):
            shard_physical = pending_physical[offset: offset + shard_sample_limit]
            if shard_physical.size == 0:
                continue
            first_uid = int(self.uids[int(shard_physical[0])])
            last_uid = int(self.uids[int(shard_physical[-1])])
            shard_name = f'{target.stem}.shard.{first_uid:012d}.{last_uid:012d}.pt'
            shard_path = shard_dir / shard_name
            shard_payload = self._state_dict_from_physical_indices(
                shard_physical,
                capacity=int(max(1, shard_physical.size)),
                pos=0,
                next_uid=self.next_uid,
            )
            self._write_atomic(shard_payload, shard_path)
            self._saved_shards.append({
                'file': shard_name,
                'count': int(shard_physical.size),
                'min_uid': first_uid,
                'max_uid': last_uid,
            })
            self._last_saved_uid = last_uid

        keep_shards: list[dict[str, Any]] = []
        keep_files = set()
        for shard in self._saved_shards:
            if int(shard.get('max_uid', 0)) < int(min_active_uid):
                stale_path = shard_dir / str(shard.get('file'))
                if stale_path.exists():
                    try:
                        stale_path.unlink()
                    except OSError:
                        pass
                continue
            keep_shards.append(shard)
            keep_files.add(str(shard.get('file')))
        self._saved_shards = keep_shards
        for child in shard_dir.glob('*.pt'):
            if child.name not in keep_files:
                try:
                    child.unlink()
                except OSError:
                    pass

        manifest = {
            'version': self.VERSION,
            'format': MANIFEST_FORMAT,
            'capacity': int(self.capacity),
            'size': int(self.size),
            'state_shape': tuple(self.state_shape),
            'policy_num_actions': int(self.policy_size),
            'next_uid': int(self.next_uid),
            'last_saved_uid': int(self._last_saved_uid),
            'min_active_uid': int(min_active_uid),
            'max_active_uid': int(max_active_uid),
            'save_shard_size': int(shard_sample_limit),
            'shard_dir': shard_dir.name,
            'shards': self._saved_shards,
        }
        self._write_atomic(manifest, target)

    @classmethod
    def _from_ordered_samples(cls, items: list[tuple[Any, Any, float]], priorities: list[float], cfg: AppConfig | ReplayConfig | None = None, uids: list[int] | None = None) -> 'ReplayBuffer':
        buffer = cls(cfg)
        if not items:
            return buffer
        if len(priorities) < len(items):
            fill_value = float(max(priorities)) if priorities else 1.0
            priorities = list(priorities) + [fill_value] * (len(items) - len(priorities))
        elif len(priorities) > len(items):
            priorities = list(priorities[: len(items)])
        if uids is None:
            uids = list(range(1, len(items) + 1))
        elif len(uids) < len(items):
            start_uid = (max(uids) + 1) if uids else 1
            uids = list(uids) + list(range(start_uid, start_uid + (len(items) - len(uids))))
        elif len(uids) > len(items):
            uids = list(uids[: len(items)])
        if len(items) > buffer.capacity:
            items = items[-buffer.capacity:]
            priorities = priorities[-buffer.capacity:]
            uids = uids[-buffer.capacity:]
        for idx, ((state, policy, value), priority, uid) in enumerate(zip(items, priorities, uids)):
            buffer._set_sample(idx, state, policy, value, max(float(priority), float(buffer.replay_cfg.eps)), uid)
        buffer.size = len(items)
        buffer.pos = buffer.size % buffer.capacity
        buffer.next_uid = (max(uids) + 1) if uids else 1
        buffer._last_saved_uid = max(uids) if uids else 0
        if buffer.size > 0:
            buffer._approx_max_priority = float(np.max(buffer._priorities[:buffer.size]))
        else:
            buffer._approx_max_priority = 1.0
        buffer._invalidate_sampling_caches()
        buffer.rebuild_seen_hashes()
        return buffer

    @classmethod
    def _ordered_samples_from_ring_state(cls, data: dict[str, Any]):
        states = list(data.get('states', []))
        policies = list(data.get('policies', []))
        values = list(data.get('values', []))
        priorities = list(data.get('priorities', []))
        uids = list(data.get('uids', []))
        inferred_capacity = max(1, int(data.get('capacity', 0) or len(states) or len(priorities) or 1))
        size = max(0, min(int(data.get('size', 0)), inferred_capacity, len(states), len(policies), len(values), len(priorities)))
        pos = int(data.get('pos', size % inferred_capacity)) % inferred_capacity
        if len(uids) < inferred_capacity:
            uids = list(uids) + [0] * (inferred_capacity - len(uids))
        else:
            uids = list(uids[:inferred_capacity])
        if size == 0:
            return [], [], [], inferred_capacity
        physical_order = list(range(size)) if size < inferred_capacity else [(pos + i) % inferred_capacity for i in range(size)]
        items = [(states[i], policies[i], float(values[i])) for i in physical_order]
        ordered_priorities = [float(priorities[i]) for i in physical_order]
        ordered_uids = [int(uids[i]) if i < len(uids) else 0 for i in physical_order]
        return items, ordered_priorities, ordered_uids, inferred_capacity

    @classmethod
    def _ordered_samples_from_compact_state(cls, data: dict[str, Any]):
        states = list(data.get('states', []))
        policies = list(data.get('policies', []))
        values = list(data.get('values', []))
        priorities = list(data.get('priorities', []))
        uids = list(data.get('uids', []))
        size = max(0, min(int(data.get('size', len(states))), len(states), len(policies), len(values), len(priorities)))
        if len(uids) < size:
            uids = list(uids) + [0] * (size - len(uids))
        else:
            uids = list(uids[:size])
        items = [(states[i], policies[i], float(values[i])) for i in range(size)]
        ordered_priorities = [float(priorities[i]) for i in range(size)]
        ordered_uids = [int(uids[i]) for i in range(size)]
        inferred_capacity = max(1, int(data.get('capacity', size or 1)))
        return items, ordered_priorities, ordered_uids, inferred_capacity

    @classmethod
    def _ordered_samples_from_sparse_state(cls, data: dict[str, Any]):
        states = np.asarray(data.get('states', np.zeros((0, 20, 8, 8), dtype=np.float16)), dtype=np.float16)
        values = np.asarray(data.get('values', np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        priorities = np.asarray(data.get('priorities', np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        uids = np.asarray(data.get('uids', np.zeros((0,), dtype=np.int64)), dtype=np.int64)
        lengths = np.asarray(data.get('policy_lengths', np.zeros((0,), dtype=np.int32)), dtype=np.int32)
        flat_indices = np.asarray(data.get('policy_indices', np.zeros((0,), dtype=np.uint16)), dtype=np.uint16)
        flat_probs = np.asarray(data.get('policy_probs', np.zeros((0,), dtype=np.float16)), dtype=np.float16)
        size = max(0, min(int(data.get('size', len(states))), len(states), len(values), len(priorities), len(lengths)))
        if len(uids) < size:
            uids = np.concatenate([uids, np.zeros(size - len(uids), dtype=np.int64)])
        else:
            uids = uids[:size]
        offset = 0
        items = []
        for idx in range(size):
            length = int(lengths[idx])
            next_offset = offset + max(0, length)
            items.append((
                states[idx],
                PackedPolicy(
                    flat_indices[offset:next_offset].astype(np.uint16, copy=True),
                    flat_probs[offset:next_offset].astype(np.float16, copy=True),
                ),
                float(values[idx]),
            ))
            offset = next_offset
        return items, priorities[:size].astype(np.float32, copy=True).tolist(), uids[:size].astype(np.int64, copy=True).tolist(), max(1, int(data.get('capacity', size or 1)))

    @classmethod
    def _from_sparse_state_dict(cls, data: dict[str, Any], cfg: AppConfig | ReplayConfig | None = None) -> 'ReplayBuffer':
        buffer = cls(cfg)
        states = np.asarray(data.get('states', np.zeros((0, *buffer.state_shape), dtype=np.float16)), dtype=np.float16)
        values = np.asarray(data.get('values', np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        priorities = np.asarray(data.get('priorities', np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        uids = np.asarray(data.get('uids', np.zeros((0,), dtype=np.int64)), dtype=np.int64)
        lengths = np.asarray(data.get('policy_lengths', np.zeros((0,), dtype=np.int32)), dtype=np.int32)
        flat_indices = np.asarray(data.get('policy_indices', np.zeros((0,), dtype=np.uint16)), dtype=np.uint16)
        flat_probs = np.asarray(data.get('policy_probs', np.zeros((0,), dtype=np.float16)), dtype=np.float16)

        if states.ndim >= 2 and states.shape[1:]:
            buffer.state_shape = tuple(int(dim) for dim in states.shape[1:])
            buffer.states = np.zeros((buffer.capacity, *buffer.state_shape), dtype=np.float16)

        size = max(0, min(int(data.get('size', len(states))), len(states), len(values), len(priorities), len(lengths)))
        if len(uids) < size:
            uids = np.concatenate([uids, np.zeros((size - len(uids),), dtype=np.int64)])
        else:
            uids = uids[:size]

        if size == 0:
            buffer.policy_size = max(buffer.policy_size, int(data.get('policy_num_actions', buffer.policy_size)))
            return buffer

        keep_size = min(size, buffer.capacity)
        start_sample = size - keep_size
        end_sample = size
        prefix = np.concatenate(([0], np.cumsum(lengths[:size], dtype=np.int64)))
        start_offset = int(prefix[start_sample])
        end_offset = int(prefix[end_sample])

        kept_states = np.ascontiguousarray(states[start_sample:end_sample], dtype=np.float16)
        if kept_states.shape[1:] != buffer.state_shape:
            buffer.state_shape = tuple(int(dim) for dim in kept_states.shape[1:])
            buffer.states = np.zeros((buffer.capacity, *buffer.state_shape), dtype=np.float16)
        buffer.states[:keep_size] = kept_states
        buffer.values[:keep_size] = values[start_sample:end_sample].astype(np.float32, copy=False)
        clipped_priorities = np.clip(
            priorities[start_sample:end_sample].astype(np.float32, copy=False),
            float(buffer.replay_cfg.eps),
            10.0,
        )
        buffer._priorities[:keep_size] = clipped_priorities
        kept_uids = uids[start_sample:end_sample].astype(np.int64, copy=False)
        buffer.uids[:keep_size] = kept_uids
        buffer.policy_size = max(buffer.policy_size, int(data.get('policy_num_actions', buffer.policy_size)))

        local_offsets = prefix[start_sample:end_sample + 1] - start_offset
        kept_flat_indices = flat_indices[start_offset:end_offset]
        kept_flat_probs = flat_probs[start_offset:end_offset]
        for row in range(keep_size):
            row_start = int(local_offsets[row])
            row_end = int(local_offsets[row + 1])
            if row_end <= row_start:
                buffer.policies[row] = EMPTY_PACKED_POLICY
                continue
            buffer.policies[row] = PackedPolicy(
                indices=np.ascontiguousarray(kept_flat_indices[row_start:row_end].astype(np.uint16, copy=True)),
                probs=np.ascontiguousarray(kept_flat_probs[row_start:row_end].astype(np.float16, copy=True)),
            )

        buffer.size = int(keep_size)
        buffer.pos = buffer.size % buffer.capacity
        buffer.next_uid = max(int(data.get('next_uid', 0) or 0), (int(kept_uids.max()) + 1) if kept_uids.size else 1)
        buffer._last_saved_uid = int(kept_uids.max()) if kept_uids.size else 0
        buffer._approx_max_priority = float(clipped_priorities.max()) if clipped_priorities.size else 1.0
        buffer._invalidate_sampling_caches()
        buffer.rebuild_seen_hashes()
        return buffer

    @classmethod
    def _from_manifest_serialized(cls, data: dict[str, Any], cfg: AppConfig | ReplayConfig | None = None, *, source_path: str | Path) -> 'ReplayBuffer':
        manifest = Path(source_path)
        shard_dir_name = data.get('shard_dir') or manifest.with_suffix(manifest.suffix + '.d').name
        shard_dir = manifest.parent / shard_dir_name

        shard_states = []
        shard_values = []
        shard_priorities = []
        shard_uids = []
        shard_lengths = []
        shard_policy_indices = []
        shard_policy_probs = []

        for shard in data.get('shards', []):
            shard_file = shard_dir / str(shard.get('file'))
            if not shard_file.exists():
                continue
            payload = torch.load(shard_file, map_location='cpu', weights_only=False)
            if not isinstance(payload, dict) or payload.get('format') != cls.FORMAT:
                continue
            shard_size = max(
                0,
                min(
                    int(payload.get('size', 0)),
                    len(payload.get('states', [])),
                    len(payload.get('values', [])),
                    len(payload.get('priorities', [])),
                    len(payload.get('policy_lengths', [])),
                ),
            )
            if shard_size <= 0:
                continue
            lengths = np.asarray(payload.get('policy_lengths', np.zeros((0,), dtype=np.int32)), dtype=np.int32)[:shard_size]
            total_nnz = int(lengths.sum())
            shard_states.append(np.asarray(payload.get('states', np.zeros((0, *cls(cfg).state_shape), dtype=np.float16)), dtype=np.float16)[:shard_size])
            shard_values.append(np.asarray(payload.get('values', np.zeros((0,), dtype=np.float32)), dtype=np.float32)[:shard_size])
            shard_priorities.append(np.asarray(payload.get('priorities', np.zeros((0,), dtype=np.float32)), dtype=np.float32)[:shard_size])
            shard_uids.append(np.asarray(payload.get('uids', np.zeros((0,), dtype=np.int64)), dtype=np.int64)[:shard_size])
            shard_lengths.append(lengths)
            shard_policy_indices.append(np.asarray(payload.get('policy_indices', np.zeros((0,), dtype=np.uint16)), dtype=np.uint16)[:total_nnz])
            shard_policy_probs.append(np.asarray(payload.get('policy_probs', np.zeros((0,), dtype=np.float16)), dtype=np.float16)[:total_nnz])

        if not shard_states:
            buffer = cls(cfg)
            buffer._saved_shards = list(data.get('shards', []))
            buffer._last_saved_uid = int(data.get('last_saved_uid', 0))
            if 'policy_num_actions' in data:
                buffer.policy_size = max(buffer.policy_size, int(data.get('policy_num_actions', buffer.policy_size)))
            return buffer

        merged_state = {
            'version': cls.VERSION,
            'format': cls.FORMAT,
            'capacity': int(data.get('capacity', sum(arr.shape[0] for arr in shard_states))),
            'size': int(sum(arr.shape[0] for arr in shard_states)),
            'pos': 0,
            'next_uid': int(data.get('next_uid', 0)),
            'state_shape': tuple(np.asarray(shard_states[0]).shape[1:]),
            'state_dtype': 'float16',
            'policy_num_actions': int(data.get('policy_num_actions', DEFAULT_POLICY_SIZE)),
            'states': np.ascontiguousarray(np.concatenate(shard_states, axis=0), dtype=np.float16),
            'policy_lengths': np.ascontiguousarray(np.concatenate(shard_lengths, axis=0), dtype=np.int32),
            'policy_indices': np.ascontiguousarray(np.concatenate(shard_policy_indices, axis=0), dtype=np.uint16) if shard_policy_indices else np.zeros((0,), dtype=np.uint16),
            'policy_probs': np.ascontiguousarray(np.concatenate(shard_policy_probs, axis=0), dtype=np.float16) if shard_policy_probs else np.zeros((0,), dtype=np.float16),
            'values': np.ascontiguousarray(np.concatenate(shard_values, axis=0), dtype=np.float32),
            'priorities': np.ascontiguousarray(np.concatenate(shard_priorities, axis=0), dtype=np.float32),
            'uids': np.ascontiguousarray(np.concatenate(shard_uids, axis=0), dtype=np.int64),
        }
        buffer = cls._from_sparse_state_dict(merged_state, cfg=cfg)
        buffer._saved_shards = list(data.get('shards', []))
        buffer._last_saved_uid = int(data.get('last_saved_uid', buffer._last_saved_uid))
        return buffer

    @classmethod
    def _ordered_samples_from_manifest(cls, data: dict[str, Any], manifest_path: str | Path):
        manifest = Path(manifest_path)
        shard_dir_name = data.get('shard_dir') or manifest.with_suffix(manifest.suffix + '.d').name
        shard_dir = manifest.parent / shard_dir_name
        items: list[tuple[Any, Any, float]] = []
        priorities: list[float] = []
        uids: list[int] = []
        for shard in data.get('shards', []):
            shard_file = shard_dir / str(shard.get('file'))
            if not shard_file.exists():
                continue
            shard_payload = torch.load(shard_file, map_location='cpu', weights_only=False)
            shard_items, shard_priorities, shard_uids, _ = cls._ordered_samples_from_sparse_state(shard_payload)
            items.extend(shard_items)
            priorities.extend(shard_priorities)
            uids.extend(shard_uids)
        return items, priorities, uids, max(1, int(data.get('capacity', len(items) or 1)))

    @classmethod
    def from_serialized(cls, data: Any, cfg: AppConfig | ReplayConfig | None = None, *, source_path: str | Path | None = None) -> 'ReplayBuffer':
        if isinstance(data, cls):
            data = data.to_state_dict()
        if isinstance(data, dict):
            fmt = data.get('format')
            if fmt == MANIFEST_FORMAT:
                if source_path is None:
                    raise ValueError('Manifest replay buffer requires source_path for shard resolution')
                return cls._from_manifest_serialized(data, cfg=cfg, source_path=source_path)
            if fmt == cls.FORMAT:
                return cls._from_sparse_state_dict(data, cfg=cfg)
            if fmt == 'ring_replay_buffer_compact':
                items, priorities, uids, _ = cls._ordered_samples_from_compact_state(data)
                return cls._from_ordered_samples(items, priorities, cfg=cfg, uids=uids)
            if {'states', 'policies', 'values', 'priorities'}.issubset(data.keys()):
                items, priorities, uids, _ = cls._ordered_samples_from_ring_state(data)
                return cls._from_ordered_samples(items, priorities, cfg=cfg, uids=uids)
            if 'buffer' in data:
                return cls._from_ordered_samples(list(data.get('buffer', [])), list(data.get('priorities', [])), cfg=cfg)
        if hasattr(data, 'buffer'):
            priorities = list(getattr(data, 'priorities', [])) if hasattr(data, 'priorities') else []
            return cls._from_ordered_samples(list(getattr(data, 'buffer', [])), priorities, cfg=cfg)
        raise ValueError('Unsupported replay buffer serialization format')

    @classmethod
    def load_from_path(
            cls,
            path: str,
            cfg: AppConfig | ReplayConfig | None = None,
            *,
            prompt_on_missing_shards: bool = False,
            strict_missing_shards: bool = False,
    ) -> 'ReplayBuffer':
        raw = torch.load(path, map_location='cpu', weights_only=False)

        if isinstance(raw, dict) and raw.get('format') == MANIFEST_FORMAT:
            shard_dir, present, missing = _inspect_manifest_shards(raw, path)

            if missing:
                print('=' * 80)
                print('WARNING: Missing replay buffer shards detected')
                print('manifest =', path)
                print('shard_dir =', shard_dir)
                print('missing_count =', len(missing))
                for shard in missing[:20]:
                    print(' -', shard.get('file'))
                if len(missing) > 20:
                    print(f"... and {len(missing) - 20} more")
                print('=' * 80)

                if strict_missing_shards and not prompt_on_missing_shards:
                    raise RuntimeError('Aborted: missing replay buffer shards detected.')

                if prompt_on_missing_shards:
                    answer = input(
                        'Continue without missing shards and remove them from manifest? [y/N]: '
                    ).strip().lower()

                    if answer not in {'y', 'yes'}:
                        raise RuntimeError('Aborted: missing replay buffer shards detected.')

                    _rewrite_manifest_without_missing_shards(path, raw, present)
                    raw = torch.load(path, map_location='cpu', weights_only=False)
                elif strict_missing_shards:
                    raise RuntimeError('Aborted: missing replay buffer shards detected.')

        return cls.from_serialized(raw, cfg=cfg, source_path=path)

    def load(self, path: str) -> None:
        loaded = self.load_from_path(path, cfg=self.cfg)
        self.cfg = loaded.cfg
        self.replay_cfg = loaded.replay_cfg
        self.training_cfg = loaded.training_cfg
        self.capacity = loaded.capacity
        self.state_shape = loaded.state_shape
        self.policy_size = loaded.policy_size
        self.states = loaded.states
        self.policies = loaded.policies
        self.values = loaded.values
        self._priorities = loaded._priorities
        self.uids = loaded.uids
        self.size = loaded.size
        self.pos = loaded.pos
        self.next_uid = loaded.next_uid
        self.seen_hashes = loaded.seen_hashes
        self.state_hash_to_physicals = loaded.state_hash_to_physicals
        self.min_keep_per_state = loaded.min_keep_per_state
        self.max_keep_per_state = loaded.max_keep_per_state
        self._last_saved_uid = loaded._last_saved_uid
        self._saved_shards = loaded._saved_shards
        self._approx_max_priority = loaded._approx_max_priority
        self._invalidate_sampling_caches()






