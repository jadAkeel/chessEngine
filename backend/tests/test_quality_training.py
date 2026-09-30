"""Multi-move policy targets in shards and the wrapper's cosine LR schedule."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from app.infra.config import AppConfig, ExternalDataConfig, SystemConfig
from app.training.external_samples import load_external_samples_with_stats
from scripts.kaggle_train_external import _cosine_lr


def _cfg() -> AppConfig:
    return AppConfig(
        external=ExternalDataConfig(dedup=False, shuffle=False),
        system=SystemConfig(max_fullmove=120),
    )


def _write_shard(path: Path, topk_indices=None, topk_probs=None) -> None:
    count = 2
    states = np.zeros((count, 20, 8, 8), dtype=np.float16)
    for i in range(count):
        states[i, 0, 0, i] = 1.0
        states[i, 19, :, :] = 20.0 / 120.0
    arrays = dict(
        states=states,
        policy_indices=np.asarray([5, 9], dtype=np.int32),
        values=np.asarray([0.25, -0.5], dtype=np.float32),
    )
    if topk_indices is not None:
        arrays["policy_topk_indices"] = np.asarray(topk_indices, dtype=np.int16)
        arrays["policy_topk_probs"] = np.asarray(topk_probs, dtype=np.float16)
    np.savez_compressed(path, **arrays)


def test_shard_without_topk_keeps_one_hot_policy(tmp_path):
    shard = tmp_path / "shard_0.npz"
    _write_shard(shard)
    samples = load_external_samples_with_stats(shard, _cfg()).samples
    assert [s[1].indices.tolist() for s in samples] == [[5], [9]]
    assert [s[1].probs.tolist() for s in samples] == [[1.0], [1.0]]


def test_topk_rows_become_normalised_multi_move_policies(tmp_path):
    shard = tmp_path / "shard_0.npz"
    _write_shard(
        shard,
        topk_indices=[[5, 7, 11, -1], [9, -1, -1, -1]],
        topk_probs=[[0.6, 0.3, 0.3, 0.0], [1.0, 0.0, 0.0, 0.0]],
    )
    first, second = load_external_samples_with_stats(shard, _cfg()).samples
    assert first[1].indices.tolist() == [5, 7, 11]
    assert float(first[1].probs.astype(np.float64).sum()) == pytest.approx(1.0, abs=2e-3)
    assert float(first[1].probs[0]) == pytest.approx(0.5, abs=2e-3)
    # A row with one real move falls back to the one-hot target.
    assert second[1].indices.tolist() == [9]
    assert second[1].probs.tolist() == [1.0]


def test_mismatched_topk_shapes_are_rejected(tmp_path):
    shard = tmp_path / "shard_0.npz"
    _write_shard(shard, topk_indices=[[5, 7], [9, -1]], topk_probs=[[1.0], [1.0]])
    with pytest.raises(ValueError, match="policy_topk"):
        load_external_samples_with_stats(shard, _cfg())


def test_cosine_lr_runs_from_start_to_final_over_the_budget():
    start, final, hours = 3e-5, 3e-6, 10.0
    assert _cosine_lr(start, final, 0.0, hours) == pytest.approx(start)
    assert _cosine_lr(start, final, hours * 1800.0, hours) == pytest.approx((start + final) / 2)
    assert _cosine_lr(start, final, hours * 3600.0, hours) == pytest.approx(final)
    # Past the budget it stays at the final value; with no budget it never decays.
    assert _cosine_lr(start, final, hours * 7200.0, hours) == pytest.approx(final)
    assert _cosine_lr(start, final, 1e9, 0.0) == pytest.approx(start)
    values = [_cosine_lr(start, final, t * 3600.0, hours) for t in range(11)]
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert not any(math.isnan(v) for v in values)
