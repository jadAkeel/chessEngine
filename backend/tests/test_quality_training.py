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


# ---- policy-only (value-masked) shards ----


def _masked_cfg(**external) -> AppConfig:
    base = dict(dedup=False, shuffle=False)
    base.update(external)
    return AppConfig(external=ExternalDataConfig(**base), system=SystemConfig(max_fullmove=120))


def test_value_mask_glob_keeps_samples_with_nan_values(tmp_path):
    masked = tmp_path / "lichess_eval_v2_shard_0.npz"
    games = tmp_path / "feb_jul_shard_0.npz"
    _write_shard(masked)
    _write_shard(games)
    cfg = _masked_cfg(value_mask_glob="lichess_eval*, other*")

    masked_samples = load_external_samples_with_stats(masked, cfg).samples
    assert len(masked_samples) == 2
    assert all(math.isnan(s[2]) for s in masked_samples)
    assert [s[1].indices.tolist() for s in masked_samples] == [[5], [9]]

    game_values = [s[2] for s in load_external_samples_with_stats(games, cfg).samples]
    assert game_values == pytest.approx([0.25, -0.5])


def test_nan_values_in_unmasked_shards_are_still_dropped(tmp_path):
    shard = tmp_path / "feb_jul_shard_0.npz"
    _write_shard(shard)
    data = dict(np.load(shard))
    data["values"] = np.asarray([np.nan, 0.5], dtype=np.float32)
    np.savez_compressed(shard, **data)
    result = load_external_samples_with_stats(shard, _masked_cfg(value_mask_glob="lichess_eval*"))
    assert [s[2] for s in result.samples] == [0.5]
    assert result.stats["bad_value"] == 1


def test_use_topk_policy_false_ignores_topk_arrays(tmp_path):
    shard = tmp_path / "shard_0.npz"
    _write_shard(
        shard,
        topk_indices=[[5, 7, 11, -1], [9, 3, -1, -1]],
        topk_probs=[[0.6, 0.3, 0.3, 0.0], [0.5, 0.5, 0.0, 0.0]],
    )
    samples = load_external_samples_with_stats(shard, _masked_cfg(use_topk_policy=False)).samples
    assert [s[1].indices.tolist() for s in samples] == [[5], [9]]


def test_masked_value_loss_ignores_nan_targets():
    import torch

    from app.training.trainer import _masked_value_loss

    pred = torch.tensor([[0.5], [0.0], [-1.0], [0.2]], requires_grad=True)
    target = torch.tensor([1.0, float("nan"), 0.0, float("nan")])
    weights = torch.ones(4)
    per_sample, loss, mask = _masked_value_loss(pred, target, weights)
    assert mask.tolist() == [True, False, True, False]
    assert per_sample.tolist() == pytest.approx([0.25, 0.0, 1.0, 0.0])
    assert loss.item() == pytest.approx((0.25 + 1.0) / 2)
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad[1].item() == 0.0 and pred.grad[3].item() == 0.0

    all_masked = torch.full((2,), float("nan"))
    _, loss, _ = _masked_value_loss(pred[:2], all_masked, torch.ones(2))
    assert loss.item() == 0.0


class _ConstantModel:
    """Uniform policy and a fixed value, enough for evaluate_model_on_samples."""

    training = False

    def __init__(self, value: float):
        self.value = value

    def eval(self):
        return self

    def train(self, mode: bool = True):
        return self

    def __call__(self, states):
        import torch

        from app.game.move_encoding import NUM_MOVES

        n = states.shape[0]
        return torch.zeros(n, NUM_MOVES), torch.full((n, 1), self.value)


def test_validation_value_loss_averages_only_samples_with_values():
    from app.training.replay_buffer import PackedPolicy
    from app.training.trainer import evaluate_model_on_samples

    state = np.zeros((20, 8, 8), dtype=np.float16)
    policy = PackedPolicy(indices=np.array([5], dtype=np.uint16), probs=np.array([1.0], dtype=np.float16))
    # Batch 1 is all policy-only; batch 2 has values 1.0 and 0.0 against a 0.5 prediction.
    samples = [(state, policy, float("nan"))] * 2 + [(state, policy, 1.0), (state, policy, 0.0)]
    stats = evaluate_model_on_samples(_ConstantModel(0.5), samples, batch_size=2, cfg=_cfg())
    assert stats["value_samples"] == 2
    assert stats["value_loss"] == pytest.approx(0.25)
    assert math.isfinite(stats["loss"])


def test_replay_buffer_accepts_and_samples_policy_only_values():
    from dataclasses import replace

    from app.training.replay_buffer import ReplayBuffer

    cfg = AppConfig()
    buffer = ReplayBuffer(replace(cfg, replay=replace(cfg.replay, capacity=16)))
    policy = np.zeros(4672, dtype=np.float32)
    policy[7] = 1.0
    for i in range(8):
        state = np.zeros((20, 8, 8), dtype=np.float16)
        state[0, 0, i] = 1.0
        buffer.add(state, policy, float("nan") if i % 2 else 0.5)
    assert len(buffer) == 8
    _, _, values, indices, weights = buffer.sample_batch(batch_size=8)
    assert np.isnan(values).sum() == 4
    assert np.all(np.isfinite(weights))
    buffer.update_priorities(indices, np.full(len(indices), 0.3, dtype=np.float32))


def test_wrapper_writes_value_mask_and_topk_switch(tmp_path):
    import argparse

    import yaml

    from app.infra.config import load_config
    from scripts.kaggle_train_external import _write_config, build_parser

    args = build_parser().parse_args([
        "--samples-path", str(tmp_path / "shards"),
        "--save-dir", str(tmp_path / "ckpt"),
        "--config-out", str(tmp_path / "cfg.yaml"),
        "--value-mask-glob", "lichess_eval*",
        "--no-topk-policy",
    ])
    assert isinstance(args, argparse.Namespace)
    path = _write_config(args)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["external"]["value_mask_glob"] == "lichess_eval*"
    assert data["external"]["use_topk_policy"] is False
    cfg = load_config(str(path))
    assert cfg.external.value_mask_glob == "lichess_eval*"
    assert cfg.external.use_topk_policy is False

    plain = build_parser().parse_args(["--config-out", str(tmp_path / "plain.yaml")])
    plain_data = yaml.safe_load(_write_config(plain).read_text(encoding="utf-8"))
    assert "value_mask_glob" not in plain_data["external"]
    assert "use_topk_policy" not in plain_data["external"]
