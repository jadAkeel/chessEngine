from __future__ import annotations

import numpy as np
import pytest
import torch

from app.game.move_encoding import NUM_MOVES
from app.infra.config import AppConfig, ModelConfig, ReplayConfig, TrainingConfig
from app.model.network import ChessNet
from app.training import trainer
from app.training.replay_buffer import ReplayBuffer


def _cfg(steps: int = 12) -> AppConfig:
    return AppConfig(
        model=ModelConfig(input_planes=20, channels=8, res_blocks=1),
        replay=ReplayConfig(capacity=64, prioritized=False),
        training=TrainingConfig(batch_size=8, epochs=1, train_steps_per_iter=steps, use_amp=False),
    )


def _filled_buffer(cfg: AppConfig) -> ReplayBuffer:
    rng = np.random.default_rng(0)
    buffer = ReplayBuffer(cfg)
    for i in range(32):
        policy = np.zeros(NUM_MOVES, dtype=np.float32)
        policy[int(rng.integers(NUM_MOVES))] = 1.0
        buffer.add(torch.from_numpy(rng.random((20, 8, 8), dtype=np.float32)), policy, float(i % 3 - 1))
    return buffer


def _poison_on_calls(model: ChessNet, bad_calls: set[int] | None):
    """Make chosen forward passes return NaN and poison BatchNorm stats, like an fp16 overflow."""
    original = model.forward
    calls = {"n": 0}

    def forward(x):
        calls["n"] += 1
        policy, value = original(x)
        if bad_calls is None or calls["n"] in bad_calls:
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.running_mean.fill_(float("nan"))
            return policy * float("nan"), value * float("nan")
        return policy, value

    model.forward = forward


def _all_finite(model: ChessNet) -> bool:
    return all(torch.isfinite(t).all() for t in model.state_dict().values() if t.is_floating_point())


def test_nan_step_rolls_back_and_training_finishes_finite():
    cfg = _cfg()
    model = ChessNet(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _poison_on_calls(model, bad_calls={5})

    stats = trainer.train_model(model, optimizer, _filled_buffer(cfg), device="cpu", cfg=cfg)

    assert stats["nonfinite_steps"] == 1
    assert stats["steps"] == 11
    assert np.isfinite(stats["loss"])
    assert _all_finite(model), "BatchNorm stats poisoned by the NaN pass must be restored"


def test_persistent_nan_raises_instead_of_training_on_garbage(monkeypatch):
    monkeypatch.setattr(trainer, "MAX_NONFINITE_STEPS", 3)
    cfg = _cfg()
    model = ChessNet(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _poison_on_calls(model, bad_calls=None)

    with pytest.raises(RuntimeError, match="diverged"):
        trainer.train_model(model, optimizer, _filled_buffer(cfg), device="cpu", cfg=cfg)


def test_refuses_to_start_from_nan_weights():
    cfg = _cfg()
    model = ChessNet(cfg)
    with torch.no_grad():
        next(model.parameters()).fill_(float("nan"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with pytest.raises(RuntimeError, match="non-finite before training"):
        trainer.train_model(model, optimizer, _filled_buffer(cfg), device="cpu", cfg=cfg)
