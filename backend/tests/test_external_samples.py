import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from app.game.move_encoding import NUM_MOVES
from app.infra.config import AppConfig, ExternalDataConfig, ReplayConfig, SystemConfig, TrainingConfig
from app.training.external_samples import load_external_samples
from app.training.replay_buffer import ReplayBuffer


class ExternalSamplesTests(unittest.TestCase):
    def _cfg(self, *, input_planes: int = 20, capacity: int = 8, min_fullmove: int = 0, max_fullmove: int = 0):
        cfg = AppConfig(
            external=ExternalDataConfig(min_fullmove=min_fullmove, max_fullmove=max_fullmove),
            replay=ReplayConfig(capacity=capacity, prioritized=False),
            system=SystemConfig(max_fullmove=120),
            training=TrainingConfig(batch_size=2),
        )
        if input_planes != cfg.model.input_planes:
            cfg = AppConfig(
                model=type(cfg.model)(
                    input_planes=input_planes,
                    channels=cfg.model.channels,
                    res_blocks=cfg.model.res_blocks,
                    value_dropout=cfg.model.value_dropout,
                ),
                training=cfg.training,
                external=cfg.external,
                replay=cfg.replay,
                mcts=cfg.mcts,
                selfplay=cfg.selfplay,
                arena=cfg.arena,
                system=cfg.system,
            )
        return cfg

    def _write_npz(self, path: Path, *, input_planes: int = 20, count: int = 3, fullmoves: list[int] | None = None):
        states = np.zeros((count, input_planes, 8, 8), dtype=np.float16)
        for i in range(count):
            states[i, 0, 0, 0] = i + 1
            if fullmoves is not None:
                states[i, 19, :, :] = float(fullmoves[i]) / 120.0
        policy_indices = np.arange(count, dtype=np.int32)
        values = np.linspace(-1.0, 1.0, count, dtype=np.float32)
        np.savez_compressed(
            path,
            states=states,
            policy_indices=policy_indices,
            values=values,
            input_planes=np.asarray([input_planes], dtype=np.int16),
            policy_size=np.asarray([NUM_MOVES], dtype=np.int32),
        )

    def test_load_external_samples_matches_runtime_shapes(self):
        cfg = self._cfg()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'ext.npz'
            self._write_npz(path, input_planes=20, count=2)
            loaded = list(load_external_samples(path, cfg))

        self.assertEqual(len(loaded), 2)
        state, policy, value = loaded[0]
        self.assertEqual(state.shape, (20, 8, 8))
        self.assertEqual(policy.indices.tolist(), [0])
        self.assertAlmostEqual(float(policy.probs[0]), 1.0, places=3)
        self.assertEqual(value, -1.0)

    def test_load_external_samples_rejects_input_plane_mismatch(self):
        cfg = self._cfg(input_planes=20)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'ext_bad.npz'
            self._write_npz(path, input_planes=24, count=1)
            with self.assertRaises(ValueError):
                list(load_external_samples(path, cfg))

    def test_external_injection_on_copy_does_not_mutate_main_buffer(self):
        cfg = self._cfg(capacity=10)
        base = ReplayBuffer(cfg)
        dense_policy = np.zeros(NUM_MOVES, dtype=np.float32)
        dense_policy[10] = 1.0
        base.add(torch.zeros((20, 8, 8)), dense_policy, 0.25)
        self.assertEqual(len(base), 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'ext.npz'
            self._write_npz(path, input_planes=20, count=2)

            train_buffer = ReplayBuffer.from_serialized(base.to_state_dict(), cfg=cfg)
            for state, policy, value in load_external_samples(path, cfg):
                train_buffer.add(state, policy, value)

            self.assertEqual(len(base), 1)
            self.assertEqual(len(train_buffer), 3)

    def test_load_external_samples_can_filter_by_fullmove_range(self):
        cfg = self._cfg(min_fullmove=2, max_fullmove=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'ext.npz'
            self._write_npz(path, input_planes=20, count=4, fullmoves=[1, 2, 3, 13])
            loaded = list(load_external_samples(path, cfg))

        self.assertEqual(len(loaded), 2)
        self.assertEqual(float(loaded[0][0][19, 0, 0]), np.float16(2 / 120))
        self.assertEqual(float(loaded[1][0][19, 0, 0]), np.float16(3 / 120))


if __name__ == '__main__':
    unittest.main()
