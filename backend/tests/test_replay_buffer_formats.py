import tempfile
import unittest
from collections import deque
from pathlib import Path

import numpy as np
import torch

from app.infra.config import AppConfig, ReplayConfig, TrainingConfig
from app.training.replay_buffer import ReplayBuffer, SparsePolicyBatch


class _LegacyObject:
    def __init__(self, samples, priorities, capacity):
        self.buffer = deque(samples, maxlen=capacity)
        self.priorities = deque(priorities, maxlen=capacity)


def _cfg(capacity: int = 4):
    return AppConfig(
        replay=ReplayConfig(capacity=capacity, prioritized=True),
        training=TrainingConfig(batch_size=2),
    )


class ReplayBufferFormatTests(unittest.TestCase):
    def _sample(self, idx: int):
        state = torch.full((2, 2), float(idx))
        policy = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        value = float(idx % 3 - 1)
        return state, policy, value

    def test_new_ring_buffer_eviction_keeps_latest(self):
        buffer = ReplayBuffer(_cfg(capacity=3))
        for idx in range(5):
            buffer.add(*self._sample(idx))

        self.assertEqual(len(buffer), 3)
        kept_values = [int(sample[0][0, 0].item()) for sample in buffer.buffer]
        self.assertEqual(kept_values, [2, 3, 4])
        self.assertEqual(buffer.pos, 2)

    def test_loads_legacy_object_and_rebuilds_dedup(self):
        samples = [self._sample(i) for i in range(3)]
        priorities = [0.5, 0.7, 0.9]
        legacy = _LegacyObject(samples, priorities, capacity=3)

        buffer = ReplayBuffer.from_serialized(legacy, cfg=_cfg(capacity=5))
        self.assertEqual(len(buffer), 3)
        self.assertEqual(buffer.capacity, 5)
        self.assertEqual(len(buffer.seen_hashes), 3)

        buffer.add(*samples[-1])
        self.assertEqual(len(buffer), 3)

    def test_loads_legacy_dict_without_dropping_items_when_priorities_shorter(self):
        samples = [self._sample(i) for i in range(3)]
        payload = {'buffer': samples, 'priorities': [0.25]}

        buffer = ReplayBuffer.from_serialized(payload, cfg=_cfg(capacity=5))
        self.assertEqual(len(buffer), 3)
        self.assertEqual(list(buffer.priorities), [0.25, 0.25, 0.25])

    def test_round_trip_new_format(self):
        buffer = ReplayBuffer(_cfg(capacity=4))
        for idx in range(3):
            buffer.add(*self._sample(idx))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'replay.pt'
            buffer.save(str(path))
            loaded = ReplayBuffer.load_from_path(str(path), cfg=_cfg(capacity=4))

        self.assertEqual(len(loaded), 3)
        self.assertEqual(loaded.capacity, 4)
        kept_values = [int(sample[0][0, 0].item()) for sample in loaded.buffer]
        self.assertEqual(kept_values, [0, 1, 2])
        self.assertEqual(len(loaded.seen_hashes), 3)

    def test_new_format_stores_sparse_policy_payload(self):
        buffer = ReplayBuffer(_cfg(capacity=4))
        for idx in range(3):
            buffer.add(*self._sample(idx))

        state_dict = buffer.to_state_dict()
        self.assertEqual(state_dict['format'], ReplayBuffer.FORMAT)
        self.assertIn('policy_lengths', state_dict)
        self.assertIn('policy_indices', state_dict)
        self.assertIn('policy_probs', state_dict)
        self.assertNotIn('policies', state_dict)


    def test_sample_batch_returns_sparse_targets_without_dense_matrix(self):
        buffer = ReplayBuffer(_cfg(capacity=8))
        for idx in range(5):
            buffer.add(*self._sample(idx))

        states, policies, values, indices, weights = buffer.sample_batch(batch_size=3)

        self.assertEqual(states.shape[0], 3)
        self.assertIsInstance(policies, SparsePolicyBatch)
        self.assertEqual(policies.lengths.shape, (3,))
        self.assertEqual(policies.indices.shape[0], 3)
        self.assertEqual(policies.probs.shape[0], 3)
        self.assertEqual(len(values), 3)
        self.assertEqual(len(indices), 3)
        self.assertEqual(len(weights), 3)


    def test_manifest_save_writes_incremental_shards_and_round_trips(self):
        buffer = ReplayBuffer(_cfg(capacity=8))
        for idx in range(3):
            buffer.add(*self._sample(idx))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'replay.pt'
            buffer.save(str(path))
            shard_dir = Path(str(path) + '.d')
            first_shards = sorted(shard_dir.glob('*.pt'))
            self.assertEqual(len(first_shards), 1)
            first_payload = torch.load(first_shards[0], map_location='cpu', weights_only=False)
            first_count = int(first_payload['size'])

            for idx in range(3, 5):
                buffer.add(*self._sample(idx))
            buffer.save(str(path))

            second_shards = sorted(shard_dir.glob('*.pt'))
            self.assertEqual(len(second_shards), 2)
            second_payload = torch.load(second_shards[-1], map_location='cpu', weights_only=False)
            self.assertEqual(first_count, 3)
            self.assertEqual(int(second_payload['size']), 2)

            loaded = ReplayBuffer.load_from_path(str(path), cfg=_cfg(capacity=8))

        self.assertEqual(len(loaded), 5)
        kept_values = [int(sample[0][0, 0].item()) for sample in loaded.buffer]
        self.assertEqual(kept_values, [0, 1, 2, 3, 4])

    def test_manifest_prunes_evicted_shards(self):
        buffer = ReplayBuffer(_cfg(capacity=3))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'replay.pt'
            for idx in range(3):
                buffer.add(*self._sample(idx))
            buffer.save(str(path))

            for idx in range(3, 6):
                buffer.add(*self._sample(idx))
            buffer.save(str(path))

            shard_dir = Path(str(path) + '.d')
            shards = sorted(shard_dir.glob('*.pt'))
            self.assertEqual(len(shards), 1)

            loaded = ReplayBuffer.load_from_path(str(path), cfg=_cfg(capacity=3))

        kept_values = [int(sample[0][0, 0].item()) for sample in loaded.buffer]
        self.assertEqual(kept_values, [3, 4, 5])
    def test_loads_ring_dict_and_resizes_to_cfg_capacity(self):
        ring_payload = {
            'version': 2,
            'format': 'ring_replay_buffer',
            'capacity': 3,
            'size': 3,
            'pos': 1,
            'next_uid': 4,
            'states': [self._sample(3)[0], self._sample(1)[0], self._sample(2)[0]],
            'policies': [self._sample(3)[1], self._sample(1)[1], self._sample(2)[1]],
            'values': [self._sample(3)[2], self._sample(1)[2], self._sample(2)[2]],
            'priorities': [0.3, 0.1, 0.2],
            'uids': [3, 1, 2],
        }

        loaded = ReplayBuffer.from_serialized(ring_payload, cfg=_cfg(capacity=2))
        kept_values = [int(sample[0][0, 0].item()) for sample in loaded.buffer]
        self.assertEqual(kept_values, [2, 3])
        self.assertEqual(list(loaded.priorities), [0.2, 0.3])


if __name__ == '__main__':
    unittest.main()
