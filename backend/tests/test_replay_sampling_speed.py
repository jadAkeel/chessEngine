"""The uniform replay sampler must stay O(batch) on a multi-million-sample buffer."""
import time
from collections import Counter

import numpy as np
import pytest

from app.training.replay_buffer import ReplayBuffer, _uniform_choice_without_replacement


@pytest.mark.parametrize("pool, k", [(3_000_000, 128), (1000, 128), (128, 128), (10, 3)])
def test_uniform_choice_is_distinct_and_in_range(pool, k):
    np.random.seed(0)
    for _ in range(50):
        chosen = _uniform_choice_without_replacement(pool, k)
        assert chosen.size == k
        assert np.unique(chosen).size == k
        assert chosen.min() >= 0 and chosen.max() < pool


def test_uniform_choice_is_uniform():
    np.random.seed(1)
    counts = Counter()
    for _ in range(20_000):
        counts.update(_uniform_choice_without_replacement(1000, 5).tolist())
    freq = np.array([counts[i] for i in range(1000)])
    # 100 draws expected per element; a biased sampler would leave some near 0.
    assert freq.min() > 55 and freq.max() < 150


def test_uniform_choice_redraws_duplicates():
    np.random.seed(2)
    # 64 of 600 positions: duplicates are likely on the first draw and must be replaced.
    for _ in range(200):
        assert np.unique(_uniform_choice_without_replacement(600, 64)).size == 64


def test_uniform_choice_is_fast_on_a_large_pool():
    _uniform_choice_without_replacement(3_000_000, 128)
    started = time.perf_counter()
    for _ in range(200):
        _uniform_choice_without_replacement(3_000_000, 128)
    assert (time.perf_counter() - started) / 200 < 0.005  # the old permutation took ~130 ms


class _Cfg:
    def __init__(self, prioritized):
        self.prioritized = prioritized


def test_pool_without_excludes_and_keeps_enough_candidates():
    buffer = ReplayBuffer.__new__(ReplayBuffer)
    pool = np.arange(3_000_000, dtype=np.int64)
    excluded = np.arange(0, 3_000_000, 30_000, dtype=np.int64)[:100]
    for prioritized in (False, True):
        buffer.replay_cfg = _Cfg(prioritized)
        np.random.seed(3)
        rest = buffer._pool_without(pool, excluded, 28)
        assert rest.size >= 28
        assert not np.isin(rest, excluded).any()
        if prioritized:
            assert rest.size == pool.size - excluded.size  # exact pool for importance weights
