import numpy as np
from agents import LatestReplayBuffer
from agents import ReplayBufferParams, ReplayBufferEntry


def test_last_elements_are_retrieved():
    lrp = ReplayBufferParams(max_len=10)
    replay_buffer = LatestReplayBuffer(lrp)
    for i in range(14):
        replay_buffer.add(ReplayBufferEntry(np.array([]), np.array([]), i, i, False))

    samples = replay_buffer.sample(5)
    for i, sample in enumerate(samples, 9):
        assert sample.a == i

    indexes = [s.a for s in samples]
    assert len(indexes) == len(set(indexes))
