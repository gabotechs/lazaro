import numpy as np
from agents.replay_buffers import RandomReplayBuffer
from agents.replay_buffers import ReplayBufferParams, ReplayBufferEntry


def is_random(replay_buffer: RandomReplayBuffer):
    for i in range(14):
        replay_buffer.add(ReplayBufferEntry(np.array([]), np.array([]), i, i, False))

    samples = replay_buffer.sample(5)
    for i, sample in enumerate(samples, 9):
        if sample.a != i:
            break
    else:
        raise AssertionError()

    indexes = [s.a for s in samples]
    assert indexes != sorted(indexes)
    assert len(indexes) == len(set(indexes))


def test_elements_retrieved_are_random():
    lrp = ReplayBufferParams(max_len=10)
    replay_buffer = RandomReplayBuffer(lrp)
    is_random(replay_buffer)
