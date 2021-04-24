import numpy as np

from . import tools
from ...replay_buffers import RandomReplayBuffer, RandomReplayBufferParams, ReplayBufferEntry


def is_random(replay_buffer: RandomReplayBuffer):
    for i in range(14):
        replay_buffer.rp_add(ReplayBufferEntry(np.array([]), np.array([]), i, i, False))

    samples = replay_buffer.rp_sample(5)
    for i, sample in enumerate(samples, 9):
        if sample.a != i:
            break
    else:
        raise AssertionError()

    indexes = [s.a for s in samples]
    assert indexes != sorted(indexes)
    assert len(indexes) == len(set(indexes))


class Agent(RandomReplayBuffer, tools.Agent):
    pass


def test_elements_retrieved_are_random():
    lrp = RandomReplayBufferParams(max_len=10)
    test_agent = Agent(rp=lrp)
    is_random(test_agent)
