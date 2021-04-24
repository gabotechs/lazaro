import numpy as np

from . import tools
from ...replay_buffers import PrioritizedReplayBuffer, PrioritizedReplayBufferParams, ReplayBufferEntry


def is_prioritized(replay_buffer: PrioritizedReplayBuffer):
    important_index = 2
    for i in range(10):
        replay_buffer.rp_add(ReplayBufferEntry(np.array([i]), np.array([i]), i, i, False))

    replay_buffer._refactor_priorities(list(range(10)), [0.5 if i == important_index else 0.1 for i in range(10)])

    dist = {}
    for _ in range(100):
        samples = replay_buffer.rp_sample(5)
        assert len(samples) == 5
        for sample in samples:
            dist[sample.a] = 1 if sample.a not in dist else dist[sample.a] + 1

    for i in dist:
        if i == important_index:
            continue
        assert dist[important_index] > dist[i]

    last_index = 11
    replay_buffer._refactor_priorities(list(range(10)), [0.1 for _ in range(10)])
    replay_buffer.rp_add(ReplayBufferEntry(np.array([last_index]), np.array([last_index]), last_index, last_index, False))
    dist = {}
    for _ in range(100):
        samples = replay_buffer.rp_sample(3)
        assert len(samples) == 3
        for sample in samples:
            dist[sample.a] = 1 if sample.a not in dist else dist[sample.a] + 1

    for i in dist:
        if i == last_index:
            continue
        assert dist[last_index] > dist[i]


class Agent(PrioritizedReplayBuffer, tools.Agent):
    pass


def test_important_transitions_are_prioritized():
    prb = PrioritizedReplayBufferParams(max_len=15, alpha=0.5, init_beta=0.9, final_beta=1.0, increase_beta=1+1e-5)
    test_agent = Agent(rp=prb)
    is_prioritized(test_agent)
