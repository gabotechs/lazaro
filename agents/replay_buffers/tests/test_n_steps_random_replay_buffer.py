import numpy as np
from agents import NStepsRandomReplayBuffer
from agents import NStepReplayBufferParams, ReplayBufferEntry
from .test_random_replay_buffer import is_random


def is_n_step(replay_buffer):
    n_step, gamma = replay_buffer.rp.n_step, replay_buffer.rp.gamma
    final_step = 6
    records = []
    for i in range(14):
        rpe = ReplayBufferEntry(np.array([i]), np.array([i]), i, i, i == final_step)
        records.append(rpe)
        replay_buffer.add(rpe)

    assert len(replay_buffer) == 14 - n_step + 1

    samples = replay_buffer.sample(8)
    for sample in samples:
        i = sample.a
        rest = 0
        if final_step - n_step + 1 <= i <= final_step:
            rest = i - (final_step - n_step + 1)
        assert sample.s_.item() == records[i + n_step - 1 - rest].s_.item()
        assert round(sample.r, 3) == round(sum([ii * (gamma ** n) for n, ii in enumerate(range(i, i + n_step - rest))]), 3)


def test_elements_are_n_step():
    nrp = NStepReplayBufferParams(max_len=15, n_step=3, gamma=0.6)
    replay_buffer = NStepsRandomReplayBuffer(nrp)
    is_n_step(replay_buffer)


def test_elements_retrieved_are_random():
    nrp = NStepReplayBufferParams(max_len=10, n_step=1, gamma=0.9)
    replay_buffer = NStepsRandomReplayBuffer(nrp)
    is_random(replay_buffer)
