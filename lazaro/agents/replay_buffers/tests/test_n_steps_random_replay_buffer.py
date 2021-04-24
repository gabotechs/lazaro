import numpy as np

from . import tools
from .test_random_replay_buffer import is_random
from ..base.n_steps_replay_buffer import NStepsReplayBuffer
from ...replay_buffers import NStepRandomReplayBufferParams, ReplayBufferEntry, NStepsRandomReplayBuffer


def is_n_step(replay_buffer: NStepsReplayBuffer):
    n_step, gamma = replay_buffer.rp.n_step, replay_buffer.hyper_params.gamma
    final_step = 6
    records = []
    for i in range(14):
        rpe = ReplayBufferEntry(np.array([i]), np.array([i]), i, i, i == final_step)
        records.append(rpe)
        replay_buffer.rp_add(rpe)

    assert replay_buffer.rp_get_length() == 14 - n_step + 1

    samples = replay_buffer.rp_sample(8)
    for sample in samples:
        i = sample.a
        rest = 0
        if final_step - n_step + 1 <= i <= final_step:
            rest = i - (final_step - n_step + 1)
        assert sample.s_.item() == records[i + n_step - 1 - rest].s_.item()
        assert round(sample.r, 3) == round(sum([ii * (gamma ** n) for n, ii in enumerate(range(i, i + n_step - rest))]), 3)


class Agent(NStepsRandomReplayBuffer, tools.Agent):
    pass


def test_elements_are_n_step():
    nrp = NStepRandomReplayBufferParams(max_len=15, n_step=3)
    test_agent = Agent(rp=nrp)
    is_n_step(test_agent)


def test_elements_retrieved_are_random():
    nrp = NStepRandomReplayBufferParams(max_len=10, n_step=1)
    test_agent = Agent(rp=nrp)
    is_random(test_agent)
