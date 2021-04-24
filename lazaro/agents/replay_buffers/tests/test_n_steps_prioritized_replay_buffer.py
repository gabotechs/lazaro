from . import tools
from .test_n_steps_random_replay_buffer import is_n_step
from .test_prioritized_replay_buffer import is_prioritized
from ...replay_buffers import NStepPrioritizedReplayBufferParams, NStepsPrioritizedReplayBuffer


class Agent(NStepsPrioritizedReplayBuffer, tools.Agent):
    pass


def test_elements_are_n_step():
    nprp = NStepPrioritizedReplayBufferParams(max_len=15, n_step=3, alpha=0.5, init_beta=0.9, final_beta=1.0, increase_beta=1e-5)
    test_agent = Agent(rp=nprp)
    is_n_step(test_agent)


def test_elements_retrieved_are_prioritized():
    nprp = NStepPrioritizedReplayBufferParams(max_len=10, n_step=1, alpha=0.5, init_beta=0.9, final_beta=1.0, increase_beta=1e-5)
    test_agent = Agent(rp=nprp)
    is_prioritized(test_agent)
