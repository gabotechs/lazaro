from agents import NStepsPrioritizedReplayBuffer
from agents import NStepPrioritizedReplayBufferParams
from .test_n_steps_random_replay_buffer import is_n_step
from .test_prioritized_replay_buffer import is_prioritized


def test_elements_are_n_step():
    nprp = NStepPrioritizedReplayBufferParams(max_len=15, n_step=3, gamma=0.6, alpha=0.5, init_beta=0.9, final_beta=1.0, increase_beta=1+1e-5)
    replay_buffer = NStepsPrioritizedReplayBuffer(nprp)
    is_n_step(replay_buffer)


def test_elements_retrieved_are_prioritized():
    nprp = NStepPrioritizedReplayBufferParams(max_len=10, n_step=1, gamma=0.9, alpha=0.5, init_beta=0.9, final_beta=1.0, increase_beta=1+1e-5)
    replay_buffer = NStepsPrioritizedReplayBuffer(nprp)
    is_prioritized(replay_buffer)
