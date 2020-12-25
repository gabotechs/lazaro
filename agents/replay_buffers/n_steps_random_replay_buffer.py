from .n_steps_replay_buffer import NStepsReplayBuffer
from .random_replay_buffer import RandomReplayBuffer


class NStepsRandomReplayBuffer(NStepsReplayBuffer, RandomReplayBuffer):
    pass
