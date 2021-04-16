from .base.n_steps_replay_buffer import NStepsReplayBuffer
from .random_replay_buffer import RandomReplayBuffer
from abc import ABC


class NStepsRandomReplayBuffer(NStepsReplayBuffer, RandomReplayBuffer, ABC):
    def rp_link(self):
        NStepsReplayBuffer.rp_link(self)
        RandomReplayBuffer.rp_link(self)
