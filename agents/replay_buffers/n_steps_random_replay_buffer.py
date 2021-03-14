from .base.n_steps_replay_buffer import NStepsReplayBuffer
from .random_replay_buffer import RandomReplayBuffer


class NStepsRandomReplayBuffer(NStepsReplayBuffer, RandomReplayBuffer):
    def link_to_agent(self, agent):
        NStepsReplayBuffer.link_to_agent(self, agent)
        RandomReplayBuffer.link_to_agent(self, agent)
