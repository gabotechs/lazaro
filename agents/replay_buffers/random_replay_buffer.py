import random
from .base.replay_buffer import ReplayBuffer
from .base.models import RandomReplayBufferParams


class RandomReplayBuffer(ReplayBuffer):
    def __init__(self, rp: RandomReplayBufferParams = RandomReplayBufferParams()):
        super(RandomReplayBuffer, self).__init__(rp)

    def sample(self, limit: int):
        indexes = random.sample(list(range(len(self))), limit)
        return [self.records[i] for i in indexes]

    def link_to_agent(self, agent):
        return
