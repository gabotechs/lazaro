import random
from .base.replay_buffer import ReplayBuffer
from .base.models import RandomReplayBufferParams


class RandomReplayBuffer(ReplayBuffer):
    rp: RandomReplayBufferParams

    def sample(self, limit: int):
        indexes = random.sample(list(range(len(self))), limit)
        return [self.records[i] for i in indexes]
