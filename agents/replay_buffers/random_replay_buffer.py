import random
from .replay_buffer import ReplayBuffer


class RandomReplayBuffer(ReplayBuffer):
    def sample(self, limit: int):
        indexes = random.sample(list(range(len(self))), limit)
        return [self.records[i] for i in indexes]
