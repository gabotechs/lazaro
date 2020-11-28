import random
from .replay_buffer import ReplayBuffer


class RandomReplayBuffer(ReplayBuffer):
    def sample(self, limit: int = None):
        return random.sample(self.records, limit if limit is not None and limit < len(self.records) else len(self.records))
