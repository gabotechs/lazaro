from .base.replay_buffer import ReplayBuffer


class LatestReplayBuffer(ReplayBuffer):
    def sample(self, limit: int = None):
        return [self.records[len(self.records)-i-1] for i in range(limit)]
