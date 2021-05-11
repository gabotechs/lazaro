import random
from abc import ABC

from .base.params import RandomReplayBufferParams
from .base.replay_buffer import ReplayBuffer


class RandomReplayBuffer(ReplayBuffer, ABC):
    def __init__(self, replay_buffer_params: RandomReplayBufferParams = RandomReplayBufferParams(), *args, **kwargs):
        if not isinstance(replay_buffer_params, RandomReplayBufferParams):
            raise ValueError("argument rp must be an instance or subclass of RandomReplayBufferParams")
        super(RandomReplayBuffer, self).__init__(replay_buffer_params, *args, **kwargs)

    def rp_sample(self, limit: int):
        indexes = random.sample(list(range(self.rp_get_length())), limit)
        return [self.records[i] for i in indexes]

    def rp_link(self):
        pass
