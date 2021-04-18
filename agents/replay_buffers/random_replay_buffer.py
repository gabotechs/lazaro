import random
from .base.replay_buffer import ReplayBuffer
from .base.models import RandomReplayBufferParams
from abc import ABC


class RandomReplayBuffer(ReplayBuffer, ABC):
    def __init__(self, rp: RandomReplayBufferParams = RandomReplayBufferParams(), *args, **kwargs):
        for sub_class in rp.__class__.mro():
            if issubclass(sub_class, RandomReplayBufferParams):
                break
        else:
            raise ValueError("argument rp must be an instance or subclass of RandomReplayBufferParams")
        super(RandomReplayBuffer, self).__init__(rp, *args, **kwargs)

    def rp_sample(self, limit: int):
        indexes = random.sample(list(range(self.rp_get_length())), limit)
        return [self.records[i] for i in indexes]

    def rp_link(self):
        pass
