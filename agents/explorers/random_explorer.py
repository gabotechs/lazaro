import typing as T
import numpy as np
from random import random, randrange
from agents.explorers.base.models import RandomExplorerParams
from agents.explorers.base.explorer import Explorer


class RandomExplorer(Explorer):
    def __init__(self, ep: RandomExplorerParams):
        self.ep: RandomExplorerParams = ep
        self.epsilon: float = ep.init_ep

    def decay(self):
        if self.epsilon > self.ep.final_ep:
            self.epsilon *= self.ep.decay_ep

    def choose(self, estimated_rewards: np.ndarray, f: T.Callable[[np.ndarray], int]) -> int:
        if random() > self.epsilon:
            return f(estimated_rewards)
        else:
            return randrange(0, len(estimated_rewards))
