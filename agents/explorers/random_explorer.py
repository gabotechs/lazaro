import typing as T
import numpy as np
from random import random, randrange
from .models import RandomExplorerParams
from .explorer import Explorer


class RandomExplorer(Explorer):
    def __init__(self, params: RandomExplorerParams):
        self.init_ep: float = params.init_ep
        self.final_ep: float = params.final_ep
        self.decay_ep: float = params.decay_ep

        self.ep: float = params.init_ep

    def decay(self):
        if self.ep > self.final_ep:
            self.ep *= self.decay_ep

    def choose(self, estimated_rewards: np.ndarray, f: T.Callable[[np.ndarray], int]) -> int:
        if random() > self.ep:
            return f(estimated_rewards)
        else:
            return randrange(0, len(estimated_rewards))
