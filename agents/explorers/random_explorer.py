import typing as T
from random import random, randrange
from agents.explorers.base.models import RandomExplorerParams
from agents.explorers.base.explorer import Explorer


class RandomExplorer(Explorer):
    def __init__(self, ep: RandomExplorerParams = RandomExplorerParams()):
        if not 0 <= ep.init_ep <= 1:
            raise ValueError("initial epsilon must be between 0 and 1")
        elif not 0 <= ep.final_ep <= 1:
            raise ValueError("final epsilon must be between 0 and 1")
        elif ep.final_ep >= ep.init_ep:
            raise ValueError("final epsilon must be less than initial epsilon")
        elif not 0 < ep.decay_ep < 1:
            raise ValueError("decay epsilon must be between 0 and 1")
        self.ep: RandomExplorerParams = ep
        self.epsilon: float = ep.init_ep

    def decay(self):
        if self.epsilon > self.ep.final_ep:
            self.epsilon -= self.ep.decay_ep
        elif self.epsilon < self.ep.final_ep:
            self.epsilon = self.ep.final_ep

    def choose(self, actions: T.List[float], f: T.Callable[[T.List[float]], int]) -> int:
        if random() > self.epsilon:
            return f(actions)
        else:
            return randrange(0, len(actions))
