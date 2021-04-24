import typing as T
from abc import ABC
from random import random, randrange

import torch

from ..explorers.base.explorer import Explorer
from ..explorers.base.params import RandomExplorerParams


class RandomExplorer(Explorer, ABC):
    def __init__(self, ep: RandomExplorerParams = RandomExplorerParams(), *args, **kwargs):
        if not isinstance(ep, RandomExplorerParams):
            raise ValueError("argument ep must be an instance of RandomExplorerParams")
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
        self.arrived_to_minimum: bool = False
        super(RandomExplorer, self).__init__(*args, **kwargs)

    def last_layers_model_modifier(self, model: torch.nn.Module) -> torch.nn.Module:
        return model

    def decay(self, *_, **__) -> None:
        self.log.debug(f"decay epsilon for {type(self).__name__} triggered")
        if self.epsilon > self.ep.final_ep:
            self.epsilon -= self.ep.decay_ep
        elif self.epsilon < self.ep.final_ep:
            self.epsilon = self.ep.final_ep
        elif not self.arrived_to_minimum:
            self.log.info("epsilon arrived to minimum")
            self.arrived_to_minimum = True

    def ex_choose(self, actions: T.List[float], f: T.Callable[[T.List[float]], int]) -> int:
        if random() > self.epsilon:
            return f(actions)
        else:
            return randrange(0, len(actions))

    def ex_link(self):
        self.add_step_callback("random explorer decay", self.decay)

    def ex_get_stats(self) -> T.Dict[str, float]:
        return {"Random Explorer Epsilon": self.epsilon}
