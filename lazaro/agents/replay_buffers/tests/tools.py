from abc import ABC

import numpy as np
import torch

from .... import agents


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


class Agent(agents.explorers.RandomExplorer, agents.DqnAgent, ABC):
    def __init__(self, *args, **kwargs):
        super(Agent, self).__init__(*args, action_space=1, **kwargs)

    def model_factory(self) -> torch.nn.Module:
        return Model()

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x)
