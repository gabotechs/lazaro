from .... import agents
from abc import ABC
import torch
import numpy as np


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


class Agent(agents.replay_buffers.RandomReplayBuffer, agents.DqnAgent, ABC):
    def __init__(self, *args, **kwargs):
        super(Agent, self).__init__(*args, action_space=1, **kwargs)

    def model_factory(self) -> torch.nn.Module:
        return Model()

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x)
