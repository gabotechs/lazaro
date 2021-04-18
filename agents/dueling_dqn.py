import typing as T
from abc import ABC

import torch

from .base.models import DuelingDqnHyperParams
from .dqn import DqnAgent
from .explorers.noisy_explorer import NoisyLinear


class DuelingDqnNetwork(torch.nn.Module):
    def __init__(self,
                 model: torch.nn.Module,
                 action_space: int,
                 last_layer_factory: T.Callable[[int, int], torch.nn.Module]):
        super(DuelingDqnNetwork, self).__init__()
        self.model = model
        last_layer = list(model.modules())[-1]
        if not isinstance(last_layer, (torch.nn.Linear, NoisyLinear)):
            raise ValueError("the model you have created must have a torch.nn.Linear or "
                             "agents.explorers.noisy_explorer.NoisyLinear in the last layer")

        if last_layer.out_features == action_space:
            print("WARNING: detected same number of features in the output of the model than the action space")

        self.value = last_layer_factory(last_layer.out_features, 1)
        self.advantage = last_layer_factory(last_layer.out_features, action_space)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        v = self.value(x)
        a = self.advantage(x)
        return v + a - a.mean(dim=-1, keepdim=True)


class DuelingDqnAgent(DqnAgent, ABC):
    def __init__(self,
                 action_space: int,
                 hp: DuelingDqnHyperParams = DuelingDqnHyperParams(),
                 use_gpu: bool = True):
        super(DuelingDqnAgent, self).__init__(action_space, hp, use_gpu)

    def dqn_model_wrapper(self, model: torch.nn.Module) -> torch.nn.Module:
        return DuelingDqnNetwork(model, self.action_space, self.last_layer_factory)
