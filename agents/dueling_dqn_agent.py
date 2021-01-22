from abc import ABC
import typing as T
import torch
from .explorers.noisy_explorer import NoisyLinear
from .base.models import DuelingDqnHyperParams, TrainingParams
from .dqn_agent import DqnAgent
from .explorers import AnyExplorer
from .replay_buffers import AnyReplayBuffer


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

        self.value = torch.nn.Sequential(
            # last_layer_factory(last_layer.out_features, last_layer.out_features),
            # torch.nn.ReLU(),
            last_layer_factory(last_layer.out_features, 1)
        )
        self.advantage = torch.nn.Sequential(
            # last_layer_factory(last_layer.out_features, last_layer.out_features),
            # torch.nn.ReLU(),
            last_layer_factory(last_layer.out_features, action_space)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        v = self.value(x)
        a = self.advantage(x)
        return v + a - a.mean(dim=-1, keepdim=True)


class DuelingDqnAgent(DqnAgent, ABC):
    def __init__(self,
                 action_space: int,
                 explorer: T.Union[AnyExplorer, None],
                 replay_buffer: AnyReplayBuffer,
                 tp: TrainingParams,
                 hp: DuelingDqnHyperParams = DuelingDqnHyperParams(),
                 use_gpu: bool = True,
                 save_progress: bool = True,
                 tensor_board_log: bool = True):
        super(DuelingDqnAgent, self).__init__(action_space, explorer, replay_buffer, tp, hp,
                                              use_gpu, save_progress, tensor_board_log)

    def build_model(self) -> torch.nn.Module:
        model = super(DuelingDqnAgent, self).build_model()
        return DuelingDqnNetwork(model, self.action_space, self.last_layer_factory)
