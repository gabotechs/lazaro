import typing as T
from abc import ABC
import torch
from .base.models import DoubleDuelingDqnHyperParams, TrainingParams
from .double_dqn_agent import DoubleDqnAgent
from .dueling_dqn_agent import DuelingDqnNetwork
from .explorers import AnyExplorer
from .replay_buffers import AnyReplayBuffer


class DoubleDuelingDqnAgent(DoubleDqnAgent, ABC):
    def __init__(self,
                 action_space: int,
                 explorer: T.Union[AnyExplorer, None],
                 replay_buffer: AnyReplayBuffer,
                 tp: TrainingParams,
                 hp: DoubleDuelingDqnHyperParams = DoubleDuelingDqnHyperParams(),
                 use_gpu: bool = True,
                 save_progress: bool = True,
                 tensor_board_log: bool = True):
        super(DoubleDuelingDqnAgent, self).__init__(action_space, explorer, replay_buffer, tp, hp,
                                                    use_gpu, save_progress, tensor_board_log)

    def build_model(self) -> torch.nn.Module:
        model = super(DoubleDuelingDqnAgent, self).build_model()
        return DuelingDqnNetwork(model, self.action_space, self.last_layer_factory)
