import typing as T
from abc import ABC

from .base.models import DoubleDuelingDqnHyperParams, TrainingParams
from .double_dqn_agent import DoubleDqnAgent
from .dueling_dqn_agent import DuelingDqnAgent
from .explorers import AnyExplorer
from .replay_buffers import AnyReplayBuffer


class DoubleDuelingDqnAgent(DuelingDqnAgent, DoubleDqnAgent, ABC):
    def __init__(self,
                 action_space: int,
                 explorer: AnyExplorer,
                 replay_buffer: AnyReplayBuffer,
                 tp: TrainingParams,
                 hp: DoubleDuelingDqnHyperParams = DoubleDuelingDqnHyperParams(),
                 use_gpu: bool = True,
                 tensor_board_log: bool = True):
        super(DoubleDuelingDqnAgent, self).__init__(action_space, explorer, replay_buffer, tp, hp, use_gpu, tensor_board_log)
