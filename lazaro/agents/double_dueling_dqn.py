from abc import ABC

from .base.models import DoubleDuelingDqnHyperParams
from .double_dqn import DoubleDqnAgent
from .dueling_dqn import DuelingDqnAgent


class DoubleDuelingDqnAgent(DuelingDqnAgent, DoubleDqnAgent, ABC):
    def __init__(self,
                 action_space: int,
                 hp: DoubleDuelingDqnHyperParams = DoubleDuelingDqnHyperParams(),
                 use_gpu: bool = True):
        super(DoubleDuelingDqnAgent, self).__init__(action_space, hp, use_gpu)
