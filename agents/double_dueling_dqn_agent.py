from abc import ABC
from .base.models import DoubleDuelingDqnHyperParams
from .double_dqn_agent import DoubleDqnAgent
from .dueling_dqn_agent import DuelingDqnNetwork


class DoubleDuelingDqnAgent(DoubleDqnAgent, ABC):
    hp: DoubleDuelingDqnHyperParams
    network_class = DuelingDqnNetwork
