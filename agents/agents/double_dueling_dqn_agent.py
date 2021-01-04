from abc import ABC
from .base.models import DqnHyperParams
from .double_dqn_agent import DoubleDqnAgent
from .dueling_dqn_agent import DuelingDqnNetwork


class DoubleDuelingDqnAgent(DoubleDqnAgent, ABC):
    hp: DqnHyperParams
    network_class = DuelingDqnNetwork
