from abc import ABC
from .base.models import DqnHyperParams
from .double_dqn_agent import DoubleDqnAgent
from .dueling_dqn_agent import DuelingDqnNetwork

DoubleDuelingDqnHyperParams = DqnHyperParams


class DoubleDuelingDqnAgent(DoubleDqnAgent, ABC):
    hp: DoubleDuelingDqnHyperParams
    network_class = DuelingDqnNetwork
