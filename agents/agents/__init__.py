import typing as T
from .advantage_actor_critic_agent import AdvantageActorCriticAgent
from .monte_carlo_advantage_actor_critic_agent import MonteCarloAdvantageActorCriticAgent
from .double_dqn_agent import DoubleDqnAgent
from .base.models import DqnHyperParams, DoubleDqnHyperParams, ACHyperParams, HyperParams, TrainingParams, TrainingProgress
AnyAgent = T.Union[DoubleDqnAgent, AdvantageActorCriticAgent, MonteCarloAdvantageActorCriticAgent]
