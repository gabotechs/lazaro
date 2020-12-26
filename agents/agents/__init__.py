import typing as T
from .advantage_actor_critic_agent import AdvantageActorCriticAgent
from .monte_carlo_advantage_actor_critic_agent import MonteCarloAdvantageActorCriticAgent
from .dqn_agent import DqnAgent
from .models import DqnHyperParams, ACHyperParams, HyperParams, TrainingParams, TrainingProgress, MDqnHyperParams
AnyAgent = T.Union[DqnAgent, AdvantageActorCriticAgent, MonteCarloAdvantageActorCriticAgent]
