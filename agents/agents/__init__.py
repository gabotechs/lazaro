import typing as T
from .advantage_actor_critic_agent import AdvantageActorCriticAgent
from .monte_carlo_advantage_actor_critic_agent import MonteCarloAdvantageActorCriticAgent
from .dqn_agent import DqnAgent
from .dqn_memory_agent import DqnMemoryAgent
from .models import DqnHyperParams, ACHyperParams, HyperParams, TrainingParams, TrainingProgress, MDqnHyperParams, MDqnTrainingParams
AnyAgent = T.Union[DqnAgent, DqnMemoryAgent, AdvantageActorCriticAgent, MonteCarloAdvantageActorCriticAgent]
