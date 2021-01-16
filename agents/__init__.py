import typing
from .advantage_actor_critic_agent import AdvantageActorCriticAgent, ACHyperParams
from .monte_carlo_advantage_actor_critic_agent import MonteCarloAdvantageActorCriticAgent
from .dqn_agent import DqnAgent, DqnHyperParams
from .double_dqn_agent import DoubleDqnAgent, DoubleDqnHyperParams
from .double_dueling_dqn_agent import DoubleDuelingDqnAgent, DoubleDuelingDqnHyperParams
from .dueling_dqn_agent import DuelingDqnAgent, DuelingDqnHyperParams
import replay_buffers
import explorers
from .base.models import TrainingStep, TrainingParams, TrainingProgress, LearningStep

AnyAgent = typing.Union[DqnAgent, DoubleDqnAgent, DuelingDqnAgent, DoubleDuelingDqnAgent, AdvantageActorCriticAgent,
                        MonteCarloAdvantageActorCriticAgent]
