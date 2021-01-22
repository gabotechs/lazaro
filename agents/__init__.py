import typing
from .advantage_actor_critic_agent import A2cAgent, A2CHyperParams
from .monte_carlo_advantage_actor_critic_agent import MonteCarloA2cCriticAgent
from .dqn_agent import DqnAgent, DqnHyperParams
from .double_dqn_agent import DoubleDqnAgent, DoubleDqnHyperParams
from .double_dueling_dqn_agent import DoubleDuelingDqnAgent, DoubleDuelingDqnHyperParams
from .dueling_dqn_agent import DuelingDqnAgent, DuelingDqnHyperParams
from .ppo_agent import PpoAgent, PpoHyperParams
from .base.models import TrainingStep, TrainingParams, TrainingProgress, LearningStep
from . import replay_buffers, explorers

AnyAgent = typing.Union[DqnAgent, DoubleDqnAgent, DuelingDqnAgent, DoubleDuelingDqnAgent, A2cAgent,
                        MonteCarloA2cCriticAgent, PpoAgent]
