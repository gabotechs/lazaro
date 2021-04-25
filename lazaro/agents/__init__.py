import typing
from .a2c import A2cAgent, A2CHyperParams
from .monte_carlo_advantage_actor_critic import MonteCarloA2c
from .dqn import DqnAgent, DqnHyperParams
from .double_dqn import DoubleDqnAgent, DoubleDqnHyperParams
from .double_dueling_dqn import DoubleDuelingDqnAgent, DoubleDuelingDqnHyperParams
from .dueling_dqn import DuelingDqnAgent, DuelingDqnHyperParams
from .ppo import PpoAgent, PpoHyperParams
from .base.models import TrainingStep, TrainingParams, TrainingProgress, LearningStep
from . import replay_buffers, explorers, loggers

AnyAgent = typing.Union[DqnAgent, DoubleDqnAgent, DuelingDqnAgent, DoubleDuelingDqnAgent, A2cAgent, MonteCarloA2c, PpoAgent]
