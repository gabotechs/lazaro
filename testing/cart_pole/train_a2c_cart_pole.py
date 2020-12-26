import numpy as np
import torch
import torch.nn.functional as F

from agents import AdvantageActorCriticAgent, MonteCarloAdvantageActorCriticAgent, ACHyperParams, TrainingParams
from agents.replay_buffers import NStepsPrioritizedReplayBuffer, NStepPrioritizedReplayBufferParams
from environments import CartPole

from testing.helpers import train

AGENT_PARAMS = ACHyperParams(c_lr=0.01, a_lr=0.001, gamma=0.99)
TRAINING_PARAMS = TrainingParams(learn_every=1, batch_size=16, episodes=300)
REPLAY_BUFFER_PARAMS = NStepPrioritizedReplayBufferParams(max_len=1000, gamma=AGENT_PARAMS.gamma, n_step=5, alpha=0.6,
                                                          init_beta=0.4, final_beta=1.0, increase_beta=1+1e-4)

USE_MONTE_CARLO = True

AgentParentClass = MonteCarloAdvantageActorCriticAgent if USE_MONTE_CARLO else AdvantageActorCriticAgent
env = CartPole()


class CustomActionEstimator(torch.nn.Module):
    def __init__(self, in_size: int, out_size: int):
        super(CustomActionEstimator, self).__init__()
        self.out_size = out_size
        self.linear1 = torch.nn.Linear(in_size, in_size*10)
        self.linear2 = torch.nn.Linear(in_size*10, out_size*10)
        self.linear3 = torch.nn.Linear(out_size*10, out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)


class CustomActorCriticAgent(AgentParentClass):
    def model_factory(self) -> torch.nn.Module:
        return CustomActionEstimator(env.get_observation_space()[0], len(env.get_action_space()))

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        return torch.unsqueeze(torch.tensor(x, dtype=torch.float32), 0)


if __name__ == "__main__":
    agent = CustomActorCriticAgent(AGENT_PARAMS, TRAINING_PARAMS, None, NStepsPrioritizedReplayBuffer(REPLAY_BUFFER_PARAMS))
    train(agent, env)
