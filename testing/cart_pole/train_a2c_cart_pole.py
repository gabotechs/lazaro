import numpy as np
import torch
import torch.nn.functional as F

from agents import AdvantageActorCriticAgent, MonteCarloAdvantageActorCriticAgent, ACHyperParams, TrainingParams
from agents.replay_buffers import NStepsPrioritizedReplayBuffer, NStepPrioritizedReplayBufferParams
from agents.explorers import NoisyExplorer, NoisyExplorerParams
from environments import CartPole


AGENT_PARAMS = ACHyperParams(c_lr=0.01, a_lr=0.01, gamma=0.97)
TRAINING_PARAMS = TrainingParams(learn_every=1, batch_size=128, episodes=300)
NOISY_EXPLORER_PARAMS = NoisyExplorerParams(extra_layers=[], std_init=0.5, reset_noise_every=1)
REPLAY_BUFFER_PARAMS = NStepPrioritizedReplayBufferParams(max_len=20000, gamma=AGENT_PARAMS.gamma, n_step=5, alpha=0.6,
                                                          init_beta=0.4, final_beta=1.0, increase_beta=1e-4)

USE_MONTE_CARLO = True

AgentParentClass = MonteCarloAdvantageActorCriticAgent if USE_MONTE_CARLO else AdvantageActorCriticAgent
env = CartPole()


class CustomActionEstimator(torch.nn.Module):
    def __init__(self, in_size: int):
        super(CustomActionEstimator, self).__init__()
        self.linear1 = torch.nn.Linear(in_size, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear1(x))


class CustomActorCriticAgent(AgentParentClass):
    def model_factory(self) -> torch.nn.Module:
        return CustomActionEstimator(env.get_observation_space()[0])

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        return torch.unsqueeze(torch.tensor(x, dtype=torch.float32), 0)


if __name__ == "__main__":
    agent = CustomActorCriticAgent(
        len(env.get_action_space()),
        AGENT_PARAMS,
        TRAINING_PARAMS,
        NoisyExplorer(NOISY_EXPLORER_PARAMS),
        NStepsPrioritizedReplayBuffer(REPLAY_BUFFER_PARAMS)
    )
    agent.train(env)
