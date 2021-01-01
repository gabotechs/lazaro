import numpy as np
import torch
import torch.nn.functional as F

from agents import DoubleDqnAgent, DoubleDqnHyperParams, TrainingParams
from agents.explorers import NoisyExplorer, NoisyExplorerParams
from agents.replay_buffers import NStepsPrioritizedReplayBuffer, NStepPrioritizedReplayBufferParams
from environments import CartPole

from testing.helpers import train


env = CartPole()


class CustomActionEstimator(torch.nn.Module):
    def __init__(self, in_size: int, out_size: int):
        super(CustomActionEstimator, self).__init__()
        self.linear1 = torch.nn.Linear(in_size, in_size*10)
        self.linear2 = torch.nn.Linear(in_size*10, out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x


class CustomDoubleDqnAgent(DoubleDqnAgent):
    def model_factory(self) -> torch.nn.Module:
        return CustomActionEstimator(env.get_observation_space()[0], 20)

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        return torch.unsqueeze(torch.tensor(x, dtype=torch.float32), 0)


EXPLORER_PARAMS = NoisyExplorerParams(extra_layers=[], std_init=0.5, reset_noise_every=1)
AGENT_PARAMS = DoubleDqnHyperParams(lr=0.01, gamma=0.995, ensure_every=10)
TRAINING_PARAMS = TrainingParams(learn_every=1, batch_size=128, episodes=500)
REPLAY_BUFFER_PARAMS = NStepPrioritizedReplayBufferParams(max_len=5000, gamma=AGENT_PARAMS.gamma, n_step=10, alpha=0.6,
                                                          init_beta=0.4, final_beta=1.0, increase_beta=1+1e-4)

if __name__ == "__main__":
    agent = CustomDoubleDqnAgent(
        len(env.get_action_space()),
        AGENT_PARAMS,
        TRAINING_PARAMS,
        NoisyExplorer(EXPLORER_PARAMS),
        NStepsPrioritizedReplayBuffer(REPLAY_BUFFER_PARAMS)
    )
    print(agent.get_info())
    train(agent, env)
