import numpy as np
import torch

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
        self.relu1 = torch.nn.ReLU()

        self.linear2 = torch.nn.Linear(in_size*10, out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.linear1(x))
        x = self.linear2(x)
        return x


class CustomDoubleDqnAgent(DoubleDqnAgent):
    def model_factory(self) -> torch.nn.Module:
        return CustomActionEstimator(env.get_observation_space()[0], 100)

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        return torch.unsqueeze(torch.tensor(x, dtype=torch.float32), 0)


EXPLORER_PARAMS = NoisyExplorerParams(layers=[len(env.get_action_space())], std_init=0.5, reset_noise_every=100)
AGENT_PARAMS = DoubleDqnHyperParams(lr=0.01, gamma=0.995, ensure_every=10)
TRAINING_PARAMS = TrainingParams(learn_every=1, batch_size=128, episodes=500)
REPLAY_BUFFER_PARAMS = NStepPrioritizedReplayBufferParams(max_len=5000, gamma=AGENT_PARAMS.gamma, n_step=3, alpha=0.6,
                                                          init_beta=0.4, final_beta=1.0, increase_beta=1+1e-3)

if __name__ == "__main__":
    agent = CustomDoubleDqnAgent(
        AGENT_PARAMS,
        TRAINING_PARAMS,
        NoisyExplorer(EXPLORER_PARAMS),
        NStepsPrioritizedReplayBuffer(REPLAY_BUFFER_PARAMS),
        use_gpu=True
    )
    train(agent, env)
