import numpy as np
import torch

from agents import DqnAgent, HyperParams, TrainingParams
from agents.explorers import RandomExplorer, RandomExplorerParams
from agents.replay_buffers import RandomReplayBuffer
from environments import CartPole

from testing.helpers import train

EXPLORER_PARAMS = RandomExplorerParams(init_ep=1, final_ep=0.01, decay_ep=1-1e-3)
AGENT_PARAMS = HyperParams(lr=0.01, gamma=0.995)
TRAINING_PARAMS = TrainingParams(learn_every=1, ensure_every=10, batch_size=128, finish_condition=lambda x: False)
MEMORY_LEN = 5000


env = CartPole()


class CustomActionEstimator(torch.nn.Module):
    def __init__(self, in_size: int, out_size: int):
        super(CustomActionEstimator, self).__init__()
        self.linear1 = torch.nn.Linear(in_size, in_size*10)
        self.relu1 = torch.nn.ReLU()

        self.linear2 = torch.nn.Linear(in_size*10, out_size*10)
        self.relu2 = torch.nn.ReLU()

        self.linear3 = torch.nn.Linear(out_size*10, out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        return self.linear3(x)


class CustomDqnAgent(DqnAgent):
    @staticmethod
    def model_factory() -> torch.nn.Module:
        return CustomActionEstimator(env.get_observation_space()[0], len(env.get_action_space()))

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        return torch.unsqueeze(torch.tensor(x, dtype=torch.float32), 0)


if __name__ == "__main__":
    agent = CustomDqnAgent(
        AGENT_PARAMS,
        TRAINING_PARAMS,
        RandomExplorer(EXPLORER_PARAMS),
        RandomReplayBuffer(5000),
        use_gpu=True
    )
    train(agent, env)
