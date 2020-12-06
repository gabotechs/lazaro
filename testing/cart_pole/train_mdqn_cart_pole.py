import typing as T
import numpy as np
import torch

from agents.dqn_memory_agent import DqnMemoryAgent, DqnHyperParams
from agents.replay_buffers import RandomReplayBuffer
from agents.explorers import RandomExplorer, RandomExplorerParams
from agents import TrainingParams

from environments import CartPole
from testing.helpers import train

EXPLORER_PARAMS = RandomExplorerParams(init_ep=1, final_ep=0.05, decay_ep=1-1e-3)
AGENT_PARAMS = DqnHyperParams(lr=0.01, gamma=0.995, ensure_every=10)
TRAINING_PARAMS = TrainingParams(learn_every=1, batch_size=128, episodes=50)
MEMORY_LEN = 5000

env = CartPole()


class CustomActionEstimator(torch.nn.Module):
    def __init__(self, in_size: int, out_size: int):
        super(CustomActionEstimator, self).__init__()
        self.linear1 = torch.nn.Linear(in_size, in_size*10)
        self.relu1 = torch.nn.ReLU()

        self.linear2 = torch.nn.Linear(in_size*10, out_size*10)
        self.relu2 = torch.nn.ReLU()

        self.hidden_size = out_size*10
        self.gru = torch.nn.GRUCell(out_size*10, self.hidden_size)

        self.linear3 = torch.nn.Linear(self.hidden_size, out_size)

    def init_memory(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros((batch_size, self.hidden_size), dtype=torch.float32)

    def forward(self, x: torch.Tensor, m: torch.Tensor = None) -> T.Tuple[torch.Tensor, torch.Tensor]:
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        m_ = self.gru(x, m) if m is None else self.gru(x)
        return self.linear3(x), m_.clone().detach()


class CustomDqnAgent(DqnMemoryAgent):
    def memory_init(self) -> torch.Tensor:
        action_estimator: CustomActionEstimator = self.action_estimator
        return action_estimator.init_memory()

    def model_factory(self) -> torch.nn.Module:
        return CustomActionEstimator(env.get_observation_space()[0], len(env.get_action_space()))

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        return torch.unsqueeze(torch.tensor(x, dtype=torch.float32), 0)


if __name__ == "__main__":
    agent = CustomDqnAgent(
        AGENT_PARAMS,
        TRAINING_PARAMS,
        RandomExplorer(EXPLORER_PARAMS),
        RandomReplayBuffer(MEMORY_LEN),
        use_gpu=True)
    train(agent, env)
