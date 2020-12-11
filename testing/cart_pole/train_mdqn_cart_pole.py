import typing as T
import numpy as np
import torch

from agents.agents.dqn_memory_agent import DqnMemoryAgent, MDqnHyperParams
from agents.replay_buffers import RandomReplayBuffer
from agents.explorers import RandomExplorer, RandomExplorerParams
from agents import MDqnTrainingParams

from environments import CartPole
from testing.helpers import train

EXPLORER_PARAMS = RandomExplorerParams(init_ep=1, final_ep=0.05, decay_ep=1-1e-3)
AGENT_PARAMS = MDqnHyperParams(a_lr=0.01, m_lr=0.001, gamma=0.995, ensure_every=10)
TRAINING_PARAMS = MDqnTrainingParams(learn_every=1, batch_size=128, episodes=5000, memory_batch_size=128, memory_learn_every=20, memory_clear_after_learn=False)
MEMORY_LEN = 1000

env = CartPole()


class CustomActionEstimator(torch.nn.Module):
    def __init__(self, in_size: int, mem_size: int, out_size: int):
        super(CustomActionEstimator, self).__init__()
        self.linear1 = torch.nn.Linear(in_size, in_size*10)
        self.relu1 = torch.nn.ReLU()

        self.linear2 = torch.nn.Linear(in_size*10, out_size*100)
        self.relu2 = torch.nn.ReLU()

        self.head = torch.nn.Linear(mem_size+out_size*100, out_size)

    def forward(self, x: torch.Tensor, m: torch.Tensor = None) -> torch.Tensor:
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        x = torch.cat([x, m], 1)
        return self.head(x)


class CustomMemoryProvider(torch.nn.Module):
    def __init__(self, in_size: int, mem_size: int, out_size: int):
        super(CustomMemoryProvider, self).__init__()
        self.linear1 = torch.nn.Linear(in_size, in_size * 100)
        self.relu1 = torch.nn.ReLU()

        self.linear2 = torch.nn.Linear(in_size * 100, out_size * 10)
        self.relu2 = torch.nn.ReLU()

        self.mem_size = mem_size
        self.gru = torch.nn.GRUCell(out_size * 10, mem_size)

    def init_memory(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros((batch_size, self.mem_size), dtype=torch.float32)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        m_ = self.gru(x, m)
        return m_


class CustomDqnAgent(DqnMemoryAgent):
    mem_size = 100

    def memory_init(self) -> torch.Tensor:
        return torch.zeros((1, self.mem_size), dtype=torch.float32)

    def memory_model_factory(self) -> torch.nn.Module:
        return CustomMemoryProvider(2, self.mem_size, len(env.get_action_space()))

    def model_factory(self) -> torch.nn.Module:
        return CustomActionEstimator(2, self.mem_size, len(env.get_action_space()))

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        return torch.unsqueeze(torch.tensor([x[0], x[2]], dtype=torch.float32), 0)


if __name__ == "__main__":
    agent = CustomDqnAgent(
        AGENT_PARAMS,
        TRAINING_PARAMS,
        RandomExplorer(EXPLORER_PARAMS),
        RandomReplayBuffer(MEMORY_LEN),
        use_gpu=True)
    train(agent, env)
