import typing as T
import numpy as np
import torch

from agents.agents.dqn_memory_agent import DqnMemoryAgent, HyperParams
from agents.explorers import RandomExplorer, RandomExplorerParams
from trainers import Trainer, TrainingParams
from environments import BeamRider

from testing.helpers import train


EXPLORER_PARAMS = RandomExplorerParams(init_ep=1, final_ep=0.05, decay_ep=1-4e-5)
AGENT_PARAMS = HyperParams(lr=0.01, gamma=0.995, memory_len=2000)
TRAINING_PARAMS = TrainingParams(learn_every=10, ensure_every=100, batch_size=64)


env: BeamRider = BeamRider()


class CustomActionEstimator(torch.nn.Module):
    def __init__(self, width: int, height: int, out_size: int):
        super(CustomActionEstimator, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=5, stride=2)
        w, h = self._conv_size_out((width, height), 5, 2)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.relu1 = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=5, stride=2)
        w, h = self._conv_size_out((w, h), 5, 2)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.relu2 = torch.nn.ReLU()

        self.conv3 = torch.nn.Conv2d(16, 16, kernel_size=5, stride=2)
        w, h = self._conv_size_out((w, h), 5, 2)
        self.bn3 = torch.nn.BatchNorm2d(16)
        self.relu3 = torch.nn.ReLU()

        self.hidden_size = w*h*2
        self.device: str = "cpu"
        self.gru = torch.nn.GRUCell(w*h*16, self.hidden_size)

        self.head = torch.nn.Linear(self.hidden_size, out_size)

    def init_memory(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros((batch_size, self.hidden_size), dtype=torch.float32).to(self.device)

    def to(self, device: str):
        self.device = device
        return super(CustomActionEstimator, self).to(device)

    @staticmethod
    def _conv_size_out(size: T.Tuple[int, int], kernel_size: int, stride: int) -> T.Tuple[int, int]:
        return (size[0] - (kernel_size - 1) - 1) // stride + 1, (size[1] - (kernel_size - 1) - 1) // stride + 1

    def forward(self, x: torch.Tensor, m: torch.Tensor = None) -> T.Tuple[torch.Tensor, torch.Tensor]:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = x.view(x.size()[0], -1)
        m_: torch.Tensor = self.gru(x, m) if m is not None else self.gru(x)
        return self.head(m_), m_.clone().detach()


class CustomDqnAgent(DqnMemoryAgent):
    @staticmethod
    def action_estimator_factory() -> torch.nn.Module:
        return CustomActionEstimator(
            env.get_action_space()[0],
            env.get_observation_space()[1],
            len(env.get_action_space())
        )

    def memory_init(self) -> torch.Tensor:
        action_estimator: CustomActionEstimator = self.action_estimator
        return action_estimator.init_memory()

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        return torch.unsqueeze(torch.tensor(x.transpose((2, 0, 1)), dtype=torch.float32) / 255, 0)


if __name__ == "__main__":
    trainer = Trainer(env, CustomDqnAgent(AGENT_PARAMS, use_gpu=True), RandomExplorer(EXPLORER_PARAMS), TRAINING_PARAMS)
    train(trainer)
