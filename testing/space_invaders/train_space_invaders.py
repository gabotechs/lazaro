import typing as T
import numpy as np
import torch

from agents.dqn_agent import Agent, HyperParams
from explorers import RandomExplorer, RandomExplorerParams
from trainers import Trainer, TrainingParams
from environments import SpaceInvaders

from testing.helpers import train


EXPLORER_PARAMS = RandomExplorerParams(init_ep=1, final_ep=0.05, decay_ep=1-1e-4)
AGENT_PARAMS = HyperParams(lr=0.01, gamma=0.999, memory_len=5000)
TRAINING_PARAMS = TrainingParams(learn_every=10, ensure_every=100, batch_size=64)


env = SpaceInvaders()


class CustomActionEstimator(torch.nn.Module):
    def __init__(self, width: int, height: int, out_size: int):
        super(CustomActionEstimator, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=5, stride=2)
        w, h = self._conv_size_out((width, height), 5, 2)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.relu1 = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5, stride=2)
        w, h = self._conv_size_out((w, h), 5, 2)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.relu2 = torch.nn.ReLU()

        self.conv3 = torch.nn.Conv2d(32, 32, kernel_size=5, stride=2)
        w, h = self._conv_size_out((w, h), 5, 2)
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.relu3 = torch.nn.ReLU()

        self.head = torch.nn.Linear(w*h*32, out_size)

    @staticmethod
    def _conv_size_out(size: T.Tuple[int, int], kernel_size: int, stride: int) -> T.Tuple[int, int]:
        return (size[0] - (kernel_size - 1) - 1) // stride + 1, (size[1] - (kernel_size - 1) - 1) // stride + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size()[0], -1))


class CustomDqnAgent(Agent):
    @staticmethod
    def action_estimator_factory() -> torch.nn.Module:
        return CustomActionEstimator(
            env.get_observation_space()[0],
            env.get_observation_space()[1],
            len(env.get_action_space())
        )

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        return torch.unsqueeze(torch.tensor(x.transpose((2, 0, 1)), dtype=torch.float32) / 255, 0)

    def postprocess(self, t: torch.Tensor) -> np.ndarray:
        return np.array(t.squeeze(0))


if __name__ == "__main__":
    trainer = Trainer(env, CustomDqnAgent(AGENT_PARAMS, use_gpu=True), RandomExplorer(EXPLORER_PARAMS), TRAINING_PARAMS)
    train(trainer)
