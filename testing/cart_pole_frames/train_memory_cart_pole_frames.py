import typing as T
import numpy as np
import torch
import torchvision

from agents.dqn_memory_agent import DqnMemoryAgent, HyperParams
from agents.explorers import RandomExplorer, RandomExplorerParams
from trainers import Trainer, TrainingParams
from environments import CartPoleFrames

from testing.helpers import train


EXPLORER_PARAMS = RandomExplorerParams(init_ep=1, final_ep=0.05, decay_ep=1-1e-3)
AGENT_PARAMS = HyperParams(lr=0.01, gamma=0.999, memory_len=1000)
TRAINING_PARAMS = TrainingParams(learn_every=1, ensure_every=10, batch_size=32)


env = CartPoleFrames()


class CustomFramesMemoryActionEstimator(torch.nn.Module):
    def __init__(self, width: int, height: int, out_size: int):
        super(CustomFramesMemoryActionEstimator, self).__init__()
        w_, h_ = 150, int(height*150/width)
        self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize((w_, h_))])

        w, h = width, height
        self.downscale = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        w, h = self._conv_size_out((w, h), 2, 2)

        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=5, stride=2)
        w, h = self._conv_size_out((w, h), 5, 2)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        w, h = self._pool_size_out((w, h), 2, 2)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.relu1 = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=5, stride=2)
        w, h = self._conv_size_out((w, h), 5, 2)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        w, h = self._pool_size_out((w, h), 2, 2)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.relu2 = torch.nn.ReLU()

        self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=5, stride=2)
        w, h = self._conv_size_out((w, h), 5, 2)
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.relu3 = torch.nn.ReLU()

        self.hidden_size = w * h * 2
        self.device: str = "cpu"
        self.gru = torch.nn.GRUCell(w * h * 32, self.hidden_size)

        self.head = torch.nn.Linear(self.hidden_size, out_size)

    def init_memory(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros((batch_size, self.hidden_size), dtype=torch.float32).to(self.device)

    def to(self, device: str) -> torch.nn.Module:
        self.device = device
        return super(CustomFramesMemoryActionEstimator, self).to(device)

    @staticmethod
    def _conv_size_out(size: T.Tuple[int, int], kernel_size: int, stride: int) -> T.Tuple[int, int]:
        return (size[0] - (kernel_size - 1) - 1) // stride + 1, (size[1] - (kernel_size - 1) - 1) // stride + 1

    @staticmethod
    def _pool_size_out(size: T.Tuple[int, int], kernel_size: int, stride: int) -> T.Tuple[int, int]:
        return (size[0] - (kernel_size - 1) - 1) // stride + 1, (size[1] - (kernel_size - 1) - 1) // stride + 1

    def forward(self, x: torch.Tensor, m: torch.Tensor = None) -> T.Tuple[torch.Tensor, torch.Tensor]:
        layers = [self.downscale,
                  self.conv1, self.pool1, self.bn1, self.relu1,
                  self.conv2, self.pool2, self.relu2,
                  self.conv3, self.relu3]

        for layer in layers:
            x = layer(x)

        x = x.view(x.size()[0], -1)
        m_: torch.Tensor = self.gru(x, m) if m is not None else self.gru(x)
        return self.head(m_), m_.clone().detach()


class CustomDqnAgent(DqnMemoryAgent):
    @staticmethod
    def action_estimator_factory() -> torch.nn.Module:
        return CustomFramesMemoryActionEstimator(
            env.get_observation_space()[0],
            env.get_observation_space()[1],
            len(env.get_action_space())
        )

    def memory_init(self) -> torch.Tensor:
        action_estimator: CustomFramesMemoryActionEstimator = self.action_estimator
        return action_estimator.init_memory()

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        return torch.unsqueeze(torch.tensor(x.transpose((2, 0, 1)), dtype=torch.float32) / 255, 0)


if __name__ == "__main__":
    trainer = Trainer(env, CustomDqnAgent(AGENT_PARAMS, use_gpu=True), RandomExplorer(EXPLORER_PARAMS), TRAINING_PARAMS)
    train(trainer)
