import typing as T
import numpy as np
import torch
import torchvision

from agents.dqn_agent import Agent, HyperParams
from agents.explorers import RandomExplorer, RandomExplorerParams
from trainers import Trainer, TrainingParams
from environments import CartPoleFrames
from agents.replay_buffers import RandomReplayBuffer

from testing.helpers import train


EXPLORER_PARAMS = RandomExplorerParams(init_ep=1, final_ep=0.05, decay_ep=1-1e-3)
AGENT_PARAMS = HyperParams(lr=0.01, gamma=0.999)
TRAINING_PARAMS = TrainingParams(learn_every=1, ensure_every=10, batch_size=32)
MEMORY_LEN = 5000


env = CartPoleFrames()


class CustomActionEstimator(torch.nn.Module):
    def __init__(self, width: int, height: int, out_size: int):
        super(CustomActionEstimator, self).__init__()
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

        self.conv3 = torch.nn.Conv2d(16, 16, kernel_size=5, stride=2)
        w, h = self._conv_size_out((w, h), 5, 2)
        self.bn3 = torch.nn.BatchNorm2d(16)
        self.relu3 = torch.nn.ReLU()

        self.head = torch.nn.Linear(w*h*16, out_size)

    @staticmethod
    def _conv_size_out(size: T.Tuple[int, int], kernel_size: int, stride: int) -> T.Tuple[int, int]:
        return (size[0] - (kernel_size - 1) - 1) // stride + 1, (size[1] - (kernel_size - 1) - 1) // stride + 1

    @staticmethod
    def _pool_size_out(size: T.Tuple[int, int], kernel_size: int, stride: int) -> T.Tuple[int, int]:
        return (size[0] - (kernel_size - 1) - 1) // stride + 1, (size[1] - (kernel_size - 1) - 1) // stride + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layers = [self.downscale,
                  self.conv1, self.pool1, self.relu1,
                  self.conv2, self.pool2, self.relu2,
                  self.conv3, self.relu3]

        for layer in layers:
            x = layer(x)

        return self.head(x.view(x.size()[0], -1))


class CustomDqnAgent(Agent):
    @staticmethod
    def model_factory() -> torch.nn.Module:
        return CustomActionEstimator(
            env.get_observation_space()[0],
            env.get_observation_space()[1],
            len(env.get_action_space())
        )

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        return torch.unsqueeze(torch.tensor(x.transpose((2, 0, 1)), dtype=torch.float32) / 255, 0)


if __name__ == "__main__":
    trainer = Trainer(env, CustomDqnAgent(AGENT_PARAMS, use_gpu=True), RandomExplorer(EXPLORER_PARAMS), RandomReplayBuffer(MEMORY_LEN), TRAINING_PARAMS)
    train(trainer)

