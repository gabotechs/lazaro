import numpy as np
import typing as T
import torch

from agents import ActorCriticAgent, ACHyperParams, TrainingParams
from agents.replay_buffers import RandomReplayBuffer
from environments import SpaceInvaders

from testing.helpers import train

AGENT_PARAMS = ACHyperParams(c_lr=0.001, a_lr=0.0001, gamma=0.995)
TRAINING_PARAMS = TrainingParams(learn_every=10, ensure_every=10, batch_size=128, finish_condition=lambda x: False)
MEMORY_LEN = 1000

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


class CustomA2CAgent(ActorCriticAgent):
    def model_factory(self) -> torch.nn.Module:
        return CustomActionEstimator(
            env.get_observation_space()[0],
            env.get_observation_space()[1],
            len(env.get_action_space())
        )

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        return torch.unsqueeze(torch.tensor(x.transpose((2, 0, 1)), dtype=torch.float32) / 255, 0)


if __name__ == '__main__':
    agent = CustomA2CAgent(AGENT_PARAMS, TRAINING_PARAMS, None, RandomReplayBuffer(MEMORY_LEN))
    train(agent, env)
