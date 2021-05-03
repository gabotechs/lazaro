import numpy as np
import torch
import torch.nn.functional as F

import lazaro as lz

env = lz.environments.CartPole()


class CustomNN(torch.nn.Module):
    def __init__(self):
        super(CustomNN, self).__init__()
        self.linear = torch.nn.Linear(lz.environments.CartPole.OBSERVATION_SPACE, 128)

    def forward(self, x):
        return F.relu(self.linear(x))


class CustomAgent(lz.agents.explorers.NoisyExplorer,
                  lz.agents.replay_buffers.RandomReplayBuffer,
                  lz.agents.loggers.TensorBoardLogger,
                  lz.agents.MonteCarloA2c):
    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x.astype("float32"))

    def model_factory(self):
        return CustomNN()


agent = CustomAgent(action_space=lz.environments.CartPole.ACTION_SPACE)
agent.train(env)
