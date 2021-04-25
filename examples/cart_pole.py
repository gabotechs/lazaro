import torch
import torch.nn.functional as F
import lazaro as lz

env = lz.environments.CartPole()


class CustomNN(torch.nn.Module):
    def __init__(self):
        super(CustomNN, self).__init__()
        self.linear = torch.nn.Linear(4, 30)

    def forward(self, x):
        return F.relu(self.linear(x))


class CustomAgent(lz.agents.explorers.NoisyExplorer,
                  lz.agents.replay_buffers.RandomReplayBuffer,
                  lz.agents.loggers.TensorBoardLogger,
                  lz.agents.MonteCarloA2c):
    def model_factory(self):
        return CustomNN()


agent = CustomAgent(action_space=2)
agent.train(env)
