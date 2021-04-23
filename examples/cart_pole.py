import torch
import torch.nn.functional as F
import agents as lz
from environments import CartPole

env = CartPole()


class CustomNN(torch.nn.Module):
    def __init__(self):
        super(CustomNN, self).__init__()
        self.linear = torch.nn.Linear(4, 30)
        self.linear2 = torch.nn.Linear(30, 10)

    def forward(self, x):
        return F.relu(self.linear2(F.relu(self.linear(x))))


class CustomAgent(lz.explorers.NoisyExplorer,
                  lz.replay_buffers.NStepsPrioritizedReplayBuffer,
                  lz.loggers.TensorBoardLogger,
                  lz.DoubleDuelingDqnAgent):
    def model_factory(self):
        return CustomNN()


agent = CustomAgent(action_space=2)
agent.train(env)
