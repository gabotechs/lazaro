import torch
import torch.nn.functional as F
import agents
from agents import explorers, replay_buffers
from environments import CartPole

env = CartPole()


class CustomNN(torch.nn.Module):
    def __init__(self):
        super(CustomNN, self).__init__()
        self.linear = torch.nn.Linear(env.get_observation_space()[0], 30)

    def forward(self, x):
        return F.relu(self.linear(x))


class CustomAgent(replay_buffers.RandomReplayBuffer, explorers.NoisyExplorer, agents.PpoAgent):
    def model_factory(self):
        return CustomNN()

    def preprocess(self, x):
        return torch.unsqueeze(torch.tensor(x, dtype=torch.float32), 0)


agent = CustomAgent(action_space=len(env.get_action_space()))
agent.train(env, agents.TrainingParams(batch_size=128, episodes=100))
input()
