import torch
import torch.nn.functional as F
import agents
from environments import CartPole

env = CartPole()


class CustomNN(torch.nn.Module):
    def __init__(self):
        super(CustomNN, self).__init__()
        self.linear = torch.nn.Linear(env.get_observation_space()[0], 30)

    def forward(self, x):
        return F.relu(self.linear(x))


class CustomAgent(agents.PpoAgent):
    def model_factory(self):
        return CustomNN()

    def preprocess(self, x):
        return torch.unsqueeze(torch.tensor(x, dtype=torch.float32), 0)


agent = CustomAgent(
    len(env.get_action_space()),
    agents.explorers.RandomExplorer(),
    agents.replay_buffers.RandomReplayBuffer(),
    agents.TrainingParams(batch_size=128, episodes=100),
)
agent.train(env)
input()
