import torch
import torch.nn.functional as F
import agents
from environments import CartPole

env = CartPole()

ACTION_SPACE = len(env.get_action_space())
OBSERVATION_SPACE = env.get_observation_space()[0]
LAYER_SIZE = 10
RANDOM_EXPLORER_PARAMS = agents.explorers.RandomExplorerParams(init_ep=1.0, final_ep=0.01, decay_ep=1e-3)
AGENT_PARAMS = agents.DoubleDuelingDqnHyperParams(lr=0.01, gamma=0.95, ensure_every=10)
TRAINING_PARAMS = agents.TrainingParams(learn_every=1, batch_size=64, episodes=100)
REPLAY_BUFFER_PARAMS = agents.replay_buffers.RandomReplayBufferParams(max_len=5000)


class CustomNN(torch.nn.Module):
    def __init__(self):
        super(CustomNN, self).__init__()
        self.linear = torch.nn.Linear(OBSERVATION_SPACE, LAYER_SIZE)

    def forward(self, x):
        return F.relu(self.linear(x))


class CustomAgent(agents.DoubleDuelingDqnAgent):
    def model_factory(self):
        return CustomNN()

    def preprocess(self, x):
        return torch.unsqueeze(torch.tensor(x, dtype=torch.float32), 0)


agent = CustomAgent(
    ACTION_SPACE,
    AGENT_PARAMS,
    TRAINING_PARAMS,
    agents.explorers.RandomExplorer(RANDOM_EXPLORER_PARAMS),
    agents.replay_buffers.RandomReplayBuffer(REPLAY_BUFFER_PARAMS)
)
agent.train(env)
