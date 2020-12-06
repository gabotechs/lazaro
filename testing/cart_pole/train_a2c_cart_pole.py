import numpy as np
import torch

from agents import ActorCriticAgent, ACHyperParams, TrainingParams
from agents.replay_buffers import RandomReplayBuffer
from environments import CartPole

from testing.helpers import train

AGENT_PARAMS = ACHyperParams(c_lr=0.001, a_lr=0.0001, gamma=0.995)
TRAINING_PARAMS = TrainingParams(learn_every=1, ensure_every=10, batch_size=128, finish_condition=lambda x: False)
MEMORY_LEN = 1000

env = CartPole()


class CustomActionEstimator(torch.nn.Module):
    def __init__(self, in_size: int, out_size: int):
        super(CustomActionEstimator, self).__init__()
        self.out_size = out_size
        self.linear1 = torch.nn.Linear(in_size, in_size*10)
        self.relu1 = torch.nn.ReLU()

        self.linear2 = torch.nn.Linear(in_size*10, out_size*100)
        self.relu2 = torch.nn.ReLU()

        self.linear3 = torch.nn.Linear(out_size*100, out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        return self.linear3(x)


class CustomActorCriticAgent(ActorCriticAgent):
    def model_factory(self) -> torch.nn.Module:
        return CustomActionEstimator(env.get_observation_space()[0], len(env.get_action_space()))

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        return torch.unsqueeze(torch.tensor(x, dtype=torch.float32), 0)


if __name__ == "__main__":
    agent = CustomActorCriticAgent(AGENT_PARAMS, TRAINING_PARAMS, None, RandomReplayBuffer(MEMORY_LEN), use_gpu=True)
    train(agent, env)
