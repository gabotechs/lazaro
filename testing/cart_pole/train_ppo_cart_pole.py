import numpy as np
import torch
import torch.nn.functional as F

from agents import PpoAgent, PpoHyperParams, TrainingParams
from agents.replay_buffers import NStepsPrioritizedReplayBuffer, NStepPrioritizedReplayBufferParams
from agents.replay_buffers import RandomReplayBuffer, RandomReplayBufferParams
from agents.explorers import NoisyExplorer, NoisyExplorerParams
from agents.explorers import RandomExplorer, RandomExplorerParams
from environments import CartPole


AGENT_PARAMS = PpoHyperParams(lr=0.01, gamma=0.99, clip_factor=0.2, ensure_every=1, entropy_factor=0.01)
TRAINING_PARAMS = TrainingParams(learn_every=1, batch_size=16, episodes=300)
NOISY_EXPLORER = NoisyExplorer(NoisyExplorerParams(extra_layers=[], std_init=0.5, reset_noise_every=1))
RANDOM_EXPLORER = RandomExplorer(RandomExplorerParams(init_ep=1, final_ep=0, decay_ep=1e-3))
N_STEPS_PRIORITIZED_REPLAY_BUFFER = NStepsPrioritizedReplayBuffer(NStepPrioritizedReplayBufferParams(
    max_len=10000, gamma=AGENT_PARAMS.gamma, n_step=3, alpha=0.6, init_beta=0.4, final_beta=1.0, increase_beta=1e-4)
)
RANDOM_REPLAY_BUFFER = RandomReplayBuffer(RandomReplayBufferParams(max_len=10000))

env = CartPole()


class CustomActionEstimator(torch.nn.Module):
    def __init__(self, in_size: int):
        super(CustomActionEstimator, self).__init__()
        self.linear1 = torch.nn.Linear(in_size, 10)
        self.linear2 = torch.nn.Linear(10, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear2(F.relu(self.linear1(x))))


class CustomActorCriticAgent(PpoAgent):
    def model_factory(self) -> torch.nn.Module:
        return CustomActionEstimator(env.get_observation_space()[0])

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        return torch.unsqueeze(torch.tensor(x, dtype=torch.float32), 0)


if __name__ == "__main__":
    agent = CustomActorCriticAgent(
        len(env.get_action_space()),
        AGENT_PARAMS,
        TRAINING_PARAMS,
        RANDOM_EXPLORER,
        RANDOM_REPLAY_BUFFER
    )
    agent.train(env)
    input()
