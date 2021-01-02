import typing as T
import numpy as np
import torch
from collections import deque
import torch.nn.functional as F

from agents import DoubleDqnAgent, DoubleDqnHyperParams, TrainingParams
from agents.explorers import NoisyExplorer, NoisyExplorerParams
from agents.replay_buffers import NStepsPrioritizedReplayBuffer, NStepPrioritizedReplayBufferParams
from environments import SpaceInvadersRam

from testing.helpers import train

FRAME_HISTORY = 4


class CustomSpaceInvadersRam(SpaceInvadersRam):
    def __init__(self, frame_history_size: int):
        super(CustomSpaceInvadersRam, self).__init__()
        self.frame_history_size: int = frame_history_size
        self.frame_history = deque(maxlen=frame_history_size)
        self.reset_history()

    def reset_history(self):
        os = self.get_observation_space()
        for _ in range(self.frame_history_size):
            self.frame_history.append(np.zeros(os, dtype=np.uint8))

    def reset(self) -> np.ndarray:
        self.reset_history()
        s = super(CustomSpaceInvadersRam, self).reset()
        self.frame_history.append(s)
        return np.array(self.frame_history)

    def step(self, action: int) -> T.Tuple[np.ndarray, float, bool]:
        s, r, f = super(CustomSpaceInvadersRam, self).step(action)
        self.frame_history.append(s)
        return np.array(self.frame_history), r, f


env = CustomSpaceInvadersRam(FRAME_HISTORY)


class CustomActionEstimator(torch.nn.Module):
    def __init__(self, in_size: int, out_size: int):
        super(CustomActionEstimator, self).__init__()

        self.linear1 = torch.nn.Linear(in_size, in_size * 10)
        self.linear2 = torch.nn.Linear(self.linear1.out_features, in_size * 10)


        self.gru = torch.nn.GRUCell(self.linear2.out_features, out_size)
        self.linear3 = torch.nn.Linear(out_size, out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = None
        for i in range(x.size()[1]):
            x1 = F.relu(self.linear1(x[:, i, ...]))
            x1 = F.relu(self.linear2(x1))
            m = self.gru(x1, m)

        return F.relu(m)


class CustomDoubleDqnAgent(DoubleDqnAgent):
    def model_factory(self) -> torch.nn.Module:
        return CustomActionEstimator(env.get_observation_space()[0], 100)

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        return torch.unsqueeze(torch.tensor(x, dtype=torch.float32), 0)


EXPLORER_PARAMS = NoisyExplorerParams(extra_layers=[100], std_init=0.5, reset_noise_every=1)
AGENT_PARAMS = DoubleDqnHyperParams(lr=0.01, gamma=0.99, ensure_every=100)
TRAINING_PARAMS = TrainingParams(learn_every=2, batch_size=64, episodes=10000)
REPLAY_BUFFER_PARAMS = NStepPrioritizedReplayBufferParams(max_len=10000, gamma=AGENT_PARAMS.gamma, n_step=5, alpha=0.6,
                                                          init_beta=0.4, final_beta=1.0, increase_beta=1+1e-5)

if __name__ == "__main__":
    agent = CustomDoubleDqnAgent(
        len(env.get_action_space()),
        AGENT_PARAMS,
        TRAINING_PARAMS,
        NoisyExplorer(EXPLORER_PARAMS),
        NStepsPrioritizedReplayBuffer(REPLAY_BUFFER_PARAMS)
    )
    train(agent, env)
