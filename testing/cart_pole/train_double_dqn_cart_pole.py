import numpy as np
import os
import typing as T
from collections import deque
import torch
import torch.nn.functional as F

from agents import DoubleDuelingDqnAgent, DoubleDqnHyperParams, TrainingParams, MonteCarloAdvantageActorCriticAgent, ACHyperParams
from agents.explorers import NoisyExplorer, NoisyExplorerParams, RandomExplorer, RandomExplorerParams
from agents.replay_buffers import NStepsPrioritizedReplayBuffer, NStepPrioritizedReplayBufferParams
from environments import CartPole

os.environ["LOG_LEVEL"] = "WARNING"


class CustomCartPole(CartPole):
    def __init__(self, frame_history_size: int):
        super(CustomCartPole, self).__init__()
        self.frame_history_size: int = frame_history_size
        self.frame_history = deque(maxlen=frame_history_size)
        self.reset_history()

    def get_observation_space(self) -> T.Tuple[int]:
        return 2,

    def reset_history(self):
        os = self.get_observation_space()
        for _ in range(self.frame_history_size):
            self.frame_history.append(np.zeros(os, dtype=np.uint8))

    def reset(self) -> np.ndarray:
        self.reset_history()
        s = super(CustomCartPole, self).reset()
        self.frame_history.append(np.array([s[0], s[2]]))
        return np.array(self.frame_history)

    def do_step(self, action: int) -> T.Tuple[np.ndarray, float, bool]:
        s, r, f = super(CustomCartPole, self).do_step(action)
        self.frame_history.append(np.array([s[0], s[2]]))
        return np.array(self.frame_history), r, f


env = CustomCartPole(2)


class CustomActionEstimator(torch.nn.Module):
    def __init__(self, in_size: int):
        super(CustomActionEstimator, self).__init__()

        INPUT = in_size
        CTX_LINEAR_1 = 20
        CTX_GRU_SIZE = 50
        GRU_ATTENTION_SIZE = 20

        LINEAR_1 = 20

        OUT_SIZE = 20

        self.ctx_linear_1 = torch.nn.Linear(INPUT, CTX_LINEAR_1)
        self.ctx_gru = torch.nn.GRUCell(self.ctx_linear_1.out_features, CTX_GRU_SIZE)
        self.attn_gru = torch.nn.GRUCell(self.ctx_linear_1.out_features, GRU_ATTENTION_SIZE)
        self.attn_condenser = torch.nn.Linear(GRU_ATTENTION_SIZE, 1)

        self.linear_1 = torch.nn.Linear(INPUT, LINEAR_1)

        self.head = torch.nn.Linear(CTX_GRU_SIZE+self.linear_1.out_features, OUT_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = None
        states = []
        for i in range(x.shape[1]):
            ctx_x = F.relu(self.ctx_linear_1(x[:, i, ...]))
            m = self.ctx_gru(ctx_x, m)
            states.append(m)
        states = torch.stack(states, dim=1)

        in_x = F.relu(self.linear_1(x[:, -1, ...]))

        attn_m = None
        attn_weights = []
        for i in range(x.shape[1]):
            attn_m = self.attn_gru(in_x, attn_m)
            attn_w = self.attn_condenser(attn_m)
            attn_weights.append(attn_w)

        attn_weights = F.softmax(torch.stack(attn_weights, dim=1), dim=1)
        attn_out = (attn_weights * states).sum(dim=1)
        out = torch.cat((in_x, attn_out), dim=1)
        return F.relu(self.head(out))


class CustomDoubleDqnAgent(MonteCarloAdvantageActorCriticAgent):
    def model_factory(self) -> torch.nn.Module:
        return CustomActionEstimator(env.get_observation_space()[0])

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        return torch.unsqueeze(torch.tensor(x, dtype=torch.float32), 0)


NOISY_EXPLORER_PARAMS = NoisyExplorerParams(extra_layers=[], std_init=0.5, reset_noise_every=1)
RANDOM_EXPLORER_PARAMS = RandomExplorerParams(init_ep=1.0, final_ep=0.01, decay_ep=1e-3)
AGENT_PARAMS = ACHyperParams(a_lr=0.01, gamma=0.95, c_lr=0.01)
TRAINING_PARAMS = TrainingParams(learn_every=1, batch_size=64, episodes=5000)
REPLAY_BUFFER_PARAMS = NStepPrioritizedReplayBufferParams(max_len=20000, gamma=AGENT_PARAMS.gamma, n_step=3, alpha=0.6,
                                                          init_beta=0.4, final_beta=1.0, increase_beta=1e-5)

if __name__ == "__main__":
    agent = CustomDoubleDqnAgent(
        len(env.get_action_space()),
        AGENT_PARAMS,
        TRAINING_PARAMS,
        RandomExplorer(RANDOM_EXPLORER_PARAMS),
        NStepsPrioritizedReplayBuffer(REPLAY_BUFFER_PARAMS)
    )
    agent.train(env)
