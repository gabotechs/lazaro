import typing as T
import numpy as np
import torch
from collections import deque
import torch.nn.functional as F

from agents import DoubleDqnAgent, DoubleDqnHyperParams, TrainingParams
from agents.explorers import NoisyExplorer, NoisyExplorerParams
from agents.replay_buffers import NStepsPrioritizedReplayBuffer, NStepPrioritizedReplayBufferParams
from environments import SpaceInvaders
from plotter import Renderer

FRAME_HISTORY = 2
PLOT_INTERNAL_STATES_EVERY = 0


class CustomSpaceInvaders(SpaceInvaders):
    def __init__(self, frame_history_size: int):
        super(CustomSpaceInvaders, self).__init__()
        self.frame_history_size: int = frame_history_size
        self.frame_history = deque(maxlen=frame_history_size)
        self.reset_history()

    def reset_history(self):
        os = self.get_observation_space()
        for _ in range(self.frame_history_size):
            self.frame_history.append(np.zeros(os, dtype=np.uint8))

    def reset(self) -> np.ndarray:
        self.reset_history()
        s = super(CustomSpaceInvaders, self).reset()
        self.frame_history.append(s)
        return np.array(self.frame_history)

    def do_step(self, action: int) -> T.Tuple[np.ndarray, float, bool]:
        s, r, f = super(CustomSpaceInvaders, self).step(action)
        self.frame_history.append(s)
        return np.array(self.frame_history), r, f


env = CustomSpaceInvaders(FRAME_HISTORY)


class CustomActionEstimator(torch.nn.Module):
    def __init__(self, width: int, height: int, out_size: int):
        super(CustomActionEstimator, self).__init__()
        self._forward_count = 0
        self.device = "cuda"

        w, h = width, height

        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        w, h = self._conv_size_out((w, h), self.max_pool.kernel_size, self.max_pool.stride)

        FILTERS_1 = 16
        self.conv1 = torch.nn.Conv2d(3, FILTERS_1, kernel_size=8, stride=4)
        w, h = self._conv_size_out((w, h), self.conv1.kernel_size[0], self.conv1.stride[0])
        self.conv1_renderer = Renderer("conv1", (int(FILTERS_1 / 4), 4))

        FILTERS_2 = 32
        self.conv2 = torch.nn.Conv2d(FILTERS_1, FILTERS_2, kernel_size=4, stride=3)
        w, h = self._conv_size_out((w, h), self.conv2.kernel_size[0], self.conv2.stride[0])
        self.conv2_renderer = Renderer("conv2", (int(FILTERS_2 / 8), 8))

        FILTERS_3 = 32
        self.conv3 = torch.nn.Conv2d(FILTERS_2, FILTERS_3, kernel_size=3, stride=2)
        w, h = self._conv_size_out((w, h), self.conv3.kernel_size[0], self.conv3.stride[0])
        self.conv3_renderer = Renderer("conv3", (int(FILTERS_3 / 8), 8))

        HIDDEN_SIZE = 1000

        self.hidden_size = HIDDEN_SIZE
        self.gru = torch.nn.GRUCell(w*h*FILTERS_3, self.hidden_size)

        UNION_SIZE = 512
        self.union = torch.nn.Linear(self.hidden_size+w*h*FILTERS_3, UNION_SIZE)
        self.head = torch.nn.Linear(UNION_SIZE, out_size)

    @staticmethod
    def _conv_size_out(size: T.Tuple[int, int], kernel_size: int, stride: int) -> T.Tuple[int, int]:
        return (size[0] - (kernel_size - 1) - 1) // stride + 1, (size[1] - (kernel_size - 1) - 1) // stride + 1

    @staticmethod
    def _pool_size_out(size: T.Tuple[int, int], kernel_size: int, stride: int) -> T.Tuple[int, int]:
        return (size[0] - (kernel_size - 1) - 1) // stride + 1, (size[1] - (kernel_size - 1) - 1) // stride + 1

    def init_hidden_state(self, batch_size: int):
        return torch.zeros((batch_size, self.hidden_size), dtype=torch.float32).to(self.gru.weight_hh.device.type)

    @staticmethod
    def render_intermediate_frame(renderer: Renderer, x: torch.Tensor):
        frames = x[0].clone().cpu().detach().numpy().transpose((1, 2, 0))
        for filter_index in range(frames.shape[2]):
            title = "filter: " + str(filter_index)
            frame = frames[:, :, filter_index]
            divisor = max(abs(frame.max()), abs(frame.min()))
            frame = frame * 255 / divisor
            renderer.render(filter_index, title, frame)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        is_normal_forward = batch_size == 1
        m = self.init_hidden_state(batch_size)
        x1 = x[:, -1, ...]
        for i in range(x.shape[1]):
            x1 = self.max_pool(x[:, i, ...])
            x1 = F.relu(self.conv1(x1))
            if is_normal_forward and PLOT_INTERNAL_STATES_EVERY and self._forward_count % PLOT_INTERNAL_STATES_EVERY == 0 and i == x.size()[1]-1:
                self.render_intermediate_frame(self.conv1_renderer, x1)

            x1 = F.relu(self.conv2(x1))
            if is_normal_forward and PLOT_INTERNAL_STATES_EVERY and self._forward_count % PLOT_INTERNAL_STATES_EVERY == 0 and i == x.size()[1]-1:
                self.render_intermediate_frame(self.conv2_renderer, x1)

            x1 = F.relu(self.conv3(x1))
            if is_normal_forward and PLOT_INTERNAL_STATES_EVERY and self._forward_count % PLOT_INTERNAL_STATES_EVERY == 0 and i == x.size()[1]-1:
                self.render_intermediate_frame(self.conv3_renderer, x1)

            m = self.gru(x1.reshape(batch_size, -1), m)

        if is_normal_forward:
            self._forward_count += 1
        return F.relu(self.head(F.relu(self.union(F.relu(torch.cat([x1.reshape(batch_size, -1), m], dim=1))))))


class CustomDoubleDqnAgent(DoubleDqnAgent):
    def model_factory(self) -> torch.nn.Module:
        return CustomActionEstimator(env.get_observation_space()[0], env.get_observation_space()[1], 20)

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        return torch.unsqueeze(torch.tensor(x.transpose((0, 3, 1, 2)), dtype=torch.float32) / 255, 0)


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
