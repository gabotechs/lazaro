import typing as T
import numpy as np
import torch
from _collections import deque

from agents.agents.dqn_agent import DqnAgent, TrainingParams, DqnHyperParams
from agents.explorers import RandomExplorer, RandomExplorerParams
from environments import CartPoleFrames
from agents.replay_buffers import RandomReplayBuffer
from plotter import Renderer

from testing.helpers import train


EXPLORER_PARAMS = RandomExplorerParams(init_ep=1, final_ep=0.01, decay_ep=1-1e-3)
AGENT_PARAMS = DqnHyperParams(lr=0.001, gamma=0.995, ensure_every=10)
TRAINING_PARAMS = TrainingParams(learn_every=1, batch_size=128, episodes=5000)
MEMORY_LEN = 5000
FRAME_HISTORY = 4
PLOT_INTERNAL_STATES_EVERY = 1000


class CustomCartPoleFrames(CartPoleFrames):
    def __init__(self, frame_history_size: int):
        super(CustomCartPoleFrames, self).__init__()
        self.frame_history_size: int = frame_history_size
        self.frame_history = deque(maxlen=frame_history_size)
        self.reset_history()

    def reset_history(self):
        os = self.get_observation_space()
        for _ in range(self.frame_history_size):
            self.frame_history.append(np.zeros(os, dtype=np.uint8))

    def reset(self) -> np.ndarray:
        self.reset_history()
        s = super(CustomCartPoleFrames, self).reset()
        self.frame_history.append(s)
        return np.array(self.frame_history)

    def step(self, action: int) -> T.Tuple[np.ndarray, float, bool]:
        s, r, f = super(CustomCartPoleFrames, self).step(action)
        self.frame_history.append(s)
        return np.array(self.frame_history), r, f


env = CustomCartPoleFrames(FRAME_HISTORY)


class CustomActionEstimator(torch.nn.Module):
    def __init__(self, width: int, height: int, out_size: int):
        super(CustomActionEstimator, self).__init__()
        self._forward_count = 0
        self.device = "cpu"

        w, h = width, height
        self.downscale = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        w, h = self._conv_size_out((w, h), 2, 2)

        FILTERS_1 = 1
        self.conv1 = torch.nn.Conv2d(3, FILTERS_1, kernel_size=5, stride=3)
        w, h = self._conv_size_out((w, h), self.conv1.kernel_size[0], self.conv1.stride[0])
        self.bn1 = torch.nn.BatchNorm2d(FILTERS_1)
        self.relu1 = torch.nn.ReLU()
        self.conv1_renderer = Renderer("conv1", FILTERS_1)

        FILTERS_2 = 2
        self.conv2 = torch.nn.Conv2d(FILTERS_1, FILTERS_2, kernel_size=5, stride=2)
        w, h = self._conv_size_out((w, h), self.conv2.kernel_size[0], self.conv2.stride[0])
        self.bn2 = torch.nn.BatchNorm2d(FILTERS_2)
        self.relu2 = torch.nn.ReLU()
        self.conv2_renderer = Renderer("conv2", (int(FILTERS_2 / 2), 2))

        FILTERS_3 = 4
        self.conv3 = torch.nn.Conv2d(FILTERS_2, FILTERS_3, kernel_size=5, stride=2)
        w, h = self._conv_size_out((w, h), self.conv3.kernel_size[0], self.conv3.stride[0])
        self.bn3 = torch.nn.BatchNorm2d(FILTERS_3)
        self.relu3 = torch.nn.ReLU()
        self.conv3_renderer = Renderer("conv3", (int(FILTERS_3 / 2), 2))

        HIDDEN_SIZE = 20

        self.hidden_size = HIDDEN_SIZE
        self.gru = torch.nn.GRUCell(w*h*FILTERS_3, self.hidden_size)
        self.head = torch.nn.Linear(self.hidden_size, out_size)

    def to(self: T, device: str) -> T:
        self.device = device
        return super(CustomActionEstimator, self).to(device)

    @staticmethod
    def _conv_size_out(size: T.Tuple[int, int], kernel_size: int, stride: int) -> T.Tuple[int, int]:
        return (size[0] - (kernel_size - 1) - 1) // stride + 1, (size[1] - (kernel_size - 1) - 1) // stride + 1

    @staticmethod
    def _pool_size_out(size: T.Tuple[int, int], kernel_size: int, stride: int) -> T.Tuple[int, int]:
        return (size[0] - (kernel_size - 1) - 1) // stride + 1, (size[1] - (kernel_size - 1) - 1) // stride + 1

    def init_hidden_state(self, batch_size: int):
        return torch.zeros((batch_size, self.hidden_size), dtype=torch.float32).to(self.device)

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
        batch_size = x.size()[0]
        is_normal_forward = batch_size == 1
        m = self.init_hidden_state(batch_size)
        for i in range(x.size()[1]):
            x1 = x[:, i, ...]
            x1 = self.downscale(x1)
            x1 = self.conv1(x1)
            x1 = self.bn1(x1)
            # x1 = self.relu1(x1)
            if is_normal_forward and PLOT_INTERNAL_STATES_EVERY and self._forward_count % PLOT_INTERNAL_STATES_EVERY == 0 and i == x.size()[1]-1:
                self.render_intermediate_frame(self.conv1_renderer, x1)

            x1 = self.conv2(x1)
            x1 = self.bn2(x1)
            # x1 = self.relu2(x1)
            if is_normal_forward and PLOT_INTERNAL_STATES_EVERY and self._forward_count % PLOT_INTERNAL_STATES_EVERY == 0 and i == x.size()[1] - 1:
                self.render_intermediate_frame(self.conv2_renderer, x1)

            x1 = self.conv3(x1)
            # x1 = self.relu3(x1)
            if is_normal_forward and PLOT_INTERNAL_STATES_EVERY and self._forward_count % PLOT_INTERNAL_STATES_EVERY == 0 and i == x.size()[1] - 1:
                self.render_intermediate_frame(self.conv3_renderer, x1)

            m = self.gru(x1.flatten(1), m)

        self._forward_count += 1
        return self.head(self.relu1(m))


class CustomMDqnAgent(DqnAgent):
    def model_factory(self) -> torch.nn.Module:
        return CustomActionEstimator(
            env.get_observation_space()[0]-170-84,
            env.get_observation_space()[1],
            len(env.get_action_space())
        )

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        return torch.unsqueeze(torch.tensor(x[:, 170:-84, :, :].transpose((0, 3, 1, 2)), dtype=torch.float32) / 255, 0)


if __name__ == "__main__":
    agent = CustomMDqnAgent(AGENT_PARAMS, TRAINING_PARAMS, RandomExplorer(EXPLORER_PARAMS), RandomReplayBuffer(MEMORY_LEN))
    train(agent, env)
