import typing as T
import numpy as np
import torch

from agents import DqnHyperParams, RandomExplorerParams, TrainingParams, DqnMemoryAgent, RandomExplorer

from environments import CartPoleFrames
from agents.replay_buffers import RandomReplayBuffer
from plotter import Renderer

from testing.helpers import train


EXPLORER_PARAMS = RandomExplorerParams(init_ep=1, final_ep=0.01, decay_ep=1-4e-3)
AGENT_PARAMS = DqnHyperParams(lr=0.001, gamma=0.995, ensure_every=10)
TRAINING_PARAMS = TrainingParams(learn_every=1, batch_size=128, episodes=5000)
MEMORY_LEN = 1000
PLOT_INTERNAL_STATES_EVERY = 1000


env = CartPoleFrames()


class CustomActionEstimator(torch.nn.Module):
    def __init__(self, width: int, height: int, out_size: int):
        super(CustomActionEstimator, self).__init__()
        w_, h_ = width, height

        self.downscale = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.downscale_renderer = Renderer("downscale", 3)

        w_, h_ = self._conv_size_out((w_, h_), 2, 2)

        # ==== action path ====
        self.filters1 = 4
        self.conv1 = torch.nn.Conv2d(3, self.filters1, kernel_size=5, stride=2)
        w, h = self._conv_size_out((w_, h_), 5, 2)
        self.bn1 = torch.nn.BatchNorm2d(self.filters1)
        self.relu1 = torch.nn.ReLU()
        self.conv1_renderer = Renderer("conv1", self.filters1)
        self.forward_count = 0

        self.filters2 = 4
        self.conv2 = torch.nn.Conv2d(self.filters1, self.filters2, kernel_size=5, stride=2)
        w, h = self._conv_size_out((w, h), 5, 2)
        self.bn2 = torch.nn.BatchNorm2d(self.filters2)
        self.relu2 = torch.nn.ReLU()
        self.conv2_renderer = Renderer("conv2", (int(self.filters2 / 2), 2))

        self.filters3 = 8
        self.conv3 = torch.nn.Conv2d(self.filters2, self.filters3, kernel_size=5, stride=2)
        w, h = self._conv_size_out((w, h), 5, 2)
        self.bn3 = torch.nn.BatchNorm2d(self.filters3)
        self.relu3 = torch.nn.ReLU()
        self.conv3_renderer = Renderer("conv3", (int(self.filters3 / 2), 2))
        self.action_path_size = 20
        self.linear1 = torch.nn.Linear(self.filters3*h*w, self.action_path_size)

        # ==== memory path ====
        self.mfilters1 = 4
        self.mconv1 = torch.nn.Conv2d(3, self.mfilters1, kernel_size=8, stride=2)
        w, h = self._conv_size_out((w_, h_), 8, 2)
        self.mbn1 = torch.nn.BatchNorm2d(self.mfilters1)
        self.mrelu1 = torch.nn.ReLU()
        self.mconv1_renderer = Renderer("mconv1", self.mfilters1)

        self.mfilters2 = 8

        self.mconv2 = torch.nn.Conv2d(self.mfilters1, self.mfilters2, kernel_size=4, stride=2)
        w, h = self._conv_size_out((w, h), 4, 2)
        self.mbn2 = torch.nn.BatchNorm2d(self.mfilters2)
        self.mrelu2 = torch.nn.ReLU()
        self.mconv2_renderer = Renderer("mconv2", (int(self.mfilters2 / 2), 2))

        self.mfilters3 = 16
        self.mconv3 = torch.nn.Conv2d(self.mfilters2, self.mfilters3, kernel_size=2, stride=2)
        w, h = self._conv_size_out((w, h), 2, 2)
        self.mbn3 = torch.nn.BatchNorm2d(self.mfilters3)
        self.mrelu3 = torch.nn.ReLU()
        self.mconv3_renderer = Renderer("mconv3", (int(self.mfilters3 / 4), 4))

        self.hidden_size = 20
        self.gru = torch.nn.GRUCell(w*h*self.mfilters3, self.hidden_size)

        # ==== common path ====
        self.head = torch.nn.Linear(self.action_path_size+self.hidden_size, out_size)

    def init_memory(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros((batch_size, self.hidden_size), dtype=torch.float32)

    @staticmethod
    def _conv_size_out(size: T.Tuple[int, int], kernel_size: int, stride: int) -> T.Tuple[int, int]:
        return (size[0] - (kernel_size - 1) - 1) // stride + 1, (size[1] - (kernel_size - 1) - 1) // stride + 1

    @staticmethod
    def _pool_size_out(size: T.Tuple[int, int], kernel_size: int, stride: int) -> T.Tuple[int, int]:
        return (size[0] - (kernel_size - 1) - 1) // stride + 1, (size[1] - (kernel_size - 1) - 1) // stride + 1

    @staticmethod
    def render_intermediate_frame(renderer: Renderer, x: torch.Tensor):
        frames = x[0].clone().cpu().detach().numpy().transpose((1, 2, 0))
        for filter_index in range(frames.shape[2]):
            title = "filter: " + str(filter_index)
            frame = frames[:, :, filter_index]
            divisor = max(abs(frame.max()), abs(frame.min()))
            frame = frame * 255 / divisor
            renderer.render(filter_index, title, frame)

    def forward(self, _x: torch.Tensor, m: torch.Tensor) -> T.Tuple[torch.Tensor, torch.Tensor]:
        _x = self.downscale(_x)
        is_normal_forward = _x.size()[0] == 1

        # ==== action path ====
        ax = self.conv1(_x)
        ax = self.relu1(ax)
        ax = self.bn1(ax)
        if is_normal_forward and PLOT_INTERNAL_STATES_EVERY and self.forward_count % PLOT_INTERNAL_STATES_EVERY == 0:
            self.render_intermediate_frame(self.conv1_renderer, ax)

        ax = self.conv2(ax)
        ax = self.relu2(ax)
        ax = self.bn2(ax)
        if is_normal_forward and PLOT_INTERNAL_STATES_EVERY and self.forward_count % PLOT_INTERNAL_STATES_EVERY == 0:
            self.render_intermediate_frame(self.conv2_renderer, ax)

        ax = self.conv3(ax)
        ax = self.relu3(ax)
        ax = self.bn3(ax)
        if is_normal_forward and PLOT_INTERNAL_STATES_EVERY and self.forward_count % PLOT_INTERNAL_STATES_EVERY == 0:
            self.render_intermediate_frame(self.conv3_renderer, ax)

        ax = ax.view(ax.size()[0], -1)
        ax = self.linear1(ax)

        # ==== memory path ====
        mx = self.mconv1(_x)
        mx = self.mrelu1(mx)
        mx = self.mbn1(mx)
        if is_normal_forward and PLOT_INTERNAL_STATES_EVERY and self.forward_count % PLOT_INTERNAL_STATES_EVERY == 0:
            self.render_intermediate_frame(self.mconv1_renderer, mx)

        mx = self.mconv2(mx)
        mx = self.mrelu2(mx)
        mx = self.mbn2(mx)
        if is_normal_forward and PLOT_INTERNAL_STATES_EVERY and self.forward_count % PLOT_INTERNAL_STATES_EVERY == 0:
            self.render_intermediate_frame(self.mconv2_renderer, mx)

        mx = self.mconv3(mx)
        mx = self.mrelu3(mx)
        mx = self.mbn3(mx)
        if is_normal_forward and PLOT_INTERNAL_STATES_EVERY and self.forward_count % PLOT_INTERNAL_STATES_EVERY == 0:
            self.render_intermediate_frame(self.mconv3_renderer, mx)

        mx = mx.view(mx.size()[0], -1)
        mx = self.gru(mx, m)
        m_ = mx.clone().detach()

        # ==== common path ====
        if is_normal_forward:
            self.forward_count += 1
        x = torch.cat([ax, mx], dim=1)
        x = self.head(x)
        return x, m_


class CustomMDqnAgent(DqnMemoryAgent):

    def memory_init(self) -> torch.Tensor:
        return self.action_estimator.init_memory()

    def model_factory(self) -> torch.nn.Module:
        return CustomActionEstimator(
            env.get_observation_space()[0]-170-84,
            env.get_observation_space()[1],
            len(env.get_action_space())
        )

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        return torch.unsqueeze(torch.tensor(x[170:-84, :, :].transpose((2, 0, 1)), dtype=torch.float32) / 255, 0)


if __name__ == "__main__":
    agent = CustomMDqnAgent(AGENT_PARAMS, TRAINING_PARAMS, RandomExplorer(EXPLORER_PARAMS), RandomReplayBuffer(MEMORY_LEN))
    train(agent, env)
