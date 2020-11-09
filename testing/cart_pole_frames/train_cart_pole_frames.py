import typing as T
import numpy as np
import torch
import torchvision

from dqn_agent import DqnAgent, HyperParams
from plotter import Plotter
from explorer import RandomExplorer, RandomExplorerParams
from trainer import Trainer, TrainingParams, TrainingProgress
from environments import CartPoleFrames


EXPLORER_PARAMS = RandomExplorerParams(init_ep=1, final_ep=0.05, decay_ep=1-1e-3)
AGENT_PARAMS = HyperParams(lr=0.01, gamma=0.999, memory_len=5000)
TRAINING_PARAMS = TrainingParams(learn_every=1, ensure_every=10, batch_size=128)


class CartPoleFramesActionEstimator(torch.nn.Module):
    def __init__(self, width: int, height: int, out_size: int):
        super(CartPoleFramesActionEstimator, self).__init__()
        w, h = 150, int(height*150/width)
        self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize((w, h))])
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=5, stride=2)
        w, h = self._conv_size_out((w, h), 5, 2)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        w, h = self._pool_size_out((w, h), 2, 2)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.relu1 = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=5, stride=2)
        w, h = self._conv_size_out((w, h), 5, 2)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        w, h = self._pool_size_out((w, h), 2, 2)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.relu2 = torch.nn.ReLU()

        self.conv3 = torch.nn.Conv2d(16, 16, kernel_size=5, stride=2)
        w, h = self._conv_size_out((w, h), 5, 2)
        self.bn3 = torch.nn.BatchNorm2d(16)
        self.relu3 = torch.nn.ReLU()

        self.head = torch.nn.Linear(w*h*16, out_size)

    @staticmethod
    def _conv_size_out(size: T.Tuple[int, int], kernel_size: int, stride: int) -> T.Tuple[int, int]:
        return (size[0] - (kernel_size - 1) - 1) // stride + 1, (size[1] - (kernel_size - 1) - 1) // stride + 1

    @staticmethod
    def _pool_size_out(size: T.Tuple[int, int], kernel_size: int, stride: int) -> T.Tuple[int, int]:
        return (size[0] - (kernel_size - 1) - 1) // stride + 1, (size[1] - (kernel_size - 1) - 1) // stride + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layers = [self.transform,
                  self.conv1, self.pool1, self.bn1, self.relu1,
                  self.conv2, self.pool2, self.bn2, self.relu2,
                  self.conv3, self.bn3, self.relu3]

        for layer in layers:
            x = layer(x)

        return self.head(x.view(x.size()[0], -1))


def main():
    env: CartPoleFrames = CartPoleFrames()
    plotter: Plotter = Plotter()

    class SpaceInvadersDqnAgent(DqnAgent):
        @staticmethod
        def action_estimator_factory() -> torch.nn.Module:
            return CartPoleFramesActionEstimator(
                env.get_observation_space()[0],
                env.get_observation_space()[1],
                len(env.get_action_space())
            )

        def preprocess(self, x: np.ndarray) -> torch.Tensor:
            return torch.unsqueeze(torch.tensor(x.transpose((2, 0, 1)), dtype=torch.float32) / 255, 0)

        def postprocess(self, t: torch.Tensor) -> np.ndarray:
            return np.array(t.squeeze(0))

    agent: SpaceInvadersDqnAgent = SpaceInvadersDqnAgent(hp=AGENT_PARAMS, use_gpu=True)

    explorer: RandomExplorer = RandomExplorer(EXPLORER_PARAMS)
    trainer = Trainer(env, agent, explorer, TRAINING_PARAMS)

    agent.set_infer_callback(lambda: explorer.decay())

    reward_record: T.List[float] = []

    def progress_callback(progress: TrainingProgress):
        reward_record.append(progress.total_reward)

        plotter.plot(reward_record, aliasing=.8)
        print(
            "lost! achieved "
            "| tries:", progress.tries,
            "| steps survived:", progress.steps_survived,
            "| reward:", progress.total_reward,
            "| epsilon:", round(explorer.ep, 2)
        )

    trainer.set_progress_callback(progress_callback)
    trainer.train(lambda progress: progress.tries >= 1000)


if __name__ == "__main__":
    main()
