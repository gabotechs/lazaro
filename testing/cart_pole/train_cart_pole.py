import typing as T
import numpy as np
import torch

from dqn_agent import DqnAgent, HyperParams
from plotter import Plotter
from explorer import RandomExplorer, RandomExplorerParams
from trainer import Trainer, TrainingParams, TrainingProgress
from environments import CartPole

EXPLORER_PARAMS = RandomExplorerParams(init_ep=1, final_ep=0.05, decay_ep=1-1e-3)
AGENT_PARAMS = HyperParams(lr=0.01, gamma=0.995)
TRAINING_PARAMS = TrainingParams(learn_every=1, ensure_every=10, batch_size=128)


class CartPoleActionEstimator(torch.nn.Module):
    def __init__(self, in_size: int, out_size: int):
        super(CartPoleActionEstimator, self).__init__()
        self.linear1 = torch.nn.Linear(in_size, in_size*10)
        self.relu1 = torch.nn.ReLU()

        self.linear2 = torch.nn.Linear(in_size*10, out_size*10)
        self.relu2 = torch.nn.ReLU()

        self.linear3 = torch.nn.Linear(out_size*10, out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        return self.linear3(x)


def main():
    env: CartPole = CartPole()
    plotter: Plotter = Plotter()

    class CartPoleDqnAgent(DqnAgent):
        @staticmethod
        def action_estimator_factory() -> torch.nn.Module:
            return CartPoleActionEstimator(env.get_observation_space()[0], len(env.get_action_space()))

        def preprocess(self, x: np.ndarray) -> torch.Tensor:
            return torch.unsqueeze(torch.tensor(x, dtype=torch.float32), 0)

        def postprocess(self, t: torch.Tensor) -> np.ndarray:
            return np.array(t.squeeze(0))

    agent: CartPoleDqnAgent = CartPoleDqnAgent(
        hp=AGENT_PARAMS,
        use_gpu=True
    )

    explorer: RandomExplorer = RandomExplorer(EXPLORER_PARAMS)
    trainer = Trainer(env, agent, explorer, TRAINING_PARAMS)

    agent.set_infer_callback(lambda: explorer.decay())

    reward_record: T.List[float] = []

    def progress_callback(progress: TrainingProgress):
        reward_record.append(progress.total_reward)

        plotter.plot(reward_record)
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
