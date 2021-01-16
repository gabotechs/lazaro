import torch
import torch.nn.functional as F
import numpy as np
import pytest
import typing as T
from agents import DqnAgent, DoubleDqnAgent, DuelingDqnAgent, DoubleDuelingDqnAgent, AdvantageActorCriticAgent, \
    MonteCarloAdvantageActorCriticAgent

from agents import DqnHyperParams, DoubleDqnHyperParams, TrainingParams, TrainingProgress, ACHyperParams
from agents.base import Agent
from environments import CartPole
from agents.replay_buffers import RandomReplayBuffer, NStepsRandomReplayBuffer, NStepsPrioritizedReplayBuffer, \
    PrioritizedReplayBuffer, AnyReplayBuffer
from agents.replay_buffers import ReplayBufferParams, NStepReplayBufferParams, NStepPrioritizedReplayBufferParams, \
    PrioritizedReplayBufferParams
from agents.explorers import RandomExplorer, NoisyExplorer, AnyExplorer
from agents.explorers import RandomExplorerParams, NoisyExplorerParams


def get_test_env():
    return CartPole()


def get_random_explorer():
    return RandomExplorer(RandomExplorerParams(init_ep=1.0, final_ep=0.0, decay_ep=1e-2))


def get_noisy_explorer():
    return NoisyExplorer(NoisyExplorerParams(extra_layers=[], std_init=0.5, reset_noise_every=1))


def get_random_replay_buffer():
    return RandomReplayBuffer(ReplayBufferParams(max_len=1000))


def get_n_step_random_replay_buffer():
    return NStepsRandomReplayBuffer(NStepReplayBufferParams(max_len=1000, n_step=5, gamma=0.9))


def get_prioritized_replay_buffer():
    return PrioritizedReplayBuffer(PrioritizedReplayBufferParams(
        max_len=1000, alpha=0.6, init_beta=0.4, final_beta=1.0, increase_beta=1e-4
    ))


def get_n_step_prioritized_replay_buffer():
    return NStepsPrioritizedReplayBuffer(NStepPrioritizedReplayBufferParams(
        max_len=1000, gamma=0.9, n_step=5, alpha=0.6, init_beta=0.4, final_beta=1.0, increase_beta=1e-4
    ))


def get_dqn_agent(replay_buffer: AnyReplayBuffer, explorer: AnyExplorer) -> Agent:
    class CustomAgent(DqnAgent):
        def model_factory(self) -> torch.nn.Module:
            return NN()

        def preprocess(self, x: np.ndarray) -> torch.Tensor:
            return torch.unsqueeze(torch.tensor(x, dtype=torch.float32), 0)

    hp = DqnHyperParams(lr=0.03, gamma=0.9)
    tp = TrainingParams(learn_every=1, batch_size=64, episodes=200)
    return CustomAgent(2, hp, tp, explorer, replay_buffer, save_progress=False, tensor_board_log=False)


def get_double_dqn_agent(replay_buffer: AnyReplayBuffer, explorer: AnyExplorer) -> Agent:
    class CustomAgent(DoubleDqnAgent):
        def model_factory(self) -> torch.nn.Module:
            return NN()

        def preprocess(self, x: np.ndarray) -> torch.Tensor:
            return torch.unsqueeze(torch.tensor(x, dtype=torch.float32), 0)

    hp = DoubleDqnHyperParams(lr=0.03, gamma=0.9, ensure_every=20)
    tp = TrainingParams(learn_every=1, batch_size=64, episodes=200)
    return CustomAgent(2, hp, tp, explorer, replay_buffer, save_progress=False, tensor_board_log=False)


def get_dueling_dqn_agent(replay_buffer: AnyReplayBuffer, explorer: AnyExplorer) -> Agent:
    class CustomAgent(DuelingDqnAgent):
        def model_factory(self) -> torch.nn.Module:
            return NN()

        def preprocess(self, x: np.ndarray) -> torch.Tensor:
            return torch.unsqueeze(torch.tensor(x, dtype=torch.float32), 0)

    hp = DoubleDqnHyperParams(lr=0.03, gamma=0.9, ensure_every=20)
    tp = TrainingParams(learn_every=1, batch_size=64, episodes=200)
    return CustomAgent(2, hp, tp, explorer, replay_buffer, save_progress=False, tensor_board_log=False)


def get_double_dueling_dqn_agent(replay_buffer: AnyReplayBuffer, explorer: AnyExplorer) -> Agent:
    class CustomAgent(DoubleDuelingDqnAgent):
        def model_factory(self) -> torch.nn.Module:
            return NN()

        def preprocess(self, x: np.ndarray) -> torch.Tensor:
            return torch.unsqueeze(torch.tensor(x, dtype=torch.float32), 0)

    hp = DoubleDqnHyperParams(lr=0.03, gamma=0.9, ensure_every=20)
    tp = TrainingParams(learn_every=1, batch_size=64, episodes=200)
    return CustomAgent(2, hp, tp, explorer, replay_buffer, save_progress=False, tensor_board_log=False)


class NN(torch.nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.linear_1 = torch.nn.Linear(4, 10)
        self.linear_2 = torch.nn.Linear(10, 10)

    def forward(self, x):
        return F.relu(self.linear_2(F.relu(self.linear_1(x))))


params = []
for r_b in [get_random_replay_buffer(),
            get_n_step_random_replay_buffer(),
            get_prioritized_replay_buffer(),
            get_n_step_prioritized_replay_buffer()]:
    for ex in [get_random_explorer(),
               get_noisy_explorer()]:
        for ag in [get_dqn_agent(r_b, ex),
                   get_double_dqn_agent(r_b, ex),
                   get_dueling_dqn_agent(r_b, ex),
                   get_double_dueling_dqn_agent(r_b, ex)]:
            params.append(pytest.param(ag, id=f"{ag.get_self_class_name()}({type(r_b).__name__}, {type(ex).__name__})"))


@pytest.mark.parametrize("agent", params)
def test_agents(agent: Agent):
    env = get_test_env()
    env.visualize = False

    record: T.List[TrainingProgress] = []

    def on_progress(progress: TrainingProgress):
        record.append(progress)
        return progress.total_reward > 100

    agent.add_progress_callback(on_progress)
    agent.train(env)
    assert any([r.total_reward > 100 for r in record])
