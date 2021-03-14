import torch
import torch.nn.functional as F
import numpy as np
import pytest
import typing as T
from agents import DqnAgent, DoubleDqnAgent, DuelingDqnAgent, DoubleDuelingDqnAgent, A2cAgent, \
    MonteCarloA2cCriticAgent, PpoAgent

from agents import DqnHyperParams, DoubleDqnHyperParams, DuelingDqnHyperParams, DoubleDuelingDqnHyperParams, \
    TrainingParams, TrainingProgress, A2CHyperParams, PpoHyperParams
from agents.base import Agent
from environments import CartPole
from agents.replay_buffers import RandomReplayBuffer, NStepsRandomReplayBuffer, NStepsPrioritizedReplayBuffer, \
    PrioritizedReplayBuffer, AnyReplayBuffer
from agents.replay_buffers import RandomReplayBufferParams, NStepReplayBufferParams, \
    NStepPrioritizedReplayBufferParams, PrioritizedReplayBufferParams
from agents.explorers import RandomExplorer, NoisyExplorer, AnyExplorer
from agents.explorers import RandomExplorerParams, NoisyExplorerParams


EXPECTED_REWARD = 25


def get_test_env():
    return CartPole()


def get_random_explorer():
    return RandomExplorer(RandomExplorerParams(init_ep=1.0, final_ep=0.05, decay_ep=5e-3))


def get_noisy_explorer():
    return NoisyExplorer(NoisyExplorerParams(extra_layers=tuple(), std_init=0.5, reset_noise_every=1))


def get_random_replay_buffer():
    return RandomReplayBuffer(RandomReplayBufferParams(max_len=10000))


def get_n_step_random_replay_buffer():
    return NStepsRandomReplayBuffer(NStepReplayBufferParams(max_len=10000, n_step=5))


def get_prioritized_replay_buffer():
    return PrioritizedReplayBuffer(PrioritizedReplayBufferParams(
        max_len=10000, alpha=0.6, init_beta=0.4, final_beta=1.0, increase_beta=1e-4
    ))


def get_n_step_prioritized_replay_buffer():
    return NStepsPrioritizedReplayBuffer(NStepPrioritizedReplayBufferParams(
        max_len=10000, n_step=5, alpha=0.6, init_beta=0.4, final_beta=1.0, increase_beta=1e-4
    ))


def get_dqn_agent(replay_buffer: AnyReplayBuffer, explorer: AnyExplorer) -> Agent:
    class CustomAgent(DqnAgent):
        def model_factory(self) -> torch.nn.Module:
            return NN()

        def preprocess(self, x: np.ndarray) -> torch.Tensor:
            return torch.unsqueeze(torch.tensor(x, dtype=torch.float32), 0)

    hp = DqnHyperParams(lr=0.003, gamma=0.99, learn_every=1)
    tp = TrainingParams(batch_size=64, episodes=200)
    return CustomAgent(2, explorer, replay_buffer, tp, hp, tensor_board_log=False)


def get_double_dqn_agent(replay_buffer: AnyReplayBuffer, explorer: AnyExplorer) -> Agent:
    class CustomAgent(DoubleDqnAgent):
        def model_factory(self) -> torch.nn.Module:
            return NN()

        def preprocess(self, x: np.ndarray) -> torch.Tensor:
            return torch.unsqueeze(torch.tensor(x, dtype=torch.float32), 0)

    hp = DoubleDqnHyperParams(lr=0.003, gamma=0.99, ensure_every=20, learn_every=1)
    tp = TrainingParams(batch_size=64, episodes=200)
    return CustomAgent(2, explorer, replay_buffer, tp, hp, tensor_board_log=False)


def get_dueling_dqn_agent(replay_buffer: AnyReplayBuffer, explorer: AnyExplorer) -> Agent:
    class CustomAgent(DuelingDqnAgent):
        def model_factory(self) -> torch.nn.Module:
            return NN()

        def preprocess(self, x: np.ndarray) -> torch.Tensor:
            return torch.unsqueeze(torch.tensor(x, dtype=torch.float32), 0)

    hp = DuelingDqnHyperParams(lr=0.003, gamma=0.99, learn_every=1)
    tp = TrainingParams(batch_size=64, episodes=200)
    return CustomAgent(2, explorer, replay_buffer, tp, hp, tensor_board_log=False)


def get_double_dueling_dqn_agent(replay_buffer: AnyReplayBuffer, explorer: AnyExplorer) -> Agent:
    class CustomAgent(DoubleDuelingDqnAgent):
        def model_factory(self) -> torch.nn.Module:
            return NN()

        def preprocess(self, x: np.ndarray) -> torch.Tensor:
            return torch.unsqueeze(torch.tensor(x, dtype=torch.float32), 0)

    hp = DoubleDuelingDqnHyperParams(lr=0.003, gamma=0.99, ensure_every=20, learn_every=1)
    tp = TrainingParams(batch_size=64, episodes=200)
    return CustomAgent(2, explorer, replay_buffer, tp, hp, tensor_board_log=False)


def get_a2c_agent(replay_buffer: AnyReplayBuffer, explorer: AnyExplorer) -> Agent:
    class CustomAgent(A2cAgent):
        def model_factory(self) -> torch.nn.Module:
            return NN()

        def preprocess(self, x: np.ndarray) -> torch.Tensor:
            return torch.unsqueeze(torch.tensor(x, dtype=torch.float32), 0)

    hp = A2CHyperParams(lr=0.003, gamma=0.99, learn_every=1)
    tp = TrainingParams(batch_size=64, episodes=200)
    return CustomAgent(2, explorer, replay_buffer, tp, hp, tensor_board_log=False)


def get_mca2c_agent(replay_buffer: AnyReplayBuffer, explorer: AnyExplorer) -> Agent:
    class CustomAgent(MonteCarloA2cCriticAgent):
        def model_factory(self) -> torch.nn.Module:
            return NN()

        def preprocess(self, x: np.ndarray) -> torch.Tensor:
            return torch.unsqueeze(torch.tensor(x, dtype=torch.float32), 0)

    hp = A2CHyperParams(lr=0.003, gamma=0.99, learn_every=1)
    tp = TrainingParams(batch_size=64, episodes=200)
    return CustomAgent(2, explorer, replay_buffer, tp, hp, tensor_board_log=False)


def get_ppo_agent(replay_buffer: AnyReplayBuffer, explorer: AnyExplorer) -> Agent:
    class CustomAgent(PpoAgent):
        def model_factory(self) -> torch.nn.Module:
            return NN()

        def preprocess(self, x: np.ndarray) -> torch.Tensor:
            return torch.unsqueeze(torch.tensor(x, dtype=torch.float32), 0)

    hp = PpoHyperParams(lr=0.003, gamma=0.99, learn_every=1, clip_factor=0.2, entropy_factor=0.01)
    tp = TrainingParams(batch_size=64, episodes=200)
    return CustomAgent(2, explorer, replay_buffer, tp, hp, tensor_board_log=False)


class NN(torch.nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.linear_1 = torch.nn.Linear(4, 10)
        self.linear_2 = torch.nn.Linear(10, 100)

    def forward(self, x):
        return F.relu(self.linear_2(F.relu(self.linear_1(x))))


params = []
for r_b_f in [get_random_replay_buffer,
              get_n_step_random_replay_buffer,
              get_prioritized_replay_buffer,
              get_n_step_prioritized_replay_buffer]:
    for ex_f in [get_random_explorer,
                 get_noisy_explorer]:
        for ag_f in [get_dqn_agent,
                     get_double_dqn_agent,
                     get_dueling_dqn_agent,
                     get_double_dueling_dqn_agent,
                     get_a2c_agent,
                     get_mca2c_agent,
                     get_ppo_agent]:
            r_b = r_b_f()
            ex = ex_f()
            ag = ag_f(r_b, ex)
            params.append(pytest.param(ag, id=f"{ag.get_self_class_name()}({type(r_b).__name__}, {type(ex).__name__})"))


@pytest.mark.parametrize("agent", params)
def test_agents(agent: Agent):
    env = get_test_env()
    env.visualize = False

    record: T.List[TrainingProgress] = []

    def on_progress(progress: TrainingProgress):
        record.append(progress)
        print(progress.total_reward)
        return progress.total_reward > EXPECTED_REWARD

    agent.add_progress_callback(on_progress)
    agent.train(env)
    assert any([r.total_reward > EXPECTED_REWARD for r in record])
