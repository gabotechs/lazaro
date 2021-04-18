import torch
import torch.nn.functional as F
import numpy as np
import pytest
import typing as T
import agents
from agents import replay_buffers, explorers, TrainingProgress, TrainingParams
from environments import CartPole

EXPECTED_REWARD = 25


class NN(torch.nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.linear_1 = torch.nn.Linear(4, 10)
        self.linear_2 = torch.nn.Linear(10, 100)

    def forward(self, x):
        return F.relu(self.linear_2(F.relu(self.linear_1(x))))


params = []
for RB in [replay_buffers.RandomReplayBuffer,
           replay_buffers.NStepsPrioritizedReplayBuffer,
           replay_buffers.PrioritizedReplayBuffer,
           replay_buffers.NStepsPrioritizedReplayBuffer]:
    for EX in [explorers.RandomExplorer,
               explorers.NoisyExplorer]:
        for AG in [agents.DqnAgent,
                   agents.DoubleDqnAgent,
                   agents.DuelingDqnAgent,
                   agents.DoubleDuelingDqnAgent,
                   agents.A2cAgent,
                   agents.MonteCarloA2c,
                   agents.PpoAgent]:

            class TestAgent(RB, EX, AG):
                def model_factory(self) -> torch.nn.Module:
                    return NN()

                def preprocess(self, x: np.ndarray) -> torch.Tensor:
                    return torch.unsqueeze(torch.tensor(x, dtype=torch.float32), 0)

            ag = TestAgent
            params.append(pytest.param(ag, id=f"{AG.__name__}({EX.__name__}, {RB.__name__})"))


@pytest.mark.parametrize("agent_class", params)
def test_agents(agent_class: T.Type[agents.AnyAgent]):
    env = CartPole()
    env.visualize = False
    agent = agent_class(action_space=len(env.get_action_space()))

    record: T.List[TrainingProgress] = []

    def on_progress(progress: TrainingProgress):
        record.append(progress)
        print(progress.total_reward)
        return progress.total_reward > EXPECTED_REWARD

    agent.add_progress_callback("testing", on_progress)
    agent.train(env, TrainingParams(batch_size=64, episodes=200))
    assert any([r.total_reward > EXPECTED_REWARD for r in record])
