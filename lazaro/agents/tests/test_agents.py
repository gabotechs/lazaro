import typing as T

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ... import agents, environments

EXPECTED_REWARD = 25


class NN(torch.nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.linear_1 = torch.nn.Linear(4, 10)
        self.linear_2 = torch.nn.Linear(10, 100)

    def forward(self, x):
        return F.relu(self.linear_2(F.relu(self.linear_1(x))))


params = []
for RB in [agents.replay_buffers.RandomReplayBuffer,
           agents.replay_buffers.NStepsPrioritizedReplayBuffer,
           agents.replay_buffers.PrioritizedReplayBuffer,
           agents.replay_buffers.NStepsPrioritizedReplayBuffer]:
    for EX in [agents.explorers.RandomExplorer,
               agents.explorers.NoisyExplorer]:
        for AG in [agents.DqnAgent,
                   agents.DoubleDqnAgent,
                   agents.DuelingDqnAgent,
                   agents.DoubleDuelingDqnAgent,
                   agents.A2cAgent,
                   agents.MonteCarloA2c,
                   agents.PpoAgent]:

            class Agent(RB, EX, AG):
                def model_factory(self) -> torch.nn.Module:
                    return NN()

            ag: T.Type[agents.AnyAgent] = Agent
            params.append(pytest.param(ag, id=f"{AG.__name__}({EX.__name__}, {RB.__name__})"))


@pytest.mark.parametrize("agent_class", params)
def test_agents(agent_class: T.Type[agents.AnyAgent]):
    env = environments.CartPole()
    env.visualize = False
    agent = agent_class(action_space=len(env.get_action_space()))

    record: T.List[agents.TrainingProgress] = []

    def on_progress(progress: agents.TrainingProgress):
        record.append(progress)
        print(progress.total_reward)
        return progress.total_reward > EXPECTED_REWARD

    agent.add_progress_callback("testing", on_progress)
    agent.train(env, agents.TrainingParams(batch_size=64, episodes=200))
    assert any([r.total_reward > EXPECTED_REWARD for r in record])
