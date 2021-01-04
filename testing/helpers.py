import typing as T
import json
import os

from agents import AnyAgent, TrainingProgress, NStepsPrioritizedReplayBuffer
from agents.explorers import RandomExplorer
from environments import Environment
from plotter import Plotter

os.environ["LOG_LEVEL"] = "INFO"


def train(agent: AnyAgent, env: Environment):
    plotter: Plotter = Plotter()
    reward_record: T.List[float] = []

    def progress_callback(progress: TrainingProgress):
        reward_record.append(progress.total_reward)

        plotter.plot(reward_record, aliasing=0)
        if isinstance(agent.explorer, RandomExplorer):
            print("| epsilon:", round(agent.explorer.epsilon, 2))
        if isinstance(agent.replay_buffer, NStepsPrioritizedReplayBuffer):
            print("| beta:", round(agent.replay_buffer.beta, 2))

    agent.add_progress_callback(progress_callback)
    agent.train(env)
    input("press any key for ending: ")
