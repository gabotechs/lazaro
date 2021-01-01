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

        plotter.plot(reward_record, aliasing=.1)
        print(
            "lost! achieved "
            "| episode:", progress.tries,
            "| steps survived:", progress.steps_survived,
            "| reward:", progress.total_reward,
            ("| epsilon:", round(agent.explorer.epsilon, 2)) if isinstance(agent.explorer, RandomExplorer) else "",
            ("| beta:", round(agent.replay_buffer.beta, 2) if isinstance(agent.replay_buffer, NStepsPrioritizedReplayBuffer) else "")
        )

    agent.add_progress_callback(progress_callback)
    agent.train(env)
    input("press any key for ending: ")
