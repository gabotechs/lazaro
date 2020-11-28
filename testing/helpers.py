import typing as T

from agents import Agent, TrainingProgress
from agents.explorers import Explorer, RandomExplorer
from environments import Environment
from plotter import Plotter


def train(agent: Agent, env: Environment):
    plotter: Plotter = Plotter()
    explorer: T.Union[Explorer, RandomExplorer, None] = agent.explorer
    if isinstance(explorer, RandomExplorer):
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
            ("| epsilon:", round(explorer.ep, 2)) if isinstance(explorer, RandomExplorer) else ""
        )

    agent.set_progress_callback(progress_callback)
    agent.train(env)
    input("press any key for ending: ")
