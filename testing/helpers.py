import typing as T
from explorers import Explorer, RandomExplorer
from plotter import Plotter
from trainers import Trainer, TrainingProgress


def train(trainer: Trainer):
    plotter: Plotter = Plotter()
    explorer: T.Union[RandomExplorer, None] = trainer.explorer
    if isinstance(explorer, RandomExplorer):
        trainer.agent.set_infer_callback(lambda: explorer.decay())

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

    trainer.set_progress_callback(progress_callback)
    trainer.train(lambda progress: progress.tries >= 1000)
    input("press any key for ending: ")
