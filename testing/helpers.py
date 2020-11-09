import typing as T

from plotter import Plotter
from trainer import Trainer, TrainingProgress


def train(trainer: Trainer):
    plotter: Plotter = Plotter()
    trainer.agent.set_infer_callback(lambda: trainer.explorer.decay())

    reward_record: T.List[float] = []

    def progress_callback(progress: TrainingProgress):
        reward_record.append(progress.total_reward)

        plotter.plot(reward_record, aliasing=.8)
        print(
            "lost! achieved "
            "| tries:", progress.tries,
            "| steps survived:", progress.steps_survived,
            "| reward:", progress.total_reward,
            "| epsilon:", round(trainer.explorer.ep, 2)
        )

    trainer.set_progress_callback(progress_callback)
    trainer.train(lambda progress: progress.tries >= 1000)
