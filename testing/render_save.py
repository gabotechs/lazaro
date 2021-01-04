import json
import os

from agents import TrainingProgress
from plotter import Plotter


PATH = "/home/gabriel/trader/OpenAITraining/data/DoubleDuelingDqnAgent/CustomCartPole/2021-01-04/19:58:17"


if __name__ == '__main__':
    plotter = Plotter()
    checkpoints_folder = os.path.join(PATH, "checkpoints")
    assert os.path.isdir(checkpoints_folder)
    reward_record = []
    for checkpoint in sorted(os.listdir(checkpoints_folder)):
        loaded_checkpoint: TrainingProgress = TrainingProgress(**json.load(open(os.path.join(checkpoints_folder, checkpoint))))
        reward_record.append(loaded_checkpoint.total_reward)
        plotter.plot(reward_record, 0.9)

    input("press a key to end: ")
