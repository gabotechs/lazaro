import typing as T


class HyperParams:
    pass


class DqnHyperParams(HyperParams):
    def __init__(self, lr: float, gamma: float, ensure_every: int):
        self.lr: float = lr
        self.ensure_every: int = ensure_every
        self.gamma: float = gamma


class ACHyperParams(HyperParams):
    def __init__(self, a_lr: float, c_lr: float, gamma: float):
        self.c_lr: float = c_lr
        self.a_lr: float = a_lr
        self.gamma: float = gamma


class TrainingProgress:
    def __init__(self, tries: int, steps_survived: int, total_reward: float):
        self.tries: int = tries
        self.steps_survived: int = steps_survived
        self.total_reward: float = total_reward


class TrainingParams:
    def __init__(self,
                 learn_every: int,
                 batch_size: int,
                 episodes: int):
        self.learn_every: int = learn_every
        self.batch_size: int = batch_size
        self.episodes: int = episodes


