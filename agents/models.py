import typing as T


class HyperParams:
    def __init__(self, lr: float, gamma: float):
        self.lr: float = lr
        self.gamma: float = gamma


class ACHyperParams(HyperParams):
    def __init__(self, a_lr: float, c_lr: float, gamma: float):
        super(ACHyperParams, self).__init__(c_lr, gamma)
        self.a_lr: float = a_lr


class TrainingProgress:
    def __init__(self, tries: int, steps_survived: int, total_reward: float):
        self.tries: int = tries
        self.steps_survived: int = steps_survived
        self.total_reward: float = total_reward


class TrainingParams:
    def __init__(self,
                 learn_every: int,
                 ensure_every: int,
                 batch_size: int,
                 finish_condition: T.Callable[[TrainingProgress], bool]):
        self.learn_every: int = learn_every
        self.ensure_every: int = ensure_every
        self.batch_size: int = batch_size
        self.finish_condition: T.Callable[[TrainingProgress], bool] = finish_condition


