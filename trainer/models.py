class TrainingParams:
    def __init__(self, learn_every: int, ensure_every: int, batch_size: int):
        self.learn_every: int = learn_every
        self.ensure_every: int = ensure_every
        self.batch_size: int = batch_size


class TrainingProgress:
    def __init__(self, tries: int, steps_survived: int, total_reward: float):
        self.tries: int = tries
        self.steps_survived: int = steps_survived
        self.total_reward: float = total_reward
