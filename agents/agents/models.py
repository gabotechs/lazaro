from dataclasses import dataclass


class HyperParams:
    pass


@dataclass
class MDqnHyperParams(HyperParams):
    a_lr: float
    m_lr: float
    ensure_every: int
    gamma: float


@dataclass
class DqnHyperParams(HyperParams):
    lr: float
    ensure_every: int
    gamma: float


@dataclass
class ACHyperParams(HyperParams):
    c_lr: float
    a_lr: float
    gamma: float


@dataclass
class TrainingProgress:
    tries: int
    steps_survived: int
    total_reward: float


@dataclass
class TrainingParams:
    learn_every: int
    batch_size: int
    episodes: int


@dataclass
class MDqnTrainingParams(TrainingParams):
    memory_batch_size: int
    memory_learn_every: int
    memory_clear_after_learn: bool
