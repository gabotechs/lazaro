import typing as T
from dataclasses import dataclass

from agents.replay_buffers import ReplayBufferEntry


@dataclass
class HyperParams:
    gamma: float


@dataclass
class DqnHyperParams(HyperParams):
    lr: float


@dataclass
class DuelingDqnHyperParams(DqnHyperParams):
    pass


@dataclass
class DoubleDqnHyperParams(DqnHyperParams):
    lr: float
    ensure_every: int


@dataclass
class DoubleDuelingDqnHyperParams(DoubleDqnHyperParams):
    pass


@dataclass
class ACHyperParams(HyperParams):
    c_lr: float
    a_lr: float


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
class TrainingStep:
    i: int
    steps_survived: int
    episode: int


@dataclass
class LearningStep:
    batch: T.List[ReplayBufferEntry]
    x: T.List[float]
    y: T.List[float]
