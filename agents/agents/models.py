import typing as T
from dataclasses import dataclass

from ..replay_buffers import ReplayBufferEntry


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
class LearningStep:
    batch: T.List[ReplayBufferEntry]
    x: T.List[float]
    y: T.List[float]
