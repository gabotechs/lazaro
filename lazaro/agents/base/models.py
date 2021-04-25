import typing as T
from dataclasses import dataclass

import numpy as np


@dataclass
class ReplayBufferEntry:
    s: np.ndarray
    s_: np.ndarray
    a: int
    r: float
    final: bool
    index: T.Union[None, int] = None
    weight: int = 1


@dataclass
class HyperParams:
    gamma: float = 0.99
    learn_every: int = 1


@dataclass
class DqnHyperParams(HyperParams):
    lr: float = 0.01


@dataclass
class DuelingDqnHyperParams(DqnHyperParams):
    pass


@dataclass
class DoubleDqnHyperParams(DqnHyperParams):
    ensure_every: int = 10


@dataclass
class DoubleDuelingDqnHyperParams(DoubleDqnHyperParams):
    pass


@dataclass
class A2CHyperParams(HyperParams):
    lr: float = 0.001


@dataclass
class PpoHyperParams(A2CHyperParams):
    clip_factor: float = 0.02
    entropy_factor: float = 0.01
    ensure_every: int = 10


@dataclass
class TrainingProgress:
    step: int
    episode: int
    steps_survived: int
    total_reward: float


TProgressCallback = T.Callable[[TrainingProgress], bool]


@dataclass
class TrainingParams:
    batch_size: int = 46
    episodes: int = 500


@dataclass
class TrainingStep:
    step: int
    episode: int


TStepCallback = T.Callable[[TrainingStep], None]


@dataclass
class LearningStep:
    batch: T.List[T.Any]
    x: T.List[float]
    y: T.List[float]


TLearnCallback = T.Callable[[LearningStep], None]
