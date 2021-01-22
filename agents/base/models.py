import typing as T
from dataclasses import dataclass

from agents.replay_buffers import ReplayBufferEntry


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
    ensure_every: int = 1


@dataclass
class DoubleDuelingDqnHyperParams(DoubleDqnHyperParams):
    pass


@dataclass
class A2CHyperParams(HyperParams):
    lr: float = 0.01


@dataclass
class PpoHyperParams(A2CHyperParams):
    clip_factor: float = 0.02
    entropy_factor: float = 0.01
    ensure_every: int = 1


@dataclass
class TrainingProgress:
    step: int
    episode: int
    steps_survived: int
    total_reward: float


@dataclass
class TrainingParams:
    batch_size: int
    episodes: int


@dataclass
class TrainingStep:
    step: int
    episode: int


@dataclass
class LearningStep:
    batch: T.List[ReplayBufferEntry]
    x: T.List[float]
    y: T.List[float]
