import typing as T
from dataclasses import dataclass

import torch

T_S = T.TypeVar("T_S")


@dataclass
class ReplayBufferEntry(T.Generic[T_S]):
    s: T_S
    s_: T_S
    a: int
    r: float
    final: bool
    index: T.Union[None, int] = None
    weight: int = 1


@dataclass
class LearningBatch:
    s: T.Union[torch.Tensor, T.Tuple[torch.Tensor, ...]]
    s_: T.Union[torch.Tensor, T.Tuple[torch.Tensor, ...]]
    a: torch.Tensor
    r: torch.Tensor
    final: torch.Tensor
    weight: torch.Tensor


@dataclass
class AgentParams:
    gamma: float = 0.99
    learn_every: int = 1


@dataclass
class DqnHyperParams(AgentParams):
    lr: float = 0.0025


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
class A2CHyperParams(AgentParams):
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
    episodes: int = 10000


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
