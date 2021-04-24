from dataclasses import dataclass
import typing as T
import numpy as np
import torch


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
class MemoryReplayBufferEntry(ReplayBufferEntry):
    m: torch.Tensor = None


@dataclass
class ReplayBufferParams:
    max_len: int = 10000


@dataclass
class RandomReplayBufferParams(ReplayBufferParams):
    pass


@dataclass
class NStepReplayBufferParams(ReplayBufferParams):
    n_step: int = 3


@dataclass
class PrioritizedReplayBufferParams(ReplayBufferParams):
    alpha: float = 0.6
    init_beta: float = 0.4
    final_beta: float = 1.0
    increase_beta: float = 1e-4


@dataclass
class NStepPrioritizedReplayBufferParams(NStepReplayBufferParams, PrioritizedReplayBufferParams):
    pass


@dataclass
class NStepRandomReplayBufferParams(NStepReplayBufferParams, RandomReplayBufferParams):
    pass
