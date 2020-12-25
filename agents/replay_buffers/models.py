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
    max_len: int


@dataclass
class NStepReplayBufferParams(ReplayBufferParams):
    n_step: int
    gamma: float


@dataclass
class PrioritizedReplayBufferParams(ReplayBufferParams):
    alpha: float
    init_beta: float
    final_beta: float
    increase_beta: float


@dataclass
class NStepPrioritizedReplayBufferParams(NStepReplayBufferParams, PrioritizedReplayBufferParams):
    pass

