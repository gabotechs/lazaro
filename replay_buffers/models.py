import numpy as np
import torch


class ReplayBufferEntry:
    def __init__(self, s: np.ndarray, s_: np.ndarray, a: int, r: float, final: bool):
        self.s: np.ndarray = s
        self.s_: np.ndarray = s_
        self.a: int = a
        self.r: float = r
        self.final: bool = final


class MemoryReplayBufferEntry(ReplayBufferEntry):
    def __init__(self, s: np.ndarray, m: torch.Tensor, s_: np.ndarray, a: int, r: float, final: bool):
        super(MemoryReplayBufferEntry, self).__init__(s, s_, a, r, final)
        self.m: torch.Tensor = m
