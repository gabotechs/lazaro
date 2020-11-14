import numpy as np


class MemoryEntry:
    def __init__(self, s: np.ndarray, s_: np.ndarray, a: int, r: float, final: bool):
        self.s: np.ndarray = s
        self.s_: np.ndarray = s_
        self.a: int = a
        self.r: float = r
        self.final: bool = final


class HyperParams:
    def __init__(self, lr: float, gamma: float, memory_len: int):
        self.lr: float = lr
        self.gamma: float = gamma
        self.memory_len: int = memory_len
