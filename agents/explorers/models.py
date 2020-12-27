import typing as T
from dataclasses import dataclass


@dataclass
class RandomExplorerParams:
    init_ep: float
    final_ep: float
    decay_ep: float


@dataclass
class NoisyExplorerParams:
    layers: T.List[int]
    reset_noise_every: int
    std_init: float
