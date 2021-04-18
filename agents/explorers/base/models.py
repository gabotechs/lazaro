import typing as T
from dataclasses import dataclass


@dataclass
class RandomExplorerParams:
    init_ep: float = 1.0
    final_ep: float = 0.01
    decay_ep: float = 1e-3


@dataclass
class NoisyExplorerParams:
    extra_layers: T.List[int] = ()
    reset_noise_every: int = 1
    std_init: float = 0.5
