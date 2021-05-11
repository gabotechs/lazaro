import typing as T
from dataclasses import dataclass, field


@dataclass
class RandomExplorerParams:
    init_ep: float = 1.0
    final_ep: float = 0.01
    decay_ep: float = 1e-3


@dataclass
class NoisyExplorerParams:
    extra_layers: T.List[int] = field(default_factory=list)
    reset_noise_every: int = 1
    std_init: float = 0.5
