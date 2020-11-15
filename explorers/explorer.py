import typing as T
from abc import ABC, abstractmethod
import numpy as np


class Explorer(ABC):
    @abstractmethod
    def choose(self, actions: np.ndarray, f: T.Callable[[np.ndarray], int]) -> int:
        raise NotImplementedError()
