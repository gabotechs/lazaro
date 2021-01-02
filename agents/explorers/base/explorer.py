import typing as T
from abc import ABC, abstractmethod
import numpy as np


class Explorer(ABC):
    @abstractmethod
    def choose(self, actions: T.List[float], f: T.Callable[[T.List[float]], int]) -> int:
        raise NotImplementedError()
