from abc import ABC, abstractmethod
import numpy as np


class Explorer(ABC):
    @abstractmethod
    def choose(self, actions: np.ndarray) -> int:
        raise NotImplementedError()
