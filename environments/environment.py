from abc import ABC, abstractmethod
import numpy as np
import typing as T


class Environment(ABC):
    @abstractmethod
    def get_observation_space(self) -> T.Tuple[int]:
        raise NotImplementedError()

    @abstractmethod
    def get_action_space(self) -> T.Tuple[int]:
        raise NotImplementedError()

    @abstractmethod
    def reset(self) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def step(self, action: int) -> T.Tuple[np.ndarray, float, bool]:
        raise NotImplementedError()

    @abstractmethod
    def render(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError()
