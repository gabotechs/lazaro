from abc import ABC, abstractmethod
import numpy as np
import typing as T


class Environment(ABC):
    last_s: T.Union[None, np.ndarray] = None

    @abstractmethod
    def get_observation_space(self) -> T.Tuple[int]:
        raise NotImplementedError()

    @abstractmethod
    def get_action_space(self) -> T.Tuple[int]:
        raise NotImplementedError()

    @abstractmethod
    def reset(self) -> np.ndarray:
        raise NotImplementedError()

    def step(self, action: int) -> T.Tuple[np.ndarray, float, bool]:
        s, r, f = self.do_step(action)
        self.last_s = s
        return s, r, f

    @abstractmethod
    def do_step(self, action: int) -> T.Tuple[np.ndarray, float, bool]:
        raise NotImplementedError()

    @abstractmethod
    def render(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError()
