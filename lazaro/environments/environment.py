from abc import ABC, abstractmethod
import typing as T


T_S = T.TypeVar("T_S")


class Environment(T.Generic[T_S], ABC):
    @abstractmethod
    def reset(self) -> T_S:
        raise NotImplementedError()

    @abstractmethod
    def step(self, action: int) -> T.Tuple[T_S, float, bool]:
        raise NotImplementedError()

    @abstractmethod
    def render(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError()
