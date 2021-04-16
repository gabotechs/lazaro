import typing as T
from abc import ABC, abstractmethod


class ExplorerInterface(ABC):
    @abstractmethod
    def ex_choose(self, actions: T.List[float], f: T.Callable[[T.List[float]], int]) -> int:
        raise NotImplementedError()

    @abstractmethod
    def ex_link(self):
        raise NotImplementedError()

    @abstractmethod
    def ex_get_stats(self) -> T.Dict[str, float]:
        raise NotImplementedError()
