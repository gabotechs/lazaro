import typing as T
from abc import ABC, abstractmethod
import logging


class Explorer(ABC):
    def __init__(self):
        self.log = logging.getLogger(type(self).__name__)

    @abstractmethod
    def choose(self, actions: T.List[float], f: T.Callable[[T.List[float]], int]) -> int:
        raise NotImplementedError()

    @abstractmethod
    def link_to_agent(self, agent):
        raise NotImplementedError()

    @abstractmethod
    def get_stats(self) -> T.Dict[str, float]:
        raise NotImplementedError()
