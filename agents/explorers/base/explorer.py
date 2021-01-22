import typing as T
from abc import ABC, abstractmethod
from logger import get_logger


class Explorer(ABC):
    def __init__(self):
        self.log = get_logger(type(self).__name__)

    @abstractmethod
    def choose(self, actions: T.List[float], f: T.Callable[[T.List[float]], int]) -> int:
        raise NotImplementedError()
