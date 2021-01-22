from abc import ABC, abstractmethod
import typing as T
from .models import ReplayBufferEntry, ReplayBufferParams


class ReplayBuffer(ABC):
    def __init__(self, rp: ReplayBufferParams):
        self.rp: ReplayBufferParams = rp
        self.records: T.List[T.Union[None, ReplayBufferEntry]] = [None for _ in range(self.rp.max_len)]
        self.ptr = 0
        self.filled = False

    def __len__(self):
        return self.ptr if not self.filled else self.rp.max_len

    def clear(self):
        self.records = [None for _ in range(self.rp.max_len)]
        self.ptr = 0
        self.filled = False

    def add(self, entry: T.Union[ReplayBufferEntry]) -> bool:
        self.records[self.ptr] = entry
        entry.index = self.ptr
        self.ptr = (self.ptr + 1) % self.rp.max_len
        if not self.filled and self.ptr == 0:
            print("FFFIIIILLLLEEEEDDDD")
            self.filled = True
        return True

    @abstractmethod
    def sample(self, limit: int):
        raise NotImplementedError()
