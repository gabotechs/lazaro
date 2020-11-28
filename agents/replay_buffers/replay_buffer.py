from abc import ABC, abstractmethod
import typing as T
from .models import ReplayBufferEntry, MemoryReplayBufferEntry
from _collections import deque


class ReplayBuffer(ABC):
    def __init__(self, max_len: int):
        self.records: T.Deque[T.Union[ReplayBufferEntry, MemoryReplayBufferEntry]] = deque(maxlen=max_len)

    def __len__(self):
        return len(self.records)

    def clear(self):
        self.records.clear()

    def add(self, entry: T.Union[ReplayBufferEntry, MemoryReplayBufferEntry]):
        self.records.append(entry)

    @abstractmethod
    def sample(self, limit: int = None):
        raise NotImplementedError()
