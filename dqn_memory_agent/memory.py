import typing as T
import random
from .models import MemoryEntry
from _collections import deque


class Memory:
    def __init__(self, max_len: int):
        self.records: T.Deque[MemoryEntry] = deque(maxlen=max_len)

    def __len__(self):
        return len(self.records)

    def clear(self):
        self.records.clear()

    def add(self, entry: MemoryEntry):
        self.records.append(entry)

    def sample(self, limit: int = None):
        return random.sample(self.records, limit if limit is not None and limit < len(self.records) else len(self.records))
