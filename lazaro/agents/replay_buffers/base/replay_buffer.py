from abc import ABC
import typing as T
from .params import ReplayBufferEntry, ReplayBufferParams
from ...base.base_agent import BaseAgent


class ReplayBuffer(BaseAgent, ABC):
    def __init__(self, rp: ReplayBufferParams, *args, **kwargs):
        super(ReplayBuffer, self).__init__(*args, **kwargs)
        self.rp: ReplayBufferParams = rp
        self.records: T.List[T.Union[None, ReplayBufferEntry]] = [None for _ in range(self.rp.max_len)]
        self.ptr = 0
        self.filled = False

    def rp_get_length(self):
        return self.ptr if not self.filled else self.rp.max_len

    def rp_clear(self) -> None:
        self.log.info("buffer is being emptied")
        self.records = [None for _ in range(self.rp.max_len)]
        self.ptr = 0
        self.filled = False

    def rp_add(self, entry: T.Union[ReplayBufferEntry]) -> bool:
        self.log.debug("adding new entry to bufer")
        self.records[self.ptr] = entry
        entry.index = self.ptr
        self.ptr = (self.ptr + 1) % self.rp.max_len
        if not self.filled and self.ptr == 0:
            self.log.info("buffer has been filled")
            self.filled = True
        return True

    def rp_get_stats(self) -> T.Dict[str, float]:
        return {"Replay Buffer Size": self.rp_get_length()}
