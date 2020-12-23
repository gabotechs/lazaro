import typing as T
from collections import deque

from .random_replay_buffer import RandomReplayBuffer
from .models import ReplayBufferEntry


class NStepsRandomReplayBuffer(RandomReplayBuffer):
    def __init__(self, max_len: int, n_step: int, gamma: float):
        super().__init__(max_len)
        self.n_step_buffer: T.Deque[ReplayBufferEntry] = deque(maxlen=n_step)
        self.n_step: int = n_step
        self.gamma: float = gamma

    def _get_n_step_info(self) -> ReplayBufferEntry:
        first_entry = self.n_step_buffer[0]
        last_entry = self.n_step_buffer[-1]
        ac_r, ac_s_, ac_final = last_entry.r, last_entry.s_, last_entry.final

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, s_, final = transition.r, transition.s_, transition.final
            ac_r = r + self.gamma * ac_r * (1 - int(final))
            if final:
                ac_s_, ac_final = (s_, final)

        return ReplayBufferEntry(first_entry.s, ac_s_, first_entry.a, ac_r, ac_final)

    def add(self, entry: T.Union[ReplayBufferEntry]) -> None:
        self.n_step_buffer.append(entry)
        if len(self.n_step_buffer) < self.n_step:
            return

        n_step_entry = self._get_n_step_info()
        super(NStepsRandomReplayBuffer, self).add(n_step_entry)
