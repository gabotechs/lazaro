import typing as T
from abc import ABC
from collections import deque

from .replay_buffer import ReplayBuffer
from .models import ReplayBufferEntry, NStepReplayBufferParams


class NStepsReplayBuffer(ReplayBuffer, ABC):
    def __init__(self, rp: NStepReplayBufferParams = NStepReplayBufferParams()):
        super().__init__(rp)
        self.rp: NStepReplayBufferParams = rp
        self.n_step_buffer: T.Deque[ReplayBufferEntry] = deque(maxlen=self.rp.n_step)
        self.accumulate_rewards: bool = True
        self.gamma = 0.99  # should be overridden

    def set_gamma(self, gamma: float) -> None:
        self.gamma = gamma

    def _get_n_step_info(self) -> ReplayBufferEntry:
        first_entry = self.n_step_buffer[0]
        last_entry = self.n_step_buffer[-1]
        ac_r, ac_s_, ac_final = last_entry.r, last_entry.s_, last_entry.final

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, s_, final = transition.r, transition.s_, transition.final
            if self.accumulate_rewards:
                ac_r = r + self.gamma * ac_r

            if final:
                ac_s_, ac_final, ac_r = (s_, final, r)

        return ReplayBufferEntry(first_entry.s, ac_s_, first_entry.a, ac_r, ac_final)

    def add(self, entry: T.Union[ReplayBufferEntry]) -> bool:
        self.n_step_buffer.append(entry)
        if len(self.n_step_buffer) < self.rp.n_step:
            return False

        n_step_entry = self._get_n_step_info()
        super(NStepsReplayBuffer, self).add(n_step_entry)
        return True

    def clear(self):
        super(NStepsReplayBuffer, self).clear()
        self.n_step_buffer.clear()
