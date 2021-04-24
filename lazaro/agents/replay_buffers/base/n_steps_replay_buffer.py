import typing as T
from abc import ABC
from collections import deque

from .replay_buffer import ReplayBuffer
from .params import ReplayBufferEntry, NStepReplayBufferParams


class NStepsReplayBuffer(ReplayBuffer, ABC):
    def __init__(self, rp: NStepReplayBufferParams = NStepReplayBufferParams(), *args, **kwargs):
        self.rp: NStepReplayBufferParams = rp
        self.n_step_buffer: T.Deque[ReplayBufferEntry] = deque(maxlen=self.rp.n_step)
        super().__init__(rp, *args, **kwargs)

    def _get_n_step_info(self) -> ReplayBufferEntry:
        first_entry = self.n_step_buffer[0]
        last_entry = self.n_step_buffer[-1]
        ac_r, ac_s_, ac_final = last_entry.r, last_entry.s_, last_entry.final

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, s_, final = transition.r, transition.s_, transition.final
            if self.accumulate_rewards:
                ac_r = r + self.hyper_params.gamma * ac_r

            if final:
                ac_s_, ac_final, ac_r = (s_, final, r)

        return ReplayBufferEntry(first_entry.s, ac_s_, first_entry.a, ac_r, ac_final)

    def rp_add(self, entry: T.Union[ReplayBufferEntry]) -> bool:
        self.n_step_buffer.append(entry)
        if len(self.n_step_buffer) < self.rp.n_step:
            return False

        n_step_entry = self._get_n_step_info()
        super(NStepsReplayBuffer, self).rp_add(n_step_entry)
        return True

    def rp_clear(self):
        super(NStepsReplayBuffer, self).rp_clear()
        self.n_step_buffer.clear()

    def rp_link(self):
        prev_gamma = self.hyper_params.gamma
        self.hyper_params.gamma = self.hyper_params.gamma ** self.rp.n_step
        self.log.info(f"refactoring gamma: {prev_gamma} -> {self.hyper_params.gamma}")

