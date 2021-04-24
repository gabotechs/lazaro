from .base.params import NStepPrioritizedReplayBufferParams
from .base.n_steps_replay_buffer import NStepsReplayBuffer
from .priorized_replay_buffer import PrioritizedReplayBuffer
from abc import ABC


class NStepsPrioritizedReplayBuffer(PrioritizedReplayBuffer, NStepsReplayBuffer, ABC):
    def __init__(self, rp: NStepPrioritizedReplayBufferParams = NStepPrioritizedReplayBufferParams(), *args, **kwargs):
        if not isinstance(rp, NStepPrioritizedReplayBufferParams):
            raise ValueError("argument rp must be an instance of NStepPrioritizedReplayBufferParams")
        super(NStepsPrioritizedReplayBuffer, self).__init__(rp, *args, **kwargs)

    def rp_link(self):
        PrioritizedReplayBuffer.rp_link(self)
        NStepsReplayBuffer.rp_link(self)
