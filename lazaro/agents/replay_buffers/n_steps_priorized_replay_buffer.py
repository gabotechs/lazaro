from abc import ABC

from .base.n_steps_replay_buffer import NStepsReplayBuffer
from .base.params import NStepPrioritizedReplayBufferParams
from .priorized_replay_buffer import PrioritizedReplayBuffer


class NStepsPrioritizedReplayBuffer(PrioritizedReplayBuffer, NStepsReplayBuffer, ABC):
    def __init__(self,
                 replay_buffer_params: NStepPrioritizedReplayBufferParams = NStepPrioritizedReplayBufferParams(),
                 *args, **kwargs):
        if not isinstance(replay_buffer_params, NStepPrioritizedReplayBufferParams):
            raise ValueError("argument rp must be an instance of NStepPrioritizedReplayBufferParams")
        super(NStepsPrioritizedReplayBuffer, self).__init__(replay_buffer_params, *args, **kwargs)

    def rp_link(self):
        PrioritizedReplayBuffer.rp_link(self)
        NStepsReplayBuffer.rp_link(self)
