from abc import ABC

from .base.n_steps_replay_buffer import NStepsReplayBuffer
from .base.params import NStepRandomReplayBufferParams
from .random_replay_buffer import RandomReplayBuffer


class NStepsRandomReplayBuffer(NStepsReplayBuffer, RandomReplayBuffer, ABC):
    def __init__(self,
                 replay_buffer_params: NStepRandomReplayBufferParams = NStepRandomReplayBufferParams(),
                 *args, **kwargs):
        if not isinstance(replay_buffer_params, NStepRandomReplayBufferParams):
            raise ValueError("argument rp must be an instance of NStepPrioritizedReplayBufferParams")
        super(NStepsRandomReplayBuffer, self).__init__(replay_buffer_params, *args, **kwargs)

    def rp_link(self):
        NStepsReplayBuffer.rp_link(self)
        RandomReplayBuffer.rp_link(self)
