from .base.models import NStepPrioritizedReplayBufferParams
from .base.n_steps_replay_buffer import NStepsReplayBuffer
from .priorized_replay_buffer import PrioritizedReplayBuffer


class NStepsPrioritizedReplayBuffer(PrioritizedReplayBuffer, NStepsReplayBuffer):
    def __init__(self, rp: NStepPrioritizedReplayBufferParams = NStepPrioritizedReplayBufferParams()):
        super(NStepsPrioritizedReplayBuffer, self).__init__(rp)
