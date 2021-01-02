from .base.models import NStepPrioritizedReplayBufferParams
from .base.n_steps_replay_buffer import NStepsReplayBuffer
from .priorized_replay_buffer import PrioritizedReplayBuffer


class NStepsPrioritizedReplayBuffer(PrioritizedReplayBuffer, NStepsReplayBuffer):
    rp: NStepPrioritizedReplayBufferParams
