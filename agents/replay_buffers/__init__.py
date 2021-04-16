import typing as T
from .random_replay_buffer import RandomReplayBuffer
from .n_steps_random_replay_buffer import NStepsRandomReplayBuffer
from .priorized_replay_buffer import PrioritizedReplayBuffer
from .n_steps_priorized_replay_buffer import NStepsPrioritizedReplayBuffer
from .base.models import ReplayBufferParams, RandomReplayBufferParams, ReplayBufferEntry, MemoryReplayBufferEntry, \
    NStepReplayBufferParams,  NStepPrioritizedReplayBufferParams, PrioritizedReplayBufferParams

AnyReplayBuffer = T.Union[RandomReplayBuffer,
                          NStepsRandomReplayBuffer,
                          NStepsPrioritizedReplayBuffer,
                          PrioritizedReplayBuffer]
