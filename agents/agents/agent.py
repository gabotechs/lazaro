from abc import ABC, abstractmethod
import typing as T
import torch
import numpy as np

from environments import Environment
from agents.explorers import AnyExplorer, RandomExplorer
from agents.replay_buffers import AnyReplayBuffer, ReplayBufferEntry, NStepsPrioritizedReplayBuffer
from .models import HyperParams, TrainingProgress, TrainingParams, LearningStep


class Agent(ABC):
    def __init__(self,
                 hp: HyperParams,
                 tp: TrainingParams,
                 explorer: T.Union[AnyExplorer, None],
                 replay_buffer: AnyReplayBuffer,
                 use_gpu: bool = True):
        self.hp: HyperParams = hp
        self.tp: TrainingParams = tp
        self.explorer: T.Union[AnyExplorer, None] = explorer
        self.replay_buffer: AnyReplayBuffer = replay_buffer
        self.device = "cuda" if use_gpu else "cpu"
        self.use_gpu = use_gpu
        self.infer_callbacks: T.List[T.Union[T.Callable[[], None]]] = []
        self.progress_callbacks: T.List[T.Union[T.Callable[[TrainingProgress], None]]] = []
        self.learning_callbacks: T.List[T.Union[T.Callable[[LearningStep], None]]] = []
        self.no_more_callbacks: bool = False

    def add_infer_callback(self, cbk: T.Callable[[], None]):
        if self.no_more_callbacks:
            raise ValueError("no more callbacks")
        self.infer_callbacks.append(cbk)

    def add_progress_callback(self, cbk: T.Callable[[TrainingProgress], None]):
        if self.no_more_callbacks:
            raise ValueError("no more callbacks")
        self.progress_callbacks.append(cbk)

    def add_learn_callback(self, cbk: T.Callable[[LearningStep], None]):
        if self.no_more_callbacks:
            raise ValueError("no more callbacks")
        self.learning_callbacks.append(cbk)

    def hook_callbacks(self):
        if isinstance(self.replay_buffer, NStepsPrioritizedReplayBuffer):
            def new_callback(learning_step: LearningStep):
                self.replay_buffer.update_priorities(
                    [e.index for e in learning_step.batch],
                    [abs(x-y)+1e-7 for x, y in zip(learning_step.x, learning_step.y)]
                )

            self.add_learn_callback(new_callback)

            def new_callback():
                self.replay_buffer.increase_beta()

            self.add_infer_callback(new_callback)

        if isinstance(self.explorer, RandomExplorer):
            def new_callback():
                self.explorer.decay()

            self.add_infer_callback(new_callback)

        self.no_more_callbacks = True

    @abstractmethod
    def model_factory(self) -> torch.nn.Module:
        raise NotImplementedError()

    @abstractmethod
    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def postprocess(self, t: torch.Tensor) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def infer(self, *args) -> T.Any:
        raise NotImplementedError()

    @abstractmethod
    def learn(self, batch: T.List[ReplayBufferEntry]) -> None:
        raise NotImplementedError()

    @abstractmethod
    def train(self, env: Environment) -> None:
        raise NotImplementedError()
