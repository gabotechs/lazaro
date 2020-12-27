from abc import ABC, abstractmethod
import typing as T
import torch
import numpy as np

from environments import Environment
from agents.explorers import AnyExplorer, RandomExplorer, NoisyExplorer
from agents.replay_buffers import AnyReplayBuffer, ReplayBufferEntry, NStepsPrioritizedReplayBuffer
from .models import HyperParams, TrainingProgress, TrainingParams, LearningStep, TrainingStep


class Agent(ABC):
    def __init__(self,
                 hp: HyperParams,
                 tp: TrainingParams,
                 explorer: T.Union[AnyExplorer, None],
                 replay_buffer: AnyReplayBuffer,
                 use_gpu: bool = True):
        self.hp: HyperParams = hp
        self.tp: TrainingParams = tp
        self.gamma: float = hp.gamma
        self.explorer: T.Union[AnyExplorer, None] = explorer
        self.replay_buffer: AnyReplayBuffer = replay_buffer
        self.device = "cuda" if use_gpu else "cpu"
        self.use_gpu = use_gpu
        self.step_callbacks: T.List[T.Union[T.Callable[[TrainingStep], None]]] = []
        self.progress_callbacks: T.List[T.Union[T.Callable[[TrainingProgress], None]]] = []
        self.learning_callbacks: T.List[T.Union[T.Callable[[LearningStep], None]]] = []
        self.link_replay_buffer()
        self.link_explorer()

    def link_replay_buffer(self):
        if isinstance(self.replay_buffer, NStepsPrioritizedReplayBuffer):
            def new_callback(learning_step: LearningStep):
                self.replay_buffer.update_priorities(
                    [e.index for e in learning_step.batch],
                    [abs(x-y)+1e-7 for x, y in zip(learning_step.x, learning_step.y)]
                )

            self.add_learn_callback(new_callback)

            def new_callback(training_step: TrainingStep):
                self.replay_buffer.increase_beta()

            self.add_step_callback(new_callback)

            self.gamma = self.hp.gamma ** self.replay_buffer.rp.n_step

    def link_explorer(self):
        if isinstance(self.explorer, RandomExplorer):
            def new_callback(training_step: TrainingStep):
                self.explorer.decay()

            self.add_step_callback(new_callback)

        if isinstance(self.explorer, NoisyExplorer):
            model_factory = self.model_factory

            models_pointers: T.List[torch.nn.Module] = []

            def new_model_factory() -> torch.nn.Module:
                model = self.explorer.wrap_model(model_factory)
                models_pointers.append(model)
                return model

            self.model_factory = new_model_factory

            def new_callback(training_step: TrainingStep):
                if training_step.i % self.explorer.ep.reset_noise_every == 0:
                    for model in models_pointers:
                        model.reset_noise()

            self.add_step_callback(new_callback)

    def add_step_callback(self, cbk: T.Callable[[TrainingStep], None]):
        self.step_callbacks.append(cbk)

    def add_progress_callback(self, cbk: T.Callable[[TrainingProgress], None]):
        self.progress_callbacks.append(cbk)

    def add_learn_callback(self, cbk: T.Callable[[LearningStep], None]):
        self.learning_callbacks.append(cbk)

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
