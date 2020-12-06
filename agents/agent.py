from abc import ABC, abstractmethod
import typing as T
import torch
import numpy as np

from environments import Environment
from agents.explorers import Explorer
from agents.replay_buffers import ReplayBuffer, ReplayBufferEntry
from .models import HyperParams, TrainingProgress, TrainingParams


class Agent(ABC):
    def __init__(self,
                 hp: HyperParams,
                 tp: TrainingParams,
                 explorer: T.Union[Explorer, None],
                 replay_buffer: ReplayBuffer,
                 use_gpu: bool = True):
        self.hp: HyperParams = hp
        self.tp: TrainingParams = tp
        self.explorer: T.Union[Explorer, None] = explorer
        self.replay_buffer: ReplayBuffer = replay_buffer
        self.device = "cuda" if use_gpu else "cpu"
        self.use_gpu = use_gpu
        self.infer_callback: T.Union[T.Callable[[], None], None] = None
        self.progress_callback: T.Union[T.Callable[[TrainingProgress], None], None] = None

    def set_infer_callback(self, cbk: T.Callable[[], None]):
        self.infer_callback = cbk

    def set_progress_callback(self, cbk: T.Callable[[TrainingProgress], None]):
        self.progress_callback = cbk

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
