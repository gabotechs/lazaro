from abc import ABC, abstractmethod
import typing as T
import torch
import numpy as np
from replay_buffers import ReplayBufferEntry
from .models import HyperParams


class Agent(ABC):
    @abstractmethod
    def __init__(self, hp: HyperParams, use_gpu: bool = True):
        self.hp: HyperParams = hp
        self.device = "cuda" if use_gpu else "cpu"
        self.use_gpu = use_gpu
        self.infer_callback: T.Callable[[], None] = lambda: None

    def set_infer_callback(self, cbk: T.Callable[[], None]):
        self.infer_callback = cbk

    @staticmethod
    @abstractmethod
    def action_estimator_factory() -> torch.nn.Module:
        raise NotImplementedError()

    @abstractmethod
    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def postprocess(self, t: torch.Tensor) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def infer(self, *args) -> T.Any:
        self.infer_callback()
        raise NotImplementedError()

    @abstractmethod
    def learn(self, batch: T.List[ReplayBufferEntry]) -> None:
        raise NotImplementedError()
