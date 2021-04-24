from abc import ABC, abstractmethod
import typing as T
import torch

from ..base import models
from ...environments import Environment


class AgentEventInterface(ABC):
    @abstractmethod
    def health_check(self, env: Environment) -> None:
        raise NotImplementedError()

    @abstractmethod
    def add_step_callback(self, label: str, cbk: models.TStepCallback) -> None:
        raise NotImplementedError()

    @abstractmethod
    def add_progress_callback(self, label: str, cbk: models.TProgressCallback) -> None:
        raise NotImplementedError()

    @abstractmethod
    def add_learn_callback(self, label: str, cbk: models.TLearnCallback) -> None:
        raise NotImplementedError()

    @abstractmethod
    def call_step_callbacks(self, training_step: models.TrainingStep) -> None:
        raise NotImplementedError()

    @abstractmethod
    def call_progress_callbacks(self, training_progress: models.TrainingProgress) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def call_learn_callbacks(self, learning_step: models.LearningStep) -> None:
        raise NotImplementedError()

    @abstractmethod
    def build_model(self) -> torch.nn.Module:
        raise NotImplementedError()

    @abstractmethod
    def get_self_class_name(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def get_info(self) -> dict:
        raise NotImplementedError()

    @abstractmethod
    def get_state_dict(self) -> dict:
        raise NotImplementedError()

    @abstractmethod
    def last_layers_model_modifier(self, model: torch.nn.Module) -> torch.nn.Module:
        raise NotImplementedError()

    @abstractmethod
    def agent_specification_model_modifier(self, model: torch.nn.Module) -> torch.nn.Module:
        raise NotImplementedError()

    @abstractmethod
    def model_factory(self) -> torch.nn.Module:
        raise NotImplementedError()

    @abstractmethod
    def preprocess(self, x: T.Iterable) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def postprocess(self, t: torch.Tensor) -> T.Iterable:
        raise NotImplementedError()

    @abstractmethod
    def infer(self, *args) -> T.Any:
        raise NotImplementedError()

    @abstractmethod
    def learn(self, batch: T.List[T.Any]) -> None:
        raise NotImplementedError()

    @abstractmethod
    def train(self, env: Environment, tp: models.TrainingParams) -> None:
        raise NotImplementedError()
