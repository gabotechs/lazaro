import typing as T
from abc import ABC, abstractmethod
from explorers import Explorer
from agents import Agent
from environments import Environment
from replay_buffers import ReplayBuffer
from .models import TrainingParams, TrainingProgress


class Trainer(ABC):
    def __init__(self,
                 env: Environment,
                 agent: Agent,
                 explorer: Explorer,
                 replay_buffer: ReplayBuffer,
                 training_params: TrainingParams):

        self.env = env
        self.agent: Agent = agent
        self.explorer: Explorer = explorer
        self.replay_buffer: ReplayBuffer = replay_buffer
        self.training_params: TrainingParams = training_params

        self.progress_callback: T.Callable[[TrainingProgress], None] = lambda x: None

    def set_progress_callback(self, cbk: T.Callable[[TrainingProgress], None]):
        self.progress_callback = cbk

    @abstractmethod
    def train(self, finish_condition: T.Callable[[TrainingProgress], bool]) -> None:
        raise NotImplementedError()
