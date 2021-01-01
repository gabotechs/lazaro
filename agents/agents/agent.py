from abc import ABC, abstractmethod
import typing as T
import torch
import os
import json
import datetime
import time
import numpy as np

from environments import Environment
from agents.explorers import AnyExplorer, RandomExplorer, NoisyExplorer
from agents.replay_buffers import AnyReplayBuffer, ReplayBufferEntry, NStepsPrioritizedReplayBuffer
from .models import HyperParams, TrainingProgress, TrainingParams, LearningStep, TrainingStep
from ..explorers.noisy_explorer import NoisyLinear


class Agent(ABC):
    def __init__(self,
                 action_space: int,
                 hp: HyperParams,
                 tp: TrainingParams,
                 explorer: T.Union[AnyExplorer, None],
                 replay_buffer: AnyReplayBuffer,
                 use_gpu: bool = True,
                 save_progress: bool = True):
        self.action_space = action_space
        self.hp: HyperParams = hp
        self.tp: TrainingParams = tp
        self.gamma: float = hp.gamma
        self.explorer: T.Union[AnyExplorer, None] = explorer
        self.replay_buffer: AnyReplayBuffer = replay_buffer
        self.device: str = "cuda" if use_gpu else "cpu"
        self.use_gpu: bool = use_gpu
        self.save_progress: bool = save_progress
        self.healthy_callbacks: T.List[T.Callable[[], None]] = []
        self.step_callbacks: T.List[T.Callable[[TrainingStep], None]] = []
        self.progress_callbacks: T.List[T.Callable[[TrainingProgress], None]] = []
        self.learning_callbacks: T.List[T.Callable[[LearningStep], None]] = []
        self.model_wrappers: T.List[T.Callable[[torch.nn.Module], torch.nn.Module]] = []
        self.save_path: T.Union[None, str] = None
        self.reward_record: T.List[float] = []
        self.loss_record: T.List[float] = []
        self.link_replay_buffer()
        self.link_explorer()
        self.link_saver()

    def link_saver(self):
        if self.save_progress:
            def init_save_callback():
                base = os.environ.get("SAVE_DIR", "data")
                if base.endswith("/"):
                    base = base[:-1]

                agent = self.get_self_class_name()
                today = str(datetime.datetime.now().date())
                now = str(datetime.datetime.now().time().strftime("%H:%M:%S"))
                folder = ""
                for sub_folder in [base, agent, today, now]:
                    folder = os.path.join(folder, sub_folder)
                    if not os.path.isdir(folder):
                        os.mkdir(folder)

                self.save_path = folder
                agent_info_path = os.path.join(folder, "agent.json")
                json.dump(self.get_info(), open(agent_info_path, "w"), indent=4)

            self.add_healthy_callback(init_save_callback)

            def checkpoint_save_callback(training_progress: TrainingProgress):
                if self.save_path is None:
                    raise ValueError("save path is None, agent has not been initialized correctly")
                folder_checkpoints = os.path.join(self.save_path, "checkpoints")
                if not os.path.isdir(folder_checkpoints):
                    os.mkdir(folder_checkpoints)

                folder_checkpoints_checkpoint = os.path.join(folder_checkpoints, str(time.time())+".json")
                json.dump(training_progress.__dict__, open(folder_checkpoints_checkpoint, "w"))

            self.add_progress_callback(checkpoint_save_callback)

    def link_replay_buffer(self):
        if isinstance(self.replay_buffer, NStepsPrioritizedReplayBuffer):
            def update_prioritized_replay_buffer_priorities(learning_step: LearningStep):
                self.replay_buffer.update_priorities(
                    [e.index for e in learning_step.batch],
                    [abs(x - y) + 1e-7 for x, y in zip(learning_step.x, learning_step.y)]
                )

            self.add_learn_callback(update_prioritized_replay_buffer_priorities)

            def increase_prioritized_replay_buffer_beta(_: TrainingStep):
                self.replay_buffer.increase_beta()

            self.add_step_callback(increase_prioritized_replay_buffer_beta)

            self.gamma = self.hp.gamma ** self.replay_buffer.rp.n_step

    def link_explorer(self):
        if isinstance(self.explorer, RandomExplorer):
            def decay_random_explorer_epsilon(_: TrainingStep):
                self.explorer.decay()

            self.add_step_callback(decay_random_explorer_epsilon)

        if isinstance(self.explorer, NoisyExplorer):
            def noisy_linear_model_factory(model: torch.nn.Module) -> torch.nn.Module:
                return self.explorer.wrap_model(model)

            self.model_wrappers.append(noisy_linear_model_factory)

            def reset_noisy_explorer_noise(training_step: TrainingStep):
                if training_step.i % self.explorer.ep.reset_noise_every == 0:
                    for attr, value in self.__dict__.items():
                        if isinstance(value, torch.nn.Module):
                            for layer in value.modules():
                                if isinstance(layer, NoisyLinear):
                                    layer.reset_noise()

            self.add_step_callback(reset_noisy_explorer_noise)

    def add_healthy_callback(self, cbk: T.Callable[[], None]):
        self.healthy_callbacks.append(cbk)

    def add_step_callback(self, cbk: T.Callable[[TrainingStep], None]):
        self.step_callbacks.append(cbk)

    def add_progress_callback(self, cbk: T.Callable[[TrainingProgress], None]):
        self.progress_callbacks.append(cbk)

    def add_learn_callback(self, cbk: T.Callable[[LearningStep], None]):
        self.learning_callbacks.append(cbk)

    def build_model(self) -> torch.nn.Module:
        model = self.model_factory()
        for wrapper in self.model_wrappers:
            model = wrapper(model)
        return model

    def get_self_class_name(self):
        return self.__class__.__bases__[0].__name__

    def get_info(self) -> dict:
        info = {
            "class": self.get_self_class_name(),
            "hyper parameters": {
                "class": type(self.hp).__name__,
                "attributes": self.hp.__dict__
            },
            "training parameters": {
                "class": type(self.tp).__name__,
                "attributes": self.tp.__dict__
            },
            "replay buffer": {
                "class": type(self.replay_buffer).__name__,
                "attributes": {
                    "class": type(self.replay_buffer.rp).__name__,
                    "attributes": self.replay_buffer.rp.__dict__
                }
            },
            "explorer": {
                "class": type(self.explorer).__name__,
                "attributes": {
                    "class": type(self.explorer.ep).__name__,
                    "attributes": self.explorer.ep.__dict__
                }
            }
        }
        return info

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
