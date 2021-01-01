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
from logger import get_logger


class Agent(ABC):
    def __init__(self,
                 action_space: int,
                 hp: HyperParams,
                 tp: TrainingParams,
                 explorer: T.Union[AnyExplorer, None],
                 replay_buffer: AnyReplayBuffer,
                 use_gpu: bool = True,
                 save_progress: bool = True):

        self.log = get_logger(self.get_self_class_name())
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
            self.log.info("linking progress callbacks...")

            def init_save_callback():
                self.log.info("initializing save callback triggered")
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
                        self.log.info(f"folder {folder} does not exists, creating it...")
                        os.mkdir(folder)
                self.log.info(f"all save folders created: {folder}")
                self.save_path = folder
                agent_info_path = os.path.join(folder, "agent.json")
                json.dump(self.get_info(), open(agent_info_path, "w"), indent=4)
                self.log.info("agent.json created correctly")

            self.add_healthy_callback(init_save_callback)

            def checkpoint_save_callback(training_progress: TrainingProgress):
                self.log.info("save callback triggered")
                if self.save_path is None:
                    raise ValueError("save path is None, agent has not been initialized correctly")
                folder_checkpoints = os.path.join(self.save_path, "checkpoints")
                if not os.path.isdir(folder_checkpoints):
                    os.mkdir(folder_checkpoints)

                folder_checkpoints_checkpoint = os.path.join(folder_checkpoints, str(time.time())+".json")
                json.dump(training_progress.__dict__, open(folder_checkpoints_checkpoint, "w"))
                self.log.info("checkpoint saved correctly")

            self.add_progress_callback(checkpoint_save_callback)
            self.log.info("progress callbacks linked correctly")
        else:
            self.log.info("progress is not going to be saved")

    def link_replay_buffer(self):
        if isinstance(self.replay_buffer, NStepsPrioritizedReplayBuffer):
            self.log.info(f"linking {type(self.replay_buffer).__name__}...")

            def update_prioritized_replay_buffer_priorities(learning_step: LearningStep):
                self.log.debug(f"update priorities for {type(self.replay_buffer).__name__} triggered")
                self.replay_buffer.update_priorities(
                    [e.index for e in learning_step.batch],
                    [abs(x - y) + 1e-7 for x, y in zip(learning_step.x, learning_step.y)]
                )

            self.add_learn_callback(update_prioritized_replay_buffer_priorities)

            def increase_prioritized_replay_buffer_beta(_: TrainingStep):
                self.log.debug(f"increase beta for {type(self.replay_buffer).__name__} triggered")
                self.replay_buffer.increase_beta()

            self.add_step_callback(increase_prioritized_replay_buffer_beta)

            new_gamma = self.hp.gamma ** self.replay_buffer.rp.n_step
            self.log.info(f"refactoring gamma for {type(self.replay_buffer).__name__}: {self.gamma} -> {new_gamma}")
            self.gamma = new_gamma
            self.log.info(f"{type(self.replay_buffer).__name__} linked correctly")
        else:
            self.log.info(f"{type(self.replay_buffer).__name__} replay buffer does not need linking")

    def link_explorer(self):
        if isinstance(self.explorer, RandomExplorer):
            self.log.info(f"linking {type(self.explorer).__name__}...")

            def decay_random_explorer_epsilon(_: TrainingStep):
                self.log.debug(f"decay epsilon for {type(self.explorer).__name__} triggered")
                self.explorer.decay()

            self.add_step_callback(decay_random_explorer_epsilon)
            self.log.info(f"{type(self.explorer).__name__} linked correctly")

        elif isinstance(self.explorer, NoisyExplorer):
            self.log.info(f"linking {type(self.explorer).__name__}...")

            def noisy_linear_model_factory(model: torch.nn.Module) -> torch.nn.Module:
                self.log.info(f"wrapping model with noisy layers triggered")
                return self.explorer.wrap_model(model)

            self.model_wrappers.append(noisy_linear_model_factory)

            def reset_noisy_explorer_noise(training_step: TrainingStep):
                self.log.debug(f"reset noise for {type(self.explorer).__name__} triggered")
                if training_step.i % self.explorer.ep.reset_noise_every == 0:
                    for attr, value in self.__dict__.items():
                        if isinstance(value, torch.nn.Module):
                            for i, layer in enumerate(value.modules()):
                                if isinstance(layer, NoisyLinear):
                                    self.log.debug(f"layer {i} for attribute {attr} is noisy, noise reset")
                                    layer.reset_noise()

            self.add_step_callback(reset_noisy_explorer_noise)

        else:
            self.log.info(f"{type(self.explorer).__name__} explorer does not need linking")

    def add_healthy_callback(self, cbk: T.Callable[[], None]):
        self.healthy_callbacks.append(cbk)
        self.log.info(f"added new healthy callback, there are {len(self.healthy_callbacks)} healthy callbacks")

    def add_step_callback(self, cbk: T.Callable[[TrainingStep], None]):
        self.step_callbacks.append(cbk)
        self.log.info(f"added new step callback, there are {len(self.step_callbacks)} step callbacks")

    def add_progress_callback(self, cbk: T.Callable[[TrainingProgress], None]):
        self.progress_callbacks.append(cbk)
        self.log.info(f"added new progress callback, there are {len(self.progress_callbacks)} progress callbacks")

    def add_learn_callback(self, cbk: T.Callable[[LearningStep], None]):
        self.learning_callbacks.append(cbk)
        self.log.info(f"added new learn callback, there are {len(self.learning_callbacks)} learn callbacks")

    def call_healthy_callbacks(self):
        self.log.debug("calling healthy callbacks...")
        for cbk in self.healthy_callbacks:
            cbk()
        self.log.debug("all healthy callbacks called")

    def call_step_callbacks(self, training_step: TrainingStep):
        self.log.debug(f"new training step: {training_step.__dict__}")
        self.log.debug("calling step callbacks...")
        for cbk in self.step_callbacks:
            cbk(training_step)
        self.log.debug("all step callbacks called")

    def call_progress_callbacks(self, training_progress: TrainingProgress):
        self.log.info(f"new training progress: {training_progress.__dict__}")
        self.log.debug("calling progress callbacks...")
        for cbk in self.progress_callbacks:
            cbk(training_progress)
        self.log.debug("all progress callbacks called")

    def call_learn_callbacks(self, learning_step: LearningStep):
        self.log.debug(f"new learning step")
        self.log.debug("calling learning callbacks...")
        for cbk in self.learning_callbacks:
            cbk(learning_step)
        self.log.debug("all learning callbacks called")

    def build_model(self) -> torch.nn.Module:
        self.log.info("building model from model factory...")
        model = self.model_factory()
        self.log.info("model built correctly")
        for i, wrapper in enumerate(self.model_wrappers):
            self.log.info(f"wrapping model with wrapper {i}")
            model = wrapper(model)
            self.log.info("model wrapped correctly")
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
