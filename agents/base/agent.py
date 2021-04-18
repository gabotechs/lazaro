from abc import ABC, abstractmethod
import typing as T
import torch
import json
import datetime
import numpy as np

from environments import Environment

from . import models
from .base_object import BaseObject
from .. import explorers as ex, replay_buffers as rp
from logger import get_logger
import os
from plotter import TensorBoard


class Agent(BaseObject, ABC):
    def __init__(self,
                 action_space: int,
                 explorer: ex.AnyExplorer,
                 replay_buffer: rp.AnyReplayBuffer,
                 tp: models.TrainingParams,
                 hp: models.HyperParams,
                 use_gpu: bool = True,
                 tensor_board_log: bool = True):
        super(Agent, self).__init__()
        self.log = get_logger(self.get_self_class_name())
        self.action_space = action_space
        self.hp = hp
        self.tp = tp
        self.gamma: float = hp.gamma
        self.explorer: ex.AnyExplorer = explorer
        self.replay_buffer: rp.AnyReplayBuffer = replay_buffer
        self.device: str = "cpu"
        self.use_gpu: bool = use_gpu
        if use_gpu:
            if not torch.cuda.is_available():
                self.log.warning("cuda is not available, CPU will be used")
                self.use_gpu = False
            else:
                self.device = "cuda"

        self.tensor_board_log: bool = tensor_board_log

        self.module_names: T.List[str] = []
        self.modules: T.Dict[torch.nn.Module, T.Dict] = {}
        self.step_callbacks: T.List[models.TStepCallback] = []
        self.progress_callbacks: T.List[models.TProgressCallback] = []
        self.learning_callbacks: T.List[models.TLearnCallback] = []
        self.model_wrappers: T.List[T.Callable[[torch.nn.Module], torch.nn.Module]] = []

        def last_layer_factory(in_features: int, out_features: int) -> torch.nn.Linear:
            return torch.nn.Linear(in_features, out_features)

        self.last_layer_factory: T.Callable[[int, int], torch.nn.Module] = last_layer_factory
        self.save_path: T.Union[None, str] = None
        self.summary_writer: T.Union[TensorBoard, None] = None
        self.sample_inputs: T.Union[None, T.List[torch.Tensor]] = None

        replay_buffer.link_to_agent(self)
        explorer.link_to_agent(self)
        self.link_tensorboard()

    def health_check(self, env: Environment):
        self.log.info("checking the model is healthy...")
        s = env.reset()
        self.log.debug(f"state for testing health is:\n{s}")
        self.log.info("testing preprocessing...")
        try:
            self.preprocess(s)
        except Exception as e:
            self.log.error("error while testing preprocessing")
            raise e
        self.log.info("preprocessing is correct")
        self.log.info("testing inference...")
        try:
            self.infer(s)
        except Exception as e:
            self.log.error("error while testing inference")
            raise e
        self.log.info("inference is correct")
        self.log.info("testing learning...")
        while len(self.replay_buffer) < 2:
            a = 0
            s_, r, final = env.step(a)
            self.replay_buffer.add(rp.ReplayBufferEntry(s, s_, a, r, final))
            s = s_
            if final:
                s = env.reset()

        batch = self.replay_buffer.sample(2)
        try:
            self.learn(batch)
        except Exception as e:
            self.log.error("error while testing learning")
            raise e

        self.log.info("learning is correct")
        self.log.info("model is healthy!")
        self.replay_buffer.clear()

        if self.tensor_board_log:
            self.create_tensor_board_folder(env)
            self.summary_writer = TensorBoard(self.save_path)
            self.tensorboard_log_model_graph()
            # self.tensorboard_log_hyper_params()

    def tensorboard_log_training_progress(self, training_progress: models.TrainingProgress) -> bool:
        if self.summary_writer:
            self.summary_writer.add_scalar("episode reward", training_progress.total_reward, training_progress.episode)

        return False

    def tensorboard_log_hyper_params(self):
        info = self.get_info()
        result = {}

        def nested(root: T.Any, acc: str = ""):
            if not isinstance(root, dict):
                result[acc] = json.dumps(root) if isinstance(root, tuple) else root
            else:
                for key in root:
                    nested(root[key], acc+" -> "+key if acc != "" else key)

        nested(info)
        for r in result:
            self.summary_writer.add_text(r, str(result[r]))

    def tensorboard_log_model_graph(self):
        models_in_self: T.Dict[str, torch.nn.Module] = {}
        for attr, value in self.__dict__.items():
            if isinstance(value, torch.nn.Module) and not attr.startswith("loss") and self.sample_inputs is not None:
                models_in_self[attr] = value

        class AllModels(torch.nn.Module):
            def __init__(self):
                super(AllModels, self).__init__()
                for name, model in models_in_self.items():
                    self.__setattr__(name, model)

            def forward(self, x):
                result_unfolded = []
                for result in [self.__getattr__(name)(x) for name in models_in_self]:
                    if isinstance(result, tuple):
                        for folded_result in result:
                            result_unfolded.append(folded_result)
                    else:
                        result_unfolded.append(result)
                return tuple(result_unfolded)

        self.summary_writer.add_graph(AllModels(), self.sample_inputs)

    def tensorboard_log_explorer_stats_progress_callback(self, training_progress: models.TrainingProgress) -> bool:
        if self.summary_writer:
            for k, v in self.explorer.get_stats().items():
                self.summary_writer.add_scalar(k, v, training_progress.episode)
        return False

    def tensorboard_log_replay_buffer_stats_progress_callback(self, training_progress: models.TrainingProgress) -> bool:
        if self.summary_writer:
            for k, v in self.replay_buffer.get_stats().items():
                self.summary_writer.add_scalar(k, v, training_progress.episode)
        return False
    
    def DEL_tensorboard_log_prioritized_replay_buffer_add_beta(self, training_progress: models.TrainingProgress) -> bool:
        if self.summary_writer:
            self.summary_writer.add_scalar("prioritized replay buffer Beta",
                                           self.replay_buffer.beta,
                                           training_progress.episode)
        return False

    def tensorboard_embedding_forward_hook(self, module: torch.nn.Module, x: T.Tuple[torch.Tensor], y: torch.Tensor) -> None:
        if self.sample_inputs is None:
            self.sample_inputs = x
        if self.summary_writer:
            if len(self.module_names) == 0:
                for attr, value in self.__dict__.items():
                    if isinstance(value, torch.nn.Module) and not attr.startswith("loss"):
                        self.module_names.append(attr)
            if module not in self.modules:
                self.modules[module] = {"name": self.module_names[len(self.modules)], "times": 0, "renders": 0}

            if self.modules[module]["times"] % 1000 == 0:
                self.summary_writer.add_embedding(y,
                                                  tag=self.modules[module]["name"],
                                                  global_step=self.modules[module]["renders"])
                self.modules[module]["renders"] += 1
            self.modules[module]["times"] += 1

    def tensorboard_model_wrapper(self, model: torch.nn.Module) -> torch.nn.Module:
        model.register_forward_hook(self.tensorboard_embedding_forward_hook)
        return model

    def link_tensorboard(self) -> None:
        if self.tensor_board_log:
            self.log.info("linking tensorboard callbacks...")
            self.add_progress_callback(self.tensorboard_log_training_progress)
            self.add_progress_callback(self.tensorboard_log_explorer_stats_progress_callback)
            self.add_progress_callback(self.tensorboard_log_replay_buffer_stats_progress_callback)
            self.model_wrappers.append(self.tensorboard_model_wrapper)
            
    def create_tensor_board_folder(self, env: Environment):
        base = os.environ.get("SAVE_DIR", "data")
        if base.endswith("/"):
            base = base[:-1]

        agent = self.get_self_class_name()
        today = str(datetime.datetime.now().date())
        now = str(datetime.datetime.now().time().strftime("%H:%M:%S"))
        folder = ""
        for sub_folder in [base, agent, type(env).__name__, today, now]:
            folder = os.path.join(folder, sub_folder)
            if not os.path.isdir(folder):
                self.log.info(f"folder {folder} does not exists, creating it...")
                os.mkdir(folder)
        self.save_path = folder
        self.log.info(f"all save folders created: {folder}")


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
