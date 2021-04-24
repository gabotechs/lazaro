import datetime
import json
import os
import typing as T
from abc import ABC

import torch

from .tensorboard_logger import TensorBoard
from ..base import models
from ..base.base_agent import BaseAgent
from ...environments import Environment


DISABLE_EMBEDDING = True


class TensorBoardLogger(BaseAgent, ABC):
    def __init__(self, *args, **kwargs):
        self.summary_writer: T.Union[None, TensorBoard] = None
        self.sample_inputs: T.Union[None, T.List[torch.Tensor]] = None
        self.module_names: T.List[str] = []
        self.modules: T.Dict[torch.nn.Module, T.Dict] = {}
        super().__init__(*args, **kwargs)
        self.link_tensorboard()

    def __del__(self):
        if self.summary_writer:
            self.summary_writer.__del__()

    def health_check(self, env: Environment):
        super(TensorBoardLogger, self).health_check(env)
        folder = self.create_tensor_board_folder(env)
        self.summary_writer = TensorBoard(folder)
        self.tensorboard_log_model_graph()

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
                    nested(root[key], acc + " -> " + key if acc != "" else key)

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
            for k, v in self.ex_get_stats().items():
                self.summary_writer.add_scalar(k, v, training_progress.episode)
        return False

    def tensorboard_log_replay_buffer_stats_progress_callback(self, training_progress: models.TrainingProgress) -> bool:
        if self.summary_writer:
            for k, v in self.rp_get_stats().items():
                self.summary_writer.add_scalar(k, v, training_progress.episode)
        return False

    def forward_hook(self, module: torch.nn.Module, x: T.Tuple[torch.Tensor], y: torch.Tensor) -> None:
        if self.sample_inputs is None:
            self.sample_inputs = x
        if DISABLE_EMBEDDING:
            return
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

    def link_tensorboard(self) -> None:
        self.log.info("linking tensorboard callbacks...")
        self.add_progress_callback("tensorboard log training progress",
                                   self.tensorboard_log_training_progress)
        self.add_progress_callback("tensorboard log explorer stats",
                                   self.tensorboard_log_explorer_stats_progress_callback)
        self.add_progress_callback("tensorboard log replay buffer stats",
                                   self.tensorboard_log_replay_buffer_stats_progress_callback)

    def create_tensor_board_folder(self, env: Environment) -> str:
        base = os.environ.get("LZ_TENSORBOARD_PATH", "data")
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
        self.log.info(f"all save folders created: {folder}")
        return folder
