import typing as T
from abc import ABC

import torch

from . import models, base_object
from .. import interfaces
from ...environments import Environment


class BaseAgent(base_object.BaseObject,
                interfaces.AgentEventInterface,
                interfaces.ExplorerInterface,
                interfaces.ReplayBufferInterface,
                ABC):
    def __init__(self, action_space: int, hp: models.HyperParams, use_gpu: bool = True):
        super(BaseAgent, self).__init__()
        self.device: str = "cpu"
        if use_gpu:
            if not torch.cuda.is_available():
                self.log.warning("cuda is not available, CPU will be used")
            else:
                self.device = "cuda"
        self.action_space = action_space
        self.hyper_params = hp
        self.step_callbacks: T.Dict[str, models.TStepCallback] = {}
        self.progress_callbacks: T.Dict[str, models.TProgressCallback] = {}
        self.learning_callbacks: T.Dict[str, models.TLearnCallback] = {}
        self.accumulate_rewards: bool = True
        self.rp_link()
        self.ex_link()

    def get_self_class_name(self):
        return self.__class__.__name__

    @staticmethod
    def last_layer_factory(in_features: int, out_features: int) -> torch.nn.Module:
        return torch.nn.Linear(in_features, out_features)

    def health_check(self, env: Environment) -> None:
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
        while self.rp_get_length() < 2:
            a = 0
            s_, r, final = env.step(a)
            self.rp_add(models.ReplayBufferEntry(s, s_, a, r, final))
            s = s_
            if final:
                s = env.reset()

        batch = self.rp_sample(2)
        try:
            self.learn(batch)
        except Exception as e:
            self.log.error("error while testing learning")
            raise e

        self.log.info("learning is correct")
        self.log.info("model is healthy!")
        self.rp_clear()

    def add_step_callback(self, label: str, cbk: models.TStepCallback) -> None:
        if label in self.step_callbacks:
            self.log.warning(f"overwriting step callback with label {label}")
        self.step_callbacks[label] = cbk
        self.log.debug(f"added step callback {label}: {cbk}")

    def add_progress_callback(self, label: str, cbk: models.TProgressCallback) -> None:
        if label in self.progress_callbacks:
            self.log.warning(f"overwriting progress callback with label {label}")
        self.progress_callbacks[label] = cbk
        self.log.debug(f"added progress callback {label}: {cbk}")

    def add_learn_callback(self, label: str, cbk: models.TLearnCallback) -> None:
        if label in self.learning_callbacks:
            self.log.warning(f"overwriting learning callback with label {label}")
        self.learning_callbacks[label] = cbk
        self.log.debug(f"added learning callback {label}: {cbk}")

    def call_step_callbacks(self, training_step: models.TrainingStep) -> None:
        self.log.debug("calling step callbacks...")
        for label, cbk in self.step_callbacks.items():
            self.log.debug(f"calling step callback {label}, {cbk}")
            cbk(training_step)
        self.log.debug("all step callbacks called")

    def call_progress_callbacks(self, training_progress: models.TrainingProgress) -> bool:
        self.log.debug("calling progress callbacks...")
        must_exit = False
        for label, cbk in self.progress_callbacks.items():
            self.log.debug(f"calling progress callback {label}, {cbk}")
            may_exit = cbk(training_progress)
            if may_exit:
                must_exit = True
                self.log.warning(f"progress callback {label} said that training should end")
        self.log.debug("all progress callbacks called")
        return must_exit

    def call_learn_callbacks(self, learning_step: models.LearningStep) -> None:
        self.log.debug("calling learning callbacks...")
        for label, cbk in self.learning_callbacks.items():
            self.log.debug(f"calling learning callback {label}, {cbk}")
            cbk(learning_step)
        self.log.debug("all learning callbacks called")

    def forward_hook(self, module: torch.nn.Module, x: T.Tuple[torch.Tensor], y: torch.Tensor) -> None:
        pass

    def build_model(self) -> torch.nn.Module:
        self.log.info("building model from model factory...")
        model = self.model_factory()
        model.register_forward_hook(self.forward_hook)
        self.log.info("model built correctly, applying last layers modifier...")
        model = self.last_layers_model_modifier(model)
        self.log.info("last layers modified correctly, applying agent specification")
        model = self.agent_specification_model_modifier(model)
        self.log.info("agent specification applied correctly")
        return model

    def preprocess(self, x: T.Iterable) -> torch.Tensor:
        return torch.unsqueeze(torch.tensor(x, dtype=torch.float32), 0)
