from abc import ABC, abstractmethod
import typing as T
import torch
import numpy as np
from replay_buffers import ReplayBufferEntry
from .models import HyperParams
from .agent import Agent


class DqnAgent(Agent, ABC):
    def __init__(self, hp: HyperParams, use_gpu: bool = True):
        super().__init__(hp, use_gpu)

        self.action_estimator = self.action_estimator_factory().to(self.device)
        self.action_evaluator = self.action_estimator_factory().to(self.device)
        self.optimizer = torch.optim.RMSprop(self.action_estimator.parameters(), lr=hp.lr)
        self.loss_f = torch.nn.MSELoss().to(self.device)
        self.gamma = hp.gamma

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

    def infer(self, x: np.ndarray) -> np.ndarray:
        self.infer_callback()
        with torch.no_grad():
            return self.postprocess(self.action_estimator.forward(self.preprocess(x).to(self.device)).cpu())

    def ensure_learning(self) -> None:
        self.action_evaluator.load_state_dict(self.action_estimator.state_dict())

    def learn(self, batch: T.List[ReplayBufferEntry]) -> None:
        batch_s = torch.cat([self.preprocess(m.s) for m in batch], 0).to(self.device).requires_grad_(True)
        batch_s_ = torch.cat([self.preprocess(m.s_) for m in batch], 0).to(self.device)
        batch_a = [m.a for m in batch]
        batch_r = torch.tensor([m.r for m in batch], dtype=torch.float32, device=self.device)
        batch_finals = torch.tensor([int(not m.final) for m in batch], device=self.device)
        actions_estimated_values: torch.Tensor = self.action_estimator(batch_s)
        with torch.no_grad():
            actions_expected_values: torch.Tensor = self.action_evaluator(batch_s_)

        x = torch.stack([t_s[t_a] for t_s, t_a in zip(actions_estimated_values, batch_a)])
        y = torch.max(actions_expected_values, 1)[0] * self.gamma * batch_finals + batch_r
        loss = self.loss_f(x, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
