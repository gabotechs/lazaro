from abc import ABC, abstractmethod
import typing as T
import torch
import numpy as np
from replay_buffers import MemoryReplayBufferEntry
from .models import HyperParams


class DqnMemoryAgent(ABC):
    def __init__(self,
                 hp: HyperParams,
                 use_gpu: bool = True):

        device = "cuda" if use_gpu else "cpu"
        self.action_estimator = self.action_estimator_factory().to(device)
        self.action_evaluator = self.action_estimator_factory().to(device)
        self.optimizer = torch.optim.RMSprop(self.action_estimator.parameters(), lr=hp.lr)
        self.loss_f = torch.nn.MSELoss().to(device)
        self.gamma = hp.gamma
        self.use_gpu = use_gpu
        self.infer_callback = lambda: None

    def set_infer_callback(self, cbk: T.Callable[[], None]):
        self.infer_callback = cbk

    @staticmethod
    @abstractmethod
    def action_estimator_factory() -> torch.nn.Module:
        raise NotImplementedError()

    @abstractmethod
    def memory_init(self) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def postprocess(self, t: torch.Tensor) -> np.ndarray:
        raise NotImplementedError()

    def infer(self, x: np.ndarray, m: torch.Tensor = None) -> T.Tuple[np.ndarray, torch.Tensor]:
        self.infer_callback()
        device = "cuda" if self.use_gpu else "cpu"
        with torch.no_grad():
            estimated_action_values, m_ = self.action_estimator.forward(self.preprocess(x).to(device), m)
            return self.postprocess(estimated_action_values.cpu()), m_

    def ensure_learning(self) -> None:
        self.action_evaluator.load_state_dict(self.action_estimator.state_dict())

    def learn(self, batch: T.List[MemoryReplayBufferEntry]) -> None:
        device = "cuda" if self.use_gpu else "cpu"
        batch_s = torch.cat([self.preprocess(m.s) for m in batch], 0).to(device).requires_grad_(True)
        batch_m = torch.cat([m.m for m in batch], 0).requires_grad_(True)
        batch_s_ = torch.cat([self.preprocess(m.s_) for m in batch], 0).to(device)
        batch_a = [m.a for m in batch]
        batch_r = torch.tensor([m.r for m in batch], dtype=torch.float32, device=device)
        batch_finals = torch.tensor([int(not m.final) for m in batch], device=device)
        actions_estimated_values, batch_m_ = self.action_estimator(batch_s, batch_m)
        with torch.no_grad():
            actions_expected_values, _ = self.action_evaluator(batch_s_, batch_m_)

        x = torch.stack([t_s[t_a] for t_s, t_a in zip(actions_estimated_values, batch_a)])
        y = torch.max(actions_expected_values, 1)[0] * self.gamma * batch_finals + batch_r
        loss = self.loss_f(x, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
