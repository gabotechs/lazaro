from abc import ABC, abstractmethod
import typing as T
import torch
import numpy as np
from replay_buffers import MemoryReplayBufferEntry
from .models import HyperParams
from .agent import Agent


class DqnMemoryAgent(Agent, ABC):
    def __init__(self, hp: HyperParams, use_gpu: bool = True):
        super().__init__(hp, use_gpu)

        self.action_estimator = self.model_factory().to(self.device)
        self.action_evaluator = self.model_factory().to(self.device)
        self.optimizer = torch.optim.RMSprop(self.action_estimator.parameters(), lr=hp.lr)
        self.loss_f = torch.nn.MSELoss().to(self.device)
        self.gamma = hp.gamma

    @abstractmethod
    def memory_init(self) -> torch.Tensor:
        raise NotImplementedError()

    def postprocess(self, t: torch.Tensor) -> np.ndarray:
        return np.array(t.squeeze(0))

    def infer(self, x: np.ndarray, m: torch.Tensor = None) -> T.Tuple[np.ndarray, torch.Tensor]:
        self.infer_callback()
        with torch.no_grad():
            estimated_action_values, m_ = self.action_estimator.forward(self.preprocess(x).to(self.device), m)
            return self.postprocess(estimated_action_values.cpu()), m_

    def ensure_learning(self) -> None:
        self.action_evaluator.load_state_dict(self.action_estimator.state_dict())

    def learn(self, batch: T.List[MemoryReplayBufferEntry]) -> None:
        batch_s = torch.cat([self.preprocess(m.s) for m in batch], 0).to(self.device).requires_grad_(True)
        batch_m = torch.cat([m.m for m in batch], 0).requires_grad_(True)
        batch_s_ = torch.cat([self.preprocess(m.s_) for m in batch], 0).to(self.device)
        batch_a = [m.a for m in batch]
        batch_r = torch.tensor([m.r for m in batch], dtype=torch.float32, device=self.device)
        batch_finals = torch.tensor([int(not m.final) for m in batch], device=self.device)
        actions_estimated_values, batch_m_ = self.action_estimator(batch_s, batch_m)
        with torch.no_grad():
            actions_expected_values, _ = self.action_evaluator(batch_s_, batch_m_)

        x = torch.stack([t_s[t_a] for t_s, t_a in zip(actions_estimated_values, batch_a)])
        y = torch.max(actions_expected_values, 1)[0] * self.gamma * batch_finals + batch_r
        loss = self.loss_f(x, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
