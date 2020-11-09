from abc import ABC, abstractmethod
import typing as T
import torch
import numpy as np
from .memory import Memory
from .models import HyperParams, MemoryEntry


class DqnAgent(ABC):
    def __init__(self,
                 hp: HyperParams,
                 memory_len: int = 5000,
                 use_gpu: bool = True):

        device = "cuda" if use_gpu else "cpu"
        self.action_estimator = self.action_estimator_factory().to(device)
        self.action_evaluator = self.action_estimator_factory().to(device)
        self.optimizer = torch.optim.RMSprop(self.action_estimator.parameters(), lr=hp.lr)
        self.loss_f = torch.nn.MSELoss().to(device)
        self.gamma = hp.gamma
        self.use_gpu = use_gpu
        self.memory = Memory(memory_len)
        self.infer_callback = lambda: None

    def set_infer_callback(self, cbk: T.Callable[[], None]):
        self.infer_callback = cbk

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
        device = "cuda" if self.use_gpu else "cpu"
        with torch.no_grad():
            return self.postprocess(self.action_estimator.forward(self.preprocess(x).to(device)).cpu())

    def memory_full(self) -> bool:
        return len(self.memory.records) == self.memory.records.maxlen

    def memorize(self, s: np.ndarray, s_: np.ndarray, a: int, r: float, final: bool) -> None:
        self.memory.add(MemoryEntry(s, s_, a, r, final))

    def ensure_learning(self) -> None:
        self.action_evaluator.load_state_dict(self.action_estimator.state_dict())

    def learn(self, batch_size: int, random: bool = True) -> None:
        device = "cuda" if self.use_gpu else "cpu"
        memories = self.memory.sample(batch_size) if random else list(self.memory.records)[-batch_size:]
        batch_s = torch.cat([self.preprocess(m.s) for m in memories], 0).to(device).requires_grad_(True)
        batch_s_ = torch.cat([self.preprocess(m.s_) for m in memories], 0).to(device)
        batch_a = [m.a for m in memories]
        batch_r = torch.tensor([m.r for m in memories], dtype=torch.float32, device=device)
        batch_finals = torch.tensor([int(not m.final) for m in memories], device=device)
        actions_estimated_values: torch.Tensor = self.action_estimator(batch_s)
        with torch.no_grad():
            actions_expected_values: torch.Tensor = self.action_evaluator(batch_s_)

        x = torch.stack([t_s[t_a] for t_s, t_a in zip(actions_estimated_values, batch_a)])
        y = torch.max(actions_expected_values, 1)[0] * self.gamma * batch_finals + batch_r
        loss = self.loss_f(x, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
