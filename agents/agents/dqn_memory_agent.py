from abc import ABC, abstractmethod
import typing as T
import torch
import numpy as np

from environments import Environment
from agents.explorers import Explorer
from agents.replay_buffers import MemoryReplayBufferEntry, ReplayBuffer
from .models import DqnHyperParams, TrainingProgress, TrainingParams
from .agent import Agent


class DqnMemoryAgent(Agent, ABC):
    hp: DqnHyperParams

    def __init__(self,
                 hp: DqnHyperParams,
                 tp: TrainingParams,
                 explorer: T.Union[Explorer, None],
                 replay_buffer: ReplayBuffer,
                 use_gpu: bool = True):
        super(DqnMemoryAgent, self).__init__(hp, tp, explorer, replay_buffer, use_gpu)

        self.action_estimator = self.model_factory().to(self.device)
        self.action_evaluator = self.model_factory().to(self.device)
        self.optimizer = torch.optim.RMSprop(self.action_estimator.parameters(), lr=hp.lr)
        self.loss_f = torch.nn.MSELoss().to(self.device)

    @abstractmethod
    def memory_init(self) -> torch.Tensor:
        raise NotImplementedError()

    def postprocess(self, t: torch.Tensor) -> np.ndarray:
        return np.array(t.squeeze(0))

    def infer(self, x: np.ndarray, m: torch.Tensor = None) -> T.Tuple[np.ndarray, torch.Tensor]:
        if self.infer_callback:
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
        y = torch.max(actions_expected_values, 1)[0] * self.hp.gamma * batch_finals + batch_r
        loss = self.loss_f(x, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, env: Environment) -> None:
        s = env.reset()
        i = 0
        episode = 1
        m = self.memory_init().to(self.device)
        steps_survived = 0
        accumulated_reward = 0
        max_reward: T.Union[None, float] = None
        reward_record: T.List[float] = []
        while True:
            estimated_rewards, m_ = self.infer(s, m)
            a = self.explorer.choose(estimated_rewards, lambda x: np.argmax(estimated_rewards).item())
            s_, r, final = env.step(a)
            self.replay_buffer.add(MemoryReplayBufferEntry(s, m, s_, a, r, final))
            m = m_

            accumulated_reward += r
            s = s_

            if i % self.tp.learn_every == 0 and i != 0:
                batch = self.replay_buffer.sample(self.tp.batch_size)
                self.learn(batch)

            if i % self.hp.ensure_every == 0:
                self.ensure_learning()

            if final:
                tp = TrainingProgress(episode, steps_survived, accumulated_reward)
                reward_record.append(accumulated_reward)
                max_reward = accumulated_reward if max_reward is None or accumulated_reward > max_reward else max_reward
                if self.progress_callback:
                    self.progress_callback(tp)
                if episode >= self.tp.episodes:
                    return

                episode += 1
                accumulated_reward = 0
                steps_survived = 0
                s = env.reset()
                m = self.memory_init().to(self.device)
            else:
                steps_survived += 1
            env.render()
            i += 1
