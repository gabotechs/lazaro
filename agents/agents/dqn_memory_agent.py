from abc import ABC, abstractmethod
import typing as T
import torch
import random
import numpy as np

from environments import Environment
from agents.explorers import AnyExplorer
from agents.replay_buffers import MemoryReplayBufferEntry, AnyReplayBuffer
from .models import MDqnHyperParams, TrainingProgress, MDqnTrainingParams
from .agent import Agent


class DqnMemoryAgent(Agent, ABC):
    hp: MDqnHyperParams
    tp: MDqnTrainingParams

    def __init__(self,
                 hp: MDqnHyperParams,
                 tp: MDqnTrainingParams,
                 explorer: T.Union[AnyExplorer, None],
                 replay_buffer: AnyReplayBuffer,
                 use_gpu: bool = True):
        super(DqnMemoryAgent, self).__init__(hp, tp, explorer, replay_buffer, use_gpu)

        self.action_estimator = self.model_factory().to(self.device)
        self.action_evaluator = self.model_factory().to(self.device)
        self.memory_provider = self.memory_model_factory().to(self.device)
        self.action_optimizer = torch.optim.Adam(self.action_estimator.parameters(), lr=hp.a_lr)
        self.memory_optimizer = torch.optim.Adam(self.memory_provider.parameters(), lr=hp.m_lr)
        self.loss_f = torch.nn.MSELoss().to(self.device)

    @abstractmethod
    def memory_init(self) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def memory_model_factory(self) -> torch.nn.Module:
        raise NotImplementedError()

    def postprocess(self, t: torch.Tensor) -> np.ndarray:
        return np.array(t.squeeze(0))

    def infer(self, x: np.ndarray, m: torch.Tensor) -> T.Tuple[np.ndarray, torch.Tensor]:
        if self.infer_callback:
            self.infer_callback()
        with torch.no_grad():
            m_ = self.memory_provider.forward(self.preprocess(x).to(self.device), m)
            estimated_action_values = self.action_estimator.forward(self.preprocess(x).to(self.device), m_)
            return self.postprocess(estimated_action_values.cpu()), m_

    def ensure_learning(self) -> None:
        self.action_evaluator.load_state_dict(self.action_estimator.state_dict())

    def get_batched_tensors(self, batch: T.List[MemoryReplayBufferEntry]) \
            -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, T.List[int], torch.Tensor, torch.Tensor]:
        batch_s = torch.cat([self.preprocess(m.s) for m in batch], 0).to(self.device).requires_grad_(True)
        batch_m = torch.cat([m.m for m in batch], 0).requires_grad_(True)
        batch_s_ = torch.cat([self.preprocess(m.s_) for m in batch], 0).to(self.device)
        batch_a = [m.a for m in batch]
        batch_r = torch.tensor([m.r for m in batch], dtype=torch.float32, device=self.device)
        batch_finals = torch.tensor([int(not m.final) for m in batch], device=self.device)
        return batch_s, batch_m, batch_s_, batch_a, batch_r, batch_finals

    def learn_memory(self, batches: T.List[T.List[MemoryReplayBufferEntry]]) -> float:
        loss_record = []
        for batch in batches:
            batch_s, batch_m, batch_s_, batch_a, batch_r, batch_finals = self.get_batched_tensors(batch)
            batch_m_ = self.memory_provider(batch_s, batch_m)
            actions_estimated_values = self.action_estimator(batch_s, batch_m_)
            with torch.no_grad():
                batch_m__ = self.memory_provider(batch_s_, batch_m_)
                actions_expected_values = self.action_evaluator(batch_s_, batch_m__)

            x = torch.stack([t_s[t_a] for t_s, t_a in zip(actions_estimated_values, batch_a)])
            y = torch.max(actions_expected_values, 1)[0] * self.hp.gamma * batch_finals + batch_r
            loss = self.loss_f(x, y)
            self.memory_optimizer.zero_grad()
            self.action_estimator.zero_grad()
            loss.backward()
            self.memory_optimizer.step()
            loss_record.append(loss.item())

        return np.mean(loss_record).item()

    def learn(self, batch: T.List[MemoryReplayBufferEntry]) -> float:
        batch_s, batch_m, batch_s_, batch_a, batch_r, batch_finals = self.get_batched_tensors(batch)
        with torch.no_grad():
            batch_m_ = self.memory_provider(batch_s, batch_m)
        actions_estimated_values = self.action_estimator(batch_s, batch_m_)
        with torch.no_grad():
            batch_m__ = self.memory_provider(batch_s_, batch_m_)
            actions_expected_values = self.action_evaluator(batch_s_, batch_m__)

        x = torch.stack([t_s[t_a] for t_s, t_a in zip(actions_estimated_values, batch_a)])
        y = torch.max(actions_expected_values, 1)[0] * self.hp.gamma * batch_finals + batch_r
        loss = self.loss_f(x, y)
        self.action_optimizer.zero_grad()
        loss.backward()
        self.action_optimizer.step()
        return loss.item()

    def get_data_for_memory_training(self, limit: int = 0) -> T.List[T.List[MemoryReplayBufferEntry]]:
        rp_bf_indexes = set(range(len(self.replay_buffer)))
        batches: T.List[T.List[MemoryReplayBufferEntry]] = []
        i = 0
        while len(rp_bf_indexes):
            sample_size = min(self.tp.memory_batch_size, len(rp_bf_indexes))
            memory_indexes = random.sample(rp_bf_indexes, sample_size)
            batches.append([self.replay_buffer.records[m_index] for m_index in memory_indexes])
            for m_index in memory_indexes:
                rp_bf_indexes.discard(m_index)
            i += 1
            if limit >= i:
                return batches
        return batches

    def train(self, env: Environment) -> None:
        s = env.reset()
        i = 0
        episode = 1
        m = self.memory_init().to(self.device)
        steps_survived = 0
        accumulated_reward = 0
        must_learn_memory = False
        while True:
            estimated_rewards, m_ = self.infer(s, m)
            a = self.explorer.choose(estimated_rewards, lambda x: np.argmax(estimated_rewards).item())
            s_, r, final = env.step(a)
            self.replay_buffer.add(MemoryReplayBufferEntry(s, m, s_, a, r, final))
            m = m_

            accumulated_reward += r
            s = s_

            if i % self.tp.learn_every == 0 and i != 0 and len(self.replay_buffer) >= self.tp.batch_size:
                batch = self.replay_buffer.sample(self.tp.batch_size)
                loss = self.learn(batch)

            if i % self.hp.ensure_every == 0:
                self.ensure_learning()

            if i % self.tp.memory_learn_every == 0 and i:
                print("learning memory...")
                batches = self.get_data_for_memory_training(limit=1)
                loss = self.learn_memory(batches)
                print("memory loss: ", round(loss, 2))
                if self.tp.memory_clear_after_learn:
                    self.replay_buffer.clear()

            if final:
                tp = TrainingProgress(episode, steps_survived, accumulated_reward)
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
