from abc import ABC
import typing as T
import torch
import numpy as np

from environments import Environment
from ..explorers import AnyExplorer
from ..replay_buffers import ReplayBufferEntry, AnyReplayBuffer
from .models import DqnHyperParams, TrainingParams, TrainingProgress, LearningStep
from .agent import Agent


class DqnAgent(Agent, ABC):
    hp: DqnHyperParams

    def __init__(self,
                 hp: DqnHyperParams,
                 tp: TrainingParams,
                 explorer: T.Union[AnyExplorer, None],
                 replay_buffer: AnyReplayBuffer,
                 use_gpu: bool = True):
        super(DqnAgent, self).__init__(hp, tp, explorer, replay_buffer, use_gpu)

        self.action_estimator = self.model_factory().to(self.device)
        self.action_evaluator = self.model_factory().to(self.device)
        self.optimizer = torch.optim.Adam(self.action_estimator.parameters(), lr=hp.lr)
        self.loss_f = torch.nn.MSELoss(reduction="none").to(self.device)

    def infer(self, x: np.ndarray) -> np.ndarray:
        for cbk in self.infer_callbacks:
            cbk()
        with torch.no_grad():
            return self.postprocess(self.action_estimator.forward(self.preprocess(x).to(self.device)).cpu())

    def postprocess(self, t: torch.Tensor) -> np.ndarray:
        return np.array(t.squeeze(0))

    def ensure_learning(self) -> None:
        self.action_evaluator.load_state_dict(self.action_estimator.state_dict())

    def learn(self, batch: T.List[ReplayBufferEntry]) -> None:
        batch_s = torch.cat([self.preprocess(m.s) for m in batch], 0).to(self.device).requires_grad_(True)
        batch_s_ = torch.cat([self.preprocess(m.s_) for m in batch], 0).to(self.device)
        batch_a = [m.a for m in batch]
        batch_r = torch.tensor([m.r for m in batch], dtype=torch.float32, device=self.device)
        batch_finals = torch.tensor([int(not m.final) for m in batch], device=self.device)
        batch_weights = torch.tensor([m.weight for m in batch], device=self.device)
        actions_estimated_values: torch.Tensor = self.action_estimator(batch_s)
        with torch.no_grad():
            actions_expected_values: torch.Tensor = self.action_evaluator(batch_s_)

        x = torch.stack([t_s[t_a] for t_s, t_a in zip(actions_estimated_values, batch_a)])
        y = torch.max(actions_expected_values, 1)[0] * self.hp.gamma * batch_finals + batch_r
        element_wise_loss = self.loss_f(x, y)
        loss = (element_wise_loss * batch_weights).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        for cbk in self.learning_callbacks:
            cbk(LearningStep(batch, [v.item() for v in x], [v.item() for v in y]))

    def train(self, env: Environment) -> None:
        self.hook_callbacks()
        s = env.reset()
        i = 0
        episode = 1
        steps_survived = 0
        accumulated_reward = 0
        while True:
            estimated_rewards = self.infer(s)
            a = self.explorer.choose(estimated_rewards, lambda x: np.argmax(estimated_rewards).item())
            s_, r, final = env.step(a)
            self.replay_buffer.add(ReplayBufferEntry(s, s_, a, r, final))
            accumulated_reward += r
            s = s_

            if i % self.tp.learn_every == 0 and i != 0 and len(self.replay_buffer) >= self.tp.batch_size:
                batch = self.replay_buffer.sample(self.tp.batch_size)
                self.learn(batch)

            if i % self.hp.ensure_every == 0:
                self.ensure_learning()

            if final:
                tp = TrainingProgress(episode, steps_survived, accumulated_reward)
                for cbk in self.progress_callbacks:
                    cbk(tp)
                if episode >= self.tp.episodes:
                    return

                episode += 1
                accumulated_reward = 0
                steps_survived = 0
                s = env.reset()

            else:
                steps_survived += 1
            env.render()
            i += 1
