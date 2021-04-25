import typing as T
from abc import ABC

import numpy as np
import torch

from ..environments import Environment
from .base.models import DoubleDqnHyperParams, TrainingProgress, LearningStep, TrainingStep, TrainingParams
from .dqn import DqnAgent
from .replay_buffers import ReplayBufferEntry


class DoubleDqnAgent(DqnAgent, ABC):
    def __init__(self,
                 action_space: int,
                 hp: DoubleDqnHyperParams = DoubleDqnHyperParams(),
                 use_gpu: bool = True):
        super(DoubleDqnAgent, self).__init__(action_space, hp, use_gpu)
        self.hyper_params = hp
        self.action_evaluator = self.build_model().to(self.device).eval()

    def get_state_dict(self) -> dict:
        state_dict = super(DoubleDqnAgent, self).get_state_dict()
        state_dict.update({"action_evaluator": self.action_evaluator.state_dict()})
        return state_dict

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
        y = torch.max(actions_expected_values, 1)[0] * self.hyper_params.gamma * batch_finals + batch_r
        element_wise_loss = self.loss_f(x, y)
        loss = (element_wise_loss * batch_weights).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.call_learn_callbacks(LearningStep(batch, [v.item() for v in x], [v.item() for v in y]))

    def train(self, env: Environment, tp: TrainingParams = TrainingParams()) -> None:
        self.health_check(env)
        s = env.reset()
        i = 0
        episode = 1
        steps_survived = 0
        accumulated_reward = 0
        while True:
            estimated_rewards = self.infer(s)
            a = self.ex_choose(list(estimated_rewards), lambda x: np.argmax(estimated_rewards).item())
            s_, r, final = env.step(a)
            self.rp_add(ReplayBufferEntry(s, s_, a, r, final))
            accumulated_reward += r
            s = s_

            self.call_step_callbacks(TrainingStep(i, episode))

            if i % self.hyper_params.learn_every == 0 and i != 0 and self.rp_get_length() >= tp.batch_size:
                batch = self.rp_sample(tp.batch_size)
                self.learn(batch)

            if i % self.hyper_params.ensure_every == 0:
                self.ensure_learning()

            if final:
                training_progress = TrainingProgress(i, episode, steps_survived, accumulated_reward)
                must_exit = self.call_progress_callbacks(training_progress)
                if episode >= tp.episodes or must_exit:
                    return

                episode += 1
                accumulated_reward = 0
                steps_survived = 0
                s = env.reset()
            else:
                steps_survived += 1
            env.render()
            i += 1
