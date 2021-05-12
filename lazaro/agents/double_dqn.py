import typing as T
from abc import ABC

import numpy as np
import torch

from ..environments import Environment
from .base.models import DoubleDqnHyperParams, TrainingProgress, LearningStep, TrainingStep, TrainingParams, ReplayBufferEntry
from .dqn import DqnAgent


class DoubleDqnAgent(DqnAgent, ABC):
    def __init__(self,
                 action_space: int,
                 agent_params: DoubleDqnHyperParams = DoubleDqnHyperParams(),
                 use_gpu: bool = True):
        super(DoubleDqnAgent, self).__init__(action_space, agent_params, use_gpu)
        self.agent_params = agent_params
        self.action_evaluator = self.build_model().to(self.device).eval()

    def get_state_dict(self) -> dict:
        state_dict = super(DoubleDqnAgent, self).get_state_dict()
        state_dict.update({"action_evaluator": self.action_evaluator.state_dict()})
        return state_dict

    def ensure_learning(self) -> None:
        self.action_evaluator.load_state_dict(self.action_estimator.state_dict())

    def learn(self, entries: T.List[ReplayBufferEntry]) -> None:
        batch = self.form_learning_batch(entries)

        actions_estimated_values: torch.Tensor = self.action_estimator(batch.s)
        with torch.no_grad():
            actions_expected_values: torch.Tensor = self.action_evaluator(batch.s_)

        x = torch.stack([t_s[t_a.item()] for t_s, t_a in zip(actions_estimated_values, batch.a)])
        y = torch.max(actions_expected_values, 1)[0] * self.agent_params.gamma * batch.final + batch.r
        element_wise_loss = self.loss_f(x, y)
        loss = (element_wise_loss * batch.weight).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.call_learn_callbacks(LearningStep(entries, [v.item() for v in x], [v.item() for v in y]))

    def train(self, env: Environment, tp: TrainingParams = None) -> None:
        if tp is None:
            tp = self.default_training_params
        self.health_check(env)
        s = env.reset()
        i = 0
        episode = 1
        steps_survived = 0
        accumulated_reward = 0
        while True:
            estimated_rewards = self.act(s)
            a = self.ex_choose(list(estimated_rewards), lambda x: np.argmax(estimated_rewards).item())
            s_, r, final = env.step(a)
            self.rp_add(ReplayBufferEntry(s, s_, a, r, final))
            accumulated_reward += r
            s = s_

            self.call_step_callbacks(TrainingStep(i, episode))

            if i % self.agent_params.learn_every == 0 and i != 0 and self.rp_get_length() >= tp.batch_size:
                batch = self.rp_sample(tp.batch_size)
                self.learn(batch)

            if i % self.agent_params.ensure_every == 0:
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
