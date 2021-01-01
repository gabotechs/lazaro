from abc import ABC
import typing as T
import torch
import numpy as np

from environments import Environment
from ..explorers import AnyExplorer
from ..replay_buffers import ReplayBufferEntry, AnyReplayBuffer
from .models import DoubleDqnHyperParams, TrainingParams, TrainingProgress, LearningStep, TrainingStep
from .dqn_agent import DqnAgent


class DoubleDqnAgent(DqnAgent, ABC):
    hp: DoubleDqnHyperParams

    def __init__(self,
                 action_space: int,
                 hp: DoubleDqnHyperParams,
                 tp: TrainingParams,
                 explorer: T.Union[AnyExplorer, None],
                 replay_buffer: AnyReplayBuffer,
                 use_gpu: bool = True):
        super(DoubleDqnAgent, self).__init__(action_space, hp, tp, explorer, replay_buffer, use_gpu)
        self.action_evaluator = self.build_model().to(self.device).eval()

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
        self.call_learn_callbacks(LearningStep(batch, [v.item() for v in x], [v.item() for v in y]))

    def train(self, env: Environment) -> None:
        s = env.reset()
        i = 0
        episode = 1
        steps_survived = 0
        accumulated_reward = 0
        is_healthy = False
        while True:
            estimated_rewards = self.infer(s)
            a = self.explorer.choose(estimated_rewards, lambda x: np.argmax(estimated_rewards).item())
            s_, r, final = env.step(a)
            self.replay_buffer.add(ReplayBufferEntry(s, s_, a, r, final))
            accumulated_reward += r
            s = s_

            self.call_step_callbacks(TrainingStep(i, steps_survived, episode))

            if i % self.tp.learn_every == 0 and i != 0 and len(self.replay_buffer) >= self.tp.batch_size:
                batch = self.replay_buffer.sample(self.tp.batch_size)
                self.learn(batch)
                if not is_healthy:
                    is_healthy = True
                    self.call_healthy_callbacks()

            if i % self.hp.ensure_every == 0:
                self.ensure_learning()

            if final:
                self.call_progress_callbacks(TrainingProgress(episode, steps_survived, accumulated_reward))
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
