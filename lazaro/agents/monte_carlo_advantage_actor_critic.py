import typing as T
from abc import ABC

import numpy as np
import torch

from ..environments import Environment
from .a2c import A2cAgent
from .base.models import TrainingProgress, LearningStep, TrainingStep, TrainingParams
from .replay_buffers import ReplayBufferEntry


class MonteCarloA2c(A2cAgent, ABC):
    def learn(self, batch: T.List[ReplayBufferEntry]) -> None:
        batch_s = torch.cat([self.preprocess(m.s) for m in batch], 0).to(self.device).requires_grad_(True)
        batch_a = torch.tensor([m.a for m in batch], device=self.device)
        batch_rt = torch.tensor([[m.r] for m in batch], dtype=torch.float32, device=self.device)
        batch_weights = torch.tensor([m.weight for m in batch], device=self.device)

        action_probabilities, state_values = self.actor_critic(batch_s)

        advantages: torch.Tensor = (batch_rt - state_values.clone().detach()).squeeze(1)
        chosen_action_log_probabilities: torch.Tensor = torch.stack(
            [torch.distributions.Categorical(p).log_prob(a) for p, a in zip(action_probabilities, batch_a)])
        actor_loss: torch.Tensor = -chosen_action_log_probabilities * advantages * batch_weights
        critic_loss: torch.Tensor = self.loss_f(state_values, batch_rt) * batch_weights
        loss = (actor_loss + critic_loss).mean()
        self.actor_critic_optimizer.zero_grad()
        loss.backward()
        self.actor_critic_optimizer.step()
        self.call_learn_callbacks(LearningStep(batch, [v.item() for v in state_values], [v.item() for v in batch_rt]))

    def train(self, env: Environment, tp: TrainingParams = TrainingParams()) -> None:
        self.health_check(env)
        self.accumulate_rewards = False

        s = env.reset()
        i = 0
        episode = 1
        steps_survived = 0
        accumulated_reward = 0
        steps_record: T.List[ReplayBufferEntry] = []
        while True:
            estimated_rewards = self.infer(s)
            def choosing_f(x): return torch.distributions.Categorical(torch.tensor(x)).sample().item()
            a = self.ex_choose(list(estimated_rewards), choosing_f)
            s_, r, final = env.step(a)
            steps_record.append(ReplayBufferEntry(s, s_, a, r, final))
            accumulated_reward += r
            s = s_

            self.call_step_callbacks(TrainingStep(i, episode))

            if i % self.hyper_params.learn_every == 0 and i != 0 and self.rp_get_length() >= tp.batch_size:
                batch = self.rp_sample(tp.batch_size)
                self.learn(batch)

            if final:
                discounted_r = 0
                reward_array = np.zeros((len(steps_record)))
                for j, step in enumerate(steps_record[::-1]):
                    discounted_r = step.r + self.hyper_params.gamma * discounted_r
                    step.r = discounted_r
                    reward_array[j] = discounted_r

                mean, std, eps = reward_array.mean(), reward_array.std(), np.finfo(np.float32).eps.item()
                for step in steps_record:
                    step.r = (step.r - mean) / (std + eps)
                    self.rp_add(step)

                training_progress = TrainingProgress(i, episode, steps_survived, accumulated_reward)
                must_exit = self.call_progress_callbacks(training_progress)
                if episode >= tp.episodes or must_exit:
                    return

                accumulated_reward = 0
                steps_survived = 0
                episode += 1
                steps_record.clear()
                s = env.reset()

            else:
                steps_survived += 1
            env.render()
            i += 1
