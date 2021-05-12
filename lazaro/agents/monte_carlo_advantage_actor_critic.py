import typing as T
from abc import ABC

import numpy as np
import torch

from .a2c import A2cAgent
from .base.models import TrainingProgress, LearningStep, TrainingStep, TrainingParams, ReplayBufferEntry
from ..environments import Environment


class MonteCarloA2c(A2cAgent, ABC):
    def learn(self, entries: T.List[ReplayBufferEntry]) -> None:
        batch = self.form_learning_batch(entries)
        batch.r = batch.r.unsqueeze(1)

        action_probabilities, state_values = self.actor_critic(batch.s)

        advantages: torch.Tensor = (batch.r - state_values.clone().detach()).squeeze(1)
        chosen_action_log_probabilities: torch.Tensor = torch.stack(
            [torch.distributions.Categorical(p).log_prob(a) for p, a in zip(action_probabilities, batch.a)])
        actor_loss: torch.Tensor = -chosen_action_log_probabilities * advantages * batch.weight
        critic_loss: torch.Tensor = self.loss_f(state_values, batch.r) * batch.weight
        loss = (actor_loss + critic_loss).mean()
        self.actor_critic_optimizer.zero_grad()
        loss.backward()
        self.actor_critic_optimizer.step()
        self.call_learn_callbacks(LearningStep(entries, [v.item() for v in state_values], [v.item() for v in batch.r]))

    def train(self, env: Environment, tp: TrainingParams = None) -> None:
        if tp is None:
            tp = self.default_training_params
        self.health_check(env)
        self.accumulate_rewards = False

        s = env.reset()
        i = 0
        episode = 1
        steps_survived = 0
        accumulated_reward = 0
        steps_record: T.List[ReplayBufferEntry] = []
        while True:
            estimated_rewards = self.act(s)
            def choosing_f(x): return torch.distributions.Categorical(torch.tensor(x)).sample().item()
            a = self.ex_choose(list(estimated_rewards), choosing_f)
            s_, r, final = env.step(a)
            steps_record.append(ReplayBufferEntry(s, s_, a, r, final))
            accumulated_reward += r
            s = s_

            self.call_step_callbacks(TrainingStep(i, episode))

            if i % self.agent_params.learn_every == 0 and i != 0 and self.rp_get_length() >= tp.batch_size:
                batch = self.rp_sample(tp.batch_size)
                self.learn(batch)

            if final:
                discounted_r = 0
                reward_array = np.zeros((len(steps_record)))
                for j, step in enumerate(steps_record[::-1]):
                    discounted_r = step.r + self.agent_params.gamma * discounted_r
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
