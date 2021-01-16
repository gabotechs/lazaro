from abc import ABC
import typing as T
import torch
import numpy as np
from environments import Environment
from .replay_buffers import ReplayBufferEntry, NStepsPrioritizedReplayBuffer, NStepsRandomReplayBuffer
from .base.models import TrainingProgress, LearningStep, TrainingStep
from .advantage_actor_critic_agent import AdvantageActorCriticAgent


class MonteCarloAdvantageActorCriticAgent(AdvantageActorCriticAgent, ABC):
    def learn(self, batch: T.List[ReplayBufferEntry]) -> None:
        batch_s = torch.cat([self.preprocess(m.s) for m in batch], 0).to(self.device).requires_grad_(True)
        batch_a = torch.tensor([m.a for m in batch], device=self.device)
        batch_rt = torch.tensor([[m.r] for m in batch], dtype=torch.float32, device=self.device)
        batch_weights = torch.tensor([m.weight for m in batch], device=self.device)

        action_probabilities: torch.Tensor = self.actor(batch_s)
        state_values: torch.Tensor = self.critic(batch_s)

        advantages: torch.Tensor = (batch_rt - state_values.clone().detach()).squeeze(1)
        chosen_action_log_probabilities: torch.Tensor = torch.stack(
            [torch.distributions.Categorical(p).log_prob(a) for p, a in zip(action_probabilities, batch_a)])
        actor_loss: torch.Tensor = (-chosen_action_log_probabilities * advantages * batch_weights).mean()
        element_wise_critic_loss: torch.Tensor = self.loss_f(state_values, batch_rt)
        critic_loss = (element_wise_critic_loss * batch_weights).mean()
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        self.call_learn_callbacks(LearningStep(batch, [v.item() for v in state_values], [v.item() for v in batch_rt]))

    def train(self, env: Environment) -> None:
        self.health_check(env)
        if isinstance(self.replay_buffer, (NStepsPrioritizedReplayBuffer, NStepsRandomReplayBuffer)):
            self.replay_buffer.accumulate_rewards = False
        s = env.reset()
        i = 0
        episode = 1
        steps_survived = 0
        accumulated_reward = 0
        steps_record: T.List[ReplayBufferEntry] = []
        while True:
            estimated_rewards = self.infer(s)
            def choosing_f(x): return torch.distributions.Categorical(torch.tensor(x)).sample().item()
            a = self.explorer.choose(estimated_rewards, choosing_f) if self.explorer else choosing_f(estimated_rewards)
            s_, r, final = env.step(a)
            steps_record.append(ReplayBufferEntry(s, s_, a, r, final))
            accumulated_reward += r
            s = s_

            self.call_step_callbacks(TrainingStep(i, steps_survived, episode))

            if i % self.tp.learn_every == 0 and i != 0 and len(self.replay_buffer) >= self.tp.batch_size:
                batch = self.replay_buffer.sample(self.tp.batch_size)
                self.learn(batch)

            if final:
                discounted_r = 0
                reward_array = np.zeros((len(steps_record)))
                for i, step in enumerate(steps_record[::-1]):
                    discounted_r = step.r + self.hp.gamma * discounted_r
                    step.r = discounted_r
                    reward_array[i] = discounted_r

                mean, std, eps = reward_array.mean(), reward_array.std(), np.finfo(np.float32).eps.item()
                for step in steps_record:
                    step.r = (step.r - mean) / (std + eps)
                    self.replay_buffer.add(step)

                must_exit = self.call_progress_callbacks(TrainingProgress(episode, steps_survived, accumulated_reward))
                if episode >= self.tp.episodes or must_exit:
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
