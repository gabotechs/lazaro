import typing as T
import torch
import numpy as np
from .trainer import Trainer
from agents import ActorCriticAgent
from replay_buffers import ReplayBufferEntry
from .models import TrainingProgress


class ActorCriticTrainer(Trainer):
    agent: ActorCriticAgent

    def train(self, finish_condition: T.Callable[[TrainingProgress], bool]) -> None:
        s = self.env.reset()
        i = 0
        tries = 0
        steps_survived = 0
        accumulated_reward = 0
        accumulated_reward_record: T.List[float] = []
        steps_record: T.List[ReplayBufferEntry] = []
        while True:
            estimated_rewards = self.agent.infer(s)
            def choosing_f(x): return torch.distributions.Categorical(torch.tensor(x)).sample().item()
            a = self.explorer.choose(estimated_rewards, choosing_f) if self.explorer else choosing_f(estimated_rewards)
            s_, r, final = self.env.step(a)
            steps_record.append(ReplayBufferEntry(s, s_, a, r, final))
            accumulated_reward += r
            s = s_

            if i % self.training_params.learn_every == 0 and i != 0 and len(self.replay_buffer) >= self.training_params.batch_size:
                batch = self.replay_buffer.sample(self.training_params.batch_size)
                self.agent.learn(batch)

            if final:
                discounted_r = 0
                reward_array = np.zeros((len(steps_record)))
                for i, step in enumerate(steps_record[::-1]):
                    discounted_r = step.r + self.agent.gamma * discounted_r
                    step.r = discounted_r
                    reward_array[i] = discounted_r

                mean, std, eps = reward_array.mean(), reward_array.std(), np.finfo(np.float32).eps.item()
                for step in steps_record:
                    step.r = (step.r - mean) / (std + eps)
                    self.replay_buffer.add(step)

                tries += 1
                tp = TrainingProgress(tries, steps_survived, accumulated_reward)
                accumulated_reward_record.append(accumulated_reward)
                self.progress_callback(tp)
                if finish_condition(tp):
                    return

                accumulated_reward = 0
                steps_survived = 0
                steps_record.clear()
                s = self.env.reset()

            else:
                steps_survived += 1
            self.env.render()
            i += 1
