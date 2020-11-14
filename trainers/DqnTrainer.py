import typing as T
from .trainer import Trainer
from agents import DqnAgent
from replay_buffers import ReplayBufferEntry
from .models import TrainingProgress


class DqnTrainer(Trainer):
    agent: DqnAgent

    def train(self, finish_condition: T.Callable[[TrainingProgress], bool]) -> None:
        s = self.env.reset()
        i = 0
        tries = 0
        steps_survived = 0
        accumulated_reward = 0
        max_reward: T.Union[None, float] = None
        reward_record: T.List[float] = []
        while True:

            estimated_rewards = self.agent.infer(s)
            a = self.explorer.choose(estimated_rewards)
            s_, r, final = self.env.step(a)
            self.replay_buffer.add(ReplayBufferEntry(s, s_, a, r, final))
            accumulated_reward += r
            s = s_

            if i % self.training_params.learn_every == 0 and i != 0:
                batch = self.replay_buffer.sample(self.training_params.batch_size)
                self.agent.learn(batch)

            if i % self.training_params.ensure_every == 0:
                self.agent.ensure_learning()

            if final:
                tries += 1
                tp = TrainingProgress(tries, steps_survived, accumulated_reward)
                reward_record.append(accumulated_reward)
                max_reward = accumulated_reward if max_reward is None or accumulated_reward > max_reward else max_reward
                self.progress_callback(tp)
                if finish_condition(tp):
                    return

                accumulated_reward = 0
                steps_survived = 0
                s = self.env.reset()

            else:
                steps_survived += 1
            self.env.render()
            i += 1
