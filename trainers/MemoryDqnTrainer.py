import typing as T
from agents import DqnMemoryAgent
from .trainer import Trainer
from .models import TrainingProgress
from replay_buffers import MemoryReplayBufferEntry


class MemoryDqnTrainer(Trainer):
    agent: DqnMemoryAgent

    def train(self, finish_condition: T.Callable[[TrainingProgress], bool]) -> None:
        s = self.env.reset()
        i = 0
        tries = 0
        m = self.agent.memory_init()
        steps_survived = 0
        accumulated_reward = 0
        max_reward: T.Union[None, float] = None
        reward_record: T.List[float] = []
        while True:
            estimated_rewards, m_ = self.agent.infer(s, m)
            a = self.explorer.choose(estimated_rewards)
            s_, r, final = self.env.step(a)
            self.replay_buffer.add(MemoryReplayBufferEntry(s, m, s_, a, r, final))
            m = m_

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
                m = self.agent.memory_init()
            else:
                steps_survived += 1
            self.env.render()
            i += 1
