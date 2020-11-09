import typing as T
from explorer import RandomExplorer
from dqn_agent import DqnAgent
from dqn_memory_agent import DqnMemoryAgent
from .models import TrainingParams, TrainingProgress
from environments import Environment


class Trainer:
    def __init__(self,
                 env: Environment,
                 agent: T.Union[DqnMemoryAgent, DqnAgent],
                 explorer: RandomExplorer,
                 training_params: TrainingParams):

        self.env = env
        self.training_params: TrainingParams = training_params
        self.explorer: RandomExplorer = explorer
        self.agent: T.Union[DqnMemoryAgent, DqnAgent] = agent

        self.progress_callback: T.Callable[[TrainingProgress], None] = lambda x: None

    def set_progress_callback(self, cbk: T.Callable[[TrainingProgress], None]):
        self.progress_callback = cbk

    def train(self, finish_condition: T.Callable[[TrainingProgress], bool]) -> None:
        s = self.env.reset()
        i = 0
        tries = 0
        m = None if isinstance(self.agent, DqnAgent) else self.agent.memory_init()
        steps_survived = 0
        accumulated_reward = 0
        max_reward: T.Union[None, float] = None
        reward_record: T.List[float] = []
        while True:
            if isinstance(self.agent, DqnAgent):
                estimated_rewards = self.agent.infer(s)
                a = self.explorer.choose(estimated_rewards)
                s_, r, final = self.env.step(a)
                self.agent.memorize(s, s_, a, r, final)
            elif isinstance(self.agent, DqnMemoryAgent):
                estimated_rewards, m_ = self.agent.infer(s, m)
                a = self.explorer.choose(estimated_rewards)
                s_, r, final = self.env.step(a)
                self.agent.memorize(s, m, s_, a, r, final)
                m = m_
            else:
                raise NotImplementedError("agent not implemented")

            accumulated_reward += r
            s = s_

            if i % self.training_params.learn_every == 0 and i != 0:
                self.agent.learn(self.training_params.batch_size)

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
                m = None if isinstance(self.agent, DqnAgent) else self.agent.memory_init()
            else:
                steps_survived += 1
            self.env.render()
            i += 1
