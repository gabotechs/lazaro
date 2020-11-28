from abc import ABC
import typing as T
import torch
import numpy as np
from agents.explorers import Explorer
from environments import Environment
from agents.replay_buffers import ReplayBuffer, ReplayBufferEntry
from .models import ACHyperParams, TrainingProgress, TrainingParams
from .agent import Agent


class Actor(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super(Actor, self).__init__()
        self.model = model
        self.softmax = torch.nn.Softmax(1)

    def forward(self, x):
        x = self.model(x)
        return self.softmax(x)


class Critic(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super(Critic, self).__init__()
        self.model = model
        if not hasattr(self.model, "out_size"):
            AttributeError("model returned from .model_factory() must have the attribute .out_size being "
                           "an integer determining the output size (action space) of the network")
        self.linear = torch.nn.Linear(self.model.out_size, 1)

    def forward(self, x):
        x = self.model(x)
        return self.linear(x)


class ActorCriticAgent(Agent, ABC):
    def __init__(self,
                 hp: ACHyperParams,
                 tp: TrainingParams,
                 explorer: T.Union[Explorer, None],
                 replay_buffer: ReplayBuffer,
                 use_gpu: bool = True):
        super(ActorCriticAgent, self).__init__(hp, tp, explorer, replay_buffer, use_gpu)

        self.actor = self.build_actor(self.model_factory()).to(self.device)
        self.critic = self.build_critic(self.model_factory()).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=hp.a_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=hp.lr)
        self.loss_f = torch.nn.MSELoss().to(self.device)

    @staticmethod
    def build_actor(model: torch.nn.Module):
        return Actor(model)

    @staticmethod
    def build_critic(model: torch.nn.Module):
        return Critic(model)

    def postprocess(self, t: torch.Tensor) -> np.ndarray:
        return np.array(t.squeeze(0))

    def infer(self, x: np.ndarray) -> np.ndarray:
        if self.infer_callback:
            self.infer_callback()
        with torch.no_grad():
            return self.postprocess(self.actor.forward(self.preprocess(x).to(self.device)).cpu())

    def learn(self, batch: T.List[ReplayBufferEntry]) -> None:
        batch_s = torch.cat([self.preprocess(m.s) for m in batch], 0).to(self.device).requires_grad_(True)
        batch_a = torch.tensor([m.a for m in batch], device=self.device)
        batch_r = torch.tensor([[m.r] for m in batch], dtype=torch.float32, device=self.device)

        action_probabilities: torch.Tensor = self.actor(batch_s)
        state_values: torch.Tensor = self.critic(batch_s)

        advantages: torch.Tensor = (batch_r - state_values.clone().detach()).squeeze(1)
        chosen_action_log_probabilities: torch.Tensor = torch.stack([torch.distributions.Categorical(p).log_prob(a) for p, a in zip(action_probabilities, batch_a)])
        actor_loss: torch.Tensor = (-chosen_action_log_probabilities * advantages).mean()
        critic_loss: torch.Tensor = self.loss_f(state_values, batch_r)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def train(self, env: Environment) -> None:
        s = env.reset()
        i = 0
        tries = 0
        steps_survived = 0
        accumulated_reward = 0
        accumulated_reward_record: T.List[float] = []
        steps_record: T.List[ReplayBufferEntry] = []
        while True:
            estimated_rewards = self.infer(s)
            def choosing_f(x): return torch.distributions.Categorical(torch.tensor(x)).sample().item()
            a = self.explorer.choose(estimated_rewards, choosing_f) if self.explorer else choosing_f(estimated_rewards)
            s_, r, final = env.step(a)
            steps_record.append(ReplayBufferEntry(s, s_, a, r, final))
            accumulated_reward += r
            s = s_

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

                tries += 1
                tp = TrainingProgress(tries, steps_survived, accumulated_reward)
                accumulated_reward_record.append(accumulated_reward)
                if self.progress_callback:
                    self.progress_callback(tp)
                if self.tp.finish_condition(tp):
                    return

                accumulated_reward = 0
                steps_survived = 0
                steps_record.clear()
                s = env.reset()

            else:
                steps_survived += 1
            env.render()
            i += 1
