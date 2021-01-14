from abc import ABC
import typing as T
import torch
import numpy as np

from ..explorers import AnyExplorer
from environments import Environment
from ..replay_buffers import AnyReplayBuffer, ReplayBufferEntry
from agents.agents.base.models import ACHyperParams, TrainingProgress, TrainingParams, LearningStep, TrainingStep
from agents.agents.base.agent import Agent


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
        last_layer = list(model.modules())[-1]
        if not isinstance(last_layer, torch.nn.Linear):
            raise ValueError("the model you have created must have a torch.nn.Linear in the last layer")
        self.linear = torch.nn.Linear(last_layer.out_features, 1)

    def forward(self, x):
        x = self.model(x)
        return self.linear(x)


class AdvantageActorCriticAgent(Agent, ABC):
    hp: ACHyperParams

    def __init__(self,
                 action_space: int,
                 hp: ACHyperParams,
                 tp: TrainingParams,
                 explorer: T.Union[AnyExplorer, None],
                 replay_buffer: AnyReplayBuffer,
                 use_gpu: bool = True):
        super(AdvantageActorCriticAgent, self).__init__(action_space, hp, tp, explorer, replay_buffer, use_gpu)

        self.actor = self.build_actor(self.model_factory().to(self.device)).to(self.device)
        self.critic = self.build_critic(self.model_factory().to(self.device)).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=hp.a_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=hp.c_lr)
        self.loss_f = torch.nn.MSELoss(reduction="none").to(self.device)

    @staticmethod
    def build_actor(model: torch.nn.Module):
        return Actor(model)

    @staticmethod
    def build_critic(model: torch.nn.Module):
        return Critic(model)

    def postprocess(self, t: torch.Tensor) -> np.ndarray:
        return np.array(t.squeeze(0))

    def infer(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.postprocess(self.actor.forward(self.preprocess(x).to(self.device)).cpu())

    def learn(self, batch: T.List[ReplayBufferEntry]) -> None:
        batch_s = torch.cat([self.preprocess(m.s) for m in batch], 0).to(self.device).requires_grad_(True)
        batch_s_ = torch.cat([self.preprocess(m.s_) for m in batch], 0).to(self.device)
        batch_a = torch.tensor([m.a for m in batch], device=self.device)
        batch_r = torch.tensor([[m.r] for m in batch], dtype=torch.float32, device=self.device)
        batch_finals = torch.tensor([[int(not m.final)] for m in batch], device=self.device)
        batch_weights = torch.tensor([m.weight for m in batch], device=self.device)

        action_probabilities: torch.Tensor = self.actor(batch_s)
        state_values: torch.Tensor = self.critic(batch_s)
        with torch.no_grad():
            next_state_values: torch.Tensor = self.critic(batch_s_)
            estimated_q_value: torch.Tensor = self.hp.gamma * next_state_values * batch_finals + batch_r

        advantages: torch.Tensor = (estimated_q_value - state_values.clone().detach()).squeeze(1)
        chosen_action_log_probabilities: torch.Tensor = torch.stack(
            [torch.distributions.Categorical(p).log_prob(a) for p, a in zip(action_probabilities, batch_a)])
        actor_loss: torch.Tensor = (-chosen_action_log_probabilities * advantages * batch_weights).mean()
        element_wise_critic_loss: torch.Tensor = self.loss_f(state_values, estimated_q_value)
        critic_loss = (element_wise_critic_loss * batch_weights).mean()
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        self.call_learn_callbacks(LearningStep(batch, [v.item() for v in state_values], [v.item() for v in estimated_q_value]))

    def train(self, env: Environment) -> None:
        s = env.reset()
        i = 0
        episode = 1
        steps_survived = 0
        accumulated_reward = 0
        is_healthy = False
        while True:
            estimated_rewards = self.infer(s)
            def choosing_f(x): return torch.distributions.Categorical(torch.tensor(x)).sample().item()
            a = self.explorer.choose(estimated_rewards, choosing_f) if self.explorer else choosing_f(estimated_rewards)
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
                    self.call_healthy_callbacks(env)

            if final:
                must_exit = self.call_progress_callbacks(TrainingProgress(episode, steps_survived, accumulated_reward))
                if episode >= self.tp.episodes or must_exit:
                    return

                accumulated_reward = 0
                steps_survived = 0
                episode += 1
                s = env.reset()

            else:
                steps_survived += 1
            env.render()
            i += 1
