from abc import ABC
import typing as T
import torch
import numpy as np
from replay_buffers import ReplayBufferEntry
from .models import HyperParams
from .agent import Agent


class ActorCriticAgent(Agent, ABC):
    def __init__(self, hp: HyperParams, use_gpu: bool = True):
        super().__init__(hp, use_gpu)

        self.actor = self.build_actor(self.model_factory()).to(self.device)
        self.critic = self.build_critic(self.model_factory()).to(self.device)
        self.actor_optimizer = torch.optim.RMSprop(self.actor.parameters(), lr=hp.lr)
        self.critic_optimizer = torch.optim.RMSprop(self.critic.parameters(), lr=hp.lr)
        self.loss_f = torch.nn.MSELoss().to(self.device)
        self.gamma = hp.gamma

    @staticmethod
    def build_actor(model: torch.nn.Module):
        class Actor(torch.nn.Module):
            def __init__(self):
                super(Actor, self).__init__()
                self.model = model
                self.softmax = torch.nn.Softmax(1)

            def forward(self, x):
                x = self.model(x)
                return self.softmax(x)

        return Actor()

    @staticmethod
    def build_critic(model: torch.nn.Module):
        class Critic(torch.nn.Module):
            def __init__(self):
                super(Critic, self).__init__()
                self.model = model
                assert hasattr(self.model, "out_size"), AttributeError("model returned from .model_factory() must have the attribute .out_size being an integer determining the output size (action space) of the network")
                self.linear = torch.nn.Linear(self.model.out_size, 1)

            def forward(self, x):
                x = self.model(x)
                return self.linear(x)

        return Critic()

    def postprocess(self, t: torch.Tensor) -> np.ndarray:
        return np.array(t.squeeze(0))

    def infer(self, x: np.ndarray) -> np.ndarray:
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
        actor_loss: torch.Tensor = (-chosen_action_log_probabilities * advantages).sum()
        critic_loss: torch.Tensor = self.loss_f(state_values, batch_r)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
