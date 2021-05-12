import typing as T
from abc import ABC

import torch

from .base.models import LearningStep, PpoHyperParams, TrainingStep, ReplayBufferEntry
from .monte_carlo_advantage_actor_critic import MonteCarloA2c


class PpoAgent(MonteCarloA2c, ABC):
    def __init__(self,
                 action_space: int,
                 agent_params: PpoHyperParams = PpoHyperParams(),
                 use_gpu: bool = True):
        super(MonteCarloA2c, self).__init__(action_space, agent_params, use_gpu)
        self.hyper_params = agent_params
        self.actor_critic_new = self.build_model().to(self.device).eval()
        self.actor_critic_new_optimizer = torch.optim.Adam(self.actor_critic_new.parameters(), lr=agent_params.lr)
        self.actor_critic_new.load_state_dict(self.actor_critic.state_dict())
        self.add_step_callback("ppo agent update state", self.ensure_learning_step_callback)

    def get_state_dict(self) -> dict:
        state_dict = super(PpoAgent, self).get_state_dict()
        state_dict.update({"actor_critic_new": self.actor_critic_new.state_dict()})
        return state_dict

    def ensure_learning_step_callback(self, training_step: TrainingStep) -> None:
        if training_step.step % self.hyper_params.ensure_every == 0:
            self.actor_critic.load_state_dict(self.actor_critic_new.state_dict())

    def learn(self, entries: T.List[ReplayBufferEntry]) -> None:
        batch = self.form_learning_batch(entries)
        batch.r = batch.r.unsqueeze(1)

        new_action_probabilities, state_values = self.actor_critic_new(batch.s)

        new_chosen_action_log_probabilities_list = []
        new_chosen_action_log_entropy_list = []

        for p, a in zip(new_action_probabilities, batch.a):
            dist = torch.distributions.Categorical(p)
            new_chosen_action_log_probabilities_list.append(dist.log_prob(a))
            new_chosen_action_log_entropy_list.append(dist.entropy())

        new_chosen_action_log_probabilities: torch.Tensor = torch.stack(new_chosen_action_log_probabilities_list)
        new_chosen_action_log_entropy: torch.Tensor = torch.stack(new_chosen_action_log_entropy_list)

        with torch.no_grad():
            old_action_probabilities, _ = self.actor_critic(batch.s)
            old_chosen_action_log_probabilities: torch.Tensor = torch.stack(
                [torch.distributions.Categorical(p).log_prob(a) for p, a in zip(old_action_probabilities, batch.a)])

        advantages: torch.Tensor = (batch.r - state_values.detach()).squeeze(1)

        ratios = torch.exp(new_chosen_action_log_probabilities-old_chosen_action_log_probabilities)
        surrogate_loss_1 = ratios * advantages
        surrogate_loss_2 = torch.clamp(ratios, 1 - self.hyper_params.clip_factor, 1 + self.hyper_params.clip_factor) * advantages

        actor_loss: torch.Tensor = -torch.min(surrogate_loss_1, surrogate_loss_2) * batch.weight
        critic_loss: torch.Tensor = self.loss_f(state_values, batch.r) * batch.weight
        entropy_loss: torch.Tensor = -self.hyper_params.entropy_factor * new_chosen_action_log_entropy * batch.weight

        loss = (actor_loss + critic_loss + entropy_loss).mean()

        self.actor_critic_new_optimizer.zero_grad()
        loss.backward()
        self.actor_critic_new_optimizer.step()
        self.call_learn_callbacks(LearningStep(entries, [v.item() for v in state_values], [v.item() for v in batch.r]))
