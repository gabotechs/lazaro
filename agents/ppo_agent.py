from abc import ABC
import typing as T
import torch
from .base.models import LearningStep, PpoHyperParams, TrainingParams, TrainingStep
from .explorers import AnyExplorer
from .replay_buffers import AnyReplayBuffer, ReplayBufferEntry
from .monte_carlo_advantage_actor_critic_agent import MonteCarloA2cCriticAgent


class PpoAgent(MonteCarloA2cCriticAgent, ABC):
    hp: PpoHyperParams

    def __init__(self,
                 action_space: int,
                 explorer: AnyExplorer,
                 replay_buffer: AnyReplayBuffer,
                 tp: TrainingParams,
                 hp: PpoHyperParams = PpoHyperParams(),
                 use_gpu: bool = True,
                 tensor_board_log: bool = True):
        super(MonteCarloA2cCriticAgent, self).__init__(action_space, explorer, replay_buffer, tp, hp, use_gpu, tensor_board_log)
        self.actor_critic_new = self.build_actor_critic().to(self.device).eval()
        self.actor_critic_new_optimizer = torch.optim.Adam(self.actor_critic_new.parameters(), lr=hp.lr)
        self.actor_critic_new.load_state_dict(self.actor_critic.state_dict())
        self.add_step_callback(self.ensure_learning_step_callback)

    def get_state_dict(self) -> dict:
        state_dict = super(PpoAgent, self).get_state_dict()
        state_dict.update({"actor_critic_new": self.actor_critic_new.state_dict()})
        return state_dict

    def ensure_learning_step_callback(self, training_step: TrainingStep) -> None:
        if training_step.step % self.hp.ensure_every == 0:
            self.actor_critic.load_state_dict(self.actor_critic_new.state_dict())

    def learn(self, batch: T.List[ReplayBufferEntry]) -> None:
        batch_s = torch.cat([self.preprocess(m.s) for m in batch], 0).to(self.device).requires_grad_(True)
        batch_a = torch.tensor([m.a for m in batch], device=self.device)
        batch_rt = torch.tensor([[m.r] for m in batch], dtype=torch.float32, device=self.device)
        batch_weights = torch.tensor([m.weight for m in batch], device=self.device)

        new_action_probabilities, state_values = self.actor_critic_new(batch_s)

        new_chosen_action_log_probabilities_list = []
        new_chosen_action_log_entropy_list = []

        for p, a in zip(new_action_probabilities, batch_a):
            dist = torch.distributions.Categorical(p)
            new_chosen_action_log_probabilities_list.append(dist.log_prob(a))
            new_chosen_action_log_entropy_list.append(dist.entropy())

        new_chosen_action_log_probabilities: torch.Tensor = torch.stack(new_chosen_action_log_probabilities_list)
        new_chosen_action_log_entropy: torch.Tensor = torch.stack(new_chosen_action_log_entropy_list)

        with torch.no_grad():
            old_action_probabilities, _ = self.actor_critic(batch_s)
            old_chosen_action_log_probabilities: torch.Tensor = torch.stack(
                [torch.distributions.Categorical(p).log_prob(a) for p, a in zip(old_action_probabilities, batch_a)])

        advantages: torch.Tensor = (batch_rt - state_values.detach()).squeeze(1)

        ratios = torch.exp(new_chosen_action_log_probabilities-old_chosen_action_log_probabilities)
        surrogate_loss_1 = ratios * advantages
        surrogate_loss_2 = torch.clamp(ratios, 1 - self.hp.clip_factor, 1 + self.hp.clip_factor) * advantages

        actor_loss: torch.Tensor = -torch.min(surrogate_loss_1, surrogate_loss_2) * batch_weights
        critic_loss: torch.Tensor = self.loss_f(state_values, batch_rt) * batch_weights
        entropy_loss: torch.Tensor = -self.hp.entropy_factor * new_chosen_action_log_entropy * batch_weights

        loss = (actor_loss + critic_loss + entropy_loss).mean()

        self.actor_critic_new_optimizer.zero_grad()
        loss.backward()
        self.actor_critic_new_optimizer.step()
        self.call_learn_callbacks(LearningStep(batch, [v.item() for v in state_values], [v.item() for v in batch_rt]))
