from abc import ABC
import typing as T
import torch
import torch.nn.functional as F
import numpy as np

from .explorers import AnyExplorer
from .explorers.noisy_explorer import NoisyLinear
from environments import Environment
from .replay_buffers import AnyReplayBuffer, ReplayBufferEntry
from .base.models import A2CHyperParams, TrainingProgress, TrainingParams, LearningStep, TrainingStep
from .base.agent import Agent


class ActorCritic(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, action_size: int, last_layer_factory: T.Callable[[int, int], torch.nn.Module]):
        super(ActorCritic, self).__init__()
        self.model = model
        last_layer = list(model.modules())[-1]
        if not isinstance(last_layer, (torch.nn.Linear, NoisyLinear)):
            raise ValueError("the model you have created must have a torch.nn.Linear or "
                             "agents.explorers.noisy_explorer.NoisyLinear in the last layer")

        if last_layer.out_features == action_size:
            print("WARNING: detected same number of features in the output of the model than the action space")

        self.actor = last_layer_factory(last_layer.out_features, action_size)
        self.critic = last_layer_factory(last_layer.out_features, 1)

    def forward(self, x):
        x = self.model(x)
        return F.softmax(self.actor(x), dim=1), self.critic(x)


class A2cAgent(Agent, ABC):
    hp: A2CHyperParams

    def __init__(self,
                 action_space: int,
                 hp: A2CHyperParams,
                 tp: TrainingParams,
                 explorer: T.Union[AnyExplorer, None],
                 replay_buffer: AnyReplayBuffer,
                 use_gpu: bool = True,
                 save_progress: bool = True,
                 tensor_board_log: bool = True):
        super(A2cAgent, self).__init__(action_space, hp, tp, explorer, replay_buffer,
                                       use_gpu, save_progress, tensor_board_log)

        self.actor_critic = self.build_actor_critic().to(self.device)
        self.actor_critic_optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=hp.lr)
        self.loss_f = torch.nn.MSELoss(reduction="none").to(self.device)

    def build_actor_critic(self) -> torch.nn.Module:
        return ActorCritic(self.build_model().to(self.device), self.action_space, self.last_layer_factory)

    def postprocess(self, t: torch.Tensor) -> np.ndarray:
        return np.array(t.squeeze(0))

    def infer(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.postprocess(self.actor_critic(self.preprocess(x).to(self.device))[0].cpu())

    def learn(self, batch: T.List[ReplayBufferEntry]) -> None:
        batch_s = torch.cat([self.preprocess(m.s) for m in batch], 0).to(self.device).requires_grad_(True)
        batch_s_ = torch.cat([self.preprocess(m.s_) for m in batch], 0).to(self.device)
        batch_a = torch.tensor([m.a for m in batch], device=self.device)
        batch_r = torch.tensor([[m.r] for m in batch], dtype=torch.float32, device=self.device)
        batch_finals = torch.tensor([[int(not m.final)] for m in batch], device=self.device)
        batch_weights = torch.tensor([m.weight for m in batch], device=self.device)

        action_probabilities, state_values = self.actor_critic(batch_s)
        with torch.no_grad():
            _, next_state_values = self.actor_critic(batch_s_)
            estimated_q_value: torch.Tensor = self.hp.gamma * next_state_values * batch_finals + batch_r

        advantages: torch.Tensor = (estimated_q_value - state_values.clone().detach()).squeeze(1)
        chosen_action_log_probabilities: torch.Tensor = torch.stack(
            [torch.distributions.Categorical(p).log_prob(a) for p, a in zip(action_probabilities, batch_a)])
        actor_loss: torch.Tensor = -chosen_action_log_probabilities * advantages * batch_weights
        critic_loss: torch.Tensor = self.loss_f(state_values, estimated_q_value) * batch_weights
        loss = (actor_loss + critic_loss).mean()
        self.actor_critic_optimizer.zero_grad()
        loss.backward()
        self.actor_critic_optimizer.step()
        self.call_learn_callbacks(LearningStep(batch, [v.item() for v in state_values], [v.item() for v in estimated_q_value]))

    def train(self, env: Environment) -> None:
        self.health_check(env)
        s = env.reset()
        i = 0
        episode = 1
        steps_survived = 0
        accumulated_reward = 0
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
