from abc import ABC
import typing as T
import torch
import numpy as np

from environments import Environment
from .explorers import AnyExplorer
from .explorers.noisy_explorer import NoisyLinear
from .replay_buffers import ReplayBufferEntry, AnyReplayBuffer
from .base.models import DqnHyperParams, TrainingParams, TrainingProgress, LearningStep, TrainingStep
from .base.agent import Agent


class DqnNetwork(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, action_space: int, last_layer_factory: T.Callable[[int, int], torch.nn.Module]):
        super(DqnNetwork, self).__init__()
        self.model = model
        last_layer = list(model.modules())[-1]
        if not isinstance(last_layer, (torch.nn.Linear, NoisyLinear)):
            raise ValueError("the model you have created must have a torch.nn.Linear or "
                             "agents.explorers.noisy_explorer.NoisyLinear in the last layer")

        if last_layer.out_features == action_space:
            print("WARNING: detected same number of features in the output of the model than the action space")

        self.head = last_layer_factory(last_layer.out_features, action_space)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return self.head(x)


class DqnAgent(Agent, ABC):
    hp: DqnHyperParams
    network_class = DqnNetwork

    def __init__(self,
                 action_space: int,
                 hp: DqnHyperParams,
                 tp: TrainingParams,
                 explorer: T.Union[AnyExplorer, None],
                 replay_buffer: AnyReplayBuffer,
                 use_gpu: bool = True,
                 save_progress: bool = True,
                 tensor_board_log: bool = True):
        super(DqnAgent, self).__init__(action_space, hp, tp, explorer, replay_buffer,
                                       use_gpu, save_progress, tensor_board_log)

        self.action_estimator = self.build_model().to(self.device)
        self.optimizer = torch.optim.Adam(self.action_estimator.parameters(), lr=hp.lr)
        self.loss_f = torch.nn.MSELoss(reduction="none").to(self.device)

    def build_model(self) -> torch.nn.Module:
        model = super(DqnAgent, self).build_model()
        return self.network_class(model, self.action_space, self.last_layer_factory)

    def infer(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.postprocess(self.action_estimator.forward(self.preprocess(x).to(self.device)).cpu())

    def postprocess(self, t: torch.Tensor) -> np.ndarray:
        return np.array(t.squeeze(0))

    def learn(self, batch: T.List[ReplayBufferEntry]) -> None:
        batch_s = torch.cat([self.preprocess(m.s) for m in batch], 0).to(self.device).requires_grad_(True)
        batch_s_ = torch.cat([self.preprocess(m.s_) for m in batch], 0).to(self.device)
        batch_a = [m.a for m in batch]
        batch_r = torch.tensor([m.r for m in batch], dtype=torch.float32, device=self.device)
        batch_finals = torch.tensor([int(not m.final) for m in batch], device=self.device)
        batch_weights = torch.tensor([m.weight for m in batch], device=self.device)
        actions_estimated_values: torch.Tensor = self.action_estimator(batch_s)
        with torch.no_grad():
            actions_expected_values: torch.Tensor = self.action_estimator(batch_s_)

        x = torch.stack([t_s[t_a] for t_s, t_a in zip(actions_estimated_values, batch_a)])
        y = torch.max(actions_expected_values, 1)[0] * self.hp.gamma * batch_finals + batch_r
        element_wise_loss = self.loss_f(x, y)
        loss = (element_wise_loss * batch_weights).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.call_learn_callbacks(LearningStep(batch, [v.item() for v in x], [v.item() for v in y]))

    def train(self, env: Environment) -> None:
        s = env.reset()
        i = 0
        episode = 1
        steps_survived = 0
        accumulated_reward = 0
        is_healthy = False
        while True:
            estimated_rewards = self.infer(s)
            a = self.explorer.choose(estimated_rewards, lambda x: np.argmax(estimated_rewards).item())
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

                episode += 1
                accumulated_reward = 0
                steps_survived = 0
                s = env.reset()

            else:
                steps_survived += 1
            env.render()
            i += 1
