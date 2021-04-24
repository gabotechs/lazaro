import typing as T
from abc import ABC

import numpy as np
import torch

from ..environments import Environment
from .base.base_agent import BaseAgent
from .base.models import DqnHyperParams, TrainingParams, TrainingProgress, LearningStep, TrainingStep
from .explorers.noisy_explorer import NoisyLinear
from .replay_buffers import ReplayBufferEntry


class DqnNetwork(torch.nn.Module):
    def __init__(self,
                 model: torch.nn.Module,
                 action_space: int,
                 last_layer_factory: T.Callable[[int, int], torch.nn.Module]):
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


class DqnAgent(BaseAgent, ABC):
    def __init__(self,
                 action_space: int,
                 hp: DqnHyperParams = DqnHyperParams(),
                 use_gpu: bool = True):
        super(DqnAgent, self).__init__(action_space, hp, use_gpu)
        self.action_estimator = self.build_model().to(self.device)
        self.optimizer = torch.optim.Adam(self.action_estimator.parameters(), lr=hp.lr)
        self.loss_f = torch.nn.MSELoss(reduction="none").to(self.device)

    def get_state_dict(self) -> dict:
        return {"action_estimator": self.action_estimator.state_dict()}

    def get_info(self) -> dict:
        return {}

    def agent_specification_model_modifier(self, model: torch.nn.Module) -> torch.nn.Module:
        self.log.info("wrapping model with simple dqn layer")
        return DqnNetwork(model, self.action_space, self.last_layer_factory)

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
        y = torch.max(actions_expected_values, 1)[0] * self.hyper_params.gamma * batch_finals + batch_r
        element_wise_loss = self.loss_f(x, y)
        loss = (element_wise_loss * batch_weights).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.call_learn_callbacks(LearningStep(batch, [v.item() for v in x], [v.item() for v in y]))

    def train(self, env: Environment, tp: TrainingParams = TrainingParams()) -> None:
        self.health_check(env)
        s = env.reset()
        i = 0
        episode = 1
        steps_survived = 0
        accumulated_reward = 0
        while True:
            estimated_rewards = self.infer(s)
            a = self.ex_choose(list(estimated_rewards), lambda x: np.argmax(estimated_rewards).item())
            s_, r, final = env.step(a)
            self.rp_add(ReplayBufferEntry(s, s_, a, r, final))
            accumulated_reward += r
            s = s_

            self.call_step_callbacks(TrainingStep(i, episode))

            if i % self.hyper_params.learn_every == 0 and i != 0 and self.rp_get_length() >= tp.batch_size:
                batch = self.rp_sample(tp.batch_size)
                self.learn(batch)

            if final:
                training_progress = TrainingProgress(i, episode, steps_survived, accumulated_reward)
                must_exit = self.call_progress_callbacks(training_progress)
                if episode >= tp.episodes or must_exit:
                    return

                episode += 1
                accumulated_reward = 0
                steps_survived = 0
                s = env.reset()

            else:
                steps_survived += 1
            env.render()
            i += 1
