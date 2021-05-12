import typing as T
from abc import ABC

import numpy as np
import torch

from ..environments import Environment
from .base.base_agent import BaseAgent
from .base.models import DqnHyperParams, TrainingParams, TrainingProgress, LearningStep, TrainingStep, ReplayBufferEntry
from .explorers.noisy_explorer import NoisyLinear


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
                 agent_params: DqnHyperParams = DqnHyperParams(),
                 use_gpu: bool = True):
        super(DqnAgent, self).__init__(action_space, agent_params, use_gpu)
        self.action_estimator = self.build_model().to(self.device)
        self.optimizer = torch.optim.Adam(self.action_estimator.parameters(), lr=agent_params.lr)
        self.loss_f = torch.nn.MSELoss(reduction="none").to(self.device)

    def get_state_dict(self) -> dict:
        return {"action_estimator": self.action_estimator.state_dict()}

    def get_info(self) -> dict:
        return {}

    def agent_specification_model_modifier(self, model: torch.nn.Module) -> torch.nn.Module:
        self.log.info("wrapping model with simple dqn layer")
        return DqnNetwork(model, self.action_space, self.last_layer_factory)

    def infer(self, preprocessed: T.Union[torch.Tensor, T.Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        return self.action_estimator.forward(preprocessed)

    def postprocess(self, t: torch.Tensor) -> np.ndarray:
        return np.array(t.squeeze(0))

    def learn(self, entries: T.List[ReplayBufferEntry]) -> None:
        batch = self.form_learning_batch(entries)

        actions_estimated_values: torch.Tensor = self.action_estimator(batch.s)
        with torch.no_grad():
            actions_expected_values: torch.Tensor = self.action_estimator(batch.s_)

        x = torch.stack([t_s[t_a.item()] for t_s, t_a in zip(actions_estimated_values, batch.a)])
        y = torch.max(actions_expected_values, 1)[0] * self.agent_params.gamma * batch.final + batch.r
        element_wise_loss = self.loss_f(x, y)
        loss = (element_wise_loss * batch.weight).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.call_learn_callbacks(LearningStep(entries, [v.item() for v in x], [v.item() for v in y]))

    def train(self, env: Environment, tp: TrainingParams = None) -> None:
        if tp is None:
            tp = self.default_training_params
        self.health_check(env)
        s = env.reset()
        i = 0
        episode = 1
        steps_survived = 0
        accumulated_reward = 0
        while True:
            estimated_rewards = self.act(s)
            a = self.ex_choose(list(estimated_rewards), lambda x: np.argmax(estimated_rewards).item())
            s_, r, final = env.step(a)
            self.rp_add(ReplayBufferEntry(s, s_, a, r, final))
            accumulated_reward += r
            s = s_

            self.call_step_callbacks(TrainingStep(i, episode))

            if i % self.agent_params.learn_every == 0 and i != 0 and self.rp_get_length() >= tp.batch_size:
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
