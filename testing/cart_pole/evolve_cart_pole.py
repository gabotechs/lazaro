import numpy as np
import torch

from agents import ActorCriticAgent, ACHyperParams, TrainingParams, TrainingProgress
from agents.replay_buffers import RandomReplayBuffer
from environments import CartPole
from evolutioners import Evolutioner, T_EParams, EvolutionerParams, EvolutionProgress


EVOLUTIONER_PARAMS = EvolutionerParams(generation_size=4, max_allowed_mutation=0.5, workers=4)


env = CartPole()


class CustomActionEstimator(torch.nn.Module):
    def __init__(self, in_size: int, out_size: int):
        super(CustomActionEstimator, self).__init__()
        self.out_size = out_size
        self.linear1 = torch.nn.Linear(in_size, in_size*10)
        self.relu1 = torch.nn.ReLU()

        self.linear2 = torch.nn.Linear(in_size*10, out_size*100)
        self.relu2 = torch.nn.ReLU()

        self.linear3 = torch.nn.Linear(out_size*100, out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        return self.linear3(x)


class CustomActorCriticAgent(ActorCriticAgent):
    @staticmethod
    def model_factory() -> torch.nn.Module:
        return CustomActionEstimator(env.get_observation_space()[0], len(env.get_action_space()))

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        return torch.unsqueeze(torch.tensor(x, dtype=torch.float32), 0)


evolve_params: T_EParams = {
    "c_lr": 0.001,
    "a_lr": 0.0001,
    "batch_size": 128,
    "memory_len": 1000
}


def agent_finish_training_condition(x: TrainingProgress):
    return x.tries >= 50


def agent_factory(params: T_EParams) -> ActorCriticAgent:
    agent_params = ACHyperParams(
        c_lr=params["c_lr"],
        a_lr=params["a_lr"],
        gamma=0.995
    )
    training_params = TrainingParams(
        learn_every=1,
        ensure_every=10,
        batch_size=params["batch_size"],
        finish_condition=agent_finish_training_condition
    )

    memory_len = params["memory_len"]

    agent = CustomActorCriticAgent(
        agent_params,
        training_params,
        None,
        RandomReplayBuffer(memory_len),
        use_gpu=True
    )

    return agent


if __name__ == "__main__":
    evolutioner = Evolutioner(
        env,
        EVOLUTIONER_PARAMS,
        evolve_params,
        agent_factory
    )

    def on_progress(progress: EvolutionProgress):
        gen_msg = f"===== ended generation {progress.generation} ====="
        print(gen_msg)
        print("best params: ", progress.params[progress.best_index])
        print("result:      ", progress.results[progress.best_index])
        print("="*len(gen_msg))

    evolutioner.set_progress_callback(on_progress)
    evolutioner.evolve(lambda x: False)
