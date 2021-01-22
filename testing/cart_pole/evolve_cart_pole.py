import numpy as np
import torch

from agents import ActorCriticAgent, A2CHyperParams, TrainingParams, TrainingProgress
from agents.replay_buffers import RandomReplayBuffer
from environments import CartPole
from evolutioners import Evolutioner, T_EParams, EvolutionerParams, EvolutionProgress
from evolutioners.models import EvolvingFloat, EvolvingInt

EVOLUTIONER_PARAMS = EvolutionerParams(generation_size=10, max_allowed_mutation=0.9, workers=4)


env = CartPole()


class CustomActionEstimator(torch.nn.Module):
    def __init__(self, in_size: int, hidden_1: int, hidden_2: int, out_size: int):
        super(CustomActionEstimator, self).__init__()
        self.out_size = out_size
        self.linear1 = torch.nn.Linear(in_size, in_size*hidden_1)
        self.relu1 = torch.nn.ReLU()

        self.linear2 = torch.nn.Linear(in_size*hidden_1, out_size*hidden_2)
        self.relu2 = torch.nn.ReLU()

        self.linear3 = torch.nn.Linear(out_size*hidden_2, out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        return self.linear3(x)


class CustomActorCriticAgent(ActorCriticAgent):
    def __init__(self, *args, hidden_1,  hidden_2, **kwargs):
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        super(CustomActorCriticAgent, self).__init__(*args, **kwargs)

    def model_factory(self) -> torch.nn.Module:
        return CustomActionEstimator(env.get_observation_space()[0], self.hidden_1, self.hidden_2, len(env.get_action_space()))

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        return torch.unsqueeze(torch.tensor(x, dtype=torch.float32), 0)


evolve_params: T_EParams = {
    "hidden_1": EvolvingInt(10, 3, 50, 0.3),
    "hidden_2": EvolvingInt(100, 10, 500, 0.3),
    "c_lr": EvolvingFloat(7.604609807959665e-05, 1e-6, 1e-1, 0.8),
    "a_lr": EvolvingFloat(0.0002442092598528454, 1e-6, 1e-1, 0.8),
    "batch_size": EvolvingInt(63, 8, 512, 0.8),
    "memory_len": EvolvingInt(11434, 256, 20000, 2.0),
}


def agent_finish_training_condition(x: TrainingProgress):
    return x.episode >= 50


def agent_factory(params: T_EParams) -> ActorCriticAgent:
    agent_params = A2CHyperParams(
        c_lr=params["c_lr"].value,
        a_lr=params["a_lr"].value,
        gamma=0.995
    )
    training_params = TrainingParams(
        learn_every=1,
        ensure_every=10,
        batch_size=params["batch_size"].value,
        finish_condition=agent_finish_training_condition
    )

    memory_len = params["memory_len"].value

    agent = CustomActorCriticAgent(
        agent_params,
        training_params,
        None,
        RandomReplayBuffer(memory_len),
        use_gpu=True,
        hidden_1=params["hidden_1"].value,
        hidden_2=params["hidden_2"].value
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
        print("best params: ", {k: v.value for k, v in progress.params[progress.best_index].items()})
        print("result:      ", progress.results[progress.best_index])
        print("="*len(gen_msg))

    evolutioner.set_progress_callback(on_progress)
    evolutioner.evolve(lambda x: False)
