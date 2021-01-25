import numpy as np
import torch
import torch.nn.functional as F

import agents as lz
from environments import CartPole
from evolutioners import Evolutioner, T_EParams, EvolutionerParams, EvolutionProgress
from evolutioners.models import EvolvingFloat, EvolvingInt


env = CartPole()


class CustomNN(torch.nn.Module):
    def __init__(self, in_size: int, linear_1: int, linear_2: int):
        super(CustomNN, self).__init__()
        self.linear1 = torch.nn.Linear(in_size, linear_1)
        self.linear2 = torch.nn.Linear(linear_1, linear_2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear2(F.relu(self.linear1(x))))


class CustomAgent(lz.DoubleDuelingDqnAgent):
    def __init__(self, *args, hidden_1,  hidden_2, **kwargs):
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        super(CustomAgent, self).__init__(*args, **kwargs)

    def model_factory(self) -> torch.nn.Module:
        return CustomNN(env.get_observation_space()[0], self.hidden_1, self.hidden_2)

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        return torch.unsqueeze(torch.tensor(x, dtype=torch.float32), 0)


evolve_params: T_EParams = {
    "hidden_1": EvolvingInt(10, 3, 50, 0.3),
    "hidden_2": EvolvingInt(100, 10, 500, 0.3),
    "lr": EvolvingFloat(0.0002442092598528454, 1e-6, 1e-1, 0.8),
    "batch_size": EvolvingInt(63, 8, 512, 0.8),
    "memory_len": EvolvingInt(11434, 256, 20000, 2.0),
}


def agent_factory(params: T_EParams) -> lz.AnyAgent:
    agent = CustomAgent(
        len(env.get_action_space()),
        lz.explorers.RandomExplorer(),
        lz.replay_buffers.RandomReplayBuffer(lz.replay_buffers.RandomReplayBufferParams(
            max_len=params["memory_len"].value
        )),
        lz.TrainingParams(
            batch_size=params["batch_size"].value,
            episodes=50
        ),
        lz.DoubleDuelingDqnHyperParams(
            learn_every=1,
            lr=params["lr"].value,
            ensure_every=10
        ),
        use_gpu=True,
        hidden_1=params["hidden_1"].value,
        hidden_2=params["hidden_2"].value
    )

    return agent


if __name__ == "__main__":
    evolutioner = Evolutioner(
        env,
        EvolutionerParams(
            generation_size=10,
            max_allowed_mutation=0.9,
            workers=4
        ),
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
