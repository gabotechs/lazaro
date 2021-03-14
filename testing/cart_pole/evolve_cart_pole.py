import numpy as np
import torch
import torch.nn.functional as F

import agents as lz
from environments import CartPole
from evolutioners import Evolutioner, T_EParams, EvolutionerParams, EvolutionProgress
from evolutioners.models import EvolvingFloat, EvolvingInt


env = CartPole()
env.visualize = False


class CustomNN(torch.nn.Module):
    def __init__(self, in_size: int, linear_1: int, linear_2: int):
        super(CustomNN, self).__init__()
        self.linear1 = torch.nn.Linear(in_size, linear_1)
        self.linear2 = torch.nn.Linear(linear_1, linear_2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear2(F.relu(self.linear1(x))))


class CustomAgent(lz.DoubleDuelingDqnAgent):
    def __init__(self, *args, **kwargs):
        self.hidden_1 = 6
        self.hidden_2 = 23
        super(CustomAgent, self).__init__(*args, **kwargs)

    def model_factory(self) -> torch.nn.Module:
        return CustomNN(env.get_observation_space()[0], self.hidden_1, self.hidden_2)

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        return torch.unsqueeze(torch.tensor(x, dtype=torch.float32), 0)


evolve_params: T_EParams = {
    "lr": EvolvingFloat(0.0025, 1e-6, 1e-1, 0.0005),
    "batch_size": EvolvingInt(46, 8, 512, 20),
    "memory_len": EvolvingInt(11246, 256, 20000, 1000),
}


class CustomEvolutioner(Evolutioner):
    def agent_factory(self, params: T_EParams, state_dict: dict) -> lz.AnyAgent:
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
            use_gpu=True
        )
        if state_dict:
            agent.action_estimator.load_state_dict(state_dict["action_estimator"], strict=False)
            agent.action_evaluator.load_state_dict(agent.action_estimator.state_dict())
        return agent


if __name__ == "__main__":
    evolutioner = CustomEvolutioner(
        env,
        evolve_params,
        EvolutionerParams(generation_size=10, workers=4),
    )

    def on_progress(progress: EvolutionProgress):
        gen_msg = f"===== ended generation {progress.generation} ====="
        print(gen_msg)
        print("best params: ", {k: v.value for k, v in progress.params[progress.best_index].items()})
        print("result:      ", progress.results[progress.best_index])
        print("="*len(gen_msg))

    evolutioner.set_progress_callback(on_progress)
    evolutioner.evolve(lambda x: False)
