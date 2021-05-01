import lazaro as lz
import game
from lazaro.agents import AnyAgent
from lazaro.evolutioners import T_EParams
from model import CustomNN
import numpy as np
import torch
import typing as T


class CustomAgent(lz.agents.explorers.NoisyExplorer,
                  lz.agents.replay_buffers.NStepsPrioritizedReplayBuffer,
                  lz.agents.DoubleDuelingDqnAgent):
    def model_factory(self):
        return CustomNN(game.SnakeEnv.SHAPE[0], game.SnakeEnv.SHAPE[1], game.SnakeEnv.CHANNELS)

    def preprocess(self, x):
        x = np.array([[[int(i == int(cell)) for i in range(4)] for cell in row] for row in x]).transpose((2, 0, 1))
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0)


env = game.SnakeEnv()


evolve_params: T_EParams = {
    "lr": lz.evolutioners.EvolvingFloat(0.0025, 1e-6, 1e-1, 0.0005),
    "batch_size": lz.evolutioners.EvolvingInt(46, 8, 512, 20),
    "memory_len": lz.evolutioners.EvolvingInt(11246, 256, 20000, 1000),
}


class SnakeEvolutioner(lz.evolutioners.Evolutioner):
    def agent_factory(self, params: T_EParams, state_dict: T.Optional[dict]) -> AnyAgent:
        hp = lz.agents.DoubleDuelingDqnHyperParams(
            lr=params["lr"].value,
            gamma=.99,
            ensure_every=10,
            learn_every=1
        )
        ep = lz.agents.explorers.NoisyExplorerParams(
            extra_layers=[],
            reset_noise_every=1,
            std_init=.5
        )
        rp = lz.agents.replay_buffers.NStepPrioritizedReplayBufferParams(
            n_step=4,
            alpha=.6,
            init_beta=.4,
            final_beta=1,
            increase_beta=1e-4,
            max_len=params["memory_len"].value
        )
        tp = lz.agents.TrainingParams(
            batch_size=params["batch_size"].value,
            episodes=10000
        )
        agent = CustomAgent(action_space=len(env.get_action_space()), hp=hp, ep=ep, rp=rp)
        agent.default_training_params = tp
        return agent


if __name__ == '__main__':
    env.visualize = False
    evoultioner = SnakeEvolutioner(
        env,
        evolve_params,
        lz.evolutioners.EvolutionerParams(generation_size=50, workers=10)
    )

    def on_progress(progress: lz.evolutioners.EvolutionProgress):
        gen_msg = f"===== ended generation {progress.generation} ====="
        print(gen_msg)
        print("best params: ", {k: v.value for k, v in progress.params[progress.best_index].items()})
        print("result:      ", progress.results[progress.best_index])
        print("="*len(gen_msg))

    agent = evoultioner.agent_factory(evolve_params, {})
    env.visualize = True
    agent.train(env)
