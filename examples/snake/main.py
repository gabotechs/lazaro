import lazaro as lz

from game import SnakeEnv
from lazaro.agents import AnyAgent
from lazaro.evolutioners import T_EParams

import torch
import torch.nn.functional as F
import typing as T


class Conv2d(torch.nn.Conv2d):
    def _get_out_size(self, value: int, ind: int):
        return (value + 2*self.padding[ind] - self.dilation[ind]*(self.kernel_size[ind] - 1) - 1) // self.stride[ind] + 1

    def out_wh(self, in_wh: T.Tuple[int, int]):
        return self._get_out_size(in_wh[0], 0), self._get_out_size(in_wh[1], 1)


class CustomNN(torch.nn.Module):
    def __init__(self, params: T_EParams):
        super(CustomNN, self).__init__()
        k_s_1 = params["kernel_size_1"].value
        self.conv1 = Conv2d(SnakeEnv.CHANNELS, 16, stride=(1, 1), kernel_size=(k_s_1, k_s_1), padding=(int(k_s_1/2), int(k_s_1/2)))
        h, w = self.conv1.out_wh(SnakeEnv.SHAPE)
        k_s_2 = params["kernel_size_2"].value
        self.conv2 = Conv2d(self.conv1.out_channels, 32, stride=(1, 1), kernel_size=(k_s_2, k_s_2), padding=(int(k_s_2/2), int(k_s_2/2)))
        h, w = self.conv2.out_wh((h, w))
        k_s_3 = params["kernel_size_3"].value
        self.conv3 = Conv2d(self.conv1.out_channels, 64, stride=(1, 1), kernel_size=(k_s_3, k_s_3), padding=(int(k_s_3 / 2), int(k_s_3 / 2)))
        self.linear = torch.nn.Linear(self.conv2.out_channels * h * w, params["linear_out"].value)

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.reshape(x, (batch_size, -1))
        return F.relu(self.linear(x))


class CustomAgent(lz.agents.explorers.NoisyExplorer,
                  lz.agents.replay_buffers.NStepsPrioritizedReplayBuffer,
                  lz.agents.DoubleDqnAgent):
    def __init__(self, params: T_EParams, *args, **kwargs):
        self.params = params
        super(CustomAgent, self).__init__(*args, **kwargs)

    def model_factory(self):
        return CustomNN(self.params)

    def preprocess(self, x):
        categorized_state = [[[int(i == int(cell)) for i in range(SnakeEnv.CHANNELS)] for cell in row] for row in x]
        x = torch.tensor(categorized_state, dtype=torch.float32)
        return x.transpose(1, 2).transpose(0, 1).unsqueeze(0)


evolve_params: T_EParams = {
    "gamma": lz.evolutioners.EvolvingFloat(0.96, 0.9, 0.999, 0.01),
    "lr": lz.evolutioners.EvolvingFloat(0.001, 1e-6, 1e-1, 0.0005),
    "batch_size": lz.evolutioners.EvolvingInt(13, 8, 512, 20),
    "memory_len": lz.evolutioners.EvolvingInt(7500, 256, 20000, 1000),
    "kernel_size_1": lz.evolutioners.EvolvingInt(3, 1, 5, 1),
    "kernel_size_2": lz.evolutioners.EvolvingInt(3, 1, 4, 1),
    "kernel_size_3": lz.evolutioners.EvolvingInt(3, 1, 4, 1),
    "linear_out": lz.evolutioners.EvolvingInt(512, 16, 2048, 200),
}


class SnakeEvolutioner(lz.evolutioners.Evolutioner):
    def agent_factory(self, params: T_EParams, state_dict: T.Optional[dict]) -> AnyAgent:
        hp = lz.agents.PpoHyperParams(
            lr=params["lr"].value,
            gamma=params["gamma"].value,
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
        agent = CustomAgent(params=params, action_space=4, hp=hp, ep=ep, rp=rp)
        agent.default_training_params = tp
        return agent


if __name__ == '__main__':
    env = SnakeEnv()
    env.visualize = False
    evoultioner = SnakeEvolutioner(
        env,
        evolve_params,
        lz.evolutioners.EvolutionerParams(generation_size=10, workers=4)
    )

    def on_progress(progress: lz.evolutioners.EvolutionProgress):
        gen_msg = f"===== ended generation {progress.generation} ====="
        print(gen_msg)
        print("best params: ", {k: v.value for k, v in progress.params[progress.best_index].items()})
        print("result:      ", progress.results[progress.best_index])
        print("="*len(gen_msg))

    evoultioner.set_progress_callback(on_progress)
    evoultioner.evolve(lambda x: False)
