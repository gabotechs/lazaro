#Lazaro

Minimalist reinforcement learning framework based on Pytorch.

```python
import torch
import torch.nn.functional as F
import agents as lz
from environments import CartPole

env = CartPole()


class CustomNN(torch.nn.Module):
    def __init__(self):
        super(CustomNN, self).__init__()
        self.linear = torch.nn.Linear(4, 30)

    def forward(self, x):
        return F.relu(self.linear(x))


class CustomAgent(lz.explorers.RandomExplorer,
                  lz.replay_buffers.PrioritizedReplayBuffer,
                  lz.DoubleDuelingDqnAgent):
    def model_factory(self):
        return CustomNN()


agent = CustomAgent(action_space=2)
agent.train(env)
```

![](docs/cartpole.gif)
