<p align="center">
    <img height="200" src="./docs/lazaro.svg">
</p>
<p align="center">
    <img src="https://github.com/GabrielMusat/lazaro/actions/workflows/test.yml/badge.svg">
    <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg"/>
</p>

Reinforcement learning framework based on Pytorch for implementing custom deep learning models
on custom environments using state of the art and modular reinforcement learning algorithms

# Installation

Lazaro package is available on PyPI for Python versions greater or equal to 3.7.
```shell
pip install lazaro
```

# Usage
Import torch and lazaro packages
```python
import torch
import torch.nn.functional as F
import lazaro as lz
```

Choose an environment to train your model. You can create one yourself or choose one from the 
environments repository.
```python
env = lz.environments.CartPole()
OBSERVATION_SPACE = 4  # position, speed, angle, rotation speed 
ACTION_SPACE = 2  # move right, move left
```

Create your custom model using PyTorch.

```python
class CustomModel(torch.nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        # your last layer can have an arbitrary number of output neurons, 
        # lazaro will handle the rest of the model for you
        self.linear = torch.nn.Linear(OBSERVATION_SPACE, 30) 

    def forward(self, x):
        return F.relu(self.linear(x))
```

Create your agent. You can select any agent and mix it with any exploration policy and any replay buffer
algorithm.

```python
# mix any agent with any exploration policy and any replay buffer
class CustomAgent(lz.agents.explorers.NoisyExplorer,
                  lz.agents.replay_buffers.NStepsPrioritizedReplayBuffer,
                  lz.agents.DoubleDuelingDqnAgent):  
    # implement the model factory. It will be used to embed your torch model into the agent
    def model_factory(self):
        return CustomModel()  
```

Instantiate your agent and train it in your environment.

```python
agent = CustomAgent(action_space=ACTION_SPACE)
agent.train(env)
```

<p align="center">
    <img src="docs/cartpole.gif">
</p>