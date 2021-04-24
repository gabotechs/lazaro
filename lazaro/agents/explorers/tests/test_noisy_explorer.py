from ..noisy_explorer import NoisyExplorer, NoisyExplorerParams, NoisyLinear
import random
import torch
from . import tools


class Agent(NoisyExplorer, tools.Agent):
    pass


def test_generated_model_works():
    nep = NoisyExplorerParams(reset_noise_every=1, std_init=0.5, extra_layers=[2, 1])

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.linear = torch.nn.Linear(2, 2)

        def forward(self, x):
            return self.linear(x)

    test_agent = Agent(ep=nep)
    model = test_agent.last_layers_model_modifier(Model())
    assert len([l for l in model.modules() if isinstance(l, torch.nn.Linear)]) == 1
    assert len([l for l in model.modules() if isinstance(l, NoisyLinear)]) == 2

    op = torch.optim.Adam(model.parameters(), lr=0.005)

    bs, ep = 8, 6000
    x = torch.zeros((bs, 2), dtype=torch.float32)
    y = torch.zeros((bs, 1), dtype=torch.float32)
    for i in range(ep):
        for j in range(bs):
            a, b = 100*random.random(), 100*random.random()
            x[j, 0], x[j, 1] = a, b
            y[j][0] = a + b
        y_ = model(x)
        loss = (y - y_).pow(2).mean()
        op.zero_grad()
        loss.backward()
        op.step()
        if i % nep.reset_noise_every == 0:
            [l.reset_noise() for l in model.modules() if isinstance(l, NoisyLinear)]

    result = model(torch.Tensor([[2., 3.]])).squeeze(0).item()
    assert 4.5 < result < 5.5
    [l.reset_noise() for l in model.modules() if isinstance(l, NoisyLinear)]
    result = model(torch.Tensor([[2., 3.]])).squeeze(0).item()
    assert 4.5 < result < 5.5
