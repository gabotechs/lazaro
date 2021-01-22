import math
import typing as T
import torch
import torch.nn.functional as F
import numpy as np

from .base.explorer import Explorer
from .base.models import NoisyExplorerParams


class NoisyLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, std_init: float):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = torch.nn.Parameter(torch.empty(out_features))
        self.bias_sigma = torch.nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=size)).to(self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self) -> None:
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x,
                        self.weight_mu + self.weight_sigma * self.weight_epsilon,
                        self.bias_mu + self.bias_sigma * self.bias_epsilon)


class ModelWithNoisyLayers(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, noisy_layers: T.List[torch.nn.Module]):
        super(ModelWithNoisyLayers, self).__init__()
        self.model = model
        self.noisy_layers = torch.nn.Sequential(*noisy_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x = self.noisy_layers(x)
        return x


class NoisyExplorer(Explorer):
    def __init__(self, ep: NoisyExplorerParams = NoisyExplorerParams()):
        self.ep: NoisyExplorerParams = ep

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        last_layer = list(model.modules())[-1]
        if not isinstance(last_layer, torch.nn.Linear):
            raise ValueError("the model you have created must have a torch.nn.Linear in the last layer")
        if len(self.ep.extra_layers) == 0:
            return model
        noisy_layers = [NoisyLinear(last_layer.out_features, self.ep.extra_layers[0], self.ep.std_init)]
        for i in range(1, len(self.ep.extra_layers)):
            noisy_layers.append(torch.nn.ReLU())
            noisy_layers.append(NoisyLinear(self.ep.extra_layers[i - 1], self.ep.extra_layers[i], self.ep.std_init))

        return ModelWithNoisyLayers(model, noisy_layers)

    def choose(self, actions: np.ndarray, f: T.Callable[[np.ndarray], int]) -> int:
        return f(actions)
