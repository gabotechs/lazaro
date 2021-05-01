import typing as T

import torch
import torch.nn.functional as F

from plotter import Renderer


class Conv2d(torch.nn.Conv2d):
    def _get_out_size(self, value: int, ind: int):
        return (value + 2*self.padding[ind] - self.dilation[ind]*(self.kernel_size[ind] - 1) - 1) // self.stride[ind] + 1

    def out_wh(self, in_wh: T.Tuple[int, int]):
        return self._get_out_size(in_wh[0], 0), self._get_out_size(in_wh[1], 1)


class CustomNN(torch.nn.Module):
    RENDER_EVERY = 0

    def __init__(self, w: int, h: int, c: int):
        super(CustomNN, self).__init__()
        self.conv1 = Conv2d(c, 16, stride=(1, 1), kernel_size=(3, 3), padding=(2, 2))
        (h, w) = self.conv1.out_wh((h, w))
        self.conv2 = Conv2d(16, 32, stride=(1, 1), kernel_size=(3, 3), padding=(2, 2))
        (h, w) = self.conv2.out_wh((h, w))
        self.linear = torch.nn.Linear(self.conv2.out_channels*h*w, self.conv2.out_channels*h*w)

        self.renderers = {}
        self._register_conv_hooks()

    def _register_conv_hooks(self):
        for attr, value in self.__dict__["_modules"].items():
            if isinstance(value, torch.nn.Conv2d):
                self.renderers[attr] = Renderer(attr, value.out_channels, self.RENDER_EVERY)
                value.register_forward_hook(self.renderers[attr].conv_forward_hook)

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.reshape(x, (batch_size, -1))
        return F.relu(self.linear(x))
