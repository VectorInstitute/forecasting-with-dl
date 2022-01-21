from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class MultiInputSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if isinstance(inputs, tuple):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class DilatedBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        h_dim: int,
        w_dim: int,
        dilation: int,
        scale_bias_index: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.h_dim = h_dim
        self.w_dim = w_dim
        self.dilation = dilation
        self.scale_bias_index = scale_bias_index
        self.cn1 = nn.Conv2d(in_channels=self.in_channels,
                             out_channels=self.in_channels,
                             kernel_size=(3, 3),
                             stride=1,
                             padding=self.dilation,
                             dilation=self.dilation,
                             bias=True)
        self.cn2 = nn.Conv2d(in_channels=self.in_channels,
                             out_channels=self.in_channels,
                             kernel_size=(3, 3),
                             stride=1,
                             padding=self.dilation,
                             dilation=self.dilation,
                             bias=True)
        self.layer_norm = nn.LayerNorm(
            [self.in_channels, self.h_dim, self.w_dim])
        self.relu = nn.ReLU()

    def forward(self, x: Tensor, scale_bias: Tensor) -> Tuple[Tensor, Tensor]:
        assert scale_bias.shape[0] == 2
        scale = scale_bias[self.scale_bias_index * 2]
        bias = scale_bias[self.scale_bias_index * 2 + 1]
        identity = x
        x = self.cn1(x)
        x = self.layer_norm(x)
        x = x * scale + bias
        x = self.relu(x)
        x = self.cn2(x)
        x = self.layer_norm(x)
        x = x * scale + bias
        x = self.relu(x)
        out = x + identity
        return out, scale_bias


class DilatedEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dilations: list,
        h_dim: int,
        w_dim: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.dilations = dilations
        self.h_dim = h_dim
        self.w_dim = w_dim

        self.layers = self._make_layers()

    def _make_layers(self):
        layers = []
        for i, dl in enumerate(self.dilations):
            layers.append(
                DilatedBlock(self.in_channels, self.h_dim, self.w_dim, dl, i))
        return MultiInputSequential(*layers)

    def forward(self, x: Tensor, scale_bias: Tensor) -> Tensor:
        out = self.layers(x, scale_bias)
        return out

"""
import matplotlib.pyplot as plt
import numpy as np
model = DilatedEncoder(3, 32, 32)
with torch.no_grad():
    for name, param in model.named_parameters():
        if param.requires_grad is True:
            param.fill_(1)

x = torch.ones(1, 3, 32, 32)
output = model(x)
plt.figure()
plt_ten = output.permute(0, 2, 3, 1).detach().numpy()
plt_ten = ((plt_ten - plt_ten.min()) * (255 / (plt_ten.max() - plt_ten.min()))).astype(np.uint8)
plt.imshow(plt_ten[0])
plt.colorbar()
import pdb; pdb.set_trace()
"""
